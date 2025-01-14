import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.models import resnet18
import torch.nn as nn
import argparse
import torchaudio
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import *
from src.training_utils import *
from src.model import ResNet_MelSpec, ResNet_LogSpec, ResNet_MFCC, ResNet_LFCC

import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config, device):

    logger.info('Set up the model...')

    if config['binary_classification']:
        config['model_name'] = f"RESNET_binary_{config['feature_set']}.pth"
        config['num_classes'] = 2
    else:
        config['model_name'] = f"RESNET_multi_{config['feature_set']}.pth"
        config['num_classes'] = 8

    models_dict = {
        'MelSpec': ResNet_MelSpec,
        'LogSpec': ResNet_LogSpec,
        'MFCC': ResNet_MFCC,
        'LFCC': ResNet_LFCC
    }

    if models_dict.get(config['feature_set']) is not None:
        model = models_dict.get(config['feature_set'])(config)
    else:
        raise ValueError(f"Unknown feature set: {config['feature_set']}")

    model = (model).to(device)
    model.apply(init_weights)

    if os.path.exists(os.path.join(config['save_model_folder'], config['model_name'])):
        model.load_state_dict(
            torch.load(os.path.join(config['save_model_folder'], config['model_name']), map_location=device))
        logger.info('Model loaded : {}'.format(config['model_name']))
    else:
        logger.info('No pretrained model loaded')


    logger.info('Load training and test data...')

    df = pd.read_csv(config['partition_file'], sep='\t', header=None)
    df = df.rename(columns={0: 'filename', 1: 'partition'})
    df[['patient', 'rec_idx', 'loc', 'ac_mode', 'setup']] = df['filename'].str.split('_', expand=True)
    df['patient'] = df['patient'].astype(int)

    df['audio_path'] = df['filename'].apply(lambda x: f'{config["dataset_folder"]}/{x}.wav')

    df_label = pd.read_csv(config['diagnosis_file'], sep='\t', header=None)
    df_label = df_label.rename(columns={0: 'patient', 1: 'diagnosis'})
    label_dict = {'Healthy': 0, 'COPD': 1, 'URTI': 2, 'Asthma': 3, 'LRTI': 4, 'Bronchiectasis': 5, 'Pneumonia': 6, 'Bronchiolitis': 7}
    df_label['label'] = df_label['diagnosis'].map(label_dict)

    if config['binary_classification']:
        df_label['label'] = df_label['label'].apply(lambda x: 1 if x != 0 else 0)

    df_merge = pd.merge(df, df_label, on='patient', how='left')

    df_train = df_merge[df_merge['partition'] == 'train']
    df_test = df_merge[df_merge['partition'] == 'test']

    # Handle classes with too few samples
    class_counts = df_train['label'].value_counts()
    to_duplicate = class_counts[class_counts < 2].index
    for class_val in to_duplicate:
        df_train = pd.concat([df_train, df_train[df_train['label'] == class_val]], ignore_index=True)

    df_train, df_val = train_test_split(df_train, test_size=0.25, stratify=df_train['label'], random_state=config['seed'])

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)


    if config['train_model']:

        logger.info('Define training and validation dataloaders...')

        d_label_trn = dict(zip(df_train['audio_path'], df_train['label']))
        file_train = list(df_train['audio_path'])

        train_set = LoadTrainData(list_IDs=file_train, labels=d_label_trn, win_len=config['win_len'])
        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True,
                                  num_workers=config['num_workers'])
        del train_set, d_label_trn

        d_label_val = dict(zip(df_val['audio_path'], df_val['label']))
        file_val = list(df_val['audio_path'])

        val_set = LoadTrainData(list_IDs=file_val, labels=d_label_val, win_len=config['win_len'])
        val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        del val_set, d_label_val

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)


        logger.info('Start training...')

        best_acc = 0
        best_loss = 100
        early_stopping = 0
        writer = SummaryWriter(log_dir=config['tensorboard_logdir'])

        for epoch in range(config['num_epochs']):
            if early_stopping < config['early_stopping']:

                start_time = time.time()

                running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, criterion, device)
                with torch.no_grad():
                    valid_loss, valid_accuracy = valid_epoch(val_loader, model, criterion, device)

                elaps_time = time.time() - start_time

                scheduler.step(valid_loss)
                logger.info(f'Epoch: {epoch} - Train Acc: {elaps_time:.2f} - Train Loss: {running_loss:.5f} - Val Loss: '
                            f'{valid_loss:.5f} - Train Acc: {train_accuracy:.2f} - Val Acc: {valid_accuracy:.2f}')

                writer.add_scalar('Elaps_Time', elaps_time, epoch)
                writer.add_scalar('Loss/Train', running_loss, epoch)
                writer.add_scalar('Loss/Validation', valid_loss, epoch)
                writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
                writer.add_scalar('Accuracy/Validation', valid_accuracy, epoch)

                if valid_loss < best_loss:
                    logger.info(f'Best model found at epoch {epoch}')
                    torch.save(model.state_dict(), os.path.join(config['save_model_folder'], config['model_name']))
                    early_stopping = 0

                    best_loss = min(valid_loss, best_loss)
                    best_acc = valid_accuracy
                else:
                    early_stopping += 1
            else:
                logger.info(f'Training stopped after {epoch} epochs - Best Val Acc {best_acc:.2f}')
                break

        writer.close()

    if config['eval_model']:

        logger.info('Evaluate model...')

        model.load_state_dict(
            torch.load(os.path.join(config['save_model_folder'], config['model_name']), map_location=device))
        logger.info('EVALUATION - Model loaded : {}'.format(config['model_name']))

        config['result_path'] = config['model_name'].replace('.pth', '.txt')

        if os.path.exists(os.path.join(config['save_results_folder'], config['result_path'])):
            logger.info("Save path exists - Deleting file")
            os.remove(os.path.join(config['save_results_folder'], config['result_path']))

        d_label_eval = dict(zip(df_test['audio_path'], df_test['label']))
        file_eval = list(df_test['audio_path'])

        eval_set = LoadEvalData(list_IDs=file_eval, labels=d_label_eval, win_len=config['win_len'])
        eval_loader = DataLoader(eval_set, batch_size=config['eval_batch_size'], shuffle=True, drop_last=False,
                                 num_workers=config['num_workers'])
        del eval_set, d_label_eval

        eval_model(model, eval_loader, os.path.join(config['save_results_folder'], config['result_path']), device)



if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--feature_set', type=str, default='MelSpec', help='Feature set to use for training')
    args.add_argument('--train_model', type=bool, default=True)
    args.add_argument('--eval_model', type=bool, default=True)
    args.add_argument('--binary_classification', type=bool, default=True, help='Binary or multi-class classification')
    args = args.parse_args()

    this_folder = Path(__file__).parent

    config_path = this_folder / 'config' / 'resnet_config.yaml'
    config = read_yaml(config_path)

    config['feature_set'] = args.feature_set
    config['train_model'] = args.train_model
    config['eval_model'] = args.eval_model
    config['binary_classification'] = args.binary_classification

    seed_everything(config['seed'])
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

    # main(config, device)

    config['binary_classification'] = False

    config['feature_set'] = 'LogSpec'
    main(config, device)

    config['feature_set'] = 'MFCC'
    main(config, device)

    config['feature_set'] = 'LFCC'
    main(config, device)

    config['feature_set'] = 'MelSpec'
    main(config, device)


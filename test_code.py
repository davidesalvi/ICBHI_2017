import logging
from pathlib import Path

from torch.utils.data import DataLoader

from src.model import *
from src.training_utils import *
from src.utils import *
from visualize_results import *


def main(config, device):

    config['model_name'] = f"LCNN_binary_MelSpec_10.0sec.pth"
    model = MelSpec_model(config)
    model = (model).to(device)

    model.load_state_dict(
        torch.load(os.path.join(config['save_model_folder'], config['model_name']), map_location=device))

    df_test = pd.read_csv('test_audio/test_data.csv')

    config['result_path'] = 'test_audio/test_results.txt'

    if os.path.exists(config['result_path']):
        os.remove(config['result_path'])

    d_label_eval = dict(zip(df_test['audio_path'], df_test['label']))
    file_eval = list(df_test['audio_path'])

    eval_set = LoadEvalData(list_IDs=file_eval, labels=d_label_eval, win_len=config['win_len'])
    eval_loader = DataLoader(eval_set, batch_size=config['eval_batch_size'], shuffle=True, drop_last=False,
                             num_workers=config['num_workers'])
    del eval_set, d_label_eval

    eval_model(model, eval_loader, config['result_path'], device, config)

    # ANALYZE RESULTS
    df = pd.read_csv(config['result_path'], sep=' ', header=None)
    df = df.rename(columns={0: 'filename', 1: 'pred', 2: 'label'})

    plt.figure(figsize=(6,6))
    plot_roc_curve(df['label'], df['pred'], legend='MelSpec')
    plt.show()

    eer, rocauc = compute_eer_auc(df['label'], df['pred'])
    print(f"EER: {eer:.2f} - ROC-AUC: {rocauc:.2f}")



if __name__ == '__main__':

    this_folder = Path(__file__).parent

    config_path = this_folder / 'config' / 'resnet_config.yaml'
    config = read_yaml(config_path)

    config['feature_set'] = 'MelSpec'
    config['model_arch'] = 'LCNN'
    config['train_model'] = False
    config['eval_model'] = True
    config['classification_type'] = 'binary'
    config['win_len'] = 10.0
    config['num_classes'] = 2

    seed_everything(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(config, device)


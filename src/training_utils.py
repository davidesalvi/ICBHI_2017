import random

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class LoadTrainData(Dataset):
    def __init__(self, list_IDs, labels, win_len, fs=44100):

        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = int(self.win_len * self.fs)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track_name = self.list_IDs[index]

        x, fs = librosa.load(track_name, sr=None)

        # convert stereo to mono
        if not len(x.shape) == 1:
            x = np.mean(x, axis=1)

        y = self.labels[track_name]
        audio_len = len(x)

        if audio_len < self.win_len_samples:
            x = pad(x, self.win_len_samples)
            audio_len = len(x)

        # evaluate a random window
        last_valid_start_sample = audio_len - self.win_len_samples
        if not last_valid_start_sample == 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        x_win = x[start_sample:start_sample + self.win_len_samples]
        x_win = Tensor(x_win)

        return x_win, y


class LoadEvalData(Dataset):
    def __init__(self, list_IDs, labels, win_len, fs=44100):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = int(self.win_len * self.fs)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track = self.list_IDs[index]
        x, fs = librosa.load(track, sr=None)

        # convert stereo to mono
        if not len(x.shape) == 1:
            x = np.mean(x, axis=1)

        y = self.labels[track]
        audio_len = len(x)

        if audio_len < self.win_len_samples:
            x = pad(x, self.win_len_samples)
            audio_len = len(x)

        # Evaluate multiple windows
        num_eval = int(audio_len / self.win_len_samples)
        start_sample_list = np.linspace(0, audio_len - self.win_len_samples, num=num_eval)

        frames = []
        for start_sample in start_sample_list:
            frames += [x[int(start_sample):int(start_sample) + self.win_len_samples]]
        x_win = np.stack(frames, axis=0).astype(float)

        x_inp = Tensor(x_win)

        return x_inp, y, track



def train_epoch(train_loader, model, optim, criterion, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0

    model.train()

    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


def valid_epoch(data_loader, model, criterion, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0

    model.eval()

    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

    valid_accuracy = (num_correct / num_total) * 100
    running_loss /= num_total
    return running_loss, valid_accuracy


def eval_model(model, data_loader, save_path, device):

    model.eval()

    for batch_x, batch_y, track_id in tqdm(data_loader, total=len(data_loader)):

        fname_list = []
        pred_list = []
        label_list = []

        batch_x = batch_x.to(device)
        if batch_x.dim() == 3:
            batch_x = batch_x[0]
        batch_out = model(batch_x)
        batch_out = batch_out.mean(dim=0).detach().cpu().numpy()

        batch_pred = batch_out.argmax()
        batch_pred = int(batch_pred)

        fname_list.extend(track_id)
        pred_list.extend([batch_pred])
        label_list.extend(batch_y.tolist())

        with open(save_path, 'a+') as fh:
            for f, pred, lab in zip(fname_list, pred_list, label_list):
                fh.write('{} {} {}\n'.format(f, pred, lab))
        fh.close()


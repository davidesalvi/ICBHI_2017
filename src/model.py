import torch
from torchvision.models import resnet18
import torch.nn as nn
import torchaudio
import torchvision
from torchvision.transforms import v2
import torchaudio.transforms as T


def modify_for_grayscale(model):
    # Create a new convolutional layer with 1 input channel instead of 3
    first_conv_layer = model.conv1
    new_first_conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=first_conv_layer.bias is not None
    )

    with torch.no_grad():
        new_first_conv_layer.weight[:, 0] = first_conv_layer.weight.mean(dim=1)
        if first_conv_layer.bias is not None:
            new_first_conv_layer.bias = first_conv_layer.bias

    model.conv1 = new_first_conv_layer
    return model

class ResNet_MelSpec(nn.Module):
    def __init__(self, config):
        super(ResNet_MelSpec, self).__init__()

        self.sample_rate = config['sample_rate']
        self.n_fft = config['n_fft']
        self.win_length = int(config['win_length_val'] * self.sample_rate)
        self.hop_length = int(config['hop_length_val'] * self.sample_rate)
        self.n_mels = config['n_mels']

        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        self.model = resnet18(pretrained=True)
        self.model = modify_for_grayscale(self.model)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, config['num_classes'])
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.melspectrogram(x)
        x = self.to_db(x)
        x = self.model(x)
        return x


class ResNet_LogSpec(nn.Module):
    def __init__(self, config):
        super(ResNet_LogSpec, self).__init__()

        self.sample_rate = config['sample_rate']
        self.n_fft = config['n_fft']
        self.win_length = int(config['win_length_val'] * self.sample_rate)
        self.hop_length = int(config['hop_length_val'] * self.sample_rate)

        self.stft = T.Spectrogram(n_fft=self.n_fft,
                                  win_length=self.win_length,
                                  hop_length=self.hop_length,
                                  power=2)

        self.model = resnet18(pretrained=True)
        self.model = modify_for_grayscale(self.model)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, config['num_classes'])
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stft(x)
        x = self.to_db(x)
        x = self.model(x)
        return x


class ResNet_MFCC(nn.Module):
    def __init__(self, config):
        super(ResNet_MFCC, self).__init__()

        self.sample_rate = config['sample_rate']
        self.n_mfcc = config['n_mfcc']
        self.n_mels = config['n_mels']
        self.n_fft = config['n_fft']
        self.win_length = int(config['win_length_val'] * self.sample_rate)
        self.hop_length = int(config['hop_length_val'] * self.sample_rate)

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": self.n_fft, "win_length": self.win_length, "hop_length": self.hop_length,
                     "n_mels": self.n_mels},
        )

        self.model = resnet18(pretrained=True)
        self.model = modify_for_grayscale(self.model)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, config['num_classes'])
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mfcc(x)
        x = self.to_db(x)
        deltas = torchaudio.functional.compute_deltas(x)
        double_deltas = torchaudio.functional.compute_deltas(deltas)
        x = torch.cat([x, deltas, double_deltas], dim=2)
        x = self.model(x)
        return x


class ResNet_LFCC(nn.Module):
    def __init__(self, config):
        super(ResNet_LFCC, self).__init__()

        self.sample_rate = config['sample_rate']
        self.n_lfcc = config['n_lfcc']
        self.n_fft = config['n_fft']
        self.n_filter = config['n_filter']
        self.win_length = int(config['win_length_val'] * self.sample_rate)
        self.hop_length = int(config['hop_length_val'] * self.sample_rate)
        self.n_mels = config['n_mels']

        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=self.sample_rate,
            n_lfcc=self.n_lfcc,
            n_filter=self.n_filter,
            speckwargs={"n_fft": self.n_fft, "win_length": self.win_length, "hop_length": self.hop_length},
        )

        self.model = resnet18(pretrained=True)
        self.model = modify_for_grayscale(self.model)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, config['num_classes'])
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.lfcc(x)
        x = self.to_db(x)
        deltas = torchaudio.functional.compute_deltas(x)
        double_deltas = torchaudio.functional.compute_deltas(deltas)
        x = torch.cat([x, deltas, double_deltas], dim=2)
        x = self.model(x)
        return x

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torchvision.models import resnet18


class LCNN(nn.Module):
    def __init__(self, config):
        super(LCNN, self).__init__()

        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5),
                               padding=(2, 2), stride=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1))
        self.batchnorm6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(3, 3),
                               padding=(1, 1), stride=(1, 1))
        self.maxpool9 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batchnorm10 = nn.BatchNorm2d(48)
        self.conv11 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(1, 1),
                                padding=(0, 0), stride=(1, 1))
        self.batchnorm13 = nn.BatchNorm2d(48)
        self.conv14 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.maxpool16 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv17 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1),
                                padding=(0, 0), stride=(1, 1))
        self.batchnorm19 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.batchnorm22 = nn.BatchNorm2d(32)
        self.conv23 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                                padding=(0, 0), stride=(1, 1))
        self.batchnorm25 = nn.BatchNorm2d(32)
        self.conv26 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.maxpool28 = nn.AdaptiveMaxPool2d((16, 8))

        # Classification Part (Second Part)
        self.fc29 = nn.Linear(32 * 16 * 8, 128)
        self.batchnorm31 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.7)
        self.fc32 = nn.Linear(64, config['num_classes'])
        self.softmax = nn.Softmax(dim=1)

    def mfm2(self, x):
        out1, out2 = torch.chunk(x, 2, 1)
        return torch.max(out1, out2)

    def mfm3(self, x):
        n, c, y, z = x.shape
        out1, out2, out3 = torch.chunk(x, 3, 1)
        res1 = torch.max(torch.max(out1, out2), out3)
        tmp1 = out1.flatten()
        tmp1 = tmp1.reshape(len(tmp1), -1)
        tmp2 = out2.flatten()
        tmp2 = tmp2.reshape(len(tmp2), -1)
        tmp3 = out3.flatten()
        tmp3 = tmp3.reshape(len(tmp3), -1)
        res2 = torch.cat((tmp1, tmp2, tmp3), 1)
        res2 = torch.median(res2, 1)[0]
        res2 = res2.reshape(n, -1, y, z)
        return torch.cat((res1, res2), 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.mfm2(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.mfm2(x)
        x = self.batchnorm6(x)
        x = self.conv7(x)
        x = self.mfm2(x)
        x = self.maxpool9(x)
        x = self.batchnorm10(x)
        x = self.conv11(x)
        x = self.mfm2(x)
        x = self.batchnorm13(x)
        x = self.conv14(x)
        x = self.mfm2(x)
        x = self.maxpool16(x)
        x = self.conv17(x)
        x = self.mfm2(x)
        x = self.batchnorm19(x)
        x = self.conv20(x)
        x = self.mfm2(x)
        x = self.batchnorm22(x)
        x = self.conv23(x)
        x = self.mfm2(x)
        x = self.batchnorm25(x)
        x = self.conv26(x)
        x = self.mfm2(x)
        x = self.maxpool28(x)
        x = x.view(-1, 32 * 16 * 8)
        x = self.mfm2((self.fc29(x)))
        x = self.batchnorm31(x)
        x = self.fc32(x)
        output = self.softmax(x)
        return output


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


class MelSpec_model(nn.Module):
    def __init__(self, config):
        super(MelSpec_model, self).__init__()

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

        if config['model_arch'] == 'ResNet':
            self.model = resnet18(pretrained=True)
            self.model = modify_for_grayscale(self.model)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(num_ftrs, 256, bias=True),
                nn.ReLU(),
                nn.Linear(256, config['num_classes']),
                nn.Softmax(dim=1)
            )
        else:
            self.model = LCNN(config)

        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.melspectrogram(x)
        x = self.to_db(x)
        x = self.model(x)
        return x


class LogSpec_model(nn.Module):
    def __init__(self, config):
        super(LogSpec_model, self).__init__()

        self.sample_rate = config['sample_rate']
        self.n_fft = config['n_fft']
        self.win_length = int(config['win_length_val'] * self.sample_rate)
        self.hop_length = int(config['hop_length_val'] * self.sample_rate)

        self.stft = T.Spectrogram(n_fft=self.n_fft,
                                  win_length=self.win_length,
                                  hop_length=self.hop_length,
                                  power=2)

        if config['model_arch'] == 'ResNet':
            self.model = resnet18(pretrained=True)
            self.model = modify_for_grayscale(self.model)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(num_ftrs, 256, bias=True),
                nn.ReLU(),
                nn.Linear(256, config['num_classes']),
                nn.Softmax(dim=1)
            )
        else:
            self.model = LCNN(config)

        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stft(x)
        x = self.to_db(x)
        x = self.model(x)
        return x


class MFCC_model(nn.Module):
    def __init__(self, config):
        super(MFCC_model, self).__init__()

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

        if config['model_arch'] == 'ResNet':
            self.model = resnet18(pretrained=True)
            self.model = modify_for_grayscale(self.model)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(num_ftrs, 256, bias=True),
                nn.ReLU(),
                nn.Linear(256, config['num_classes']),
                nn.Softmax(dim=1)
            )
        else:
            self.model = LCNN(config)

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


class LFCC_model(nn.Module):
    def __init__(self, config):
        super(LFCC_model, self).__init__()

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

        if config['model_arch'] == 'ResNet':
            self.model = resnet18(pretrained=True)
            self.model = modify_for_grayscale(self.model)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(num_ftrs, 256, bias=True),
                nn.ReLU(),
                nn.Linear(256, config['num_classes']),
                nn.Softmax(dim=1)
            )
        else:
            self.model = LCNN(config)

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

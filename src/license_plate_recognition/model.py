from typing import Any

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        dictionary_size: int,
        ninp: int,
        nhid: int,
        nlayers: int,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=ninp, hidden_size=nhid, num_layers=nlayers, bidirectional=True
        )
        self.decoder = nn.Linear(2 * nhid, dictionary_size)

        self.dictionary_size = dictionary_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(input)
        seq_len, batch_size, _ = output.size()
        decoded = self.decoder(output)
        decoded = decoded.view(seq_len, batch_size, self.dictionary_size)
        return decoded


class CRNN(nn.Module):
    def conv_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        batch_norm: bool = False,
    ) -> nn.Sequential:
        layers: list[Any] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def __init__(self, dictionary_size: int):
        super().__init__()

        self.conv0 = self.conv_layer(
            in_channels=1, out_channels=64, kernel_size=3, padding=1
        )  # 1x32x128 -> 64x32x128
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x32x128 -> 64x16x64
        self.conv1 = self.conv_layer(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )  # 64x16x64 -> 128x16x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x16x64 -> 128x8x32
        self.conv2 = self.conv_layer(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )  # 128x8x32 -> 256x8x32
        self.conv3 = self.conv_layer(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )  # 256x8x32 -> 256x8x32
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)  # 256x8x32 -> 256x4x16
        self.conv4 = self.conv_layer(
            in_channels=256, out_channels=512, kernel_size=3, padding=1, batch_norm=True
        )  # 256x4x16 -> 512x4x16
        self.conv5 = self.conv_layer(
            in_channels=512, out_channels=512, kernel_size=3, padding=1, batch_norm=True
        )  # 512x4x16 -> 512x4x16
        self.pool3 = nn.MaxPool2d(
            kernel_size=(1, 2), stride=(2, 1), padding=(0, 1)
        )  # 512x4x16 -> 512x2x17
        self.conv6 = self.conv_layer(
            in_channels=512, out_channels=512, kernel_size=2, padding=0, batch_norm=True
        )  # 512x2x17 -> 512x1x16

        self.rnn = BidirectionalLSTM(
            dictionary_size=dictionary_size, ninp=512, nhid=256, nlayers=2
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # CNN layers
        l0 = self.conv0(input)
        l1 = self.conv1(self.pool0(l0))
        l2 = self.conv2(self.pool1(l1))
        l3 = self.conv3(l2)
        l4 = self.conv4(self.pool2(l3))
        l5 = self.conv5(l4)
        l6 = self.conv6(self.pool3(l5))

        # (seq_len=16, batch_size, features=512)
        map_to_sequence = l6.squeeze(2).permute(2, 0, 1)

        # RNN layers
        output = self.rnn(map_to_sequence)

        return output

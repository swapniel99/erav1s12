import torch.nn as nn
import torchinfo


class ConvLayer(nn.Module):
    def __init__(self, input_c, output_c, bias=False, padding=0, dws=False, skip=False, dilation=1, dropout=0):
        super(ConvLayer, self).__init__()

        # Member Variables
        self.skip = skip

        # If Depthwise Separable is True
        if dws and input_c == output_c:
            self.convlayer = nn.Sequential(
                nn.Conv2d(input_c, output_c, 3, bias=bias, padding=padding, groups=input_c, dilation=dilation,
                          padding_mode='replicate'),
                nn.BatchNorm2d(output_c),
                nn.ReLU(),
                nn.Conv2d(output_c, output_c, 1, bias=bias)
            )
        else:
            self.convlayer = nn.Conv2d(input_c, output_c, 3, bias=bias, padding=padding, groups=1, dilation=dilation,
                                       padding_mode='replicate')

        self.normlayer = nn.BatchNorm2d(output_c)

        self.skiplayer = None
        if self.skip and input_c != output_c:
            self.skiplayer = nn.Conv2d(input_c, output_c, 1, bias=bias)

        self.actlayer = nn.ReLU()

        self.droplayer = None
        if dropout > 0:
            self.droplayer = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.convlayer(x)
        x = self.normlayer(x)
        x = self.actlayer(x)
        if self.skip:
            if self.skiplayer is None:
                x = x + x_
            else:
                x = x + self.skiplayer(x_)
        if self.droplayer is not None:
            x = self.droplayer(x)
        return x


class Model(nn.Module):
    def __init__(self, dropout=0, skip=False):
        super(Model, self).__init__()

        # Member Variables
        self.dropout = dropout

        self.cblock1 = self.get_conv_block(3, 25, padding=1, dws=True, skip=False, reps=2, dropout=self.dropout)
        self.tblock1 = self.get_trans_block(25, 32, padding=0, dws=False, skip=False, dilation=1, dropout=self.dropout)
        self.cblock2 = self.get_conv_block(32, 32, padding=1, dws=True, skip=skip, reps=2, dropout=self.dropout)
        self.tblock2 = self.get_trans_block(32, 65, padding=0, dws=False, skip=False, dilation=2, dropout=self.dropout)
        self.cblock3 = self.get_conv_block(65, 65, padding=1, dws=True, skip=skip, reps=2, dropout=self.dropout)
        self.tblock3 = self.get_trans_block(65, 95, padding=0, dws=False, skip=False, dilation=4, dropout=self.dropout)
        self.cblock4 = self.get_conv_block(95, 95, padding=1, dws=True, skip=skip, reps=2, dropout=self.dropout)
        self.tblock4 = self.get_trans_block(95, 95, padding=0, dws=False, skip=False, dilation=8, dropout=0)

        self.oblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(95, 10, 1, bias=True),
            nn.Flatten(),
            nn.LogSoftmax(-1)
        )

    @staticmethod
    def get_conv_block(input_c, output_c, bias=False, padding=0, dws=True, skip=True, reps=2, dilation=1,
                       dropout=0):
        block = list()
        for i in range(0, reps):
            block.append(
                ConvLayer(output_c if i > 0 else input_c, output_c, bias=bias, padding=padding, dws=dws, skip=skip,
                          dilation=dilation, dropout=dropout)
            )
        return nn.Sequential(*block)

    @staticmethod
    def get_trans_block(input_c, output_c, bias=False, padding=0, dws=False, skip=False, dilation=1, dropout=0):
        return ConvLayer(input_c, output_c, bias=bias, padding=padding, dws=dws, skip=skip, dilation=dilation,
                         dropout=dropout)

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.tblock3(x)
        x = self.cblock4(x)
        x = self.tblock4(x)
        x = self.oblock(x)
        return x

    def summary(self, input_size=None):
        return torchinfo.summary(self, input_size=input_size, depth=5,
                                 col_names=["input_size", "output_size", "num_params", "params_percent"])

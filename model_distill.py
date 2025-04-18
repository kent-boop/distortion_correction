import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class plainEncoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
        super(plainEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class plainDecoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):

        super(plainDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChannel)

        if stride == 1:
            self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outChannel)
        else:
            self.conv2 = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2)
            self.bn2 = nn.BatchNorm2d(outChannel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class resEncoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):

        super(resEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outChannel))

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


class resDecoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride=1):

        super(resDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChannel)

        self.downsample = None

        if stride == 1:
            self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outChannel)
        else:
            self.conv2 = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2)
            self.bn2 = nn.BatchNorm2d(outChannel)

            self.downsample = nn.Sequential(
                nn.ConvTranspose2d(inChannel, outChannel, kernel_size=1, stride=2, output_padding=1),
                nn.BatchNorm2d(outChannel))

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


class EncoderNet(nn.Module):
    def __init__(self, layers):
        super(EncoderNet, self).__init__()

        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)

        self.en_layer1 = self.make_encoder_layer(plainEncoderBlock, 32, 32, layers[0], stride=1)
        self.en_layer2 = self.make_encoder_layer(resEncoderBlock, 32, 64, layers[1], stride=2)
        self.en_layer3 = self.make_encoder_layer(resEncoderBlock, 64, 128, layers[2], stride=2)
        self.en_layer4 = self.make_encoder_layer(resEncoderBlock, 128, 256, layers[3], stride=2)
        self.en_layer5 = self.make_encoder_layer(resEncoderBlock, 256, 256, layers[4], stride=2)

        # weight initializaion with Kaiming method
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_encoder_layer(self, block, inChannel, outChannel, block_num, stride):
        layers = []
        layers.append(block(inChannel, outChannel, stride=stride))
        for i in range(1, block_num):
            layers.append(block(outChannel, outChannel, stride=1))
        # 在每个Encoder层级末尾添加Dropout（前4层加、最后一层不加）
        if outChannel != 256:  # 根据各层输出通道调整
            layers.append(nn.Dropout2d(p=0.2))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = F.relu(self.bn(self.conv(x)))

        x = self.en_layer1(x)  # 128
        x = self.en_layer2(x)  # 64
        x = self.en_layer3(x)  # 32
        x = self.en_layer4(x)  # 16
        x = self.en_layer5(x)  # 8

        return x


class DecoderNet(nn.Module):
    def __init__(self, layers):
        super(DecoderNet, self).__init__()

        self.de_layer5 = self.make_decoder_layer(resDecoderBlock, 256, 256, layers[4], stride=2)
        self.de_layer4 = self.make_decoder_layer(resDecoderBlock, 256, 128, layers[3], stride=2)
        self.de_layer3 = self.make_decoder_layer(resDecoderBlock, 128, 64, layers[2], stride=2)
        self.de_layer2 = self.make_decoder_layer(resDecoderBlock, 64, 32, layers[1], stride=2)
        self.de_layer1 = self.make_decoder_layer(plainDecoderBlock, 32, 32, layers[0], stride=1)

        self.conv_end = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

        # weight initializaion with Kaiming method
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_decoder_layer(self, block, inChannel, outChannel, block_num, stride):

        layers = []
        for i in range(0, block_num - 1):
            layers.append(block(inChannel, inChannel, stride=1))
        layers.append(block(inChannel, outChannel, stride=stride))
        # 在前4层添加Dropout，最后一层不加
        if outChannel != 2:  # 最后一层输出无需Dropout
            layers.append(nn.Dropout2d(p=0.2))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.de_layer5(x)  # 8
        x = self.de_layer4(x)  # 16
        x = self.de_layer3(x)  # 32
        x = self.de_layer2(x)  # 64
        x = self.de_layer1(x)  # 128

        x = self.conv_end(x)
        return x


class ClassNet(nn.Module):
    def __init__(self):
        super(ClassNet, self).__init__()

        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        # 添加Dropout
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(16384, 6)

    def forward(self, x):
        # print("ClassNet:",x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output - target + 1e-16, p=2, dim=1).mean()
        return lossvalue

class GeoNetM(nn.Module):
    def __init__(self):
        super(GeoNetM,self).__init__()
        self.encoder = EncoderNet([1,1,1,1,2])
        self.decoder = DecoderNet([1,1,1,1,2])
        self.classifier = ClassNet()
        self.epeLoss = EPELoss()

    def forward(self, img):
        middle = self.encoder(img)
        flow_output = self.decoder(middle)
        clas = self.classifier(middle)
        return flow_output, clas
import torch
from torchvision import transforms
from torchvision import models
import os
from torch import nn
from torch.nn import functional as F

class ResBlK(nn.Module):
    """
    残差块
    """
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlK, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1) #第一步改变这个特征图的大小
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)  #不改变特征图的大小
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()  #先假设一个空的，若维度不同，则加上其他的
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=stride),  #与第一步的stride设置为一致
                nn.BatchNorm2d(ch_out)
            )
    def forward(self, x):

        x_out = F.relu(self.bn1(self.conv1(x)))
        x_out = F.relu(self.bn2(self.conv2(x_out)))
        #short_cut
        out = self.extra(x) + x_out
        return out

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3,padding=0), #预处理层，把channel修改为64
            nn.BatchNorm2d(64)
        )
        #四个block
        self.blk1 = ResBlK(64, 128, stride=2)
        self.blk2 = ResBlK(128, 256, stride=2)
        self.blk3 = ResBlK(256, 512, stride=2)
        # [b,512,h,w] => [b.1024.h.w]
        self.blk4 = ResBlK(512, 1024, stride=2) # [b,1024,5,5]
        self.outlayer_1= nn.Linear(1024, 512)
        self.outlayer = nn.Linear(512*1*1, 5)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)

        x = self.blk4(x)
       # print('after conv:', x.shape)
        # [b,512,h,w] >= [b,512,1,1]
        x = F.adaptive_avg_pool2d(x, [1, 1])  #
        print('after avg_pool:', x.shape)
        x = x.view(x.size(0), -1)
        print('after flatten:', x.shape)
        x = self.outlayer_1(x)
        x = self.outlayer(x)
        return x

def main():
    blk = ResBlK(64, 128, stride=2)
    tmp = torch.rand(2, 3, 224, 224)
    resnet = ResNet18()
    out = resnet(tmp)
    print(out.shape)

if __name__ == '__main__':
    main()











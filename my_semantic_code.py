from torch import nn
import os
import json
import pickle
import random
import math
import matplotlib.pyplot as plt
import torch
import numpy as np
from channel.my_channel import AWGN_channel
from torch.nn import functional as F
from torchvision import models
model = models.resnet50(pretrained=True)
 # 提取fc层中固定的参数
fc_features = model.fc.in_features  # in_features是全连接层的输入层的意思
# 修改类别为9
model.fc = nn.Linear(fc_features, 5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SemanticCommunication(nn.Module):
    def __init__(self):
        super(SemanticCommunication, self).__init__()
        self.dense_channel_encoder1 = nn.Linear(1024, 512)
        self.dense_channel_encoder2 = nn.Linear(512, 256)

        self.dense_channel_decoder1 = nn.Linear(256, 512)
        self.dense_channel_decoder2 = nn.Linear(512, 1024)
        self.dense_channel_classify = nn.Linear(1024, 5)
    def forward(self, x):
        x = x.view(x.size(0), -1) #先转成了[batch_size, features]
        #print(x.shape)
        x_out = F.relu(self.dense_channel_encoder1(x))
        x_out = F.relu(self.dense_channel_encoder2(x_out))
        #通过信道
        x_out = AWGN_channel(x_out, -10)  #假设高斯信道snr = 12dB
        x_out = F.relu(self.dense_channel_decoder1(x_out))
        x_out = F.relu(self.dense_channel_decoder2(x_out))
        x_out = self.dense_channel_classify(x_out)
        return x_out
def main():
    x = torch.rand(2, 1024)
    model = SemanticCommunication()
    x = x.to(device)
    model =model.to(device)
    x = model(x)
    print('x.shape:', x.shape) #[2,5]
    print(x, x.dtype)  #torch.float32

    a = torch.tensor([1., 0.], dtype=torch.int64) #这里不应该是onehot或者其他的
    a= a.to(device)
    print(a.shape)
    print(a, a.dtype)
    criteon = nn.CrossEntropyLoss()
    loss = criteon(x, a)
    print(loss)

if __name__=='__main__':
    main()




class semantic_encode(nn.Module):
    def __init__(self):
        super(semantic_encode, self).__init__()
        model = models.resnet50(pretrained=True)
        # 提取fc层中固定的参数
        fc_features = model.fc.in_features  # in_features是全连接层的输入层的意思
        # 修改类别为9
        model.fc = nn.Linear(fc_features, 1024)
        #self.linear = nn.Sequential(nn.Linear(1024, 5), nn.Softmax(dim=1))
        self.channel = SemanticCommunication()
    def forward(self,x):
        out_x = model(x)
        out_x = self.channel(out_x)
       # out_x = self.linear(out_x)
        return out_x





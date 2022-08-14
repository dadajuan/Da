from torchvision import models
import torch
import torchvision
from torch import nn

vgg19 = models.vgg19_bn(pretrained=True)
vgg19.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 5), nn.Softmax(dim=1))

model = models.resnet50(pretrained=True)
#提取fc层中固定的参数
fc_features = model.fc.in_features  #in_features是全连接层的输入层的意思
print(fc_features)
#修改类别为9
model.fc = nn.Linear(fc_features, 9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def main():
    x = torch.rand(2, 3, 256, 256)
    x = x.to(device)
    net = model
    out = net(x)
    print(out)
    print(type(out))
    print(out.shape)
if __name__ == '__main__':
    main()


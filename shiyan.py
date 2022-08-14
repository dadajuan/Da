import torch
from torchvision import models
from torch import nn
from torchvision import models
from my_semantic_code import SemanticCommunication

# model = models.resnet50(pretrained=True)
# #print(model)
# fc_features = model.fc.in_features #就是最后一层全连接的输出层是，2048
# print(fc_features) #
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.fc = nn.Linear(fc_features, 1024)
#
# #model.layer4[1].conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))#更改网络中某一层的参数，首先打印网络结构，修改指定层
# print(model.layer4[1].conv1)
# print(type(model.parameters()))
# print(type([model.parameters()]))
# #print(list(model.parameters()))
# print(model.parameters())
#
# count = 0
# para_optim = []
# #for k in nn.Sequential(*list(model.children())):
# for k in model.children():
#     count += 1
#     #print(k)
#     print(count)
#     if count > 9:
#         for param in k.parameters():
#             #para_optim.append(param)
#             param.requires_grad = True
#     else:
#         for param in k.parameters():
#             param.requires_grad = False
# print(model.parameters())
# print(list(model.parameters()))
# # optimizer = optim.RMSprop(para_optim, lr)
# """实验新模型"""
# class A(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(2, 2, 3)
#         self.conv2 = nn.Conv2d(2, 2, 3)
#         self.conv3 = nn.Conv2d(2, 2, 3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         return x
# a = A()
# print(a.parameters())
# #<generator object Module.parameters at 0x7f7b740d2360>
# print(list(a.parameters()))
# count = 0
# for k in a.children():
#     print(k)
#     count+=1
#     print(count)
#     if count > 2:
#         for param in k.parameters():
#             #para_optim.append(param)
#             param.requires_grad = True
#     else:
#         for param in k.parameters():
#             param.requires_grad = False
# print(list(a.parameters()))
"""如何加载模型"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
    # 提取fc层中固定的参数
fc_features = model.fc.in_features  # in_features是全连接层的输入层的意思
    # 修改类别为9
model.fc = nn.Linear(fc_features, 1024)
count = 0
for k in model.children():
    count += 1
    if count > 9: # 全连接是第10层
        for param in k.parameters():
            param.requires_grad = True
    else:
        for param in k.parameters():
            param.requires_grad = False
model1 = model.to(device)
model2 = SemanticCommunication()
model2 = model2.to(device)
load_name = './filemodel.pth'
checkpoint = torch.load(load_name)
model1.load_state_dict(checkpoint['model1'])
model2.load_state_dict(checkpoint['model2'])
print(model2)
print(list(model2.parameters()))
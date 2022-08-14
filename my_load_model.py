import torch
from torchvision import models
from torch import nn
from torchvision import models
from my_semantic_code import SemanticCommunication

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
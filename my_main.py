import os
import torch
from torchvision import transforms
from torchvision import models
from my_dataset import MyDataSet
from utils import read_split_data, plot_data_loader_image
from torch import nn
import torch.optim as optim

from my_pytorchtools import EarlyStopping
from my_semantic_code import SemanticCommunication
root = "D:/Desktop/6.2deeplearning/custom_dataset/flower_photos"  # 数据集所在根目录
epochs = 10
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    val_data_set = MyDataSet(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform["val"])

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    #train_loader 包含一个batch里的image和labels,其长度就是数据集的总长度/batch_size
    #这里返回的是一个idx和一个batch（idx，batch),idx表示这个batch的index，batch一般是长度为2的列表,列表的两个值都是tensor,分别表示数据和标签
    # batch = torch.utils.data.DataLoader(train_data_set,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            num_workers=0,
    #                                            collate_fn=train_data_set.collate_fn)

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=train_data_set.collate_fn)
   #print(type(train_loader)) #torch.utils.data.dataloader.DataLoader
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=val_data_set.collate_fn) #self.collate_fn函数就是将batch size个分散的Tensor封装成一个Tensor
    """ resnet除了最后一层之外的不更新,只更新最后一层全连接"""
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
    criteon = nn.CrossEntropyLoss()
    #criteon = nn.MSELoss()

    optimizer = optim.Adam([
        {'params': model1.parameters()},
        {'params': model2.parameters()}],
        lr=1e-3)
    #print(model)
    #print(model2)
    save_path = ".\\"  # 当前目录下
    patience = int(10)
    early_stopping = EarlyStopping(patience, verbose=True)
    for epoch in range(epochs):
        model.train()
        for batchidx , (x_train, label_train) in enumerate(train_loader):
            x_train, label_train = x_train.to(device), label_train.to(device)
            feature = model1(x_train)
            logits = model2(feature)
            loss = criteon(logits, label_train) #返回的已经是平均值
            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, loss.item()) #打印的是最后一步的,loss是 tensor scalar类型,标量转换为numpy
        #test
        model.eval()  #比如在测试时dropout是不发挥作用的
        with torch.no_grad():#用来说明测试时候不需要反向传播
            total_correct = 0
            total_number = 0
            for batchidx, (x_val, label_val) in enumerate(val_loader):
                #print('x_val:', x_val.shape, 'label_val:', label_val.shape)
                x_val, label_val = x_val.to(device), label_val.to(device)
                #[b,5]
                feature = model1(x_val)
                logits = model2(feature)
                pred = logits.argmax(dim=1)

                total_correct += torch.eq(pred, label_val).float().sum().item()
                total_number += x_val.size(0)
            acc = total_correct / total_number
            print('epoch:', epoch, 'acc_val:', acc)
            early_stopping(-acc, model1, model2)
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
    model.load_state_dict(torch.load('checkpoint.pt'))

if __name__ == '__main__':
    main()
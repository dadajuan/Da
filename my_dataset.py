from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
    #该数据集下所有样本的个数
    def __len__(self):
        return len(self.images_path) #计算该列表的元素个数

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    #静态方法
    #self.collate_fn函数就是将batch size个分散的Tensor封装成一个Tensor,
    #通过self.collate_fn函数将batch size个tuple（每个tuple长度为2，其中第一个值是数据，Tensor类型，第二个值是标签，int类型）封装成一个list;
    #这个list长度为2，两个值都是Tensor，一个是batch size个数据组成的FloatTensor，另一个是batch size个标签组成的LongTensor。
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)  #image变成（8,3,224,224）
        labels = torch.as_tensor(labels)     #labels转化为tensot（
        return images, labels
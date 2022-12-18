import os
import torch
from torchvision import models
import time
import numpy as np
from tqdm import tqdm
from collections import deque
from statistics import mean

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]

    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """计算预测正确的数量

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


def get_first_feature(dataset: torch.utils.data.Dataset):
    '''获得一个Dataset中的第一个数据的feature'''
    iter = torch.utils.data.DataLoader(dataset)
    for feature, label in iter:
        return feature

def save_state_dict(net,dir,name):
    '''保存网络权重，如果目标文件夹不存在就自动创建'''
    os.makedirs(dir,exist_ok=True)
    file_path=os.path.join(dir,name+'.pth')
    torch.save(net.state_dict(), file_path)


def evaluate_accuracy(net, data_iter):
    '''计算acc'''
    all_nums = len(data_iter)
    right_nums = 0
    for feature, label in tqdm(data_iter):
        #网络预测
        outputs = net(feature)
        _, predicted = torch.max(outputs, 1)
        if predicted[0] == label[0]:
            right_nums += 1
    return right_nums/all_nums


class AverageMeter(object):
    """Computes and stores the average"""

    #num:需要计算平均值的数据的个数，maxlen:队列最大长度
    def __init__(self, num, maxlen):
        self.data = [deque([], maxlen=maxlen) for _ in range(num)]

    #添加数据
    def append(self, *args):
        for i in range(len(args)):
            self.data[i].append(args[i])

    #根据索引获得平均值，不指定索引则返回所有平均值
    def get_mean(self, start=None, end=None):
        return np.array([mean(list(self.data[i])[start:end]) for i in range(len(self.data))])
    
    def len(self):
        return len(self.data[0])
    
    
def get_model(name,pretrained=True,**kwargs):
    '''根据名字获得模型'''
    match name:
        case "GoogLeNet":
            return models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT if pretrained else None, **kwargs)
        case "Inception_V3":
            return models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT if pretrained else None, **kwargs)
        case "ConvNeXt_Tiny":
            return models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None, **kwargs)
        case "ConvNeXt_Base":
            return models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None, **kwargs)
        case "DenseNet121":
            return models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None, **kwargs)
        case "DenseNet169":
            return models.densenet169(weights=models.DenseNet169_Weights.DEFAULT if pretrained else None, **kwargs)
        case "EfficientNet_V2_L":
            return models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT if pretrained else None, **kwargs)
        case "EfficientNet_V2_M":
            return models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None, **kwargs)
        case "MNASNet1_0":
            return models.mnasnet1_0(weights=models.MNASNet1_0_Weights.DEFAULT if pretrained else None, **kwargs)
        case "MNASNet1_3":
            return models.mnasnet1_3(weights=models.MNASNet1_3_Weights.DEFAULT if pretrained else None, **kwargs)
        case "MaxVit_T":
            return models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT if pretrained else None, **kwargs)
        case "MobileNet_V3_Small":
            return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None, **kwargs)
        case "MobileNet_V3_Large":
            return models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None, **kwargs)
        case "RegNet_X_400MF":
            return models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.DEFAULT if pretrained else None, **kwargs)
        case "RegNet_X_1_6GF":
            return models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.DEFAULT if pretrained else None, **kwargs)
        case "RegNet_Y_400MF":
            return models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None, **kwargs)
        case "RegNet_Y_1_6GF":
            return models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.DEFAULT if pretrained else None, **kwargs)
        case "ResNet18":
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None, **kwargs)
        case "ResNet50":
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None, **kwargs)
        case "ShuffleNet_V2_X1_5":
            return models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.DEFAULT if pretrained else None, **kwargs)
        case "ShuffleNet_V2_X2_0":
            return models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT if pretrained else None, **kwargs)
        case "SqueezeNet1_1":
            return models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None, **kwargs)
        case "Swin_V2_T":
            return models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT if pretrained else None, **kwargs)
        case "Swin_V2_S":
            return models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT if pretrained else None, **kwargs)
        case "VGG11_BN":
            return models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT if pretrained else None, **kwargs)
        case "VGG16_BN":
            return models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None, **kwargs)
        case "ViT_B_16":
            return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None, **kwargs)
        case "ViT_L_16":
            return models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT if pretrained else None, **kwargs)
        case "Wide_ResNet50_2":
            return models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None, **kwargs)
        case _:
            assert False, "Unknown model name: {}".format(name)
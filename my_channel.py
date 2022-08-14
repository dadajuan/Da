import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
torch.rand和torch.randn有什么区别？ 
一个均匀分布，一个是标准正态分布。
tf.random_normal  正太分布
"""
def AWGN_channel(x, snr_dB):  # used to simulate additive white gaussian noise channel,单位是db
    #输如之后对x进行归一化,输入的是未归一化的

    t = torch.sum(x ** 2, dim=1, keepdim=True)
    normalizer = t ** 0.5
    x = x / normalizer
    x_power = 1
    n_power = x_power / (10 ** (snr_dB / 10.0)) #snr,db为单位 ,噪声的功率
    noise = torch.randn_like(x, device=device) * (n_power**0.5) #均值为0，方差为1的标准正太分布，然后乘以这个就变成方法为n_power
    y = x + noise
    #yhat = y * normalizer
    return y

def RayleighChannel(x, snr_dB):
    # normalizer = tf.math.sqrt(tf.math.reduce_sum(x ** 2))
    # x = x / normalizer
    #进行归一化
    t = torch.sum(x ** 2, dim=1, keepdim=True)
    normalizer = t ** 0.5
    x = x / normalizer
    h = torch.randn_like(x, device=device)
    x_power = 1
    n_power = x_power / (10 ** (snr_dB / 10.0))
    noise = torch.randn_like(x, device=device) * torch.sqrt(n_power)
    y = h*x + noise
    return y





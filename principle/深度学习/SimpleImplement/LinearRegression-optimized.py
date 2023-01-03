import torch
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
import numpy as np
from collections import OrderedDict
import torch.utils.data as Data
from torch import nn
# 原本的实现方法
# 定义模型（函数）,此处的模型是最基本的矩阵乘法
# def linMul(x,w,b):
#     return torch.mm(x,w) + b

#第二种写法
# 这种写法的好处是使多层神经网络清晰明了
# class LinearNet(nn.Module):
#     def __init__(self, n_feature:int) -> None:
#         super().__init__()
#         # 定义输入参数个数与输出参数个数，bias默认自动生成
#         self.linear = nn.Linear(n_feature, 1)
#     # 向前传播函数（线性函数）
#     def forward(self, x):
#         # 获取判断值
#         y = self.linear(x)
#         return y

# 第三种写法
def LinearNet(n_feature):
    # Sequential是一个容器，可以容纳多层模型
    return nn.Sequential(
        # 有序字典
        OrderedDict([
            ('linear', nn.Linear(n_feature, 1))
            # 还可以传入其他层
        ])
    )
    # 或者直接 return nn.Sequential(nn.Linear(n_feature,1))
    
    # 可以查看容器中所有可学习参数（w&b）
    # net = LinearNet(2)    
    # for param in net.parameters():
    #     print(param)
net = LinearNet(2)
# 定义数据及其迭代器
# 把数据分批次
def data_iter(features: torch.tensor, batch_size:int, labels:torch.tensor):
    # 原本的实现
    # total_size = len(features)
    # indexes = list(range(0,total_size))
    # random.shuffle(indexes)
    # for i in range(0, total_size, batch_size):
    #     choice = torch.tensor(indexes[i:min(i+batch_size, total_size)])
    #     yield features.index_select(0,choice), labels.index_select(0,choice)
    
    # 优化的实现
    # import torch.utils.data as Data
    # 先组合特征值和真值
    dataSet = Data.TensorDataset(features, labels)
    # 将数据分批
    # 返回一个迭代器
    return Data.DataLoader(dataSet, batch_size, True)
# 原本的实现
# 定义Loss函数
# def Loss(y_hat, y):
#     # 1/2*(y_hat - y)^2
#     return (y_hat - y.view(y_hat.size()))**2/2

# 优化的实现
# nn中内置了均方损失函数
Loss = nn.MSELoss()

# 原本的实现
# 定义优化算法
# def sgd(params, lr, batch_size):
#     for param in params:
#         param.data -= lr*param.grad / batch_size

#优化的实现
# 会直接对net中的训练参数优化
# 设置学习率为0.03
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
# 定义训练函数
# def train(num_epoches, lr, features, batch_size, labels, hat_w, hat_b, true_w, true_b):
#     for epoch in range(num_epoches):
#         for x,y in data_iter(features, batch_size, labels):
#             lab_tmp = linMul(x, hat_w.T, hat_b)
#             # print(lab_tmp)
#             l = Loss(lab_tmp, y).sum()
#             l.backward()
#             sgd([hat_w, hat_b], lr, batch_size)
#             hat_w.grad.data.zero_()
#             hat_b.grad.data.zero_()
#         train_l = Loss(linMul(features, hat_w.T, hat_b), labels).sum()
#         print(f"epoch {epoch}, loss:{train_l}")
#     print(f"true weight:{true_w}, hat weight:{hat_w}")
#     print(f"true bias:{true_b}, hat bias:{hat_b}")

def train(num_epoch, features: torch.tensor, batch_size:int, labels:torch.tensor):
    for epoch in range(num_epoch):
        for x,y in data_iter(features, batch_size, labels):
            # 获取模拟结果
            output = net(x)
            # 计算损失
            l = Loss(output, y.view(output.size()))
            # 梯度清零
            optimizer.zero_grad()
            # 反向求导（求偏导）
            # 这样可以求出使Loss下降最快的梯度
            l.backward()
            # 优化
            optimizer.step()
        print(f"epoch {epoch}, loss:{l}")
            
maker = lambda x,y,z: torch.tensor([[round(ele.item(),10) for ele in np.random.normal(0,x,y)]],dtype=torch.float32, requires_grad=z)

def main():
    # 先定义真正的w&b，还有应该得到的结果
    true_w = maker(10,2,False)
    true_b = maker(5,1,False)
    print(f"true weight:{true_w}, true bias:{true_b}")
    #定义1000个数据(特征值)
    features = torch.randn((1000, 2))
    print(f"features:{features}")
    # 定义真值
    labels = torch.mm(features,true_w.T) + true_b
    # net是一张网，包含了神经网络各层的训练模型
    # 初始化模型参数
    # 不必再使用自定义的hat_w和hat_b，而是直接内置到模型中
    nn.init.normal_(net[0].weight, 0, 0.01)
    nn.init.constant_(net[0].bias, 0)
    # 定义迭代次数
    num_epoches = 10
    # train(num_epoches, lr, features, 10, labels, hat_w, hat_b, true_w, true_b)
    train(num_epoches, features, 10, labels)
    print(f"true weight:{true_w}, hat weight:{net[0].weight}")
    print(f"true bias:{true_b}, hat bias:{net[0].bias}")

if __name__ == "__main__":
    main()

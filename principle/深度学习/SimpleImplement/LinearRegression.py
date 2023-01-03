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
import numpy as np
import random


# 定义模型（函数）,此处的模型是最基本的矩阵乘法
def linMul(x,w,b):
    return torch.mm(x,w) + b
# 定义数据及其迭代器
# 把数据分批次
def data_iter(features: torch.tensor, batch_size:int, labels:torch.tensor):
    total_size = len(features)
    # print(f"total size:{total_size}")
    indexes = list(range(0,total_size))
    random.shuffle(indexes)
    for i in range(0, total_size, batch_size):
        choice = torch.tensor(indexes[i:min(i+batch_size, total_size)])
        yield features.index_select(0,choice), labels.index_select(0,choice)
# 定义Loss函数
def Loss(y_hat, y):
    # 1/2*(y_hat - y)^2
    return (y_hat - y.view(y_hat.size()))**2/2
# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr*param.grad / batch_size
# 定义训练函数
def train(num_epoches, lr, features, batch_size, labels, hat_w, hat_b, true_w, true_b):
    for epoch in range(num_epoches):
        for x,y in data_iter(features, batch_size, labels):
            lab_tmp = linMul(x, hat_w.T, hat_b)
            # print(lab_tmp)
            l = Loss(lab_tmp, y).sum()
            l.backward()
            sgd([hat_w, hat_b], lr, batch_size)
            hat_w.grad.data.zero_()
            hat_b.grad.data.zero_()
        train_l = Loss(linMul(features, hat_w.T, hat_b), labels).sum()
        print(f"epoch {epoch}, loss:{train_l}")
    print(f"true weight:{true_w}, hat weight:{hat_w}")
    print(f"true bias:{true_b}, hat bias:{hat_b}")
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
    labels = linMul(features, true_w.T, true_b)
    # 初始化参数
    hat_w = maker(0.01,2,True)
    hat_b = maker(0.01,1,True)
    # 定义学习率和迭代次数
    lr = 0.03
    num_epoches = 10
    train(num_epoches, lr, features, 10, labels, hat_w, hat_b, true_w, true_b)

if __name__ == "__main__":
    main()

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
dispose: true
dispose: true
dispose: true
dispose: true
dispose: true
import torchvision
import numpy as np
import torchvision.transforms as transforms

# 定义softmax


def softmax(Oi: torch.Tensor):
    Oi_exp = Oi.exp()
    return Oi_exp/(Oi_exp.sum(dim=1, keepdim=True))

# 定义模型


def net(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    return softmax(torch.mm(X.view((-1, W.size()[0])), W)+b)

# 定义Loss函数
# 交叉熵损失函数
# 表现为对数损失函数


def Loss(y: torch.Tensor, y_hat: torch.Tensor):
    # 这里的y是n*1的矩阵，而y_hat是n*j
    return - torch.log(y_hat.gather(1, y.view(-1, 1))).sum()

# 定义优化算法


def sgd(params, lr, batch_size):
    for param in params:
        # 学习率就是一个用于控制下降幅度的常数
        # 权重和偏移优化
        param.data -= lr*param.grad / batch_size

# 定义数据及其迭代器
# 把数据分批次


def data_iter(mnist_test, batch_size):
    return torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=0)
# 定义准确率


def accuracy(y: torch.Tensor, y_hat: torch.Tensor):
    # 如果某一次模型预测结果与真实结果相同，就返回1，否则返回0
    # 用argmax函数获取y_hat中最大项的索引（0~9），与真实结果对比
    # 转化为float后加和，取其中的值返回
    return (y_hat.argmax(1) == y).float().sum().item()


def train(n_epoches: int, lr: float, w_hat: torch.Tensor, b_hat: torch.Tensor,
          batch: int, data: torch.Tensor) -> torch.Tensor:
    n_features = len(data)
    # 一共拟合训练数据n_epoch次
    for epoch in range(n_epoches):
        # 初始化某一次的经验误差，准确率
        train_l, ac = 0.0, 0.0
        # 随机梯度下降法，将数据分批
        for f, lb in data_iter(data, batch):
            y_hat = net(f, w_hat, b_hat)
            # 计算当前批次经验损失并与上一批的损失加和
            l = Loss(lb, y_hat).sum()
            train_l += l
            # 计算准确率
            ac += accuracy(lb, y_hat)
            # 反向求偏导
            l.backward()
            # 反向优化模型
            sgd([w_hat, b_hat], lr, batch)
            w_hat.grad.data.zero_()
            b_hat.grad.data.zero_()
        # 将误差总和，准确率总和除以训练样本容量
        train_l /= n_features
        ac /= n_features
        print(f"epoch {epoch}, loss:{train_l}, accuracy:{ac}")
    print(f"hat weight:{w_hat}")
    print(f"hat bias:{b_hat}")
    return w_hat, b_hat


def test(test_iter: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    ac = 0.0
    batch_size = len(test_iter)
    for f, lb in data_iter(test_iter, batch_size):
        y_hypo = net(f, w, b)
        ac += accuracy(lb, y_hypo)
    ac /= batch_size
    print(f"test accuracy:{ac}")


def main():
    batch_size = 256
    # 得到的是训练组和测试组
    # 用训练组来训练模型，用测试组来测试模型准确度
    train_iter = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False,
                                                   transform=transforms.ToTensor())
    test_iter = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False,
                                                  transform=transforms.ToTensor())
    # 行数*列数，是一张图的像素总量
    num_inputs = 28*28
    # 标注问题，只有0~9一共10个标注
    num_outputs = 10

    # 初始化权重与偏移
    w = torch.tensor(np.random.normal(
        0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
    b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
    # 训练模型
    w, b = train(10, 0.03, w, b, batch_size, train_iter)
    test(test_iter, w, b)


if __name__ == "__main__":
    main()

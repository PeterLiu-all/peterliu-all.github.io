import matplotlib.pyplot as plt
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
import torchvision
import torchvision.transforms as transforms

# 获得的是Mnist中已经训练好的数据
trained_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False,
                                                transform=transforms.ToTensor())


# 设置每一个数字的映射值
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 定义画图函数
def draw(features, labels):
    # 设置图表大小
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    # 创建子图
    _, axs = plt.subplots(10, len(features) // 10)
    # 调整间距
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=2)
    assert isinstance(axs, np.ndarray)  # 类型断言
    # 将多维数组转化为一维数组
    axs = axs.reshape((1, -1))[0]
    # print(axs)
    # 对每一个子图单独设置它们的像素值和标签
    for ax, img, lbl in zip(axs, features, labels):
        # 像素值放置
        ax.imshow((img.view((28, 28))).numpy())
        # 设置标签
        ax.set_title(lbl)
        # 取消显示轴
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()


# 把训练好的数据连同每一张图对应的识别值打印出来
draw([trained_set[i][0] for i in range(30)], get_fashion_mnist_labels([trained_set[i][1] for i in range(30)]))

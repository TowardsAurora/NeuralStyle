import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import gram_matrix

"""
这段代码定义了一个名为 "ContentLoss "的类，计算输入和目标张量之间的平均平方误差（MSE）损失。目标张量被作为参数传递给类的构造函数，并被存储在类的实例中。
 一步一步的解释：
1. 类 "ContentLoss "定义了一个构造函数，该函数以目标张量为参数。
2. 该构造函数使用 "super "关键字调用父类 "nn.Module "的构造函数。
3. 目标张量使用 "detach "方法存储在类实例中，该方法创建了一个新的张量，共享相同的基础数据，但不需要梯度。这样做是为了防止目标张量在训练过程中被更新。
4. 定义了 "前进 "方法，它需要一个输入张量作为参数。
5. 在 "正向 "方法中，使用PyTorch功能模块中的 "F.mse_loss "函数计算输入张量和目标张量之间的MSE损失。
6. 损失值作为一个名为 "损失 "的变量存储在类实例中。
7. 输入张量从 "前进 "方法中返回。
"""


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


"""
该代码定义了一个名为StyleLoss的类，该类计算输入张量和目标张量的克矩阵之间的平均平方误差损失。
 分步解释：
1. StyleLoss类被定义为nn.Module的一个子类。
2. 该类的构造函数将一个目标张量作为输入，并使用函数gram_matrix()计算其克氏矩阵。然后使用detach()方法将克矩阵从计算图中分离出来，并作为该类实例的一个属性存储。
3. 该类的forward()方法将一个输入张量作为输入，并使用gram_matrix()函数计算其克氏矩阵。
4. 使用F.mse_loss()函数计算输入张量的克氏矩阵和目标张量的克氏矩阵之间的平均平方误差损失，并将其作为类的实例的一个属性存储。
5. 输入张量作为forward()方法的输出被返回。
"""


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


"""
这段代码定义了一个名为 "归一化 "的PyTorch模块，该模块接收平均值和标准差，并使用这些值对输入图像进行归一化。
 分步解释：
1. 定义一个名为 "归一化 "的PyTorch模块。
2. 在构造函数中，将平均值和标准差值初始化为张量，并将其重塑为3维（以匹配输入图像的形状）。
3. 在正向方法中，输入图像并从中减去平均值，然后除以标准差。这就利用所提供的平均值和标准差值对图像进行归一化处理。
4. 返回规范化的图像。
"""


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

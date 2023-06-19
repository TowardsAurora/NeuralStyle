import torch

"""
该代码定义了一个名为 "gram_matrix "的函数，该函数接受一个输入张量并返回该张量的Gram矩阵。
 逐步解释：
1. 函数 "gram_matrix "接受一个输入张量作为参数。
2. 使用 "size "方法提取输入张量的大小，并分配给变量a、b、c和d。
3. 使用 "视图 "方法将输入张量重塑为尺寸为（a * b, c * d）的二维张量。
4. PyTorch库的 "mm "方法被用来在重塑的输入张量和它的转置之间进行矩阵乘法。
5. 得到的矩阵除以a、b、c、d的乘积，得到格拉姆矩阵。
6. Gram矩阵作为该函数的输出被返回。
"""


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

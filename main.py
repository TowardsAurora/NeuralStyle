import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from Functions import ContentLoss, StyleLoss, Normalization
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128  # use small size if not GPU

transformer = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()]
)


# 加载图片
def image_loader(image):
    # fake batch dimension required to fit network's input dimensions
    image = transformer(image).unsqueeze(0)
    return image.to(device, torch.float)


"""
加载图片:
content_img:
style_img:
"""

content_img_name = 'redPanda.jpg'  # 'flower.jpg'  # 'redPanda.jpg'   #'figures.jpg'
content_img_path = './data/content_img/' + content_img_name
content_img_raw = Image.open(content_img_path)

style_img_name = 'water.jpg'  # 'vg_starry_night.jpg'  # 'vg_houses.jpg'
style_img_path = './data/style_img/' + style_img_name
style_img_raw = Image.open(style_img_path)
style_img_raw = style_img_raw.resize(content_img_raw.size, Image.ANTIALIAS)

content_img = image_loader(content_img_raw)
style_img = image_loader(style_img_raw)

print("style_img_size:", style_img.size(), "content_img_size:", content_img.size())

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

unloader = transforms.ToPILImage()


# 展示图像
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


"""
加载模型
"""

"""
该代码初始化了一个名为VGG19的卷积神经网络模型，加载了预先训练好的权重，将该模型移动到指定的设备上，并将该模型设置为评估模式。
 分步解释：
1. `models.vgg19(pretrained=True)`初始化一个带有预训练权重的VGG19模型。
2. `.features`只选择模型的卷积层。
3. `.to(device)`将模型移动到指定的设备上，如GPU或CPU。
4. `.eval()`将模型设置为评估模式，这将禁用dropout和批量规范化层。
"""
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# 1. 变量 "content_layers_default "被设置为一个包含字符串 "conv_4 "的数组。
# 2.变量 "style_layers_default "被设置为一个包含
# "conv_1"、"conv_2"、"conv_3"、"conv_4 "和 "conv_5 "字符串的数组。

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

"""
根据Functions 里面的函数归结出content_loss,style_loss 和模型
"""

"""
下面的代码定义了一个名为 "get_style_model_and_losses "的函数，它接收了一个预先训练好的卷积神经网络（cnn）、归一化平均值和标准偏差、样式和内容图像、以及默认的内容和样式层。该函数创建了一个新的神经网络模型，包括归一化层、内容和风格损失，以及在内容和风格层中指定的来自预训练的cnn的层。然后，该函数返回新模型、样式损失和内容损失。
 分步解释：
1. 函数 "get_style_model_and_losses "接收以下参数：预训练的卷积神经网络（cnn）、归一化平均值和标准偏差、样式和内容图像以及默认的内容和样式层。
2. 该函数使用 "归一化 "函数创建一个归一化层，并将平均值和标准差设置为输入归一化的平均值和标准差。然后，归一化层被移到脚本中指定的设备（如CPU或GPU）。
3. 创建两个空的列表来存储内容和样式的损失。
4. 使用 "nn.Sequential "函数创建一个新的神经网络模型，并将归一化层添加到模型中。
5. 一个计数器变量 "i "被初始化为0。
6. 一个for循环遍历预训练的cnn中的每个层。
7. 如果该层是卷积层，计数器变量 "i "将被递增，并以 "conv_i "的格式为该层命名，其中i是计数器变量的当前值。
8. 如果该层是ReLU激活层，则使用 "relu_i "的格式为其命名，并且用一个不修改输入的新ReLU层替换该层（inplace=False）。
9. 如果该层是一个最大池化层，则使用 "pool_i "的格式分配名称。
10. 如果该层是一个批处理归一化层，则使用 "bn_i "的格式分配名称。
11. 如果该层未被识别，将产生一个运行时错误。
12. 当前层使用指定的名称被添加到模型中。
13. 如果指定的名称在内容层列表中，内容图像通过模型，输出从计算图中分离出来，创建一个目标张量。然后使用目标张量创建一个内容损失，并使用指定的名称以 "content_loss_i "的格式添加到模型中。该内容损失也被添加到content_losses列表中。
14. 如果指定的名称在样式层列表中，那么样式图像就会通过模型，并将输出从计算图中分离出来以创建一个目标张量。然后，使用目标张量创建一个样式损失，并使用格式为 "style_loss_i "的指定名称添加到模型中。风格损失也被添加到style_losses列表中。
15. 一个反向for循环被用来寻找模型中最后一个内容或样式损失层。
16. 然后，模型被截断以移除最后一个内容或样式损失层之后的任何层。
17. 该函数返回新的模型、样式损失和内容损失。
"""


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    torch.save(model.state_dict(), "saved_model/model.pth")
    return model, style_losses, content_losses


"""
定义优化器 optimizer
"""

"""
该代码定义了一个名为 "get_input_optimizer "的函数，该函数接收一个输入图像作为参数，并返回一个优化器对象。
 逐步的解释：
1. 函数 "get_input_optimizer "是用输入参数 "input_img "定义的。
2. 使用PyTorch "优化 "模块中的LBFGS优化器创建优化器对象。
3. 输入图像以列表形式传递给优化器对象。
4. 优化器对象由该函数返回。
"""


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


"""
定义图像风格迁移方法，即训练图像风格迁移
"""

"""
下面的代码是一个函数，它使用预先训练的卷积神经网络（cnn）、内容图像和风格图像对输入图像进行风格转换。该函数对输入图像进行优化，使内容图像与输入图像之间的差异最小，同时使风格图像与输入图像之间的相似度最大。
 一步一步的解释： 
 1.  该函数接收一个Cnn模型、归一化的平均值和标准值、内容图像、样式图像、输入图像、步骤数、样式权重和内容权重作为输入。
 2.  该函数然后调用另一个函数get_style_model_and_losses()，返回预训练的模型、风格损失和内容损失。
 3.  输入图像被设置为需要梯度，而模型被设置为不需要梯度。
 4.  使用get_input_optimizer()函数创建一个优化器。
 5.  该函数然后进入一个循环，运行指定的步数。
 6.  在这个循环中，定义了一个闭合函数，该函数根据当前的输入图像计算出风格分数和内容分数。
 7.  风格分数和内容分数然后分别乘以风格权重和内容权重。
 8.  总损失被计算为风格分数和内容分数之和。
 9.  对损失调用backward()函数来计算梯度。
 10. 运行计数器被递增。
 11. 如果运行计数器是50的倍数，则打印出当前的风格损失和内容损失。
 12. 闭合函数返回总损失。
 13. 以闭合函数为参数调用optimizer.step()函数。
 14. 循环完成后，输入图像被夹在0和1之间并返回。
"""


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):

    print('Building the style transfer model..')
    model, style_losses, content_losses = \
        get_style_model_and_losses(cnn,
                                   normalization_mean, normalization_std, style_img,
                                   content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img




"""
代码显示两张图片，一张是样式图片，另一张是内容图片。
 分步解释：
1. `plt.ion()`激活了matplotlib的交互式模式，可以实时更新绘图。
2. `plt.figure()`创建一个新的图形窗口来显示图像。
3. `imshow(style_img, title='风格图像')`在图形窗口中显示风格图像，标题为'风格图像'。
4. `plt.figure()`创建另一个新的图形窗口来显示下一个图像。
5. `imshow(content_img, title='内容图片')`在新的图形窗口中显示内容图片，标题为'内容图片'。
"""
plt.ion()

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

"""
该代码定义了一个名为saveImg的函数，该函数将一个output_img作为输入，将其转换为CPU张量，删除大小为1的第一维，应用一个卸载函数，提示用户输入文件名，并将生成的图像保存为JPEG文件，放在一个名为 "result "的目录中。
 逐步的解释：
 1. 函数saveImg被定义为有一个输入参数output_img。
 2. 使用cpu()方法将output_img张量转换为CPU张量。
 3. 使用clone()方法创建一个张量的副本。
 4. 使用squeeze()方法从张量中去除第一个尺寸为1的维度。这样做是因为张量很可能有一个批处理的维度，这对于保存单一图像来说是不需要的。
 5. 卸载器函数被应用于张量。这个函数将张量转换为PIL图像，并将其缩减为原始尺寸。
 6. input()函数用来提示用户输入保存图像的文件名。
 7. save()方法用于将图像作为JPEG文件保存在"./result/"目录下，并给出文件名。
 8. 该函数结束，不返回任何东西。
"""


def saveImg(output_img):
    image = output_img.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    # print(type(image))
    name = input("请输入保存后的文件名:")
    image.save('./result/' + name + '.jpg')


"""
main 函数进行运行测试
"""
if __name__ == '__main__':
    input_img = content_img.clone()

    """
    该代码使用预先训练好的卷积神经网络（CNN）运行风格转移算法，将一张图片的风格与另一张图片的内容相结合，并产生一个输出图片。
 分步解释：
     1. 函数 "run_style_transfer "被调用，参数如下：
       - "cnn"：用于风格转换的预训练的CNN模型
       - "cnn_normalization_mean"：CNN模型中用于归一化的均值
       - "cnn_normalization_std"：CNN模型中用于规范化的标准偏差值。
       - "content_img"：包含要转移的内容的图像
       - "style_img": 包含要转移的样式的图片
       - "input_img"：初始图像，作为风格转换的起点。
     2. run_style_transfer "函数执行以下步骤：
       - 用给定的参数初始化一个 "StyleTransfer "对象
       - 调用 "StyleTransfer "对象的 "优化 "方法，以执行样式转移优化。
       - 返回由优化过程产生的输出图像
     3. 输出图像被分配到变量 "output_img"。
    """
    output_img = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img)

    # 这里 output_img输出 type 为 <class 'torch.Tensor'>
    # print(type(output_img))

    plt.figure()
    imshow(output_img, title='Output Image')

    saveImg(output_img)
    print("done!!!")

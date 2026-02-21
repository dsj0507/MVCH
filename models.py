# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
import torch.nn.utils.prune as prune
import math
import os
import joblib
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window,\
                  camel_to_snake
import attention
import time
import einops
import numpy as np
from torch import nn, cat
from einops.layers.torch import Rearrange, Reduce
from ChebConv import _ResChebGC
from CIT import CIT
from Embeddings import PatchEmbeddings, PositionalEmbeddings
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
def get_model(name, **kwargs):#**kwargs：接受任意的关键字参数，会将其参数整体组装变成一个字典进行存储，整体赋值给变量名为**kwargs
    """
    Instantiate and obtain a model with adequate hyperparameters
                实例化并获得具有足够超参数的模型
    Args:
        name: string of the model name
        kwargs: hyperparameters超参数
#超参数也是一个参数，是一个未知变量，但是它不同于在训练过程中的参数，它是可以对训练得到的参数有影响的参数，需要训练者人工输入，并作出调整，以便优化训练模型的效果
    Returns:
        model: PyTorch network  网络优化器
        optimizer: PyTorch optimizer 优化器准则
        criterion: PyTorch loss Function  损失函数
        kwargs: hyperparameters with sane defaults具有正常默认值的超参数
    """
    device = kwargs.setdefault('device', torch.device('cpu'))#往kwargs字典中添加默认值，键名是‘device',键值是'cpu',同时将键值传给device
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    weights = torch.ones(n_classes)#返回一个全为1的张量，输出形状由里面的参数定义
    #weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)

    if name == 'nn':
        kwargs.setdefault('patch_size', 1)#setdefault()是往字典kwargs中传递键名为patch_size，键值为1
        center_pixel = True
        model = Baseline(n_bands, n_classes,
                         kwargs.setdefault('dropout', False))
        lr = kwargs.setdefault('learning_rate', 0.0003)
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])#交叉熵损失函数
        kwargs.setdefault('epoch', 50)
        kwargs.setdefault('batch_size', 64)
    elif name == 'MVCH':
        patch_size = kwargs.setdefault('patch_size', 11)  # 空间领域的大小
        center_pixel = True
        weights = torch.ones(kwargs['n_classes'])
        weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
        weights = weights.to(kwargs['device'])
        kwargs.setdefault('weights', weights)
        kwargs.setdefault('epoch', 10)
        kwargs.setdefault('patch_size', 11)
        kwargs.setdefault('lr', 0.0005)
        kwargs.setdefault('batch_size', 128)
        model = MVCH(HSI_Data_Shape_C=kwargs['n_bands'], classes=kwargs['n_classes'],
                       patch_size=kwargs['patch_size'])
        criterion = FocalLoss()
        model = model.to(kwargs['device'])
        optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=0.0001)  # 权重衰减，让权重衰减到更小的值
        kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                                             patience=kwargs['epoch'] // 10,
                                                                             verbose=True))
        kwargs.setdefault('scheduler', optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs['epoch'], eta_min=1e-9))
        kwargs.setdefault('supervision', 'full')
        kwargs.setdefault('center_pixel', True)
    return model, optimizer, criterion, kwargs


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """


    def __init__(self, num_class=17,alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        if idx.max() >= self.num_class:
            raise ValueError(f"Target index {idx.max().item()} is out of bounds for num_class={self.num_class}")

        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=5):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
            # groups=32
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out

class MVCH(nn.Module):
    def __init__(self, classes,HSI_Data_Shape_C,
                 patch_size, emb_dim: int = 64, num_layers: int = 1,hidden_dim=128,num_heads: int =4, head_dim = 64,dropout = 0.1,
                 act_cfg=dict(type='GELU')):  # band:103  classes=9
        super(MVCH, self).__init__()
        self.classes = classes
        self.band = HSI_Data_Shape_C
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.channels = HSI_Data_Shape_C
        self.hidden_dim=hidden_dim
        self.image_size = 121
        self.num_patches = (self.image_size // patch_size) ** 2
        self.num_patch = int(math.sqrt(self.num_patches))
        patch_dim = 128
        drop=0
        self.relu = nn.ReLU()
        self.dropout = dropout
        """Pixel branch"""
        self.graconv1 = _ResChebGC(input_dim=128, hid_dim=64, n_seq=self.num_patches,
                                p_dropout=0)
        self.graconv2 = _ResChebGC(input_dim=64, hid_dim=emb_dim, n_seq=self.num_patches,
                                  p_dropout=0)

        self.patchify1 = Rearrange('b (h w) c -> b c h w', h=11, w=11)
        self.patchify2 = Rearrange('b c h w -> b (h w) c')

        self.bnc2 = nn.BatchNorm2d(128)
        self.bnc3 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(inplace=True)
        self.SSConv3 = SSConv(64, 64, kernel_size=5)
        self.patch_embeddings = PatchEmbeddings(patch_size=patch_size, patch_dim=patch_dim, emb_dim=emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=emb_dim)
        self.CIT = CIT(dim=emb_dim, num_layers=num_layers, num_heads=num_heads,
                                         head_dim=head_dim, hidden_dim=hidden_dim, num_patch=self.num_patch,
                                         patch_size=patch_size)
        self.Dconv11 = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(emb_dim),
        )
        self.drop = nn.Dropout(drop)
        #self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.finally_fc_classification = nn.Linear(64, self.classes)
        self.depthwise_conv = nn.Sequential(

            nn.Conv2d(self.channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU())

        self.squeeze_conv = nn.Sequential(

            nn.Conv2d(128,64, kernel_size=1, padding=0,bias=False),
            nn.BatchNorm2d(64), nn.GELU())

        self.expand_conv = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size=1, padding=0,bias=False),
            nn.BatchNorm2d(128), nn.GELU())

    def forward(self,X): 
        X = X.cuda()
        X = np.squeeze(X, axis=1)
        out1 = self.depthwise_conv(X)
        out2= self.squeeze_conv(out1)
        out3 = self.expand_conv(out2)
        output_11 = self.graconv1(self.act(self.bnc2(out3)))
        output_2 = self.patchify1(output_11)
        output_3 = self.graconv2(self.act(self.bnc3(output_2)))
        output_222 = self.patchify1(output_3)
        output_33 = self.SSConv3(output_222)
        output_333 = self.patchify2(output_33)
        x31 = self.patch_embeddings(out3)
        x32 = self.pos_embeddings(x31)
        output_4 = self.CIT(x32)
        output5 = output_4 + output_333
        output5= einops.rearrange(output5, "b (h d) co -> b co d h", h=self.patch_size)
        output6 = self.Dconv11(output5)
        output6 = self.drop(output6)
        output6 = torch.mean(output6, dim=(2,3))
        output7 = self.finally_fc_classification(output6)
        output = F.softmax(output7, dim=1)
        return output

##############################################################################################################################
class Pooling(nn.Module):
    """
    @article{ref-vit,
	title={An image is worth 16x16 words: Transformers for image recognition at scale},
	author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai,
            Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
	journal={arXiv preprint arXiv:2010.11929},
	year={2020}
    }
    """

    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

#####################
#########################################################

def train(net, optimizer, criterion, data_loader, epoch, scheduler=None,
          display_iter=100, device=torch.device('cpu'), display=None,
          val_loader=None, supervision='full'):
    """
    Training loop to optimize a network for several epochs and a specified loss
通过几轮epoch和特定的损失，循环训练去优化网络
    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader 数据集加载器
        epoch: int specifying the number of training epochs 指定训练epoch次
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss   损失函数，比如交叉熵损失函数
        device (optional): torch device to use (defaults to CPU)  指要使用的设备
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off). 刷新显示器之前的迭代次数。
        scheduler (optional): PyTorch scheduler  调度器
        val_loader (optional): validation dataset  验证数据集
        supervision (optional): 'full' or 'semi'  半监督或者全监督
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)#指定设备，将模型加载到相应模型中

    save_epoch = epoch // 20 if epoch > 20 else 1


    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []
    Train_Time_ALL = []

    #一个batch是这么训练的
    for e in tqdm(range(1, epoch + 1), desc="Training the network"):#对于迭代对象进行封装，实现可视化进度条
        # Set the network to training mode
        net.train()
        tic1 = time.perf_counter()#计算机系统时间，运行时间
        avg_loss = 0.

        # Run the training loop for one epoch运行一个周期的训练循环
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):#enumerate()在遍历中可以获得索引和元素值，类似for循环
            # Load the data into the GPU if required如果需要将数据加载到GPU中
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()#每一轮batch设置一遍zero_grad，清空过往梯度，梯度置零
            if supervision == 'full':
                output = net(data)
                # target = target - 1
                loss = criterion(output, target)

            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
                #target = target - 1
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[1](rec, data)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))
            loss.backward()#用损失函数来像后传播更新参数梯度值
            optimizer.step()#执行一次优化步骤，通过梯度下降法来更新参数的值

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(#上面需要打印的参数
                    e, epoch, batch_idx *
                    len(data), len(data) * len(data_loader),
                    100. * batch_idx / len(data_loader), mean_losses[iter_])
                update = None if loss_win is None else 'append'
                loss_win = display.line(#画直线图并显示
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter:iter_],
                    win=loss_win,
                    update=update,
                    opts={'title': "Training loss",
                          'xlabel': "Iterations",#迭代次数
                          'ylabel': "Loss"
                         }
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(Y=np.array(val_accuracies),#
                                           X=np.arange(len(val_accuracies)),#每一次epoch有一个精度
                                           win=val_win,
                                           opts={'title': "Validation accuracy",
                                                 'xlabel': "Epochs",
                                                 'ylabel': "Accuracy"
                                                })
            iter_ += 1
            del(data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:#val_loader验证数据集
            val_acc = val(net, val_loader, device=device, supervision=supervision)#val()将一组字符型数据的数字部分转换成相应的数值型数据
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
#isinstance()函数来判断一个对象是否是一个已知的类型，通常用于判断两个类型是否相同
#optim.lr_scheduler.ReduceLROnPlateau 当网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能
#class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#factor=0.1, patience=10,verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
#optimer指的是网络的优化器
# mode (str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
# factor 学习率每次降低多少，new_lr = old_lr * factor
# patience=10，容忍网路的性能不提升的次数，高于这个次数就降低学习率
# verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
# threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
# cooldown： 减少lr后恢复正常操作之前要等待的时期数。 默认值：0。
# min_lr,学习率的下限
# eps ，适用于lr的最小衰减。 如果新旧lr之间的差异小于eps，则忽略更新。 默认值：1e-8。
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e % save_epoch == 0:#训练结束后就保存模型
            save_model(net, camel_to_snake(str(net.__class__.__name__)), data_loader.dataset.name, epoch=e, metric=abs(metric))
        toc1 = time.perf_counter()#返回当前的计算机系统时间
        training_time = toc1 - tic1
        Train_Time_ALL.append(training_time)


    Train_Time_ALL = np.array(Train_Time_ALL)
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("all training time:{}".format(np.sum(Train_Time_ALL)))

def save_model(model, model_name, dataset_name, **kwargs):
     model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
     if not os.path.isdir(model_dir):
         os.makedirs(model_dir, exist_ok=True)
     if isinstance(model, torch.nn.Module):
         filename = str('run') + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
         tqdm.write("Saving neural network weights in {}".format(filename))
         torch.save(model.state_dict(), model_dir + filename + '.pth')
     else:
         filename = str('run')
         tqdm.write("Saving model params in {}".format(filename))
         joblib.dump(model, model_dir + filename + '.pkl')


def test(net, img, hyperparams):
    """
    Test a model on a specific image在特定图像上测试模型
    """
    Test_Time_ALL = []
    net.eval()
    tic2 = time.perf_counter()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),#tqdm是进度条模块
        #sliding_window（）
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():#torch.no_grad()可以屏蔽梯度计算，表示当前计算不需要反向传播
            if patch_size == 1:#patch_size空间领域大小
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)#把数组转换成张量
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)#在第二维 增加维度

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            tic3 = time.perf_counter()
            Testining_time = tic3 - tic2
            # print(Testining_time)
            Test_Time_ALL.append(Testining_time)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')
            if patch_size == 1 or center_pixel:
            #if patch_size == 1:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out
            # tic3 = time.perf_counter()
            # Testining_time = tic3 - tic2
            # # print(Testining_time)
            # Test_Time_ALL.append(Testining_time)

    Test_Time_ALL = np.array(Test_Time_ALL)
    print("Average testining time:{}".format(np.mean(Test_Time_ALL)))
    print("All testining time:{}".format(np.sum(Test_Time_ALL)))

    return probs

def val(net, data_loader, device='cpu', supervision='full'):
# TODO : fix me using metrics()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            #target = target - 1
            for pred, out in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total








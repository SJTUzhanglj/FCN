"""
FCN
"""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision

from dataset import SBDClassSeg, MyTestData
from transform import Colorize
from criterion import CrossEntropyLoss2d
from model import FCN8s
from myfunc import imsave

import visdom
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--param', type=str, default=None, help='path to pre-trained parameters')
parser.add_argument('--data', type=str, default='./train', help='path to input data')
parser.add_argument('--out', type=str, default='./out', help='path to output data')
opt = parser.parse_args()
print(opt)

vis = visdom.Visdom()
win0 = vis.image(torch.zeros(3, 100, 100))
win1 = vis.image(torch.zeros(3, 100, 100))
win2 = vis.image(torch.zeros(3, 100, 100))
win3 = vis.image(torch.zeros(3, 100, 100))
color_transform = Colorize()
"""parameters"""
iterNum = 30

"""data loader"""
# dataRoot = '/media/xyz/Files/data/datasets'
# checkRoot = '/media/xyz/Files/fcn8s-deconv'
dataRoot = opt.data
if not os.path.exists(opt.out):
    os.mkdir(opt.out)
if opt.phase == 'train':
    checkRoot = opt.out
    loader = torch.utils.data.DataLoader(
        SBDClassSeg(dataRoot, split='train', transform=True),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
else:
    outputRoot = opt.out
    loader = torch.utils.data.DataLoader(
        MyTestData(dataRoot, transform=True),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

"""nets"""
model = FCN8s()
if opt.param is None:
    vgg16 = torchvision.models.vgg16(pretrained=True)
    model.copy_params_from_vgg16(vgg16, copy_fc8=False, init_upscore=True)
else:
    model.load_state_dict(torch.load(opt.param))

criterion = CrossEntropyLoss2d()
optimizer = torch.optim.Adam(model.parameters(), 0.0001, betas=(0.5, 0.999))

model = model.cuda()

if opt.phase == 'train':
    """train"""
    for it in range(iterNum):
        epoch_loss = []
        for ib, data in enumerate(loader):
            inputs = Variable(data[0]).cuda()
            targets = Variable(data[1]).cuda()
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss.append(loss.data[0])
            loss.backward()
            optimizer.step()
            if ib % 2 == 0:
                image = inputs[0].data.cpu()
                image[0] = image[0] + 122.67891434
                image[1] = image[1] + 116.66876762
                image[2] = image[2] + 104.00698793
                title = 'input (epoch: %d, step: %d)' % (it, ib)
                vis.image(image, win=win1, env='fcn', opts=dict(title=title))
                title = 'output (epoch: %d, step: %d)' % (it, ib)
                vis.image(color_transform(outputs[0].cpu().max(0)[1].data),
                          win=win2, env='fcn', opts=dict(title=title))
                title = 'target (epoch: %d, step: %d)' % (it, ib)
                vis.image(color_transform(targets.cpu().data),
                          win=win3, env='fcn', opts=dict(title=title))
                average = sum(epoch_loss) / len(epoch_loss)
                print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))
                epoch_loss.append(average)
                x = np.arange(1, len(epoch_loss) + 1, 1)
                title = 'loss (epoch: %d, step: %d)' % (it, ib)
                vis.line(np.array(epoch_loss), x, env='fcn', win=win0,
                         opts=dict(title=title))
        filename = ('%s/FCN-epoch-%d-step-%d.pth' \
                    % (checkRoot, it, ib))
        torch.save(model.state_dict(), filename)
        print('save: (epoch: %d, step: %d)' % (it, ib))
else:
    for ib, data in enumerate(loader):
        print('testing batch %d' % ib)
        inputs = Variable(data[0]).cuda()
        outputs = model(inputs)
        hhh = color_transform(outputs[0].cpu().max(0)[1].data)
        imsave(os.path.join(outputRoot, data[1][0] + '.png'), hhh)

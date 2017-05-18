"""
why so hard to find code, I just want the simplest function and visulization
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import SBDClassSeg
from transform import Relabel, ToLabel, Colorize
from criterion import CrossEntropyLoss2d
from model import FCN32s

import visdom
import numpy as np
import pdb
vis = visdom.Visdom()
win0 = vis.image(torch.zeros(3,100,100))
win1 = vis.image(torch.zeros(3,100,100))
win2 = vis.image(torch.zeros(3,100,100))
win3 = vis.image(torch.zeros(3,100,100))
color_transform = Colorize()
"""parameters"""
iterNum = 3000

"""data loader"""
dataRoot = '/media/xyz/Files/data/datasets'
loader = torch.utils.data.DataLoader(
        SBDClassSeg(dataRoot, split='train', transform=True),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
	
"""nets"""
model = FCN32s()
vgg16 = torchvision.models.vgg16(pretrained=True)
model.copy_params_from_vgg16(vgg16, copy_fc8=False, init_upscore=True)

criterion = CrossEntropyLoss2d()
"""optimizer SGD dosen't work!!!'"""
#optimizer = torch.optim.SGD(
#	model.parameters(),
#	lr=1e-8, momentum=0.99, weight_decay=0.0005)
optimizer = torch.optim.Adam(model.parameters(), 0.0001, betas=(0.5, 0.999))

model = model.cuda()
#model.load_state_dict(torch.load('/media/xyz/Files/fcn-new/FCN-epoch-12-step-11354.pth'))
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
			title = 'input (epoch: {%d}, step: {%d})' %(it, ib)
			vis.image(image, win=win1, env='fcn', opts=dict(title=title))
#			pdb.set_trace()
			title = 'output (epoch: {%d}, step: {%d})' %(it, ib)
#			pdb.set_trace()
			vis.image(color_transform(outputs[0].cpu().max(0)[1].data),
				win=win2, env='fcn', opts=dict(title=title))
			title = 'target (epoch: {%d}, step: {%d})' %(it, ib)
			vis.image(color_transform(targets.cpu().data),
				win=win3, env='fcn', opts=dict(title=title))
			average = sum(epoch_loss) / len(epoch_loss)
			print('loss: {%.4f} (epoch: {%d}, step: {%d})' %(loss.data[0], it, ib))
			epoch_loss.append(average)
			x = np.arange(1, len(epoch_loss)+1, 1)
			title = 'loss (epoch: {%d}, step: {%d})' %(it, ib)
			vis.line(np.array(epoch_loss), x, env='fcn', win = win0, 
			opts=dict(title=title))
	filename = ('/media/xyz/Files/fcn-new/FCN-epoch-%d-step-%d.pth' \
		%(it, ib))
	torch.save(model.state_dict(), filename)
	print('save: (epoch: {%d}, step: {%d})' %(it, ib))



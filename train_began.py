import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
#from torchvision.datasets import MNIST
#from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F

torch.manual_seed(123)


def plot_loss(loss_array,name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_array)
    plt.savefig('loss_'+name)


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def run_trainer(train_loader, netD, netG, args):

    gamma = args.gamma
    lambda_k = args.lambda_k


    free_params(netD)
    free_params(netG)


    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.9,0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4,betas=(0.9,0.999))

    
    #D_scheduler = StepLR(optimizerD, step_size=25, gamma=0.5)
    #G_scheduler = StepLR(optimizerG, step_size=25, gamma=0.5)
    
    one = torch.Tensor([1])
    mone = one * -1

    one = one.cuda()
    mone = mone.cuda()


    noise = torch.FloatTensor(args.batch_size, args.n_z, 1, 1)
    noise = noise.cuda()
    noise = Variable(noise)
    
    #netG.apply(weights_init)
    #netD.apply(weights_init)

    k_t = 0
    

    for epoch in range(1000):
        G_loss_epoch = 0
        recon_loss_epoch = 0
        D_loss_epoch = 0
    
        for i, (images, labels) in enumerate(train_loader):

            #train disc
            for p in netD.parameters():
                p.requires_grad = True

            for p in netG.parameters():
                p.requires_grad = False
            
            netD.zero_grad()

            images = Variable(images)
            images = images.cuda()

            #train disc with real
            output = netD(images)-images
            output = torch.abs(output)
            errD_real = output.mean()

            #train disc with fake
            noise.data.uniform_(-1,1)
            fake = netG(noise)
            output_fake =  netD(fake)-fake
            output = torch.abs(output_fake)

            errD_fake = output.mean()
            errD = errD_real - k_t * errD_fake
            errD.backward()
            optimizerD.step()
            
            D_loss_epoch += errD.data.cpu().item()

            #train G
            #might have to freeze disc params
            for p in netG.parameters():
                p.requires_grad = True

            for p in netD.parameters():
                p.requires_grad = False

            netG.zero_grad()
            
            noise.data.uniform_(-1,1)
            fake = netG(noise)


            errG = netD(fake)-fake
            errG = torch.abs(errG)
            errG = errG.mean()

            errG.backward()
            optimizerG.step()

            G_loss_epoch += errG.data.cpu().item()

            convergence_metric = errD_real.detach()
            convergence_metric += torch.abs(gamma * errD_real.detach() - errG.detach())
            convergence_metric = convergence_metric.data.cpu().item()

            k_t += lambda_k * (gamma * errD_real.detach() - errG.detach())
            k_t = max(min(1.0, k_t),0)

            if epoch % 1 == 0 and i == 5 :
                print('saving images')
                save_image(fake[0:6].data.cpu().detach(), './recon.png')
                save_image(images[0:6].data.cpu().detach(), './orig.png')

                
        if(epoch % 1 == 0):
            print("Epoch, G_loss, D_loss, convergence_metric" 
                  ,epoch + 1, G_loss_epoch, D_loss_epoch,convergence_metric)


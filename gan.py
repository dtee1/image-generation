import argparse
import os
import torch
from torch import nn, optim
import torchvision
from torchvision.utils import make_grid, save_image
import matplotlib
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import torchvision.transforms as transforms

torch.manual_seed(100)
torch.backends.cudnn.deterministic = True


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
class Generator(nn.Module):
    def __init__(self, latent_space, dataset='MNIST'):
        super(Generator, self).__init__()
        self.latent_space = latent_space
        if dataset == 'MNIST':
            self.num_channels = 1
            self.image_size = 28
            self.main = nn.Sequential(
                nn.ConvTranspose2d(self.latent_space, 256, 7, 1, 0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, self.num_channels, 4, 2, 1),
                nn.Tanh()
            )
        elif dataset == 'CIFAR10':
            self.num_channels = 3
            self.image_size = 32
            self.main = nn.Sequential(
                nn.ConvTranspose2d(self.latent_space, 256, 4, 1, 0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, self.num_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        self.apply(weights_init)

    def forward(self, x):
        if dataset == 'MNIST':
            return self.main(x.view(-1, self.latent_space, 1, 1))
        else:
            return self.main(x.view(-1, self.latent_space, 1, 1)).view(-1, self.num_channels, self.image_size, self.image_size)


class Discriminator(nn.Module):
    def __init__(self, dataset='MNIST'):
        super(Discriminator, self).__init__()
        if dataset == 'MNIST':
            self.num_channels = 1
            self.image_size = 28
            self.main = nn.Sequential(
                nn.Conv2d(self.num_channels, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 1, 3, 1, 0),
                nn.Sigmoid()
            )
        elif dataset == 'CIFAR10':
            self.num_channels = 3
            self.image_size = 32
            self.main = nn.Sequential(
                nn.Conv2d(self.num_channels, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)
    
class Gan:
    def __init__(self, dataset='MNIST', device='cuda', gpu='0'):
        self.dataset = dataset
        self.device = torch.device(device)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        self.batch_size = 64
        self.epochs = 100
        self.latent_space = 512
        self.learning_rate = 1e-4
        self.criterion = nn.BCELoss()
        
     
        if self.dataset == 'MNIST':
            self.root = './DATA_MNIST'
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
            self.train_dataset = torchvision.datasets.MNIST(root=self.root, train=True, transform=transform, download=True)
        elif self.dataset == 'CIFAR10':
            self.root = './DATA_CIFAR10'
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5, 0.5),(0.5, 0.5, 0.5))
        ])
            self.train_dataset = torchvision.datasets.CIFAR10(root=self.root, train=True, transform=transform, download=True)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        self.dataloader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.generator = Generator(self.latent_space, self.dataset).to(self.device)
        self.discriminator = Discriminator(self.dataset).to(self.device)

        self.optim_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate,betas=(0.5, 0.999))
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,betas=(0.5, 0.999))


        self.losses_g = []
        self.losses_d = []
        self.images = []
        self.epoch_list = []
        self.jsd_losses_g = []
        self.jsd_losses_d = []

        self.losses_dict = {
        'generator_loss': [],
        'discriminator_loss': [],
        'jsd_loss_discriminator': [],
        'epoch': []
        }
    def plot_loss(self, y_data_g=None,y_data_d=None, y_label_g=None,y_label_d=None, y_label=None,save_path=None):
        fig = plt.figure()
        if y_data_g:
            plt.plot(self.losses_dict['epoch'], y_data_g, color='black', label=y_label_g)
        if y_data_d:
            plt.plot(self.losses_dict['epoch'], y_data_d, color='blue', label=y_label_d)
        plt.legend(loc='best', prop={'size': 10})
        plt.xlabel('epoch')
        plt.ylabel(y_label)
        plt.title(f'GAN_{self.dataset}_{y_label}')
        plt.savefig(save_path)
        plt.close()
        
    def main(self):
        for epoch in range(self.epochs):
            loss_g1 = 0.0
            loss_d1 = 0.0
            jsd_loss_d = 0.0
            jsd_loss_g = 0.0

            for batch_idx, (images, _) in enumerate(tqdm(self.dataloader)):
                real_images = images.to(self.device)
                batch_size = real_images.size(0)

                # Train Discriminator
                self.optim_d.zero_grad()
               
                output_real = self.discriminator(real_images)
                # real_labels = torch.ones_like(output_real).to(self.device)
                real_labels = torch.ones(batch_size, 1, 1, 1).to(self.device)
                loss_real = self.criterion(output_real, real_labels)
               
                fake_images = self.generator(torch.randn(batch_size, self.latent_space).to(self.device))
                fake_labels = torch.zeros(batch_size, 1, 1, 1).to(self.device)

                output_fake = self.discriminator(fake_images.detach())
                loss_fake = self.criterion(output_fake, fake_labels)
                loss_d = loss_real + loss_fake
                loss_d.backward()
                self.optim_d.step()
                self.optim_g.zero_grad()
                output_gen = self.discriminator(fake_images)
                loss_g = self.criterion(output_gen, real_labels)
                loss_g.backward()
                self.optim_g.step()
                loss_d1 += loss_d.item()
                jsd_loss_d += 0.5 * (-loss_d + math.log(4))
                loss_g1 += loss_g.item()

            epoch_loss_g = loss_g1 / len(images)
            epoch_loss_d = loss_d1 / len(images)
            epoch_jsd_loss_d = (jsd_loss_d / len(images)).item()

            self.losses_dict['generator_loss'].append(epoch_loss_g)
            self.losses_dict['discriminator_loss'].append(epoch_loss_d)
            self.losses_dict['jsd_loss_discriminator'].append(epoch_jsd_loss_d)
            self.losses_dict['epoch'].append(epoch + 1)

            self.plot_loss(y_data_g=self.losses_dict['generator_loss'],y_data_d=self.losses_dict['discriminator_loss'], y_label_g='Generator loss',y_label_d='Discriminator loss', y_label='Generator and Discriminator Loss',save_path=f'gan_loss_{self.dataset.lower()}.png')
            self.plot_loss(y_data_d=self.losses_dict['jsd_loss_discriminator'],y_label_d='JSD loss', y_label='JSD Loss',save_path=f'gan_jsd_loss_{self.dataset.lower()}.png')
            last_epoch = epoch == (self.epochs - 1)
            last_batch = batch_idx == len(self.dataloader) - 1
            if last_epoch and last_batch:
                with torch.no_grad():
                    fake_samples = self.generator(torch.randn(batch_size, self.latent_space).to(self.device))
                    fake_grid = make_grid(fake_images.cpu(), padding=2, normalize=True)
                    save_image(fake_grid, f"gan_generated_last_epoch.png")

                    original_grid = make_grid(real_images.cpu(), padding=2, normalize=True)
                    save_image(original_grid, f"gan_original_last_epoch.png")
                
if __name__ == "__main__":
    dataset="CIFAR10"

    gan = Gan(dataset=dataset)
    gan.main()
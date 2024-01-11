import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import torchvision.transforms as transforms
import os


torch.manual_seed(100)
torch.backends.cudnn.deterministic = True

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

class Generator(nn.Module):
    def __init__(self, latent_space, channels=1):
        super(Generator, self).__init__()
        self.latent_space = latent_space
        self.num_channels = channels

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
        # self.main = nn.Sequential(
        #     nn.ConvTranspose2d(self.latent_space, 256, 4, 1, 0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 128, 4, 2, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, 64, 4, 2, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, self.num_channels, 4, 2, 1),
        #     nn.Tanh()
        # )
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x.view(-1, self.latent_space, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator, self).__init__()
        self.num_channels = channels

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
        )
        # self.main = nn.Sequential(
        #     nn.Conv2d(self.num_channels, 64, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(64, 128, 4, 2, 1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(128, 256, 4, 2, 1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(256, 1, 4, 1, 0)
        # )
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)

class WGAN_GP:
    def __init__(self, dataset="MNIST", latent_space=2, device='cuda', gpu='0'):
        self.dataset=dataset
        self.latent_space = latent_space
        
        self.device = torch.device(device)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        self.batch_size = 64
        self.epochs = 20
        self.learning_rate = 5e-5
        self.gradient_penalty_coe = 0.5
        
        if dataset == 'MNIST':
            root='./DATA_MNIST'
            input_dim = 1 * 28 * 28
            self.num_channels = 1
            img_size = 28
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.train_dataset = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
            
        elif dataset == 'CIFAR10':
            root='./DATA_CIFAR10'
            input_dim = 3 * 32 * 32
            self.num_channels = 3
            img_size = 32
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5, 0.5),(0.5, 0.5, 0.5))
            ])
            self.train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=True)

        self.dataloader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.generator = Generator(self.latent_space, channels=self.num_channels).to(self.device)
        self.discriminator = Discriminator(channels=self.num_channels).to(self.device)

        self.optim_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.losses_g = []
        self.losses_d = []
        self.emd_distances = []
    def gradient_penalty(self, real_data, generated_data):
        alpha = torch.rand(real_data.size(0), 1, 1, 1).to(self.device)
        interpolates = alpha * real_data + (1 - alpha) * generated_data
        interpolates.requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones_like(d_interpolates).to(self.device)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                        grad_outputs=fake, create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self):
        for epoch in range(self.epochs):
            print(epoch)
            g_loss = 0.0
            d_loss = 0.0
            emd_distance = 0.0
            for batch_idx, (images, _) in enumerate(tqdm(self.dataloader)):
                # real_images = real_data[0].to(self.device)
                real_images = images.to(self.device)
                self.optim_d.zero_grad()

                fake_images = self.generator(torch.randn(real_images.size(0), self.latent_space, 1, 1).to(self.device))
                real_logits = self.discriminator(real_images)
                fake_logits = self.discriminator(fake_images.detach())

                d_loss = (fake_logits.mean() - real_logits.mean() + self.gradient_penalty_coe * self.gradient_penalty(real_images, fake_images))
                d_loss.backward()
                self.optim_d.step()

                if batch_idx % 5 == 0:
                    self.optim_g.zero_grad()
                    generated_images = self.generator(torch.randn(real_images.size(0), self.latent_space, 1, 1).to(self.device))
                    gen_logits = self.discriminator(generated_images)
                    g_loss = -gen_logits.mean()
                    g_loss.backward()
                    self.optim_g.step()

            self.losses_d.append(d_loss.item())
            self.losses_g.append(g_loss.item())
            emd_distance = (torch.abs(fake_logits.mean() - real_logits.mean()))/batch_idx
                
  
            self.emd_distances.append(emd_distance.item())
            last_epoch = epoch == (self.epochs - 1)
            last_batch = batch_idx == len(self.dataloader) - 1
            if last_epoch and last_batch:
                with torch.no_grad():
                    fake_grid = make_grid(fake_images.cpu(), padding=2, normalize=True)
                    save_image(fake_grid, f"wgan_generated_last_epoch.png")

                    original_grid = make_grid(real_images.cpu(), padding=2, normalize=True)
                    save_image(original_grid, f"wgan_original_last_epoch.png")

    def plot_losses(self):
        plt.plot(self.emd_distances, label='EMD Distance')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.title(f'WGAN_EMD Loss')
        plt.legend()
        plt.savefig(f"emd_distance_{self.dataset}")
        plt.close()

if __name__ == "__main__":
    dataset="MNIST"
    latent_space=64
    wgan_gp = WGAN_GP(dataset=dataset,latent_space=latent_space)
    wgan_gp.train()
    wgan_gp.plot_losses()

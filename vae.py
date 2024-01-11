import argparse
import os

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image, make_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms

torch.manual_seed(100)
torch.backends.cudnn.deterministic = True

class VAE(nn.Module):
    def __init__(self, input_dim, latent_space, batch_size=128, num_channels=1, img_size=28, dataset='MNIST'):
        super(VAE, self).__init__()
        
        # Model parameters
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.to(self.device)

         # Encoder layers
        self.encoder_layer_1 = nn.Linear(input_dim, 512)
    
        self.encoder_layer_2 = nn.Linear(512, 256)
       
        self.mean_layer = nn.Linear(256, latent_space)
        self.log_variance_layer = nn.Linear(256, latent_space)


        # Decoder layers
        self.decoder_layer_one = nn.Linear(latent_space, 256)
    
        self.decoder_layer_two = nn.Linear(256, 512)

        self.decoder_layer_three = nn.Linear(512, input_dim)

    def encoder(self, x):
        h = F.relu(self.encoder_layer_1(x))

        h = F.relu(self.encoder_layer_2(h))

        mean = self.mean_layer(h)
        log_variance = self.log_variance_layer(h)
        return mean, log_variance

    def reparameterize(self, mean, log_variance):
        std = torch.exp(0.5*log_variance) 
        eps = torch.randn_like(std)
        sample = mean + eps * std
        return sample

    def decoder(self, z):
        h = F.relu(self.decoder_layer_one(z))

        h = F.relu(self.decoder_layer_two(h))

        reconstruction = torch.sigmoid(self.decoder_layer_three(h))
        return reconstruction


    def reparameterize(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        sample = mean + eps * std
        return sample


    def forward(self, x):
        mean, log_variance = self.encoder(x.view(-1,  self.num_channels * self.img_size * self.img_size ))
        z = self.reparameterize(mean, log_variance)
        return self.decoder(z), mean, log_variance 
    
    def train(self, model, optimizer, train_loader):
        """
            Training the VAE model.

            Parameters:
            - model (VAE): VAE model to train
            - optimizer (torch.optim.Optimizer): Optimizer for training
            - train_loader (torch.utils.data.DataLoader): DataLoader for training data

            Returns:
            - avg_binary_cross_entropy (float): Training loss for Binary Cross Entropy
            - avg_kl_divergence (float): Training loss for Kullback-Leibler Divergence
            - avg_total_loss (float): Total training loss
        """
        avg_mse_loss= 0.0
        avg_kl_divergence = 0.0

        for images, _ in tqdm(train_loader):
            images = images.to(self.device)
            images = images.reshape(-1, self.num_channels * self.img_size * self.img_size)
        
            
            reconstructed_input, mean, log_variance = model(images)
            reconstructed_input = reconstructed_input.view(-1, self.num_channels * self.img_size * self.img_size)
            
            mse_loss = F.mse_loss(reconstructed_input, images, reduction='sum')
            kl_divergence = -0.5 * torch.sum(1+ log_variance - mean.pow(2) - log_variance.exp())

            total_loss = mse_loss + kl_divergence
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            avg_mse_loss += mse_loss.item()
            avg_kl_divergence += kl_divergence.item()
      
        avg_mse_loss= avg_mse_loss / len(train_loader.dataset)
        avg_kl_divergence = avg_kl_divergence / len(train_loader.dataset)
        avg_total_loss = avg_mse_loss + avg_kl_divergence

        return avg_mse_loss, avg_kl_divergence, avg_total_loss

    def evaluate(self, model, test_loader, last_epoch):
        """
        Evaluating the VAE model on the test set.

        Parameters:
        - model (VAE): VAE model to evaluate.
        - test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        - epoch (int): Current epoch number.

        Returns:
        - avg_mse_loss (float): MSE Loss.
        - avg_kl_divergence (float): Test loss for Kullback-Leibler Divergence.
        - avg_total_loss (float): Total test loss.
        """
        avg_mse_loss = 0.0
        avg_kl_divergence = 0.0

        with torch.no_grad():
            for batch_idx, (images, _ ) in enumerate(test_loader):
                images = images.to(self.device)
                
                reconstructed_input, mean, log_variance = model(images)
                reconstructed_input = reconstructed_input.view(-1, self.num_channels * self.img_size * self.img_size)
                images = images.view(-1, self.num_channels * self.img_size * self.img_size)
                mse_loss = F.mse_loss(reconstructed_input, images, reduction='sum')
                kl_divergence = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

                avg_mse_loss += mse_loss.item()
                avg_kl_divergence += kl_divergence.item()

                if last_epoch and batch_idx == int(len(test_loader.dataset)/self.batch_size) - 1:
                    reconstructed_input = reconstructed_input.view(self.batch_size, self.num_channels, self.img_size, self.img_size)[:64]
                    images = images.view(self.batch_size, self.num_channels, self.img_size, self.img_size)[:64]
                    generated_img = make_grid(reconstructed_input, padding=2, normalize=True)
                    original_img = make_grid(images, padding=2, normalize=True)
                    save_image(generated_img, f"vae_{self.dataset}_reconstructed.png")
                    save_image(original_img, f"vae_{self.dataset}_original.png")

        avg_mse_loss = avg_mse_loss / len(test_loader.dataset)
        avg_kl_divergence = avg_kl_divergence / len(test_loader.dataset)
        avg_total_loss = avg_mse_loss + avg_kl_divergence

        return avg_mse_loss, avg_kl_divergence, avg_total_loss

def main(dataset='MNIST'):
    batch_size = 128
    epochs = 100
    learning_rate = 1e-3
    latent_space = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 'MNIST':
        root='./DATA_MNIST'
        input_dim = 1 * 28 * 28
        num_channels = 1
        img_size = 28
        train_dataset = torchvision.datasets.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=True)
    elif dataset == 'CIFAR10':
        root='./DATA_CIFAR10'
        input_dim = 3 * 32 * 32
        num_channels = 3
        img_size = 32
     
        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=transforms.ToTensor(), download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = VAE(input_dim=input_dim, latent_space=latent_space, batch_size=batch_size, num_channels = num_channels, img_size=img_size, dataset=dataset).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_metrics = {'avg_mse_loss': [], 'kl_divergence': [], 'total_loss': []}
    test_metrics = {'avg_mse_loss': [], 'kl_divergence': [], 'total_loss': []}
    epoch_list = []

    for epoch in range(epochs):
        print(epoch)
        train_avg_mse_loss, train_kl_divergence, train_total_loss = model.train(model, optimizer, train_loader)
        test_avg_mse_loss, test_kl_divergence, test_total_loss = model.evaluate(model, test_loader, last_epoch=(epoch == epochs - 1))
        
        train_metrics['avg_mse_loss'].append(train_avg_mse_loss)
        train_metrics['kl_divergence'].append(train_kl_divergence)
        train_metrics['total_loss'].append(train_total_loss)

        test_metrics['avg_mse_loss'].append(test_avg_mse_loss)
        test_metrics['kl_divergence'].append(test_kl_divergence)
        test_metrics['total_loss'].append(test_total_loss)
        
        epoch_list.append(epoch + 1)

        plot_metrics(epoch_list, train_metrics, test_metrics, dataset)

def plot_loss(epoch_list, train_loss, test_loss, y_label, save_path, dataset):
    fig = plt.figure()
    plt.plot(epoch_list, train_loss, color='black', label="train loss")
    plt.plot(epoch_list, test_loss, color='blue', label="test loss")
    plt.legend(loc='best', prop={'size': 10})
    plt.xlabel('epoch')
    plt.ylabel(y_label)
    plt.title(f'VAE {dataset} {y_label}')
    plt.savefig(save_path)
    plt.close()

def plot_metrics(epoch_list, train_metrics, test_metrics, dataset):
    plot_loss(epoch_list, train_metrics['total_loss'], test_metrics['total_loss'], 'Reconstruction Loss and KL Divergence', f'vae_total_loss_{dataset.lower()}.png', dataset)
    plot_loss(epoch_list, train_metrics['avg_mse_loss'], test_metrics['avg_mse_loss'], 'MSE', f'vae_MSE_{dataset.lower()}.png', dataset)
    plot_loss(epoch_list, train_metrics['kl_divergence'], test_metrics['kl_divergence'], 'KL Divergence', f'vae_KLD_{dataset.lower()}.png', dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="MNIST", type=str, choices=["MNIST", "CIFAR10"])
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(dataset=args.dataset)
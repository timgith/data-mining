import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
import numpy as np
import pickle

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


classes = dataset.classes  # ['airplane', 'automobile', 'bird', ..., 'truck']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 20
vae = VAE(latent_dim=latent_dim).to(device)
vae.eval()

all_embeddings = []
all_labels = []
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        z, _, _ = vae(images)
        all_embeddings.append(z.cpu().numpy()) 
        all_labels.extend(labels.numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)
all_label_names = [classes[label] for label in all_labels]

output_data = {
    'embeddings': all_embeddings,
    'names': all_label_names
}

output_path = './cifar10_embeddings.pickle'
with open(output_path, 'wb') as f:
    pickle.dump(output_data, f)

print(f"Embeddings & labels save as {output_path}")


embeddings_path = './cifar10_embeddings.pickle'
with open(embeddings_path, 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings']
labels = data['names']


print(f"Embeddings shape: {embeddings.shape}")  
print(f"First 10 labels: {labels[:10]}")  

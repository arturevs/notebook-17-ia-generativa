import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from icrawler.builtin import BingImageCrawler
import numpy as np
from tqdm import tqdm
import logging

logging.getLogger("icrawler").setLevel(logging.ERROR)
logging.getLogger("parser").setLevel(logging.ERROR)
logging.getLogger("downloader").setLevel(logging.ERROR)
logging.getLogger("feeder").setLevel(logging.ERROR)

DATA_DIR = "data/images"
OUTPUT_DIR = "output"
KEYWORD = "landscape painting"
IMAGE_SIZE = 64
BATCH_SIZE = 8
Z_DIM = 100
NUM_EPOCHS = 150
LR = 5e-5
CLIP_VALUE = 0.01
N_CRITIC = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_images(keyword, folder, n_total=200):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    existing = len([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if existing >= n_total:
        print(f"Found {existing} images, skipping download.")
        return

    print(f"Downloading images for '{keyword}'...")
    try:
        crawler = BingImageCrawler(storage={'root_dir': folder})
        crawler.crawl(keyword=keyword, max_num=n_total)
    except Exception as e:
        print(f"Crawler failed: {e}")

    existing = len([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if existing == 0:
        print("Download failed or no images found. Generating synthetic data for testing...")
        for i in range(n_total):
            img = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(folder, f"synthetic_{i}.png"))
    
    print("Data preparation complete!")

class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_g=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, feature_g * 8, 4, 1, 0),
            self._block(feature_g * 8, feature_g * 4, 4, 2, 1),
            self._block(feature_g * 4, feature_g * 2, 4, 2, 1),
            self._block(feature_g * 2, feature_g, 4, 2, 1),
            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(feature_d, feature_d * 2, 4, 2, 1),
            self._block(feature_d * 2, feature_d * 4, 4, 2, 1),
            self._block(feature_d * 4, feature_d * 8, 4, 2, 1),
            nn.Conv2d(feature_d * 8, 1, 4, 1, 0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def train():
    download_images(KEYWORD, DATA_DIR, n_total=1000)
    
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    
    dataset = ImageDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    gen = Generator(Z_DIM).to(DEVICE)
    critic = Critic().to(DEVICE)
    initialize_weights(gen)
    initialize_weights(critic)
    
    opt_gen = optim.RMSprop(gen.parameters(), lr=LR)
    opt_critic = optim.RMSprop(critic.parameters(), lr=LR)
    
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
    
    print("Starting Training...")
    step = 0
    
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(dataloader, leave=True)
        for batch_idx, real in enumerate(loop):
            real = real.to(DEVICE)
            cur_batch_size = real.shape[0]
            
            for _ in range(N_CRITIC):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)
                fake = gen(noise)
                
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()
                
                for p in critic.parameters():
                    p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
            
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(loss_d=loss_critic.item(), loss_g=loss_gen.item())
            
            step += 1
            
        with torch.no_grad():
            fake = gen(fixed_noise)
            img_grid = vutils.make_grid(fake[:32], normalize=True)
            vutils.save_image(img_grid, f"{OUTPUT_DIR}/epoch_{epoch}.png")

    print("Training Finished!")

if __name__ == "__main__":
    train()

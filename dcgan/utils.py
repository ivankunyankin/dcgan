import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class SimpsonsDataset(Dataset):
    def __init__(self, data_dir, img_size, num_channels):
        super(SimpsonsDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(img_size),
             transforms.Normalize([0.5 for _ in range(num_channels)], [1 for _ in range(num_channels)])
            ])
        self.collection = []
        for filename in os.listdir(data_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                self.collection.append(os.path.join(data_dir, filename))

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, item):
        image = Image.open(self.collection[item])
        return self.transform(image)


def load_checkpoint(gen, disc, opt_gen, opt_disc, checkpoint_path, device):
    if os.path.exists(os.path.join(checkpoint_path, "disc_model.pt")):
        disc.load_state_dict(torch.load(os.path.join(checkpoint_path, "disc_model.pt"), map_location=device))
        opt_disc.load_state_dict(torch.load(os.path.join(checkpoint_path, "disc_optim.pt"), map_location=device))
        gen.load_state_dict(torch.load(os.path.join(checkpoint_path, "gen_model.pt"), map_location=device))
        opt_gen.load_state_dict(torch.load(os.path.join(checkpoint_path, "gen_optim.pt"), map_location=device))
        print("=> Checkpoints loaded!")
        return gen, disc, opt_gen, opt_disc
    else:
        print("!No checkpoint found. Training from scratch...")
        return gen, disc, opt_gen, opt_disc


def save_checkpoints(gen, disc, opt_gen, opt_disc, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(disc.state_dict(), os.path.join(checkpoint_path, f"disc_model.pt"))
    torch.save(opt_disc.state_dict(), os.path.join(checkpoint_path, f"disc_optim.pt"))
    torch.save(gen.state_dict(), os.path.join(checkpoint_path, f"gen_model.pt"))
    torch.save(opt_gen.state_dict(), os.path.join(checkpoint_path, f"gen_optim.pt"))

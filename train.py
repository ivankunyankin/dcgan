import os
import yaml
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
from IPython.display import clear_output
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

from model import Discriminator, Generator, initialize_weights
from utils import load_checkpoint, save_checkpoints, SimpsonsDataset


def main(config):

    # Hyperparameters etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SimpsonsDataset(config["DATA_DIR"], config["IMAGE_SIZE"], config["NUM_CHANNELS"])
    dataloader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=True)

    gen = Generator(config["NOISE_DIM"], config["NUM_CHANNELS"], config["FEATURES_GEN"]).to(device)
    disc = Discriminator(config["NUM_CHANNELS"], config["FEATURES_DISC"]).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=float(config["LEARNING_RATE"]), betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=float(config["LEARNING_RATE"]), betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    if config["FROM_CHECKPOINT"]:
        gen, disc, opt_gen, opt_disc = load_checkpoint(gen, disc, opt_gen, opt_disc, config["CHECKPOINT_PATH"], device)

    fixed_noise = torch.randn(32, config["NOISE_DIM"], 1, 1).to(device)
    writer_real = SummaryWriter(os.path.join(config["LOG_PATH"], "real"))
    writer_fake = SummaryWriter(os.path.join(config["LOG_PATH"], "fake"))

    for epoch in range(1, config["NUM_EPOCHS"]+1):
        loop = tqdm(dataloader)
        for batch_idx, real in enumerate(loop):
            loop.set_description(f"Epoch {epoch}")

            real = real.to(device)
            noise = torch.randn(config["BATCH_SIZE"], config["NOISE_DIM"], 1, 1).to(device)
            fake = gen(noise)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            loop.set_postfix(loss_gen=loss_gen.item(), disc_gen=loss_disc.item())

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    writer_real.add_image(f"Real_epoch_{epoch}", img_grid_real, global_step=batch_idx)
                    writer_fake.add_image(f"Fake_epoch_{epoch}", img_grid_fake, global_step=batch_idx)
    print("=> Saving checkpoints")
    save_checkpoints(gen, disc, opt_gen, opt_disc, config["CHECKPOINT_PATH"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="Configuration file path")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    main(config)

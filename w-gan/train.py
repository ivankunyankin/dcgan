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

from model import Critic, Generator, initialize_weights
from utils import load_checkpoint, save_checkpoints, SimpsonsDataset, get_gradient, gradient_penalty


def main(config, upsample):

    # Hyperparameters etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SimpsonsDataset(config["DATA_DIR"], config["IMAGE_SIZE"])
    dataloader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=True)

    gen = Generator(config["NOISE_DIM"], config["NUM_CHANNELS"], config["FEATURES_GEN"], upsample=upsample).to(device)
    crit = Critic(config["NUM_CHANNELS"], config["FEATURES_CRIT"]).to(device)
    initialize_weights(gen)
    initialize_weights(crit)

    opt_gen = optim.Adam(gen.parameters(), lr=float(config["LEARNING_RATE"]), betas=(config["BETA_1"], config["BETA_2"]))
    opt_crit = optim.Adam(crit.parameters(), lr=float(config["LEARNING_RATE"]), betas=(config["BETA_1"], config["BETA_2"]))
    # criterion = nn.BCELoss()

    if config["FROM_CHECKPOINT"]:
        gen, crit, opt_gen, opt_crit = load_checkpoint(gen, crit, opt_gen, opt_crit, config["CHECKPOINT_PATH"], device)

    fixed_noise = torch.randn(32, config["NOISE_DIM"], 1, 1).to(device)
    writer_real = SummaryWriter(os.path.join(config["LOG_PATH"], "real"))
    writer_fake = SummaryWriter(os.path.join(config["LOG_PATH"], "fake"))

    for epoch in range(1, config["NUM_EPOCHS"]+1):
        loop = tqdm(dataloader)
        for batch_idx, real in enumerate(loop):
            loop.set_description(f"Epoch {epoch}")

            real = real.to(device)
            repeat_critic_loss = 0
            for _ in range(config["CRIT_REPEATS"]):
                noise = torch.randn(len(real), config["NOISE_DIM"], 1, 1).to(device)
                fake = gen(noise)
                crit_real = crit(real)
                crit_fake = crit(fake.detach())

                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)

                loss_crit = -(crit_real.mean() - crit_fake.mean()) + config["C_LAMBDA"] * gp

                opt_crit.zero_grad()
                loss_crit.backward()
                opt_crit.step()

                repeat_critic_loss += loss_crit
            critic_loss = repeat_critic_loss / config["CRIT_REPEATS"]

            noise = torch.randn(len(real), config["NOISE_DIM"], 1, 1).to(device)
            fake = gen(noise)
            output = crit(fake)
            loss_gen = -output.mean()
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            loop.set_postfix(loss_gen=loss_gen.item(), loss_crit=critic_loss.item())

            # Print losses occasionally and print to tensorboard
            if batch_idx == len(dataloader) - 1:
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    writer_real.add_image("Real", img_grid_real, global_step=epoch)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=epoch)

    print("=> Saving checkpoints")
    save_checkpoints(gen, crit, opt_gen, opt_crit, config["CHECKPOINT_PATH"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="Configuration file path")
    parser.add_argument("--upsample", action="store_true", help="Whether to use upsampling + conv instead of deconv")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    upsample = args.upsample
    main(config, upsample)

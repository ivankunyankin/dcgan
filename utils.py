import os
import torch


def save_checkpoint(self, path): #, postfix=""):

    if not os.path.exists(path):
        os.mkdir(path)

    torch.save(self.model.module.state_dict(), path) # os.path.join(path, f"model_{postfix}.pt"))
    torch.save(self.optimizer.state_dict(), path) # os.path.join(path, f"optimizer_{postfix}.pt"))

import sys

import torch
import lightning as L

from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins.environments import KubeflowEnvironment

from model import LitEfficientNet
from datamodule import ISICDataModule


torch.set_float32_matmul_precision("high")


def main():
    model = LitEfficientNet()
    datamodule = ISICDataModule()

    environment = KubeflowEnvironment()
    strategy = DDPStrategy(cluster_environment=environment)
    trainer = L.Trainer(
        max_epochs=4,
        precision="bf16-mixed",
        accelerator="gpu",
        num_nodes=2,
        strategy=strategy)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    sys.exit(main())


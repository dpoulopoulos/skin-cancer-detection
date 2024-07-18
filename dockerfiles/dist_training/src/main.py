import sys

import torch
import lightning as L

from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import KubeflowEnvironment

from model import LitEfficientNet
from datamodule import ISICDataModule


torch.set_float32_matmul_precision("high")


def main():
    model = LitEfficientNet()
    datamodule = ISICDataModule()

    # setup the TensorBoard logger
    logger = TensorBoardLogger(
        "/logs/tb_logs/",
        name="isic-competition",
        flush_secs=60)

    # setup distributed training on a Kubeflow cluster
    environment = KubeflowEnvironment()
    strategy = DDPStrategy(cluster_environment=environment)

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="valid_loss",
        mode="min",
        dirpath="/logs/checkpoints/",
        filename="sample-isic-{epoch:02d}-{val_loss:.5f}")

    trainer = L.Trainer(
        max_epochs=4,
        precision="bf16-mixed",
        accelerator="gpu",
        num_nodes=2,
        strategy=strategy,
        logger=logger,
        callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    sys.exit(main())


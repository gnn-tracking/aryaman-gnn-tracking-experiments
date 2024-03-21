import torch
from gnn_tracking.training.callbacks import ExpandWandbConfig, PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.nomenclature import random_trial_name
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

name = random_trial_name()


logger = WandbLogger(
    project="aryaman-gnn-experiments",
    group="nc-fcnn",
    offline=True,
    version=name,
    tags=["first-fcnn-nc-run"],
)


tb_logger = TensorBoardLogger(".", version=name)

def cli_main():
    torch.set_float32_matmul_precision("medium")

    # noinspection PyUnusedLocal
    cli = LightningCLI(  # noqa: F841
        datamodule_class=TrackingDataModule,
        trainer_defaults={
            "callbacks": [
                RichProgressBar(leave=True),
                TriggerWandbSyncLightningCallback(),
                PrintValidationMetrics(),
                ExpandWandbConfig(),
                EarlyStopping(monitor="total", mode="min", patience=20),
                ModelCheckpoint(save_top_k=2, monitor="total", mode="min"),
            ],
            "logger": [tb_logger, logger],
            "plugins": [SLURMEnvironment(auto_requeue=False)],
        },
    )


if __name__ == "__main__":
    cli_main()

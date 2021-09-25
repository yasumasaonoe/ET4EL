from argparse import ArgumentParser
from os import makedirs, path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from et4el.data import EntityTypingDataModule
from et4el.typer import FineGrainedEntityTyper

LOGS_DIR = path.normpath(path.join(path.dirname(__file__), "../logs/"))
MODELS_DIR = path.normpath(path.join(path.dirname(__file__), "../models/"))
makedirs(LOGS_DIR, exist_ok=True)
makedirs(MODELS_DIR, exist_ok=True)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = FineGrainedEntityTyper.add_model_specific_args(parser)
    parser = EntityTypingDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dict_args = vars(args)
    dm = EntityTypingDataModule(**dict_args)

    # ------------
    # model
    # ------------
    dict_args = vars(args)
    model = FineGrainedEntityTyper(**dict_args)

    # ------------
    # logger
    # ------------
    wandb_logger = WandbLogger(project="ET4EL", log_model=True, save_dir=LOGS_DIR)

    # ------------
    # callbacks
    # ------------
    callbacks = [
        EarlyStopping(monitor="validation_accuracy"),
        ModelCheckpoint(monitor="validation_accuracy", dirpath=MODELS_DIR)
    ]

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=wandb_logger)

    # log gradients and model topology
    wandb_logger.watch(model)

    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(datamodule=dm)
    print(result)


if __name__ == '__main__':
    cli_main()

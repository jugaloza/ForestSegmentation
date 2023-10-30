from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from models import UNet,UNetModel
from Loss import DiceLoss
from dataset import GenerateData,ForestDataModule
from torch.utils.data import DataLoader
import numpy as np
import lightning.pytorch as pl
import argparse
from lightning.pytorch.callbacks import EarlyStopping



if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()

    ap.add_argument("--isRestart","-re",type=bool,default=False,help="Specify true if training needs to be started from previous checkpoints")
    ap.add_argument("--CheckPointPath","-path",type=str,help="Specify path of checkpoint from where to start training")
    args = vars(ap.parse_args())
    isRestart = args['isRestart']
    chk_path = args['CheckPointPath']

    data = ForestDataModule(batch_size=4)
    model = UNetModel()

    #trainer = pl.Trainer(max_epochs=50,accelerator="gpu",devices=1,callbacks=[EarlyStopping(monitor="val_loss",mode="min",min_delta=0.001,patience=3)])
    trainer = pl.Trainer(max_epochs=500,accelerator="gpu",devices=1)
    if isRestart:
        trainer.fit(model,data,ckpt_path=chk_path)

    else:
        trainer.fit(model,data)


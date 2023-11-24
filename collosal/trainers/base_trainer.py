import cv2
import torch
import imageio
import time
from loguru import logger
import wandb
import numpy as np
from collosal.models.Image import ImageModel

class BaseTrainer:
    def __init__(self,args,model,optimizer,scheduler=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_metrics = {}
        self.test_metrics = {}
        self.epoch_counter = 0
    
    def train(self):
        start_t = time.time()
        
        for epoch in range(self.args.epochs):
            logger.info(f"Epoch {epoch} of {self.args.epochs}")
            self.epoch_counter = epoch
            train_metrics = self.train_one_epoch(epoch)
            wandb.log(train_metrics)
        
        self.save_checkpoint()
        end_t = time.time()
        wandb.log({
            "timing/total_time":(end_t-start_t)
        })
    def save_checkpoint(self):
        import os
        os.makedirs(f"/home/paulj/checkpoints/collosal/{self.args.workspace}",exist_ok=True)
        torch.save(self.model.state_dict(),f"/home/paulj/checkpoints/collosal/{self.args.workspace}/model.pt")
        
        
            
    
    def train_one_epoch(self,epoch):
        pass
        
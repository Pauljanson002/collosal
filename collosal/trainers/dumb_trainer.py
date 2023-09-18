import cv2
import torch
import imageio
import time
from loguru import logger
import wandb
import numpy as np
from torchvision import transforms
from PIL import Image


class Trainer:
    def __init__(self,args,model,guidance,optimizer,lr_scheduler=None):
        self.args = args
        self.model = model
        self.guidance = guidance
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.embeddings = {}
        self.prepare_embeddings()
        self.all_preds = []
        transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
        image_path = '/home/paulj/projects/collosal/output.png'
        img = Image.open(image_path)
        img = transform(img)
        self.ground_truth = img.cuda()
        print(img.shape)
        
    
    def train(self):
        start_t = time.time()
        
        for epoch in range(self.args.epochs):
            logger.info(f"Epoch {epoch} of {self.args.epochs}")
            self.epoch = epoch
            self.train_one_epoch(epoch)
        
        self.save_checkpoint()
        self.all_preds = np.stack(self.all_preds, axis=0)
        imageio.mimwrite(f"video.mp4", self.all_preds, fps=4, quality=8)
        end_t = time.time()
    
    def save_checkpoint(self):
        import os
        os.makedirs(f"/home/paulj/checkpoints/collosal/{self.args.workspace}",exist_ok=True)
        torch.save(self.model.state_dict(),f"/home/paulj/checkpoints/collosal/{self.args.workspace}/model.pt")
        
        
            
    @torch.no_grad()
    def prepare_embeddings(self):
        if self.args.text is not None:
            self.embeddings["default"] = self.guidance.get_text_embeds([self.args.text])
            self.embeddings["unconditional"] = self.guidance.get_text_embeds([self.args.text])
    
    def train_one_epoch(self,epoch):
        logger.info(f"Training epoch {epoch}")
        loss = 0
        self.optimizer.zero_grad()
        
        image = self.model()
        
        # text_z = [self.embeddings["unconditional"],self.embeddings["default"]]
        # text_z = torch.cat(text_z, dim=0)
        
        # loss = loss + self.guidance.train_step(text_z,image,as_latent=False,guidance_scale=self.args.guidance_scale,grad_scale=self.args.lambda_guidance)
        #mse loss
        from torch.nn import functional as F
        loss = F.mse_loss(image, self.ground_truth)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        pred_255 = image * 255
        pred_255 = pred_255.detach().cpu().permute(1, 2, 0).cpu().numpy().astype('uint8')
        logger.info(f"Loss: {loss.item()}")
        self.all_preds.append(pred_255)
        wandb.log({
            "loss":loss.item(),
            "image":wandb.Image(pred_255)
        })
        
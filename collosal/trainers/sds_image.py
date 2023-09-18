import cv2
import torch
import imageio
import time
from loguru import logger
import wandb
import numpy as np

class Trainer:
    def __init__(self,args,model,guidance,optimizer,ref_image_path,lr_scheduler=None):
        self.args = args
        self.model = model
        self.guidance = guidance
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.embeddings = {}
        self.prepare_embeddings()
        self.all_preds = []
        self.ref_image = self.load_image(ref_image_path)
        # from collosal.models.Image import ImageModel
        # self.model = ImageModel(self.ref_image).to(args.device)
    
    def load_image(self,path):
        from PIL import Image
        from torchvision import transforms
        img = Image.open(path)
        preprocess = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        return input_batch[:,:3,:,:].cuda()
        
    
    def train(self):
        start_t = time.time()
        
        for epoch in range(self.args.epochs):
            logger.info(f"Epoch {epoch} of {self.args.epochs}")
            self.epoch = epoch
            self.train_one_epoch(epoch)
        
        self.save_checkpoint()
        self.all_preds = np.concatenate(self.all_preds, axis=0)
        imageio.mimwrite(f"video.mp4", self.all_preds, fps=10, quality=10)
        end_t = time.time()
    
    def save_checkpoint(self):
        import os
        os.makedirs(f"/home/paulj/checkpoints/collosal/{self.args.workspace}",exist_ok=True)
        torch.save(self.model.state_dict(),f"/home/paulj/checkpoints/collosal/{self.args.workspace}/model.pt")
        
        
            
    @torch.no_grad()
    def prepare_embeddings(self):
        if self.args.text is not None:
            self.embeddings["default"] = self.guidance.get_text_embeds([self.args.text])
            self.embeddings["unconditional"] = self.guidance.get_text_embeds([self.args.neg_text])
    
    def train_one_epoch(self,epoch):
        logger.info(f"Training epoch {epoch}")
        loss = 0
        self.optimizer.zero_grad()
        
        image = self.model()
        
        text_z = [self.embeddings["unconditional"],self.embeddings["default"]]
        text_z = torch.cat(text_z, dim=0)
        
        loss = loss + self.guidance.train_step(text_z,image,as_latent=False,guidance_scale=self.args.guidance_scale,grad_scale=self.args.lambda_guidance,save_guidance_path="/home/paulj/projects/collosal/guidance/image.png")
        loss += self.args.lambda_rgb * torch.mean((image - self.ref_image) ** 2)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        pred = image.clone()
        pred = pred.mul(255).add_(0.5).clamp_(0, 255).permute(0,2, 3, 1).to("cpu", torch.uint8).numpy()
        # pred = pred.detach().cpu().permute(0, 2, 3, 1).numpy().astype('uint8')
        logger.info(f"Loss: {loss.item()}")
        self.all_preds.append(pred)
        wandb.log({
            "loss":loss.item(),
            "image":wandb.Image(pred)
        })
        
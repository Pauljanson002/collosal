import cv2
import torch
import imageio
import time
from loguru import logger
import wandb
import numpy as np
from collosal.models.Image import ImageModel
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import functional as F

class SpecifyGradient(torch.autograd.Function):
    """
    This code defines a custom gradient function using PyTorch's `torch.autograd.Function` class. It is particularly helpful when you want to manipulate gradients manually in a deep learning model that relies on automatic differentiation. The class is called `SpecifyGradient`, and contains two essential methods: `forward` and `backward`.

1. The `@staticmethod` decorator indicates that these are static methods and can be called on the class itself, without instantiating an object from the class.

2. The `forward` method takes two input arguments: `ctx` and `input_tensor`. `ctx` is a context object used to store information needed for backward computation. `input_tensor` is the input tensor to this layer in the neural network. The purpose of this method is to compute the forward pass and store any required information for the backward pass.

3. The `@custom_fwd` decorator is a user-defined decorator (not provided here) which presumably wraps or modifies the forward method in some way, most likely to add functionality like logging, error checking or other custom behavior.

4. Inside the `forward` method, the ground truth gradient `gt_grad` is saved using `ctx.save_for_backward()`. This stored information will be used later in the backward function. The forward function then returns a tensor of ones with the same device and data type as the input tensor. This tensor will be used in the backward pass as a scaling factor to adjust the gradients.

5. The `backward` method takes two input arguments: `ctx` and `grad_scale`. `ctx` is the same context object used in the forward pass. `grad_scale` is the gradient scaling factor used to adjust the gradients. The purpose of this method is to compute the gradient updates with respect to the input during backpropagation. 

6. The `@custom_bwd` decorator is another user-defined decorator (not provided here) which performs a similar role for the backward method as the `@custom_fwd` decorator does for the forward method.

7. Inside the `backward` method, the ground truth gradient `gt_grad` is retrieved from the saved tensors. It is then scaled by multiplying it with `grad_scale`. The method returns the scaled gradient `gt_grad` and `None`. The `None` value is returned because there are no gradients to compute for `gt_grad` with respect to the input tensor – it is assumed to be an external property that doesn't require gradient computation.

This custom gradient function can be used in situations where you need to have fine-grained control over the gradients in a neural network. For example, if you want to perform gradient clipping or apply noise to the gradients, you would use this `SpecifyGradient` function in place of a standard PyTorch layer.
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


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
        if args.as_latent:
            self.ref_image = self.guidance.encode_imgs(self.ref_image)
        # self.model = ImageModel(self.ref_image.clone()).to(args.device)
        self.animation = []
    def load_image(self,path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
        img_tensor /= 255
        input_batch = img_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        return input_batch.cuda()
 
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

        animation = np.stack(self.animation, axis=0)
        imageio.mimwrite(f"animation.mp4", animation, fps=10, quality=10)
    
    def save_checkpoint(self):
        import os
        os.makedirs(f"/home/paulj/checkpoints/collosal/{self.args.workspace}",exist_ok=True)
        torch.save(self.model.state_dict(),f"/home/paulj/checkpoints/collosal/{self.args.workspace}/model.pt")
        
        
            
    @torch.no_grad()
    def prepare_embeddings(self):
        if self.args.text is not None:
            self.embeddings["default"] = self.guidance.get_text_embeds([self.args.text])
            self.embeddings["unconditional"] = self.guidance.get_text_embeds([self.args.neg_text])
            self.embeddings["reference"] =- self.guidance.get_text_embeds([self.args.ref_text])
    
    def train_one_epoch(self,epoch):
        logger.info(f"Training epoch {epoch}")
        self.optimizer.zero_grad()
        
        image = self.model()


        
        
        text_z = [self.embeddings["unconditional"],self.embeddings["default"]]
        text_z_ref = [self.embeddings["unconditional"],self.embeddings["reference"]]
        text_z = torch.cat(text_z, dim=0)
        text_z_ref = torch.cat(text_z_ref, dim=0)

        noise_pred, t, noise = self.guidance.predict_noise(text_z, image, guidance_scale=self.args.guidance_scale, as_latent=True)
        with torch.no_grad():
            noise_pred_ref, _, _ = self.guidance.predict_noise(text_z_ref, self.ref_image, guidance_scale=self.args.guidance_scale, as_latent=True, t=t, noise=noise)

        w =  (1 - self.guidance.alphas[t])
        grad = w * (noise_pred - noise_pred_ref)
        grad = torch.nan_to_num(grad)

        loss = SpecifyGradient.apply(image, grad)

        loss.backward()
        if epoch % 20 == 0:
            wandb.log(
                {
                    "gradient/visualize":wandb.Image(self.model.image_representation.grad),
                    "gradient/norm":self.model.image_representation.grad.norm()
                }
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        # pred = image.clone()
        # pred = pred.mul(255).add_(0.5).clamp_(0, 255).permute(0,2, 3, 1).to("cpu", torch.uint8).numpy()
        # # pred = pred.detach().cpu().permute(0, 2, 3, 1).numpy().astype('uint8')
        if epoch % 20 == 0:
            with torch.no_grad():
                wandb.log({"similarity/cos sim": F.cosine_similarity(noise_pred - noise, noise_pred_ref - noise).mean().item()})
                wandb.log({"result/image": wandb.Image(self.guidance.decode_latents(image)[0])})
                from torchvision.utils import save_image
                save_image(self.guidance.decode_latents(image)[0], "result.png")
                pred = self.guidance.decode_latents(image)[0].detach()
                pred = pred.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                self.animation.append(pred)
                self.all_preds.append(pred)
                wandb.log({
                "loss":loss.item(),
                "image":wandb.Image(pred)
            })
        logger.info(f"Loss: {loss.item()}")

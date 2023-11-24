from argparse import Namespace
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from collosal.models.Image import ImageModel

from utils import seed_everything
import numpy as np
from custom_optimizer import Adan



print(f"Pytorch {torch.__version__}")
print(f"CUDA status {torch.cuda.is_available()} : {torch.cuda.get_device_name()}")
print(
    f"Number of devices {torch.cuda.device_count()} , CUDA Version {torch.version.cuda}"
)


def load_image(path):
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
    img_tensor /= 255
    input_batch = img_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch.cuda()

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    args = OmegaConf.to_container(cfg, resolve=True)
    args = Namespace(**args)
    

    from collosal.guidance.sd_utils import StableDiffusion
    guidance = StableDiffusion(args.device, False, False, "2.1", None)
    guidance.eval()
    for p in guidance.parameters():
        p.requires_grad = False

    wandb.init(
        project="collosal", config=vars(args), name=args.workspace, mode=args.wandb
    )
    ref_image_tensor = load_image(args.ref_image)
    ref_image_tensor.requires_grad = False
    ref_image_latent = guidance.encode_imgs(ref_image_tensor)
    model = ImageModel(args,guidance,ref_image_latent).to(args.device)
    print(model)

    # optimizer = torch.optim.Adam(model.parameters(),args.lr, betas=(0.9, 0.99), eps=1e-15)
    if args.optimizer == "adan":
        optimizer = Adan(
            model.parameters(),
            lr=args.lr,
            eps=1e-8,
            weight_decay=2e-5,
            max_grad_norm=5.0,
            foreach=False,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.9)

    seed_everything(args.seed)
    if args.method == "sds":

        from collosal.trainers.sds_image import Trainer

        trainer = Trainer(args, model, guidance, optimizer,args.ref_image, lr_scheduler)
    elif args.method == "dumb":
        from collosal.trainers.dumb_trainer import Trainer

        guidance = None
        trainer = Trainer(args, model, guidance, optimizer, lr_scheduler)
    elif args.method == "dds":
        from collosal.trainers.dds_trainer import Trainer

        trainer = Trainer(
            args, model, guidance, optimizer, args.ref_image, lr_scheduler
        )
    wandb.watch(model)
    trainer.train()


if __name__ == "__main__":
    main()

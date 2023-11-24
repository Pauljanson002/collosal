from torch import nn
import torch


class ImageModel(nn.Module):
    def __init__(self,config,guidance,init_tensor=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if init_tensor == None:
            self.image_representation = torch.nn.Parameter(
                torch.zeros(1, 3, 512, 512), requires_grad=True
            )
        else:
            self.image_representation = torch.nn.Parameter(
                init_tensor, requires_grad=True
            )
    def forward(self):
        return self.image_representation

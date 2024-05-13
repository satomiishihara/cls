import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_224 as create_model


class VIT_model(nn.Module):
    def __init__(self, num_classes=10):
        super(VIT_model, self).__init__()
        vit_model = create_model(pretrained=False)
        model_state = torch.load("jx_vit_base_p16_224-80ecf9dd.pth", map_location='cpu')
        vit_model.load_state_dict(model_state, strict=False)
        self.my_vit = vit_model
        self.head = nn.Sequential(
            nn.Linear(1000, num_classes),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.head(self.my_vit(x))

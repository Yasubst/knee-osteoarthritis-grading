
import torch.nn as nn
import torchvision.models as models


class KneeOsteoarthritisModel(nn.Module):

    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()

        model_options = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT)
        }

        model_function, weights = model_options[backbone]

        self.model = model_function(
            weights=weights if pretrained else None
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, 5)

    def forward(self, xray_images):
        return self.model(xray_images)

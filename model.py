import torch.nn as nn
import torchvision.models as models
from Xception_model import xception


class CNN(nn.Module):

    def __init__(self, pretrained=True, finetuning=False, architecture='Xception', frozen_params=50, output_features=False):
        super(CNN, self).__init__()

        self.output_features = output_features

        if architecture == 'Xception':
            self.model = xception(pretrained=pretrained)
            self.model.fc = nn.Sequential()
            self.fc_xc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                       nn.Dropout(p=0.25), nn.Linear(256, 1), nn.Sigmoid())

        elif architecture == 'resnet-50':
            self.model = models.resnet50(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential()
            self.fc_xc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                       nn.Dropout(p=0.3), nn.Linear(256, 1), nn.Sigmoid())

        if finetuning:
            # 154 parameters for XceptionNET
            i = 0
            for param in self.model.parameters():
                i += 0
                if i < frozen_params:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def forward(self, x):

        x = self.model(x)
        out = self.fc_xc(x)

        if self.output_features:
            return x
        else:
            return out

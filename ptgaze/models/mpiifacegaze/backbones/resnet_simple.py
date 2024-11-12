import torch
import torchvision
from omegaconf import DictConfig





class Model(torchvision.models.ResNet):
    
    
    
    
    def __init__(self, config: DictConfig):
        block_name = config.model.backbone.resnet_block
        if block_name == 'basic':
            block = torchvision.models.resnet.BasicBlock
        elif block_name == 'bottleneck':
            block = torchvision.models.resnet.Bottleneck
        else:
            raise ValueError
        layers = list(config.model.backbone.resnet_layers) + [1]
        super().__init__(block, layers)
        del self.layer4
        del self.avgpool
        del self.fc

        pretrained_name = config.model.backbone.pretrained
        if pretrained_name:
            state_dict = torch.hub.load_state_dict_from_url(
                torchvision.models.resnet.model_urls[pretrained_name])
            self.load_state_dict(state_dict, strict=False)
            # While the pretrained models of torchvision are trained
            # using images with RGB channel order, in this repository
            # images are treated as BGR channel order.
            # Therefore, reverse the channel order of the first
            # convolutional layer.
            module = self.conv1
            module.weight.data = module.weight.data[:, [2, 1, 0]]

        with torch.no_grad():
            data = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            features = self.forward(data)
            self.n_features = features.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# import torch
# import torchvision
# from omegaconf import DictConfig

# class Model(torchvision.models.ResNet):
#     def __init__(self, config: DictConfig):
#         block_name = config.model.backbone.resnet_block
#         if block_name == 'basic':
#             block = torchvision.models.resnet.BasicBlock
#         elif block_name == 'bottleneck':
#             block = torchvision.models.resnet.Bottleneck
#         else:
#             raise ValueError
#         layers = list(config.model.backbone.resnet_layers) + [1]
#         super().__init__(block, layers)
#         del self.layer4
#         del self.avgpool
#         del self.fc

#         pretrained_name = config.model.backbone.pretrained
#         if pretrained_name:
#             model_urls = {
#                 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#                 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#                 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#                 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#                 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#             }
#             if pretrained_name not in model_urls:
#                 raise ValueError(f"Pretrained model '{pretrained_name}' not supported.")
#             state_dict = torch.hub.load_state_dict_from_url(model_urls[pretrained_name])
#             self.load_state_dict(state_dict, strict=False)
#             # While the pretrained models of torchvision are trained
#             # using images with RGB channel order, in this repository
#             # images are treated as BGR channel order.
#             # Therefore, reverse the channel order of the first
#             # convolutional layer.
#             module = self.conv1
#             module.weight.data = module.weight.data[:, [2, 1, 0]]

#         with torch.no_grad():
#             data = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
#             features = self.forward(data)
#             self.n_features = features.shape[1]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x


# class Model(torchvision.models.ResNet):
#     def __init__(self, config: DictConfig):
#         block_name = config.model.backbone.resnet_block
#         if block_name == 'basic':
#             block = torchvision.models.resnet.BasicBlock
#         elif block_name == 'bottleneck':
#             block = torchvision.models.resnet.Bottleneck
#         else:
#             raise ValueError("Invalid block name. Expected 'basic' or 'bottleneck'.")
        
#         layers = list(config.model.backbone.resnet_layers) + [1]
#         super().__init__(block, layers)
#         del self.layer4
#         del self.avgpool
#         del self.fc

#         pretrained_name = config.model.backbone.pretrained
#         if pretrained_name:
#             model_urls = {
#                 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#                 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#                 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#                 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#                 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#             }
#             if pretrained_name not in model_urls:
#                 raise ValueError(f"Pretrained model '{pretrained_name}' not available.")
            
#             state_dict = torch.hub.load_state_dict_from_url(model_urls[pretrained_name])
#             self.load_state_dict(state_dict, strict=False)
#             # Reverse the channel order of the first convolutional layer.
#             self.conv1.weight.data = self.conv1.weight.data[:, [2, 1, 0]]

#         with torch.no_grad():
#             data = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
#             features = self.forward(data)
#             self.n_features = features.shape[1]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x

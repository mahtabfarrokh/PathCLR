import torch.nn as nn
import torchvision.models as models
import torch
from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):

        super(ResNetSimCLR, self).__init__()
        print("HEREEEEEE?")
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet34": models.resnet34(pretrained=True, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
                            "resnet101": models.resnet101(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def my_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
            print("Xavier")

        if isinstance(m, nn.Conv2d):
            #             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.xavier_normal_(m.weight)
            print("Xavier 2")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            print("Xavier 3")

    def load_checkpoint(self, model):
        checkpoint = torch.load('./runs/Jan24_23-46-07_mahtab-gpu-2/best_checkpoint_0500.pth.tar',
                                map_location=torch.device('cuda'))
        optimizer = torch.optim.Adam(model.parameters(), 0.0003, weight_decay=1e-4)
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('backbone.') and not k.startswith('backbone.fc'):
                if k.startswith('backbone'):
                    # remove prefix
                    print("HERE!")
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
        log = model.load_state_dict(state_dict, strict=False)
        print("===")
        print(log)
        print("===")
        print(log.missing_keys)
        assert log.missing_keys == ['fc.weight', 'fc.bias']
        optimizer.load_state_dict(state_dict)
        print(log.missing_keys)
        print(log)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print(model_name)
            model.apply(self.my_init_weights)
        #             optimizer = model.apply(self.load_checkpoint)
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)

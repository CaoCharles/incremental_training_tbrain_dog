import torch
from copy import deepcopy
class ModelEMA(object):
    def __init__(self, model, decay=0.001):
        self.ema = deepcopy(model)
        self.ema.to('cuda')
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

# class VGGish(nn.Module):
#     """
#     PyTorch implementation of the VGGish model.
#     Adapted from: https://github.com/harritaylor/torch-vggish
#     The following modifications were made: (i) correction for the missing ReLU layers, (ii) correction for the
#     improperly formatted data when transitioning from NHWC --> NCHW in the fully-connected layers, and (iii)
#     correction for flattening in the fully-connected layers.
#     """

#     def __init__(self, params):
#         super(VGGish, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 64, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(64, 128, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(128, 256, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(256, 512, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(512 * 31 * 4, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 128),
#             nn.ReLU(inplace=True)
#         )
#         self.final_fc = nn.Linear(128, params.num_class, bias=True)

#     def forward(self, x):
#         x = self.features(x).permute(0, 2, 3, 1).contiguous()
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = self.final_fc(x)
#         x = torch.sigmoid(x)

#         return x
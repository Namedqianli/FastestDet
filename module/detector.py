from turtle import back
import torch
import torch.nn as nn

from module.mobilenetv3 import MobileNetV3

from .shufflenetv2 import ShuffleNetV2
from .custom_layers import DetectHead, SPP
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import MobileNetV3
from .repvgg import *
from .mobileone import PARAMS, mobileone

__all__ = [
    "ShuffleNet_V2",
    "MobileNet_V2",
]

class Detector(nn.Module):
    def __init__(self, category_num, load_param, backbone='ShuffleNet_V2'):
        super(Detector, self).__init__()

        if backbone == 'ShuffleNet_V2':
            self.stage_repeats = [4, 8, 4]
            self.stage_out_channels = [-1, 24, 48, 96, 192]
            self.backbone = ShuffleNetV2(self.stage_repeats, self.stage_out_channels, load_param)

            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.SPP = SPP(sum(self.stage_out_channels[-3:]), self.stage_out_channels[-2])
            
            self.detect_head = DetectHead(self.stage_out_channels[-2], category_num)
        elif backbone == 'MobileNet_V2':
            self.backbone = MobileNetV2()

            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.SPP = SPP(448, 224)
            
            self.detect_head = DetectHead(224, category_num)
        elif backbone == 'MobileNet_V3_LARGE':
            self.backbone = MobileNetV3(mode='large')

            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.SPP = SPP(280, 140)

            self.detect_head = DetectHead(140, category_num)
        elif backbone == 'RepVGG_A1':
            self.backbone = get_RepVGG_func_by_name('RepVGG-A1')(deploy=False, use_checkpoint=False)

            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.SPP = SPP(896, 448)

            self.detect_head = DetectHead(448, category_num)
        elif backbone == 'RepVGG_A0':
            self.backbone = get_RepVGG_func_by_name('RepVGG-A0')(deploy=False, use_checkpoint=False)

            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.SPP = SPP(672, 336)

            self.detect_head = DetectHead(336, category_num)
        elif 'MobileOne' in backbone:
            variant = backbone.split('_')[1]
            self.backbone = mobileone(variant=variant)

            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            scal = PARAMS[variant]['width_multipliers']
            totalChannel = int(128*scal[1] + 256*scal[2] + 512*scal[3])
            self.SPP = SPP(totalChannel, totalChannel // 2)

            self.detect_head = DetectHead(totalChannel // 2, category_num)

    def forward(self, x):
        P1, P2, P3 = self.backbone(x)
        P3 = self.upsample(P3)
        P1 = self.avg_pool(P1)
        P = torch.cat((P1, P2, P3), dim=1)

        y = self.SPP(P)

        return self.detect_head(y)

if __name__ == "__main__":
    model = Detector(80, False, 'MobileNet_V3_LARGE')
    test_data = torch.rand(1, 3, 352, 352)
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     "./test.onnx",             # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=11,          # the ONNX version to export the model to
                     do_constant_folding=True)  # whether to execute constant folding for optimization


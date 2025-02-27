import torch
import torch.nn as nn
from nets.Dysampling import DySample

from nets.Vanillanet import vanillanet_L
from nets.SPConv_3 import SPConv_3x3

def calculate_in_filters(feature_chanel, out_filters):
    if len(feature_chanel) < 3:
        raise ValueError("feature_chanel must have at least 3 elements")
    if len(out_filters) < 2:
        raise ValueError("out_filters must have at least 2 elements")
    # 计算 in_filters 的第一部分，使用切片操作并相加
    first_part = [feature_chanel[i] + out_filters[i + 1] for i in range(len(feature_chanel) - 2)]
    # 计算 in_filters 的第二部分，使用最后两个元素相加
    second_part = feature_chanel[-1] + feature_chanel[-2]
    # 组合两部分得到 in_filters
    in_filters = first_part + [second_part]
    # 计算 fs_channel 的元素
    fs_channel = out_filters[1:] + [feature_chanel[-1]]
    return in_filters, fs_channel

class unetUp(nn.Module):
    def __init__(self, in_size, out_size,in_channels,Dyupsample=False, SPConv= False):
        super(unetUp, self).__init__()

        if SPConv:
            self.conv1 = SPConv_3x3(inplanes=in_size, outplanes=out_size, stride=1, ratio=0.5)
            self.conv2 = SPConv_3x3(inplanes=out_size, outplanes=out_size, stride=1, ratio=0.5)
        else:
            self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)


        if Dyupsample:
            self.up =  DySample( in_channels = in_channels)
        else:
            self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)

        self.relu   = nn.ReLU(inplace = True)



    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg', Dyupsample = False,SPConv= False):
        super(Unet, self).__init__()




        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            feature_channel =  [64, 128, 256, 512,512]
            out_filters = [64, 128, 256, 512]
            in_filters, fs_channel = calculate_in_filters(feature_channel, out_filters)
            #  fs_channel = [128, 256, 512, 512]
            #  in_filters  = [192, 384, 768, 1024]

        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained, SPConv= SPConv)
            feature_channel = [64, 256, 512, 1024, 2048]
            out_filters = [64, 128, 256, 512]
            in_filters, fs_channel = calculate_in_filters(feature_channel, out_filters)
            #  fs_channel = [128 256 512  2048]
            #  in_filters  = [192, 512, 1024, 3072]

        elif backbone == "vaillanetL":
            self.vaillanet  = vanillanet_L(pretrained = pretrained)
            feature_channel =  [64,  256, 512, 1024, 2048]
            out_filters = [64, 128, 256, 512]
            in_filters, fs_channel = calculate_in_filters(feature_channel, out_filters)
            # fs_channel = [256 512 2048]
            # in_filters  = [384, 1024, 3072]
        elif backbone == "EFnet":
            self.EFnet  = enetb4(pretrained = pretrained)
            feature_channel =  [24, 32, 48, 96, 1536]
            out_filters = [64, 128, 256, 512]
            in_filters, fs_channel = calculate_in_filters(feature_channel, out_filters)

        elif backbone == "FasterNet":
            self.FasterNet = fasternet_l(pretrained = pretrained)
            feature_channel =  [64, 128, 256, 512, 1024]
            out_filters = [64, 128, 256, 512]
            in_filters, fs_channel = calculate_in_filters(feature_channel, out_filters)
            # fs_channel = [256 512 2048]
            # in_filters  = [384, 1024, 3072]


        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        if Dyupsample:
            self.up2end = DySample(out_filters[0])
        else:
            self.up2end = nn.UpsamplingBilinear2d(scale_factor=2)

       # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2], fs_channel[2], Dyupsample,  SPConv= SPConv)
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1], fs_channel[1], Dyupsample,  SPConv= SPConv)
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0], fs_channel[0], Dyupsample,  SPConv= SPConv)

        # if backbone == 'resnet50' or backbone == 'vgg':
        self.up_concat4 = unetUp(in_filters[3], out_filters[3], fs_channel[3], Dyupsample,  SPConv= SPConv)

        if backbone != 'vgg':
            self.up_conv = nn.Sequential(
                self.up2end,
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        # elif backbone == "vaillanetL":
        #     self.up_conv = nn.Sequential(
        #         self.up2end,  # 第一次上采样，扩大2倍
        #         nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
        #         nn.ReLU(),
        #
        #         self.up2end,  # 第二次上采样，扩大2倍
        #         nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
        #         nn.ReLU(),
        #     )

        else:
            self.up_conv = None



        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        elif self.backbone == "vaillanetL":
            [feat1, feat2, feat3, feat4, feat5] = self.vaillanet.forward(inputs)
        elif self.backbone == "EFnet":
            [feat1, feat2, feat3, feat4, feat5] = self.EFnet.forward(inputs)
        elif self.backbone == "FasterNet":
            [feat1, feat2, feat3, feat4, feat5] = self.FasterNet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)


        # if self.backbone == 'resnet50' or self.backbone == 'vgg':
        #     up4 = self.up_concat4(feat4, feat5)
        #     up3 = self.up_concat3(feat3, up4)
        #     up2 = self.up_concat2(feat2, up3)
        #     up1 = self.up_concat1(feat1, up2)
        #
        # elif self.backbone == "vaillanetL":
        #     up3 = self.up_concat3(feat3, feat4)
        #     up2 = self.up_concat2(feat2, up3)
        #     up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)  #1,64,512,512->1,64,1024,1024

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False
        elif self.backbone == "vaillanetL":
            for param in self.vaillanet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
        elif self.backbone == "vaillanetL":
            for param in self.vaillanet.parameters():
                param.requires_grad = True


if __name__ == '__main__':
    inputs = torch.randn((1, 3, 1024, 1024))
    flag = 0
    if flag==1:
        model = Unet(backbone= 'resnet50')
    elif flag==2:
        model = Unet(backbone='EFnet')
    elif flag==3:
        model = Unet(backbone='FasterNet')
    else:
        model = Unet(backbone= 'vaillanetL')
    pred = model.forward(inputs)
    print(pred.size())





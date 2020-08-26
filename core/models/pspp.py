"""PSPP Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
# from .base_models.vgg import vgg16
from .customize import _PSPP

__all__ = ['PSPPNet', 'get_pspp', 'get_pspp_resnet50_citys',
           'get_pspp_resnet101_citys']

class PSPPNet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50', aux=False, pretrained_base=True,  norm_layer=nn.BatchNorm2d,**kwargs):
        super(PSPPNet, self).__init__(nclass, aux, backbone,norm_layer=norm_layer,  pretrained_base=pretrained_base, **kwargs)
        self.head = _PSPPHead(2048,norm_layer=nn.BatchNorm2d,**kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(512, nclass, 1)
        )
        # self.sp = _SPHead(2048, nclass, aux, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['head'])

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)   ##2048
        outputs = []
        x = self.head(c4)       ##2048
        x = torch.cat([x,c4],1)    ##4096
        x = self.block(x)
        # x = self.sp(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs = None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )
class _PSPPHead(nn.Module):
    def __init__(self, in_channels,  norm_layer=nn.BatchNorm2d, norm_kwargs=None,):
        super(_PSPPHead, self).__init__()


        out_channels = int(in_channels/4)        #512
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv1 = _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs=None)   ##2048 ==>512
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs=None)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs=None)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs=None)

        
        ###Adding SPmodule 
        inter_channels = in_channels // 4
        up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        self.strip_pool1 =StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)               # 2048 =>512              
        self.strip_pool2 = StripPooling(inter_channels, (20, 12),norm_layer, up_kwargs)                             
        self.strip_pool3 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)                               
        self.strip_pool4 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
                                    
        ### 4096 ==> 2048
        self.conv6 = nn.Sequential(
        nn.Conv2d(4096, 2048, 1, bias=False),
        nn.BatchNorm2d(2048),
        nn.ReLU(True)
    )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.strip_pool1(self.conv1(self.pool1(x))), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.strip_pool2(self.conv2(self.pool2(x))), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.strip_pool3(self.conv3(self.pool3(x))), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.strip_pool4(self.conv4(self.pool4(x))), size, mode='bilinear', align_corners=True)
        feat5 = torch.cat([x, feat1, feat2, feat3, feat4], 1)
        feat6 =  self.conv6(feat5)
        return feat6
class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs = {'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)

class _SPHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_SPHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        # self.conv_c1 = nn.Sequential(
        #     nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #     norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
        #     nn.ReLU(True)
        # )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        # self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        # self.conv_c2 = nn.Sequential(
        #     nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #     norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
        #     nn.ReLU(True)
        # )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            # self.conv_c3 = nn.Sequential(
            #     nn.Dropout(0.1),
            #     nn.Conv2d(inter_channels, nclass, 1)
            # )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        # feat_c = self.conv_c1(x)
        # feat_c = self.cam(feat_c)
        # feat_c = self.conv_c2(feat_c)

        # feat_fusion = feat_p + feat_c
        feat_fusion = feat_p 


        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            # c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            # outputs.append(c_out)

        return tuple(outputs)


def get_pspp(dataset='citys', backbone='resnet50', pretrained=False,
              root='~/.torch/models', pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = PSPPNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('danet_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model




def get_pspp_resnet50_citys(**kwargs):
    return get_pspp('citys', 'resnet50', **kwargs)


def get_pspp_resnet101_citys(**kwargs):
    return get_pspp('citys', 'resnet101', **kwargs)


# def get_pspp_vgg16_citys(**kwargs):
#     return get_pspp('citys', 'vgg16', **kwargs)


if __name__ == '__main__':
    img = torch.randn(2, 3, 480, 480)
    model = get_pspp_resnet50_citys()
    outputs = model(img)
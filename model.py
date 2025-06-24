
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter


class SegBaseModel(nn.Module):
    def __init__(self, aux=False, backbone='mobilenet_v3_large', pretrained_backbone=True, **kwargs):
        super(SegBaseModel, self).__init__()
        # self.nclass = nclass
        self.aux = aux
        self.mode = backbone.split('_')[-1]
        assert self.mode in ['large', 'small']
        if backbone == 'mobilenet_v3_large':
            self.pretrained = models.mobilenetv3.__dict__[backbone](pretrained=pretrained_backbone, dilated=True).features 

        elif backbone == 'mobilenet_v3_small':
            self.pretrained = models.mobilenetv3.__dict__[backbone](pretrained=pretrained_backbone, dilated=True).features 
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
   
        stage_indices = [0] + [i for i, b in enumerate(self.pretrained) if getattr(b, "_is_cn", False)] + [len(self.pretrained) - 1]
        
        pos_c5 = stage_indices[-1]  # use C5 which has output_stride = 16; high_pos
        pos_c4 = stage_indices[-2]  # use C4 
        pos_c3 = stage_indices[-3]  # use C3 
        pos_c2 = stage_indices[-4]  # use C2 here which has output_stride = 8; low_pos
        pos_c1 = stage_indices[-5]  # use C1 
        pos_c0 = stage_indices[-6]  # use C0
        # print(pos_c5, pos_c4, pos_c3, pos_c2, pos_c1, pos_c0)
        
        channels_c5 = self.pretrained[pos_c5].out_channels
        channels_c4 = self.pretrained[pos_c4].out_channels
        channels_c3 = self.pretrained[pos_c3].out_channels
        channels_c2 = self.pretrained[pos_c2].out_channels
        channels_c1 = self.pretrained[pos_c1].out_channels
        channels_c0 = self.pretrained[pos_c0].out_channels
        # print(channels_c5, channels_c4, channels_c3, channels_c2, channels_c1, channels_c0)
        
        backbone_layers = IntermediateLayerGetter(self.pretrained, return_layers={str(pos_c5): 'c5', str(pos_c4): 'c4', str(pos_c3): 'c3', str(pos_c2): 'c2', str(pos_c1): 'c1', str(pos_c0): 'c0'})
        
        return backbone_layers(x), channels_c5, channels_c4, channels_c3, channels_c2, channels_c1, channels_c0 
    

                
                
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        # self.do = nn.Dropout(0.3)
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) # original
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x)))) # original
        

        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
    
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        p = self.avg_pool(x).view(b, c)
        y = self.fc(p).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c)

        self.relu = nn.ReLU(inplace=True)
        
        # self.do = nn.Dropout(0.3) # remove for no dropout

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)
        # x4 = self.do(x4) # remove for no dropout

        return x4



class SegRExtNet(nn.Module):
    def __init__(self, out_cc = [16, 24, 40, 80, 160, 960], filters=[32, 64, 128, 256, 512]):
        super(SegRExtNet, self).__init__()
        
        """mobilenetv3 encoder"""
        self.encoder = SegBaseModel()

        """ Intermediate connection between encoder and decoder using attention"""
        self.ca5 = ChannelAttention(out_cc[5])
        self.ca2 = ChannelAttention(out_cc[2])
        self.ca1 = ChannelAttention(out_cc[1])
        self.ca0 = ChannelAttention(out_cc[0])
        self.sa = SpatialAttention()

        """ Decoder 1 """
        self.t1 = nn.ConvTranspose2d(960, 128, kernel_size=(4, 4), stride=2, padding=1)
        self.r1 = ResidualBlock(168, 64)
        
        self.t2 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2, padding=1)
        self.r2 = ResidualBlock(56, 32)
        
        self.t3 = nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=2, padding=1)
        self.r3 = ResidualBlock(32, 16)
        
        
        self.t4 = nn.ConvTranspose2d(16, 16, kernel_size=(4, 4), stride=2, padding=1)
 


        """ Output """
        self.output = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        layers_extracted, c5, c4, c3, c2, c1, c0 = self.encoder.base_forward(x)
        layers_extracted = [(k, v, v.shape) for k, v in layers_extracted.items()]
        s0 = layers_extracted[0][1] # torch.Size([2, 16, 128, 128]) ==>
        s1 = layers_extracted[1][1] # [2, 24, 64, 64]==>
        s2 = layers_extracted[2][1] # [2, 40, 32, 32]==>
        s3 = layers_extracted[3][1] # [2, 80, 16, 16]
        s4 = layers_extracted[4][1] # [2, 160, 16, 16]
        s5 = layers_extracted[5][1] # [2, 960, 16, 16] ==>
    

        s5 = s5 * self.ca5(s5)
        s5 = s5 * self.sa(s5)

        s2 = s2 * self.ca2(s2)
        s2 = s2 * self.sa(s2)

        s1 = s1 * self.ca1(s1)
        s1 = s1 * self.sa(s1)

        s0 = s0 * self.ca0(s0)
        s0 = s0 * self.sa(s0)

        

        t1 = self.t1(s5)
        t1 = torch.cat([t1, s2], axis=1)
        r1 = self.r1(t1)

        
        t2 = self.t2(r1)
        t2 = torch.cat([t2, s1], axis=1)
        r2 = self.r2(t2)

        t3 = self.t3(r2)
        t3 = torch.cat([t3, s0], axis=1)
        r3 = self.r3(t3)

        t4 = self.t4(r3)

        output = self.output(t4)
        # print('output', output.shape)

        return output

if __name__ == "__main__":
    model = SegRExtNet().cuda()
    from torchsummary import summary
    summary(model, (3, 256, 256))
    # print(summary)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.smt import  smt_t
from Model.Modules import (
    CoordinateAttention,
    ConvBNReLU,
    Conv1x1,
    DepthwiseSeparableConv,
    DilatedConvBlock,
    Haar,
    ChannelAttention,
    ConvBN,
    IConv1x1,
    SpatialAttention,
)


class FDFDM(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 16):

        super(FDFDM, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        reduced_channels = max(1, in_channels // reduction)

        self.haar = Haar()

        self.coord_att = CoordinateAttention(in_channels, reduction)

        self.conv_reduce = ConvBNReLU(in_channels, reduced_channels, kernel_size=1)

        self.conv1x1_x = IConv1x1(reduced_channels, in_channels,bias=True)
        self.conv1x1_y = IConv1x1(reduced_channels, in_channels,bias=True)
        self.HH_depthwise = DepthwiseSeparableConv(in_channels, in_channels)
        self.hf_reduce = Conv1x1(3 * in_channels, in_channels)
        self.hf_refine = DepthwiseSeparableConv(in_channels, in_channels)

        self.ll_depthwise = DepthwiseSeparableConv(in_channels, in_channels)
        self.ll_dilated = DilatedConvBlock(in_channels, in_channels, kernel_size=3, dilation=2)

        self.channel_att = ChannelAttention(in_channels)

    def forward(self, x: torch.Tensor):

        LL, LH, HL, HH = self.haar(x)
        _, _, H2, W2 = LL.shape

        sum_lh_hl = LH + HL
        x_att_raw, y_att_raw = self.coord_att(sum_lh_hl)
        x_att_flat = x_att_raw.permute(0, 1, 3, 2)
        y_att_flat = y_att_raw
        att_cat = torch.cat([x_att_flat, y_att_flat], dim=3)
        att_reduced = self.conv_reduce(att_cat)
        part_x = att_reduced[:, :, :, :H2]
        part_y = att_reduced[:, :, :, H2:]
        part_x = part_x.permute(0, 1, 3, 2)
        att_x = self.conv1x1_x(part_x)
        att_y = self.conv1x1_y(part_y)
        weight_x = torch.sigmoid(att_x)
        weight_y = torch.sigmoid(att_y)
        LH_enh = LH * weight_x + LH
        HL_enh = HL * weight_y + HL
        avg_pool = HH.mean(dim=1, keepdim=True)
        max_pool, _ = HH.max(dim=1, keepdim=True)
        hh_att = max_pool - avg_pool

        weight_hh = torch.sigmoid(hh_att)
        HH_enh = HH * weight_hh + HH
        HH_enh = self.HH_depthwise(HH_enh)
        HF_sum = torch.cat([LH_enh, HL_enh, HH_enh], dim=1)  # [B, 3C, H, W]
        HF_sum = self.hf_reduce(HF_sum)

        high_freq_enh = self.hf_refine(HF_sum)

        ll_mid = self.ll_dilated(LL)
        ll_mid = self.channel_att(ll_mid) * ll_mid + ll_mid

        low_freq_enh = self.ll_depthwise(ll_mid)

        return high_freq_enh, low_freq_enh


class MultiScaleFreqFusion(nn.Module):

    def __init__(self, C1: int, C2: int, C4: int, out_channels: int = None):

        super(MultiScaleFreqFusion, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.C4 = C4

        self.out_channels = out_channels if out_channels is not None else C1

        self.h4_reduce = IConv1x1(C4, C1)
        self.h2_up = IConv1x1(C2, C1)
        self.in_channels_cat_high = C1 + C1 + C1
        self.high_fuse = ConvBN(self.in_channels_cat_high, C1, kernel_size=1)
        self.high_refine = ConvBNReLU(in_channels=C1,out_channels=C1,kernel_size=3)
        self.high_reduce = IConv1x1(C1, C1,bias=True)

        self.in_channels_cat_low = C1 + C2 + C4
        self.low_fuse = ConvBN(self.in_channels_cat_low, C2, kernel_size=1)
        self.low_refine = ConvBNReLU(in_channels=C2,out_channels=C2,kernel_size=3)
        self.low_reduce = IConv1x1(C2, C1,bias=True)
    def forward(
        self,
        high_freq_enh: torch.Tensor,
        high_freq_enh_2: torch.Tensor,
        high_freq_enh_4: torch.Tensor,
        low_freq_enh: torch.Tensor,
        low_freq_enh_2: torch.Tensor,
        low_freq_enh_4: torch.Tensor,
    ):

        h2 = self.h2_up(high_freq_enh_2)
        h2 = F.interpolate(h2, size=high_freq_enh.shape[2:], mode='bilinear', align_corners=False)
        h4_reduce = self.h4_reduce(high_freq_enh_4)
        h4 = F.interpolate(h4_reduce, size=high_freq_enh.shape[2:], mode='bilinear', align_corners=False)
        high_cat = torch.cat([high_freq_enh, h2, h4], dim=1)
        fused_high = self.high_reduce(self.high_refine(self.high_fuse(high_cat)))
        l2 = F.interpolate(low_freq_enh_2, size=low_freq_enh.shape[2:], mode='bilinear', align_corners=False)
        l4 = F.interpolate(low_freq_enh_4, size=low_freq_enh.shape[2:], mode='bilinear', align_corners=False)
        low_cat = torch.cat([low_freq_enh, l2, l4], dim=1)
        fused_low = self.low_reduce(self.low_refine(self.low_fuse(low_cat)))

        return fused_high, fused_low

class BRAM(nn.Module):

    def __init__(self,C1: int,C2: int,C4: int):
        super(BRAM, self).__init__()

        self.C1 = C1
        self.C2 = C2
        self.C4 = C4

        self.conv4_reduce = ConvBN(C4, C2, kernel_size=1)
        self.conv2_reduce = ConvBN(C2, C1, kernel_size=1)

        self.region_delta = nn.Sequential(
            IConv1x1(C1 + C1, C1),
            ConvBNReLU(C1, C1, kernel_size=3),
        )
        self.obj_head = IConv1x1(C1, C1, bias=True)

        self.edge_in_channels = 2 * C1 + C2 + C4 // 2
        self.low4_reduce = ConvBN(C4, C4 // 2, kernel_size=1)

        self.edge_delta = nn.Sequential(
            IConv1x1(self.edge_in_channels, C2),
            ConvBNReLU(C2, C1, kernel_size=3),
        )
        self.edge_head = IConv1x1(C1,1,bias=True)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x4: torch.Tensor,
        fused_high: torch.Tensor,
        fused_low: torch.Tensor,
    ):
        x4_reduced = self.conv4_reduce(x4)
        up4_final = F.interpolate(x4_reduced, size=x2.shape[2:], mode="bilinear", align_corners=False)

        f2 = up4_final + x2
        f2_reduced = self.conv2_reduce(f2)
        up2_final = F.interpolate(f2_reduced, size=x1.shape[2:], mode="bilinear", align_corners=False)

        feat_deep = up2_final + x1
        if fused_low.shape[2:] != x1.shape[2:]:
            fused_low = F.interpolate(fused_low, size=x1.shape[2:], mode="bilinear", align_corners=False)

        region_delta = self.region_delta(torch.cat([feat_deep, fused_low], dim=1))
        obj_map = self.obj_head(region_delta)

        x4 = self.low4_reduce(x4)
        if fused_high.shape[2:] != x1.shape[2:]:
            fused_high = F.interpolate(fused_high, size=x1.shape[2:], mode="bilinear", align_corners=False)
        up4_to_x1 = F.interpolate(x4, size=x1.shape[2:], mode='bilinear', align_corners=False)
        up2_to_x1 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        edge_feat = torch.cat([x1, up2_to_x1, up4_to_x1, fused_high], dim=1)
        edge_map =  self.edge_delta(edge_feat)

        edge_map = self.edge_head(edge_map)

        return obj_map, edge_map

#MFFC
class MFFM(nn.Module):

    def __init__(self,in_channels_img: int,in_channels_depth: int):
        super(MFFM, self).__init__()
        self.in_channels_img = in_channels_img
        self.in_channels_depth = in_channels_depth

        self.rgb_gate = ChannelAttention(in_channels_img)
        self.depth_gate = IConv1x1(self.in_channels_depth, 1,bias=True)

        cat_channels = in_channels_img  + in_channels_depth
        self.reduce = Conv1x1(cat_channels, in_channels_img)

        self.refine = DepthwiseSeparableConv(in_channels_img, in_channels_img,kernel_size=3)

    def forward(self,x: torch.Tensor,d: torch.Tensor,edge):

        edge = F.interpolate(edge, size=x.shape[2:], mode="bilinear", align_corners=False)
        x_guide = x * (1.0 + edge)
        d_guide = d * (1.0 + edge)

        rgb_gate = self.rgb_gate(x_guide)

        depth_gate = self.depth_gate(d_guide)
        depth_gate = torch.sigmoid(depth_gate)

        x_en = x_guide * depth_gate
        depth_en = d_guide * rgb_gate

        concat_feat = torch.cat([x_en,depth_en], dim=1)
        fused = self.reduce(concat_feat)

        fused =self.refine (fused)

        return fused

class FFRM(nn.Module):
    def __init__(self, channels: int, r: int = 4, eca_k: int = 3):
        super().__init__()
        cr = max(16, channels // r)
        self.reduce = ConvBN(channels,cr,kernel_size=1)

        self.dw1 = DilatedConvBlock(cr,cr,kernel_size=3, dilation=1)
        self.dw2 = DilatedConvBlock(cr,cr,kernel_size=3, dilation=2)
        self.dw3 = DilatedConvBlock(cr,cr,kernel_size=3, dilation=3)
        self.ConBnRelu = ConvBNReLU(cr,cr,kernel_size=3)

        self.expand = IConv1x1(cr, channels)

        self.conv = nn.Conv1d(1, 1, kernel_size=eca_k, padding=(eca_k - 1) // 2, bias=False)
        self.spatial = SpatialAttention()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B,C,H,W)
        residual = x

        z = self.reduce(x)
        z = self.dw1(z) + self.dw2(z) + self.dw3(z)
        z = self.ConBnRelu(z)
        z = self.expand(z)

        y = z.mean(dim=(2, 3), keepdim=False)
        y = self.conv(y.unsqueeze(1)).squeeze(1)
        y = torch.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
        z = z * y
        s = self.spatial(z) * z
        out = residual + self.alpha * s
        return out


#HFAM
class HFAM(nn.Module):

    def __init__(
        self,
        in_channels_low: int,
        in_channels_high: int,
        obj_channels: int,
        mid_channels: int = None,
    ):
        super(HFAM, self).__init__()

        self.in_channels_low = in_channels_low
        self.in_channels_high = in_channels_high

        self.fuse_channels = mid_channels if mid_channels is not None else in_channels_low

        self.guide_high = IConv1x1(in_channels_high + obj_channels, self.fuse_channels, bias=True)
        self.guide_low = IConv1x1(in_channels_low + obj_channels, self.fuse_channels, bias=True)

        fuse_channels_reduce = max(32,self.fuse_channels // 4)
        self.gate =nn.Sequential(
            IConv1x1(2 * self.fuse_channels, fuse_channels_reduce),
            ConvBNReLU(fuse_channels_reduce, fuse_channels_reduce,kernel_size=3),
            IConv1x1(fuse_channels_reduce,2,bias=True)
        )
        self.channel_att = ChannelAttention(self.fuse_channels)

        self.refine = ConvBNReLU(self.fuse_channels, self.in_channels_low, kernel_size=3)

    def forward(self,feat_low: torch.Tensor,feat_high: torch.Tensor,obj_map: torch.Tensor,) -> torch.Tensor:

        obj_down = F.interpolate(obj_map, size=feat_low.shape[2:], mode='bilinear',align_corners=False)
        feat_high = F.interpolate(feat_high, size=feat_low.shape[2:], mode='bilinear',align_corners=False)

        guided_low = self.guide_low(torch.cat([feat_low, obj_down], dim=1))
        guided_high = self.guide_high(torch.cat([feat_high, obj_down], dim=1))

        w_gate = self.gate(torch.cat([guided_low, guided_high], dim=1))
        w_norm = torch.softmax(w_gate, dim=1)
        w_low_norm, w_high_norm = torch.chunk(w_norm, chunks=2, dim=1)

        fused_low_part = guided_low * w_low_norm
        fused_high_part = guided_high * w_high_norm
        fused_sum = fused_low_part + fused_high_part

        y = fused_sum * self.channel_att(fused_sum)
        fused = y + fused_sum
        out = self.refine(fused)
        return out


class MultiScalePrediction(nn.Module):

    def __init__(self, c1: int, c2: int, c3: int, obj:int):
        super(MultiScalePrediction, self).__init__()

        self.out_conv1 = IConv1x1(c1, 1,bias=True)
        self.out_conv2 = IConv1x1(c2, 1,bias=True)
        self.out_conv3 = IConv1x1(c3, 1,bias=True)
        self.out_conv4 = IConv1x1(obj, 1,bias=True)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor, feat3: torch.Tensor, obj_map: torch.Tensor):

        pred1_up = self.out_conv1(feat1)
        pred2_up = self.out_conv2(feat2)
        pred3_up = self.out_conv3(feat3)
        obj = self.out_conv4(obj_map)

        return pred1_up, pred2_up, pred3_up, obj



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.encoder_rgb = smt_t()
        self.encoder_d = smt_t()

        stage_channels = [16, 64, 112, 128, 196]
        new_channels = [16, 64, 128, 256, 512]

        self.conv2 = IConv1x1(new_channels[2], stage_channels[2])
        self.conv3 = IConv1x1(new_channels[3], stage_channels[3])
        self.conv4 = IConv1x1(new_channels[4], stage_channels[4])


        self.m1 = FDFDM(in_channels=stage_channels[1], reduction=4)
        self.m2 = FDFDM(in_channels=stage_channels[2], reduction=8)
        self.m4 = FDFDM(in_channels=stage_channels[4], reduction=14)

        self.m = MultiScaleFreqFusion(C1=stage_channels[1], C2=stage_channels[2], C4=stage_channels[4], out_channels=stage_channels[1])

        self.EdgeAndObj = BRAM(C1=stage_channels[1], C2=stage_channels[2], C4=stage_channels[4])

        self.fusion_layer_1 = MFFM(
            in_channels_img=stage_channels[1],
            in_channels_depth=stage_channels[1],
        )
        self.fusion_layer_2 = MFFM(
            in_channels_img=stage_channels[2],
            in_channels_depth=stage_channels[2],
        )
        self.fusion_layer_3 = MFFM(
            in_channels_img=stage_channels[3],
            in_channels_depth=stage_channels[3],
        )
        self.fusion_layer_4 = MFFM(
            in_channels_img=stage_channels[4],
            in_channels_depth=stage_channels[4],
        )

        self.Fen_1 = FFRM(channels=stage_channels[1])
        self.Fen_2 = FFRM(channels=stage_channels[2])
        self.Fen_3 = FFRM(channels=stage_channels[3])
        self.Fen_4 = FFRM(channels=stage_channels[4])

        self.fusion_block_3 = HFAM(
            in_channels_low=stage_channels[3],
            in_channels_high=stage_channels[4],
            obj_channels=stage_channels[1],
            mid_channels=None,
        )

        self.fusion_block_2 = HFAM(
            in_channels_low=stage_channels[2],
            in_channels_high=stage_channels[3],
            obj_channels=stage_channels[1],
            mid_channels=None,
        )

        self.fusion_block_1 = HFAM(
            in_channels_low=stage_channels[1],
            in_channels_high=stage_channels[2],
            obj_channels=stage_channels[1],
            mid_channels=None,
        )
        self.head = MultiScalePrediction(c1=stage_channels[1], c2=stage_channels[2], c3=stage_channels[3], obj=stage_channels[1])

    def forward(self, x, depth):
        x1, x2, x3, x4 = self.encoder_rgb(x)
        d1, d2, d3, d4 = self.encoder_d(depth)

        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        d2 = self.conv2(d2)
        d3 = self.conv3(d3)
        d4 = self.conv4(d4)

        high_freq_enh, low_freq_enh = self.m1(x1)
        high_freq_enh_2, low_freq_enh_2 = self.m2(x2)
        high_freq_enh_4, low_freq_enh_4 = self.m4(x4)

        fused_high, fused_low = self.m(
            high_freq_enh,
            high_freq_enh_2,
            high_freq_enh_4,
            low_freq_enh,
            low_freq_enh_2,
            low_freq_enh_4,
        )

        obj_map, edge_map = self.EdgeAndObj(x1, x2, x4, fused_high, fused_low)
        edge = torch.sigmoid(edge_map)

        #多模态特征融合
        fused_1 = self.fusion_layer_1(x1, d1,edge)
        fused_2 = self.fusion_layer_2(x2, d2,edge)
        fused_3 = self.fusion_layer_3(x3, d3,edge)
        fused_4 = self.fusion_layer_4(x4, d4,edge)

        fused_1_en = self.Fen_1(fused_1)
        fused_2_en = self.Fen_2(fused_2)
        fused_3_en = self.Fen_3(fused_3)
        fused_4_en = self.Fen_4(fused_4)

        out_3 = self.fusion_block_3(
            feat_low=fused_3_en,
            feat_high=fused_4_en,
            obj_map=obj_map,
        )

        out_2 = self.fusion_block_2(
            feat_low=fused_2_en,
            feat_high=out_3,
            obj_map=obj_map,
        )

        out_1 = self.fusion_block_1(
            feat_low=fused_1_en,
            feat_high=out_2,
            obj_map=obj_map,
        )

        pred1_up, pred2_up, pred3_up, obj= self.head(out_1, out_2, out_3, obj_map)

        return pred1_up, pred2_up, pred3_up, obj, edge_map







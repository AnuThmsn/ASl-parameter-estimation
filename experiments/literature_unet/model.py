import torch
import torch.nn as nn

class LuciwUNet3D(nn.Module):
    """
    Inspired by Luciw et al. 2022 (DOI:10.1002/mrm.29193), adapted to 4 simulated PLDs and synthetic ASL data.
    Input: [B, 4, D, H, W]
    Output: [B, 2, D, H, W] (CBF + ATT)
    """
    def __init__(self, in_channels=4, base_channels=16):
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )
            
        self.enc1 = conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        
        self.bottleneck = conv_block(base_channels * 4, base_channels * 8)
        
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec3 = conv_block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec2 = conv_block(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec1 = conv_block(base_channels * 2, base_channels)
        
        self.final_conv = nn.Conv3d(base_channels, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        b = self.bottleneck(p3)
        
        u3 = self.up3(b)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)
        
        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        return self.final_conv(d1)

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {count}")
    return count

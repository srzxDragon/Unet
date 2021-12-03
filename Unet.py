import torch
import torch.nn as nn


class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoConv, self).__init__()
        self.twoconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.twoconv(x)


class TwoConvDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoConvDown, self).__init__()
        self.twoconvdown = nn.Sequential(
            nn.MaxPool2d(2),
            TwoConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.twoconvdown(x)


class UpCat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpCat, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.twoconv = TwoConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2, mode="pad"):
        '''
        :param x1: Unet右半部分，尺寸较小
        :param x2: Unet左半部分，尺寸较大
        :return:
        '''
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if mode == "pad":
            x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        elif mode == "crop":
            left = diffX // 2
            right = diffX - left
            up = diffY // 2
            down = diffY - up

            x2 = x2[:, :, left:x2.size()[2]-right, up:x2.size()[3]-down]

        x = torch.cat([x2, x1], dim=1)
        x = self.twoconv(x)
        return x



class Unet(nn.Module):
    def __init__(self, in_channels,
                 channel_list: list = [64, 128, 256, 512, 1024],
                 length = 5,
                 mode = "pad"):
        super(Unet, self).__init__()
        self.twoconv = TwoConv(in_channels, channel_list[0])
        self.len = length
        self.mode = mode


        twoconvdown_list = []
        for i in range(1, self.len):
            twoconvdown_list.append(TwoConvDown(channel_list[i-1], channel_list[i]))
        self.twoconvdown = nn.Sequential(*twoconvdown_list)

        upcat_list = []
        for i in range(self.len-1, 0, -1):
            upcat_list.append(UpCat(channel_list[i], channel_list[i-1]))
        self.upcat = nn.Sequential(*upcat_list)

        self.outconv = nn.Conv2d(in_channels=channel_list[0], out_channels=2, kernel_size=1)

    def forward(self, x):
        down_result = []
        x = self.twoconv(x)
        down_result.append(x)

        for layers in self.twoconvdown:
            x = layers(x)
            down_result.append(x)

        up_result = []
        x = self.upcat[0](down_result[self.len-1], down_result[self.len-2], self.mode)
        up_result.append(x)
        for i in range(1, self.len-1):
            x = self.upcat[i](x, down_result[self.len-i-2], self.mode)
            up_result.append(x)

        x = self.outconv(x)
        return x




if __name__ == "__main__":
    model = Unet(in_channels=3, mode="crop")
    # print(model)
    img = torch.rand([4, 3, 572, 572])
    result = model(img)
    print(result.size())


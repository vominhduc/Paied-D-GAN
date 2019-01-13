import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


class VisualSemanticEmbedding(nn.Module):
    def __init__(self, embed_ndim):
        super(VisualSemanticEmbedding, self).__init__()
        self.embed_ndim = embed_ndim

        # image feature
        self.img_encoder = models.vgg16(pretrained=True)
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.feat_extractor = nn.Sequential(*(self.img_encoder.classifier[i] for i in range(6)))
        self.W = nn.Linear(4096, embed_ndim, False)

        # text feature
        self.txt_encoder = nn.GRU(embed_ndim, embed_ndim, 1)

    def forward(self, img, txt):
        # image feature
        img_feat = self.img_encoder.features(img)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.feat_extractor(img_feat)
        img_feat = self.W(img_feat)

        # text feature
        h0 = torch.zeros(1, img.size(0), self.embed_ndim)
        h0 = Variable(h0.cuda() if txt.data.is_cuda else h0)
        _, txt_feat = self.txt_encoder(txt, h0)
        txt_feat = txt_feat.squeeze()

        return img_feat, txt_feat


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

    def forward(self, x):
        return F.relu(x + self.encoder(x))



class Generator(nn.Module):
    def __init__(self, use_vgg=True):
        super(Generator, self).__init__()

        
        #1st conv layer
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        #3rd conv layer
        self.encoder3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

       
        # residual blocks
        self.residual_blocks = nn.Sequential(
            nn.Conv2d(512 + 128, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

        # decoder

        #1st decoder
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        #2nd decoder
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256*2, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        #3rd decoder
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128*2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, padding=0),
            nn.Tanh()
        )

        # conditioning augmentation
        self.mu = nn.Sequential(
            nn.Linear(300, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(300, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.apply(init_weights)

    def forward(self, img, txt_feat, z=None):
        # image encoder
        img_feat1 = self.encoder1(img)
        img_feat2 = self.encoder2(img_feat1)
        img_feat3 = self.encoder3(img_feat2)

        #text encoder
        z_mean = self.mu(txt_feat)
        z_log_stddev = self.log_sigma(txt_feat)
        z = torch.randn(txt_feat.size(0), 128)
        if next(self.parameters()).is_cuda:
            z = z.cuda()
        txt_feat = z_mean + z_log_stddev.exp() * Variable(z)

  
        # residual blocks
        txt_feat = txt_feat.unsqueeze(-1).unsqueeze(-1)
        txt_feat = txt_feat.repeat(1, 1, img_feat3.size(2), img_feat3.size(3))
        fusion = torch.cat((img_feat3, txt_feat), dim=1)
        fusion = self.residual_blocks(fusion)

        # decoder
        output1 = self.decoder1(fusion)
        output2 = self.decoder2(torch.cat((output1,img_feat2), dim=1))
        output = self.decoder3(torch.cat((output2,img_feat1), dim=1))

        return output, (z_mean, z_log_stddev)


class FG_Discriminator(nn.Module):
    def __init__(self):
        super(FG_Discriminator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.residual_branch = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 128, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4)
        )

        self.compression = nn.Sequential(
            nn.Linear(300, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.apply(init_weights)
        

    def forward(self, img, txt_feat):
        img_feat = self.encoder(img)
        img_feat = F.leaky_relu(img_feat + self.residual_branch(img_feat), 0.2)
        txt_feat = self.compression(txt_feat)

        txt_feat = txt_feat.unsqueeze(-1).unsqueeze(-1)
        txt_feat = txt_feat.repeat(1, 1, img_feat.size(2), img_feat.size(3))
        fusion = torch.cat((img_feat, txt_feat), dim=1)

        output = self.classifier(fusion)

        return output.squeeze()

class BG_Discriminator(nn.Module):
    def __init__(self):
        super(BG_Discriminator, self).__init__()
        # shared-parameter full-connected layers
        self.fc = nn.Sequential(
            nn.Linear(1196, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 10),
            nn.ReLU(inplace=True),         
        )

        # joint full-connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(20, 1)
        )
        
        self.apply(init_weights)


    def forward_once(self, img_feat, text_feat):
        
        img_feat = img_feat.view(img_feat.size()[0], -1)
        text_feat = text_feat.view(text_feat.size()[0], -1)

        # concat image and text features
        img_text_feat = torch.cat((img_feat, text_feat), dim=1)

        output = self.fc(img_text_feat)

        return output

    def forward(self, input1, text1, input2, text2):
        output1 = self.forward_once(input1, text1)
        output2 = self.forward_once(input2, text2)
        output = self.fc1(torch.cat((output1,output2),1))
        return output.view(-1, 1).squeeze(1) 

        
class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        

        relu1_2 = relu1_2.view(relu1_2.data.shape[0], 
            relu1_2.data.shape[1],
            relu1_2.data.shape[2]*relu1_2.data.shape[2])

        mean1 = torch.mean(relu1_2, 2)
        std1 = torch.std(relu1_2, 2)

        relu2_2 = relu2_2.view(relu2_2.data.shape[0], 
            relu2_2.data.shape[1],
            relu2_2.data.shape[2]*relu2_2.data.shape[2])

        mean2 = torch.mean(relu2_2, 2)
        std2 = torch.std(relu2_2, 2)

        relu3_3 = relu3_3.view(relu3_3.data.shape[0], 
            relu3_3.data.shape[1],
            relu3_3.data.shape[2]*relu3_3.data.shape[2])

        mean3 = torch.mean(relu3_3, 2)
        std3 = torch.std(relu3_3, 2)

        output = torch.cat((mean1, std1, mean2, std2, mean3, std3), 1)

        return output

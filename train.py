import argparse
import fastText as fasttext
# import fasttext
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import VisualSemanticEmbedding, Generator, FG_Discriminator, BG_Discriminator, Vgg16
from data import ReedICML2016


parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--caption_root', type=str, required=True,
                    help='root directory that contains captions')
parser.add_argument('--trainclasses_file', type=str, required=True,
                    help='text file that contains training classes')
parser.add_argument('--fasttext_model', type=str, required=True,
                    help='pretrained fastText model (binary file)')
parser.add_argument('--text_embedding_model', type=str, required=True,
                    help='pretrained text embedding model')
parser.add_argument('--vgg_model', type=str, required=True,
                    help='pretrained VGG-16 model')
parser.add_argument('--save_filename', type=str, required=True,
                    help='checkpoint file')
parser.add_argument('--resume_filename', type=str,
                    help='resume checkpoint file')
parser.add_argument('--num_threads', type=int, default=4,
                    help='number of threads for fetching data (default: 4)')
parser.add_argument('--num_epochs', type=int, default=600,
                    help='number of threads for fetching data (default: 600)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size (default: 26)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate (dafault: 0.0002)')
parser.add_argument('--lr_decay', type=float, default=0.5,
                    help='learning rate decay (dafault: 0.5)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='beta1 for Adam optimizer (dafault: 0.5)')
parser.add_argument('--embed_ndim', type=int, default=300,
                    help='dimension of embedded vector (default: 300)')
parser.add_argument('--max_nwords', type=int, default=50,
                    help='maximum number of words (default: 50)')
parser.add_argument('--use_vgg', action='store_true',
                    help='use pretrained VGG network for image encoder')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
args = parser.parse_args()

if not args.no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    args.no_cuda = True



def preprocess(img, desc, len_desc, txt_encoder):
    img = Variable(img.cuda() if not args.no_cuda else img)
    desc = Variable(desc.cuda() if not args.no_cuda else desc)
    len_desc = len_desc.numpy()
    sorted_indices = np.argsort(len_desc)[::-1]
    original_indices = np.argsort(sorted_indices)
    packed_desc = nn.utils.rnn.pack_padded_sequence(
        desc[sorted_indices.tolist(), ...].transpose(0, 1),
        len_desc[sorted_indices]
    )

    _, txt_feat = txt_encoder(packed_desc)
    txt_feat = txt_feat.squeeze()
    txt_feat = txt_feat[original_indices, ...]

    txt_feat_np = txt_feat.data.cpu().numpy() if not args.no_cuda else txt_feat.data.numpy()
    txt_feat_mismatch = torch.Tensor(np.roll(txt_feat_np, 1, axis=0))
    txt_feat_mismatch = Variable(txt_feat_mismatch.cuda() if not args.no_cuda else txt_feat_mismatch)
    txt_feat_np_split = np.split(txt_feat_np, [txt_feat_np.shape[0] // 2])
    txt_feat_relevant = torch.Tensor(np.concatenate([
        np.roll(txt_feat_np_split[0], -1, axis=0),
        txt_feat_np_split[1]
    ]))
    txt_feat_relevant = Variable(txt_feat_relevant.cuda() if not args.no_cuda else txt_feat_relevant)
    return img, txt_feat, txt_feat_mismatch, txt_feat_relevant

def generateMask(b, c, h, w, k):
    mask_fg = torch.Tensor(np.random.binomial(1, 0.2, size=(h,w)))
    
    mask_fg = mask_fg.unsqueeze(0)
    
    mask_fg = mask_fg.repeat(c,1,1).unsqueeze(0)
    mask_fg = mask_fg.repeat(b,1,1,1)

    mask_fg = Variable(mask_fg.cuda() if not args.no_cuda else mask_fg)

    mask_bg = torch.ones((b,c,h,w))
    mask_bg = Variable(mask_bg.cuda() if not args.no_cuda else mask_bg)
    mask_bg = mask_bg - mask_fg

    return mask_fg, mask_bg


if __name__ == '__main__':
    print('Loading a pretrained fastText model...')
    word_embedding = fasttext.load_model(args.fasttext_model)

    print('Loading a dataset...')
    train_data = ReedICML2016(args.img_root,
                              args.caption_root,
                              args.trainclasses_file,
                              word_embedding,
                              args.max_nwords,
                              transforms.Compose([
                                  transforms.Scale(74),
                                  transforms.RandomCrop(64),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor()
                              ]))

    vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),

    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads)

    word_embedding = None

    # pretrained text embedding model
    print('Loading a pretrained text embedding model...')
    txt_encoder = VisualSemanticEmbedding(args.embed_ndim)
    txt_encoder.load_state_dict(torch.load(args.text_embedding_model))
    txt_encoder = txt_encoder.txt_encoder
    for param in txt_encoder.parameters():
        param.requires_grad = False

    # load the pre-trained vgg-16 and extract features
    print('Loading a pretrained VGG-16...')
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(args.vgg_model))


    G = Generator()
    D_FG = FG_Discriminator()
    D_BG = BG_Discriminator()
    

    if not args.no_cuda:
        txt_encoder.cuda()
        vgg.cuda()
        G.cuda()
        D_FG.cuda()
        D_BG.cuda()

    g_optimizer = torch.optim.Adam([x for x in G.parameters() if x.requires_grad],
                                   lr=args.learning_rate, betas=(args.momentum, 0.999))
    dfg_optimizer = torch.optim.Adam([x for x in D_FG.parameters() if x.requires_grad],
                                   lr=args.learning_rate, betas=(args.momentum, 0.999))
    dbg_optimizer = torch.optim.Adam([x for x in D_BG.parameters() if x.requires_grad],
                                   lr=args.learning_rate, betas=(args.momentum, 0.999))

    g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, 100, args.lr_decay)
    dfg_lr_scheduler = lr_scheduler.StepLR(dfg_optimizer, 100, args.lr_decay)
    dbg_lr_scheduler = lr_scheduler.StepLR(dbg_optimizer, 100, args.lr_decay)

    
    step = 0
   
    for epoch in range(args.num_epochs):
        dfg_lr_scheduler.step()
        dbg_lr_scheduler.step()
        g_lr_scheduler.step()

        # training loop
        avg_DFG_real_loss = 0
        avg_DFG_real_m_loss = 0
        avg_DFG_fake_loss = 0
        avg_DBG_same_loss = 0
        avg_DBG_diff_loss = 0
        avg_DBG_input_diff_loss = 0
        avg_G_fake_loss = 0
        avg_G_bg_loss = 0
        avg_G_re_loss = 0
        avg_G_loss = 0
        avg_kld = 0


        for i, (img, desc, len_desc) in enumerate(train_loader):
            img, txt_feat, txt_feat_mismatch, txt_feat_relevant = \
                preprocess(img, desc, len_desc, txt_encoder)
        
            real_img = img * 2 - 1

            img_G = Variable(vgg_normalize(img.data)) if args.use_vgg else real_img

            ONES = Variable(torch.ones(img.size(0)))
            ZEROS = Variable(torch.zeros(img.size(0)))
            if not args.no_cuda:
                ONES, ZEROS = ONES.cuda(), ZEROS.cuda()


            # UPDATE GENERATOR
            G.zero_grad()
            fake, (z_mean, z_log_stddev) = G(img_G, txt_feat_relevant)
            kld = torch.mean(-z_log_stddev + 0.5 * (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1))
            avg_kld += kld.item()
            fake_logit = D_FG(fake, txt_feat_relevant)
            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, ONES)
            avg_G_fake_loss += fake_loss.item()

            fake1,(z_mean, z_log_stddev) = G(img_G, txt_feat_relevant)
            kld += torch.mean(-z_log_stddev + 0.5 * (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1))
            avg_kld += kld.item()
            fake2,(z_mean, z_log_stddev) = G(img_G, txt_feat_relevant)
            kld += torch.mean(-z_log_stddev + 0.5 * (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1))
            avg_kld += kld.item()
            img_feat_1 = vgg(fake1)
            img_feat_2 = vgg(fake2)
            out = D_BG(img_feat_1, txt_feat_relevant, img_feat_2, txt_feat_relevant)
            bg_loss = F.binary_cross_entropy_with_logits(out, ONES)
            avg_G_bg_loss += bg_loss.item()

            # reconstruction loss
            re_loss = 1/3*F.l1_loss(img_G, fake) + 1/3*F.l1_loss(img_G, fake1) + 1/3*F.l1_loss(img_G, fake2)
            avg_G_re_loss += re_loss.item()
            
            G_loss = fake_loss + bg_loss + kld + 0.0001*re_loss
            avg_G_loss += G_loss.item()

            G_loss.backward()
            g_optimizer.step()

            # UPDATE DISCRIMINATOR
             # generate random binary filter
            mask_fg, mask_bg = generateMask(real_img.size(0), real_img.size(1), real_img.size(2), real_img.size(3), int(real_img.size(2)/4))

            D_FG.zero_grad()
            # real image with matching text
            real_logit = D_FG(real_img * mask_fg, txt_feat)
            real_loss = F.binary_cross_entropy_with_logits(real_logit, ONES)
            avg_DFG_real_loss += real_loss.item()
            real_loss.backward()

            # real image with mismatching text
            real_m_logit = D_FG(real_img, txt_feat_mismatch)
            real_m_loss = 0.5 * F.binary_cross_entropy_with_logits(real_m_logit, ZEROS)
            avg_DFG_real_m_loss += real_m_loss.item()
            real_m_loss.backward()

            # synthesized image with semantically relevant text
            fake, _ = G(img_G, txt_feat_relevant)
            fake_logit = D_FG(fake.detach(), txt_feat_relevant)
            fake_loss = 0.5 * F.binary_cross_entropy_with_logits(fake_logit, ZEROS)
            avg_DFG_fake_loss += fake_loss.item()
            fake_loss.backward()
            dfg_optimizer.step()

            # UPDATE BACKGROUND DISCRIMINATOR
            D_BG.zero_grad()
            # same background
            img_feat_1 = vgg(real_img)
            img_feat_2 = vgg(real_img)
            out = D_BG(img_feat_1, txt_feat, img_feat_2, txt_feat_mismatch)            
            same_loss = F.binary_cross_entropy_with_logits(out, ONES)
            avg_DBG_same_loss += same_loss.item()
            same_loss.backward()
            
                      
            # different background
            fake1,_ = G(img_G, txt_feat_relevant)
            fake2,_ = G(img_G, txt_feat_relevant)
            img_feat_1 = vgg(fake1.detach())
            img_feat_2 = vgg(fake2.detach())

            out = D_BG(img_feat_1, txt_feat_relevant, img_feat_2, txt_feat_relevant)
            diff_loss = F.binary_cross_entropy_with_logits(out, ZEROS)
            avg_DBG_diff_loss += diff_loss.item()
            diff_loss.backward()

            dbg_optimizer.step()
            
            
            if i % 10 == 0:
                print(('Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_mis: %.4f, D_fake: %.4f, G_fake: %.4f, KLD: %.4f, Same: %.4f, Diff: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_DFG_real_loss / (i + 1),
                      avg_DFG_real_m_loss / (i + 1), avg_DFG_fake_loss / (i + 1), 
                      avg_G_loss / (i + 1), avg_kld / (i + 1), avg_DBG_same_loss / (i + 1), 
                      avg_DBG_diff_loss / (i + 1))))


        save_image((fake.data + 1) * 0.5, './examples/epoch_%d.png' % (epoch + 1))

        torch.save(G.state_dict(), args.save_filename)
       

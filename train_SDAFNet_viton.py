import os
import math
import tqdm
import random
import torch
import argparse
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from datasets import VITONDataset
from models.sdafnet import SDAFNet_Tryon
from models import external_function
from utils import lpips
from utils.utils import AverageMeter
from torchvision import transforms, utils
from torch.utils import data

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default=192)
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--dataset_imgpath', type=str, default='VITON/VITON_train')
    parser.add_argument('--dataset_list', type=str, default='VITON/train.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--display_freq', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--multi_flows', type=int, default=6)

    opt = parser.parse_args()
    return opt

def train(opt, net):
    train_dataset = VITONDataset(opt)
    train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size,drop_last=True, shuffle=opt.shuffle,num_workers=opt.workers)

    #criterion
    criterion_L1 = nn.L1Loss()
    criterion_percept = lpips.exportPerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
    criterion_style = external_function.VGGLoss().cuda()
    #optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=opt.lr)
    #scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    epoches = 200
    iterations = 0
    for epoch in range(epoches):
        loss_l1_avg = AverageMeter()
        loss_vgg_avg = AverageMeter()
        for i, inputs in enumerate(tqdm.tqdm(train_loader)):
            iterations+=1
            img_names = inputs['img_name']
            c_names = inputs['c_name']['paired']
            img = inputs['img'].cuda()
            img_agnostic = inputs['img_agnostic'].cuda()
            pose = inputs['pose'].cuda()
            cloth_img = inputs['cloth']['paired'].cuda()
            img =  F.interpolate(img, size=(256, 192), mode='bilinear')
            ref_input = torch.cat((pose, img_agnostic), dim=1)
            result_tryon, results_all = net(ref_input, cloth_img, img_agnostic, return_all=True)
            epsilon = 0.001
            loss_all = 0
            num_layer = 5
            for num in range(num_layer):
                cur_img = F.interpolate(img, scale_factor=0.5**(4-num), mode='bilinear')
                loss_l1 = criterion_L1(results_all[num], cur_img.cuda())
                if num == 0:
                    cur_img = F.interpolate(cur_img, scale_factor=2, mode='bilinear')
                    results_all[num] = F.interpolate(results_all[num], scale_factor=2, mode='bilinear')
                loss_perceptual = criterion_percept(cur_img.cuda(),results_all[num]).mean()
                loss_content, loss_style = criterion_style(results_all[num], cur_img.cuda())
                loss_vgg = loss_perceptual+100*loss_style+0.1*loss_content
                loss_all = loss_all + (num+1) * loss_l1 + (num + 1)  * loss_vgg
            loss = loss_all
            loss_l1_avg.update(loss.item())
            loss_vgg_avg.update(loss_vgg.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations%opt.display_freq ==1:
                parse_pred = torch.cat([result_tryon,img],2)
                parse_pred = F.interpolate(parse_pred, size=(512, 192), mode='bilinear')
                utils.save_image(
                        parse_pred,
                        f"{os.path.join(opt.save_dir, opt.name)}/log_sample/{str(iterations).zfill(6)}_daf_viton_s1fine_{str(opt.name)}.jpg",
                        nrow=6,
                        normalize=True,
                        range=(-1, 1),
                    )
            if iterations%opt.save_freq ==1:
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "optim": optimizer.state_dict(),
                        "opt": opt,
                    },
                    opt.save_dir+f"/{str(iterations).zfill(6)}_daf_viton_s1fine_{str(opt.name)}.pt",
                )
            print("[%d %d][%d] l1_loss:%.4f %.4f vgg_loss:%.4f %.4f "%(epoch,epoches,iterations,loss_l1.item(),loss_l1_avg.avg,loss_vgg.item(),loss_vgg_avg.avg))
        scheduler.step()

def main():
    opt = get_opt()
    print(opt)
    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))
        os.makedirs(os.path.join(opt.save_dir, opt.name,'log_sample'))
    sdafnet = SDAFNet_Tryon(ref_in_channel=6)
    sdafnet = sdafnet.cuda()
    sdafnet.train()
    #sdafnet.load_state_dict(torch.load("ckpt_viton.pt"))
    sdafnet = torch.nn.DataParallel(sdafnet,device_ids=range(torch.cuda.device_count()))

    cudnn.benchmark = True
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    train(opt, sdafnet)



if __name__ == '__main__':
    main()

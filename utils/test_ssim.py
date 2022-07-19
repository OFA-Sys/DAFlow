import cv2
import glob
import sys
sys.path.append('/home/baishuai.bs/work/DAFlow_opensource/DAFlow')
import  pytorch_ssim
from torch.autograd import Variable
import numpy as np
SSIMS =[]
PSNRS = []
import torch
ssim_loss = pytorch_ssim.SSIM(window_size = 11)
files = glob.glob(sys.argv[1]+"/*")
for file_ in files:
    img1 = cv2.imread(file_)
    img2 = cv2.imread(sys.argv[2]+'/'+file_.split('/')[-1])
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0 
    img1 = img1.cuda()
    img2 = img2.cuda()
   
    img1 = Variable( img1, requires_grad=False)
    img2 = Variable( img2, requires_grad = False)
    SSIM = ssim_loss(img1, img2).data.item()
    print('SSIM: ', SSIM)
    SSIMS.append(SSIM)
print(len(SSIMS),np.array(SSIMS).mean())

import argparse
import random 
import copy
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_warmup as warmup
import numpy as np
import pandas as pd
from kornia.losses import ssim_loss

from ISFDataset import ISFDataset, _default_transform
from ssim_map import pos_similarity_ratio
from models.space_net import SPACE as SPACE
from models.race import RACE, ACE
from models.hdrnet import PICE_B
from saliency_losses import nss, corr_coeff, kld_loss, log_softmax, softmax
from utils import contrast_loss_G, contrast_loss_L, R_imgs, R_sclr 
from utils import decompose_imgs
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

random.seed(7)
np.random.seed(7)
torch.random.manual_seed(7)

# Process command line arguments.
parser = argparse.ArgumentParser()
train_img_path = "C:/Users/lps3090/Desktop/dataset with saliency map train/IMAGES"   
train_msk_path = "C:/Users/lps3090/Desktop/dataset with saliency map train/SALIENCY"
train_fix_path = "C:/Users/lps3090/Desktop/dataset with saliency map train/FIXATIONS" # convert from MAT to tensor
val_img_path = "C:/Users/lps3090/Desktop/dataset with saliency map train/IMAGES"
val_msk_path = "C:/Users/lps3090/Desktop/dataset with saliency map train/SALIENCY"
val_fix_path = "C:/Users/lps3090/Desktop/dataset with saliency map train/FIXATIONS"
parser.add_argument('--train_img_path', action='store', type=str, help="Training image path.", default=train_img_path)
parser.add_argument('--val_img_path', action='store', type=str, help="Validation image path.", default=val_img_path)
parser.add_argument('--train_msk_path', action='store', type=str, help="Training mask path.", default=train_msk_path)
parser.add_argument('--val_msk_path', action='store', type=str, help="Validation mask path.", default=val_msk_path)
parser.add_argument('--train_fix_path', action='store', type=str, help="Training fixation path.", default=train_fix_path)
parser.add_argument('--val_fix_path', action='store', type=str, help="Validation fixation path.", default=val_fix_path)
parser.add_argument('--save_name', action='store', type=str, help="File name to save.", default='results/space_v3_model')
parser.add_argument('--num_epoch', action='store', type=int, help="File name to save.", default=30)
parser.add_argument('--model_version', type=int, required=True)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, num_epochs, dataloaders, device, optimizer, warmup_scheduler, lr_scheduler, save_name='results/file_result', fold_num=0):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.0e12
    gamma_corr = 2.2 

    loss_hist_header = ['epoch', 'time_taken', 'finished_at', 'phase', 'total', 'power', 'ssim', 'g_contrast', 'l_contrast', 'nss', 'cc', 'kldiv']
    loss_hist = pd.DataFrame(columns=loss_hist_header) 

    mse_loss = nn.MSELoss() 
    l1_loss = nn.L1Loss() 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval() 
            since_phase = time.time()
            
            # Running loss
            running_loss = torch.zeros(8)

            # Iterate over data.
            for i, (inputs, masks, fixations) in enumerate(dataloaders[phase]):
                power_k = float(random.randint(10, 90)) / 100.
                inputs = inputs.to(device)
                masks = masks.to(device)
                fixations = fixations.to(device)

                y_inputs, _ = decompose_imgs(inputs)
                reduced_inputs = y_inputs * power_k

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    gt_reduction = R_sclr(power_k, gamma=gamma_corr) * torch.ones(inputs.size(0), 1).to(device=device)
                    outputs = model(y_inputs, R=R_sclr(power_k, gamma=gamma_corr))

                    _ssim_loss = 2 * ssim_loss(outputs, y_inputs, window_size=5, reduction='mean') # adjustable_ssim_loss(outputs, y_inputs, window_size=5, reduction='mean', luminance_factor=.25) # ssim_loss(outputs, y_inputs, window_size=5, reduction='mean')   
                    
                    _power_loss = ( 5. * mse_loss(R_imgs(y_inputs, outputs, gamma=gamma_corr), gt_reduction) + \
                        5. * l1_loss(R_imgs(y_inputs, outputs, gamma=gamma_corr), gt_reduction) )
                    _gc_loss = (1./2) * contrast_loss_G(outputs, reduced_inputs, R=gt_reduction).mean() 
                    _lc_loss = (1./4) * contrast_loss_L(outputs, reduced_inputs, R=gt_reduction).mean() 
                    print(_gc_loss.item(), _lc_loss.item())

                    salmap = pos_similarity_ratio(outputs, reduced_inputs, y_inputs)
                    
                    # Debugging
                    # plt.imshow(salmap[0].detach().cpu().squeeze())
                    # plt.show()
                    # print("Salmap {:.3f} {:.3f} {:.3f}".format(salmap.max().item(), salmap.min().item(), salmap.mean().item()))
                    # plt.imshow(masks[0].detach().cpu().squeeze())
                    # plt.show()
                    # print("Mask {:.3f} {:.3f} {:.3f}".format(masks.max().item(), masks.min().item(), masks.mean().item()))
                    
# The Original Pave         
                    if args.model_version == 0:
                        _nss_loss = (-0.25) * nss(salmap, fixations > 0.5).nanmean()
                        _cc_loss = (-0.25) * corr_coeff(salmap, masks).mean() # torch.Tensor([0]).cuda()
                        _kldiv_loss = 2.5 * kld_loss(log_softmax(salmap), softmax(masks)).mean() # torch.Tensor([0]).cuda()
                        
                        loss = (_ssim_loss + _power_loss + _gc_loss + _lc_loss + _nss_loss + _cc_loss + _kldiv_loss)

                    
# increasing NSS whhile the others fix
                    elif args.model_version == 3:
                        _nss_loss = (-0.1) * nss(salmap, fixations > 0.5).nanmean()
                        _cc_loss = (-0.25) * corr_coeff(salmap, masks).mean() # torch.Tensor([0]).cuda()
                        _kldiv_loss = 2.5 * kld_loss(log_softmax(salmap), softmax(masks)).mean() # torch.Tensor([0]).cuda()
                        
                        loss = (_ssim_loss + _power_loss + _gc_loss + _lc_loss + _nss_loss + _cc_loss + _kldiv_loss)
                       
                       
# increasing CC whhile the others fix                       
                    elif args.model_version == 4:
                        _nss_loss = (-0.25) * nss(salmap, fixations > 0.5).nanmean()
                        _cc_loss = (-0.1) * corr_coeff(salmap, masks).mean() # torch.Tensor([0]).cuda()
                        _kldiv_loss = 2.5 * kld_loss(log_softmax(salmap), softmax(masks)).mean() # torch.Tensor([0]).cuda()
                        
                        loss = (_ssim_loss + _power_loss + _gc_loss + _lc_loss + _nss_loss + _cc_loss + _kldiv_loss)
                    

# increasing KLDiv whhile the others fix
                    elif args.model_version == 8:
                        _nss_loss = (-0.25) * nss(salmap, fixations > 0.5).nanmean()
                        _cc_loss = (-0.25) * corr_coeff(salmap, masks).mean() # torch.Tensor([0]).cuda()
                        _kldiv_loss = 3.5 * kld_loss(log_softmax(salmap), softmax(masks)).mean() # torch.Tensor([0]).cuda()
                        
                        loss = (_ssim_loss + _power_loss + _gc_loss + _lc_loss + _nss_loss + _cc_loss + _kldiv_loss)

                        
# further increasing NSS & CC whhile decreasing the KLDiv
                    elif args.model_version == 10:
                        _nss_loss = (-0.001) * nss(salmap, fixations > 0.5).nanmean()
                        _cc_loss = (-0.001) * corr_coeff(salmap, masks).mean() # torch.Tensor([0]).cuda()
                        _kldiv_loss = 0.2 * kld_loss(log_softmax(salmap), softmax(masks)).mean() # torch.Tensor([0]).cuda()
                        
                        loss = (_ssim_loss + _power_loss + _gc_loss + _lc_loss + _nss_loss + _cc_loss + _kldiv_loss)





                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        with warmup_scheduler.dampening():
                            lr_scheduler.step()
                    
                    if i % 50 == 0:
                        print(" #{} k:{:.2f} LR:{:.2e} Loss SSIM: {:.3f} Power: {:.3f} GC: {:.3f} LC: {:.3f} NSS: {:.3f} CC: {:.3f} KLD: {:.3f}".format(
                            str(i).zfill(4), power_k, optimizer.param_groups[0]['lr'], float(_ssim_loss), float(_power_loss), float(_gc_loss), float(_lc_loss), float(_nss_loss), float(_cc_loss), float(_kldiv_loss)))
                        
                _r_loss = torch.tensor([loss.item(), _power_loss.item(), _ssim_loss.item(), _gc_loss.item(), _lc_loss.item(), _nss_loss.item(), _cc_loss.item(), _kldiv_loss.item()]) * inputs.size(0)
                running_loss += _r_loss

            running_loss /= len(dataloaders[phase].dataset)

            print('{} | Loss Total: {:.3f} Power: {:.3f} SSIM: {:.3f} GC: {:.3f} LC: {:.3f} NSS: {:.3f} CC: {:.3f} KLD: {:.3f}'.format(
                phase, running_loss[0].item(), running_loss[1].item(), running_loss[2].item(), running_loss[3].item(), running_loss[4].item(), running_loss[5].item(), running_loss[6].item(), running_loss[7].item()))
            now = datetime.now()
            finished_at_str = now.strftime("%m/%d/%Y, %H:%M:%S")
            print("Now", finished_at_str)
            print()

            if phase == 'val' and running_loss[0].item() < best_loss:
                best_loss = running_loss[0].item()
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best loss achieved. Saving model...')
                torch.save(best_model_wts, save_name + '_' + str(fold_num) + '.pth')
                print()

            time_taken = time.time() - since_phase
            time_taken_str = "{}:{}:{}".format(int(time_taken//3600),int(time_taken//60),int(time_taken%60))

            loss_hist.loc[len(loss_hist)] = [epoch, time_taken_str, finished_at_str, phase] + ["{:.4f}".format(i) for i in running_loss.detach().clone().cpu().tolist()]

        loss_hist.sort_values(by=['phase', 'epoch']).to_csv(save_name + '_' + str(fold_num)  + '_sm.csv', index=False)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_ds = ISFDataset(args.train_img_path, args.train_msk_path, args.train_fix_path, resize=(640,480), color_mode="grayscale", transform=_default_transform)
    # test_ds = ISFDataset(args.val_img_path, args.val_msk_path, args.val_fix_path, resize=(640,480))
    
    batch_size = 4
    # dataset = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
        
  
    if args.model_version < 100:
        model = SPACE(apply_center_bias=True, apply_gfcorrection=True, apply_len=True).to(device=device)

    num_epochs = args.num_epoch
    # dataloaders = {'train':train_loader, 'val':test_loader}
    optimizer = optim.AdamW(params=model.parameters(), lr=3e-3) # lr=1e-3 3e-3 5e-3)
    num_steps = int((len(train_ds) / batch_size) * num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_ds)):
        # Print
        print(f'Start FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4, sampler=train_subsampler)
        test_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4, sampler=test_subsampler)
        dataloaders = {'train':train_loader, 'val':test_loader}
        train(model, num_epochs, dataloaders, device, optimizer, warmup_scheduler, lr_scheduler, save_name=args.save_name, fold_num=fold)
        
        print(f'End FOLD {fold}')
        print('--------------------------------')
    print('Finished.')




#............................................................END........................................................................................

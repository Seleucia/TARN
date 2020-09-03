import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import helper.io_utils as iot
import helper.utils  as ut
import helper.mhelper as mhe
import argparse
from tarn import TARN as m_tarn
from helper.ds_loader  import dsLoader
import h5py


#CUDA_VISIBLE_DEVICES=0 python3 train_svg_lp_drivesim.py
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1986, help='Random seed')
parser.add_argument('--nupdate', type=int, default=1000000, help='Random seed')
parser.add_argument('--bsize', type=int, default=5, help='Batch Size')
parser.add_argument('--feats_gru_hidden_size', type=int, default=256, help='hidden size')
parser.add_argument('--dml_gru_hidden_size', type=int, default=256, help='hidden size')
parser.add_argument('--input_size', type=int, default=4096, help='hidden size')
parser.add_argument('--inp_s', type=int, default=4096, help='hidden size')
parser.add_argument('--nclass', type=int, default=5, help='hidden size')
parser.add_argument('--kshot', type=int, default=10, help='hidden size')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.0005, help='W. Decay')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


dsL=dsLoader(iot.get_c3d_feats_hdf5())
mdl_tarn=m_tarn(opt.input_size,opt.feats_gru_hidden_size,opt.dml_gru_hidden_size).cuda()
mm_opt=optim.Adam(mdl_tarn.parameters(), lr=opt.lr, weight_decay=opt.wd)


criterion = nn.BCELoss()

# --------- training funtions ------------------------------------
def train(c3d_feat_Q,lns_Q,c3d_feat_S_pos,lns_S_pos,c3d_feat_S_neg,lns_S_neg,target_ones,target_zeros,uidx):
    mdl_tarn.train()
    mm_opt.zero_grad()
    q_kc_pos, q_kc_neg= mdl_tarn(c3d_feat_Q,lns_Q,c3d_feat_S_pos,lns_S_pos,c3d_feat_S_neg,lns_S_neg,target_ones,target_zeros,uidx)
    pos_loss = criterion(q_kc_pos, target_ones)
    neg_loss = criterion(q_kc_neg, target_zeros)
    loss=neg_loss+pos_loss
    loss.backward()
    mm_opt.step()
    acc=(q_kc_pos>0.5).cpu().numpy().mean()/2.0+(neg_loss<=0.5).cpu().numpy().mean()/2.0
    return neg_loss,pos_loss,acc


writer = SummaryWriter(iot.get_wd()+'/runs/mdl')
writer.add_text('DataSet', 'Training Samples:{0}'.format(len(dsL.train_samples)))
writer.add_text('DataSet', 'Test Samples:{0}'.format(len(dsL.test_samples)))
print('Training Stated....')
moving_avg_loss=[]
moving_avg_prec=[]
target_zeros = torch.zeros((opt.bsize, 1)).cuda()
target_ones = torch.ones((opt.bsize, 1)).cuda()
for uidx in range(opt.nupdate):
    c3d_feat_Q,anames_Q,lns_Q,Q_kys,c3d_feat_S_pos,anames_S_pos,lns_S_pos,c3d_feat_S_neg,anames_S_neg,lns_S_neg= dsL.get_batch(opt.bsize,uidx,stream_mode=0)
    neg_loss,pos_loss,prec1=train(c3d_feat_Q,lns_Q,c3d_feat_S_pos,lns_S_pos,c3d_feat_S_neg,lns_S_neg,target_ones,target_zeros,uidx)
    moving_avg_loss.append([pos_loss.item(),neg_loss.item()])
    moving_avg_prec.append(prec1)
    if uidx%5000==0:
        writer.add_scalar('Loss/Train_pos',
                      np.mean(moving_avg_loss,0)[0],
                      uidx)

        writer.add_scalar('Loss/Train_neg',
                      np.mean(moving_avg_loss,0)[1],
                      uidx)

        writer.add_scalar('Loss/Train_acc',
                      np.mean(moving_avg_prec,0),
                      uidx)

        print('Upd: {0}| Loss Train pos/neg: {1} / {2}, acc: {3}'.format(uidx, np.mean(moving_avg_loss,0)[0], np.mean(moving_avg_loss,0)[1],np.mean(moving_avg_prec,0)))
        moving_avg_loss = []
    if uidx%50000==0:
        mhe.save_model(uidx,opt,mdl_tarn,mm_opt)
writer.close()


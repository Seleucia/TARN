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
parser.add_argument('--nupdate', type=int, default=500000, help='Random seed')
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



N = 11;
M = 13;



criterion = nn.BCELoss()


#ds_mode=0 TRAINING
# --------- training funtions ------------------------------------
def train(S,S_batch_ln,Q,Q_batch_ln,target_ones,target_zeros,uidx):
    mdl_tarn.train()
    mm_opt.zero_grad()
    loss = 0
    neg_loss = 0
    pos_loss = 0
    A, q_kc = mdl_tarn(S, S_batch_ln, Q, Q_batch_ln, )
    for nc in range(2):
    # for nc in range(opt.nclass):
        if nc==0:
            pos_loss += criterion(q_kc[nc], target_ones)
        else:
            neg_loss += criterion(q_kc[nc], target_zeros)
        # loss += ut.CrossEntropy(q_kc, y=1)
    loss=neg_loss+pos_loss
    loss.backward()
    mm_opt.step()
    return neg_loss,pos_loss,0


writer = SummaryWriter(iot.get_wd()+'/runs/mdl')
writer.add_text('DataSet', 'Training Samples:{0}'.format(len(dsL.train_samples)))
writer.add_text('DataSet', 'Test Samples:{0}'.format(len(dsL.test_samples)))
print('Training Stated....')
moving_avg_loss=[]
target_zeros = torch.zeros((opt.bsize, 1)).cuda()
target_ones = torch.ones((opt.bsize, 1)).cuda()
for uidx in range(opt.nupdate):
    S_batch=[torch.randn(np.random.choice(range(N-5,N)), opt.input_size) for n in range(opt.bsize)]
    Q_batch=[torch.randn(np.random.choice(range(M-5,M)), opt.input_size) for n in range(opt.bsize)]
    c3d_feat_Q,anames_Q,lns_Q,Q_kys,c3d_feat_S,anames_S,lns_S = dsL.get_batch(opt.bsize,uidx,stream_mode=0)
    # print(c3d_feat_Q.shape,c3d_feat_S.shape)

    # S_batch_ln=[len(s) for s in S_batch]
    # Q_batch_ln=[len(q) for q in Q_batch]
    # S = torch.nn.utils.rnn.pad_sequence(S_batch).permute(1,0,2)
    # Q = torch.nn.utils.rnn.pad_sequence(Q_batch).permute(1,0,2)
    # S_mask = (S != 0)
    # Q_mask = (Q != 0)
    neg_loss,pos_loss,prec1=train(c3d_feat_S,lns_S,c3d_feat_Q,lns_Q,target_ones,target_zeros,uidx)
    moving_avg_loss.append([pos_loss.item(),neg_loss.item(),])
    if uidx%10==0:
        writer.add_scalar('Loss/Train_pos',
                      np.mean(moving_avg_loss,0)[0],
                      uidx)

        writer.add_scalar('Loss/Train_neg',
                      np.mean(moving_avg_loss,0)[1],
                      uidx)

        print('Upd: {0}| Loss Train pos/neg: {1} / {2}'.format(uidx, np.mean(moving_avg_loss,0)[0], np.mean(moving_avg_loss,0)[1]))
        moving_avg_loss = []
    if uidx%5000==0:
        mhe.save_model(uidx,opt,mdl_tarn,mm_opt)
writer.close()


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
parser.add_argument('--nupdate', type=int, default=10000000, help='Random seed')
parser.add_argument('--bsize', type=int, default=5, help='Batch Size')
parser.add_argument('--feats_gru_hidden_size', type=int, default=256, help='hidden size')
parser.add_argument('--dml_gru_hidden_size', type=int, default=256, help='hidden size')
parser.add_argument('--input_size', type=int, default=4096, help='hidden size')
parser.add_argument('--inp_s', type=int, default=4096, help='hidden size')
parser.add_argument('--nclass', type=int, default=5, help='Number of Class for Test')
parser.add_argument('--kshot', type=int, default=10, help='Number of samples')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.0005, help='W. Decay')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


dsL=dsLoader(iot.get_c3d_feats_hdf5())


def test(mdl_tarn,mm_opt,uidx):
    mdl_tarn.eval()
    aname_lst=dsL.kshot_class_set
    dic_results=[]
    for bidx in range(len(dsL.kshot_set_test)):
        c3d_feat_Q, anames_Q, lns_Q=dsL.get_test_samples('Nan',bidx,stream_mode=2)
        dic_score={}
        # print('Query Set',anames_Q)
        for aname in aname_lst:
            c3d_feat_S, anames_S, lns_S=dsL.get_test_samples(aname,0,stream_mode=0)
            q_kc = mdl_tarn(c3d_feat_Q, lns_Q, c3d_feat_S, lns_S)
            mean_score=q_kc.detach().cpu().numpy().mean()
            dic_score[aname]=mean_score
            # print('Support Set',aname,anames_S)
        prediction=sorted(dic_score.items(), key=lambda x: x[1], reverse=True)[0][0]
        dic_results.append(prediction==anames_Q[0])
        # print(prediction,anames_Q[0])
    return dic_results
# --------- training funtions ------------------------------------
def train(mdl_tarn,mm_opt,criterion,c3d_feat_Q,lns_Q,c3d_feat_S_pos,lns_S_pos,c3d_feat_S_neg,lns_S_neg,target_ones,target_zeros,uidx):
    mdl_tarn.train()
    mm_opt.zero_grad()
    q_kc_pos =mdl_tarn(c3d_feat_Q,lns_Q,c3d_feat_S_pos,lns_S_pos)
    q_kc_neg =mdl_tarn(c3d_feat_Q,lns_Q,c3d_feat_S_neg,lns_S_neg)
    pos_loss = criterion(q_kc_pos, target_ones)
    neg_loss = criterion(q_kc_neg, target_zeros)
    loss=neg_loss+pos_loss
    loss.backward()
    mm_opt.step()
    pos_acc=(q_kc_pos>0.5).cpu().numpy().mean()
    neg_acc=(q_kc_neg<=0.5).cpu().numpy().mean()
    return neg_loss,pos_loss,neg_acc,pos_acc


def finetune(mdl_tarn,mm_opt,uidx):
    criterion = nn.BCELoss()
    moving_avg_prec=[]
    moving_avg_loss=[]
    for fuidx in range(2000):
        c3d_feat_Q, anames_Q, lns_Q, Q_kys, c3d_feat_S_pos, anames_S_pos, lns_S_pos, c3d_feat_S_neg, anames_S_neg, lns_S_neg = dsL.get_batch(
            opt.bsize, fuidx, stream_mode=2)
        neg_loss, pos_loss, neg_acc, pos_acc = train(mdl_tarn,mm_opt,criterion,c3d_feat_Q, lns_Q, c3d_feat_S_pos, lns_S_pos, c3d_feat_S_neg,
                                                     lns_S_neg, target_ones, target_zeros, fuidx)
        moving_avg_loss.append([pos_loss.item(), neg_loss.item()])
        moving_avg_prec.append([pos_acc, neg_acc])

    return moving_avg_loss,moving_avg_prec



def eval_model(mpath, uidx):
    nrepeat=10
    results=[]
    for nrep_idx in range(nrepeat):
        _, mdl_tarn, mm_opt = mhe.load_model(mpath)
        dsL.kshot_sample_set(kshot_seed=nrep_idx)
        moving_avg_loss,moving_avg_prec=finetune(mdl_tarn, mm_opt, uidx)
        dic_results=test(mdl_tarn, mm_opt, uidx)

        avg_loss=np.mean( moving_avg_loss,0)
        avg_prec=np.mean( moving_avg_prec,0)
        results.append(np.mean(dic_results))
        print('Mdl: {0}| Train Loss: {1:4f} / {2:4f}, acc: {3:4f} / {4:4f} '.format(uidx,avg_loss[0],avg_loss[1],avg_prec[0],avg_prec[1]),
              ' >>> Test Acc: {0:4f}'.format(results[-1]),
              ' Class: [{0}]'.format(' ,'.join(dsL.kshot_class_set)))
        print('Over all Test Acc: {0:4f}'.format(np.mean(results)))

def Run():
    moving_avg_loss = []
    moving_avg_prec = []
    mdl_tarn=m_tarn(opt.input_size,opt.feats_gru_hidden_size,opt.dml_gru_hidden_size).cuda()
    mm_opt=optim.Adam(mdl_tarn.parameters(), lr=opt.lr, weight_decay=opt.wd)
    criterion = nn.BCELoss()
    for uidx in range(opt.nupdate):
        c3d_feat_Q,anames_Q,lns_Q,Q_kys,c3d_feat_S_pos,anames_S_pos,lns_S_pos,c3d_feat_S_neg,anames_S_neg,lns_S_neg= dsL.get_batch(opt.bsize,uidx,stream_mode=0)
        neg_loss,pos_loss,neg_acc,pos_acc=train(mdl_tarn,mm_opt,criterion,c3d_feat_Q,lns_Q,c3d_feat_S_pos,lns_S_pos,c3d_feat_S_neg,lns_S_neg,target_ones,target_zeros,uidx)
        moving_avg_loss.append([pos_loss.item(),neg_loss.item()])
        moving_avg_prec.append([pos_acc,neg_acc])
        if uidx%50000==0:
            writer.add_scalar('Loss/Train_pos',
                          np.mean(moving_avg_loss,0)[0],
                          uidx)

            writer.add_scalar('Loss/Train_neg',
                          np.mean(moving_avg_loss,0)[1],
                          uidx)

            writer.add_scalar('Loss/Train_pos_acc',
                          np.mean(moving_avg_prec,0)[0],
                          uidx)
            writer.add_scalar('Loss/Train_neg_acc',
                          np.mean(moving_avg_prec,0)[1],
                          uidx)
            print('Upd: {0}| Loss Train pos/neg  Loss: {1:4f} / {2:4f}, acc: {3:4f} / {4:4f} '.format(uidx, np.mean(moving_avg_loss,0)[0], np.mean(moving_avg_loss,0)[1],
                                                                                    np.mean(moving_avg_prec,0)[0],np.mean(moving_avg_prec,0)[1]))
            moving_avg_loss = []
            moving_avg_prec = []
        if uidx % 100000 == 0:
            #We should save and reaload model.
            mpath=mhe.save_model(uidx, opt, mdl_tarn, mm_opt,show_txt=False)
            eval_model(mpath, uidx)
            _, mdl_tarn, mm_opt=mhe.load_model(mpath)


writer = SummaryWriter(iot.get_wd()+'/runs/mdl')
writer.add_text('DataSet', 'Training Samples:{0}'.format(len(dsL.train_samples)))
writer.add_text('DataSet', 'Test Samples:{0}'.format(len(dsL.test_samples)))
print('Training Stated....')

target_zeros = torch.zeros((opt.bsize, 1)).cuda()
target_ones = torch.ones((opt.bsize, 1)).cuda()
Run()
writer.close()


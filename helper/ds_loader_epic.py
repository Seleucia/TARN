import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import numpy as np
import h5py
import pickle
import torch
import helper.io_utils as iot
import random
from helper.io_utils import get_data_folder
class dsLoader():
    def __init__(self,hdf_feats_file,kshot=5,nclass=5,kshot_seed=1986,device='cuda'):
        # self.test_subject=['S01', 'S10', 'S16', 'S23', 'S30']
        print('Loading file: {0}'.format(hdf_feats_file))
        self.test_subject = ['P01', 'P10', 'P16', 'P23', 'P30']
        self.train_subject = ['P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P12', 'P13', 'P14', 'P15', 'P17',
                                 'P19', 'P20', 'P21', 'P22', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P31']

        # self.test_subject = ['P01', 'P10', 'P16', 'P23', 'P30','P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P12', 'P13', 'P14', 'P15', 'P17',
        #                          'P19', 'P20', 'P21', 'P22', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P31']
        # self.train_subject = ['P01', 'P10', 'P16', 'P23', 'P30','P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P12', 'P13', 'P14', 'P15', 'P17',
        #                          'P19', 'P20', 'P21', 'P22', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P31']

        self.train_classes = ['close', 'cut', 'mix', 'move', 'open', 'pour', 'put', 'remove', 'take', 'throw',
                              'turn-on', 'wash']
        self.test_classes = ['adjust', 'check', 'dry', 'empty', 'fill', 'flip', 'insert', 'peel', 'press', 'scoop',
                             'shake', 'squeeze', 'turn', 'turn-off']

        self.device=device


        self.kshot_seed=kshot_seed
        self.kshot=kshot
        self.nclass=nclass
        self.train_samples={}
        self.test_samples={}
        self.train_action_samples={}
        self.test_action_samples={}
        self.load_hdf(hdf_feats_file)
        print('Done...')

    def load_hdf(self,hdf_prot_file):
        with h5py.File(hdf_prot_file, "r") as f:
            # List all groups
            seq_lst = list(f.keys())
            for seq_ky in seq_lst:
                subj = seq_ky.split('_')[0]
                c3d_vectors = f[seq_ky]['c3d_features'].value
                if len(c3d_vectors.shape)==1:
                    c3d_vectors=np.expand_dims(c3d_vectors,0)
                aname = f[seq_ky]['aname'].value.split('_')[1]
                if aname in self.train_classes and subj in self.train_subject:
                    self.train_samples[seq_ky] = [c3d_vectors, aname]
                    if aname not in self.train_action_samples:
                        self.train_action_samples[aname] = []
                    self.train_action_samples[aname].append(seq_ky)
                else:
                    if aname in self.test_classes and subj in self.test_subject:
                        self.test_samples[seq_ky] = [c3d_vectors, aname]
                        if aname not in self.test_action_samples:
                            self.test_action_samples[aname] = []
                        self.test_action_samples[aname].append(seq_ky)


        aname_lst=list(self.test_action_samples.keys())

        for aname in aname_lst:
            if len(self.test_action_samples[aname]) <self.kshot+1:
                print('We are deleting {0} due to insufficent number of samples: {1}, kshot: {2}'.format(aname,len(self.test_action_samples[aname]),self.kshot))
                tmp_seq_ky_lst = self.test_action_samples[aname]
                for seq_ky in tmp_seq_ky_lst:
                    del self.test_samples[seq_ky]

                self.test_classes.remove(aname)
                del self.test_action_samples[aname]

        self.test_aname_cnt={aname:len(self.test_action_samples[aname]) for aname in self.test_action_samples}
        print([len(self.test_action_samples[aname]) for aname in self.test_action_samples])

    def kshot_sample_set(self,kshot_seed):
        random.seed(kshot_seed)
        self.kshot_set_train = {}
        self.kshot_set_test = {}

        self.kshot_class_set=random.sample(self.test_classes,self.nclass)
        self.kshot_train_action_samples={}
        self.kshot_test_action_samples={}
        for aname in self.kshot_class_set:
            full_lst=  self.test_action_samples[aname]
            random.shuffle(full_lst)
            self.kshot_train_action_samples[aname]=full_lst[:self.kshot]
            self.kshot_test_action_samples[aname]=full_lst[self.kshot:]

            for seq_ky in self.kshot_test_action_samples[aname]:
                self.kshot_set_test[seq_ky]=self.test_samples[seq_ky]

            for seq_ky in self.kshot_train_action_samples[aname]:
                self.kshot_set_train[seq_ky]=self.test_samples[seq_ky]

    def get_niter(self,bsize,stream_mode=1):
        if stream_mode == 0:
           niter=-1
        elif stream_mode == 1:  # data stream for training
            sample_set = len(self.kshot_set_train)
            niter =  int(sample_set/bsize)+1
        elif stream_mode == 2:  # data stream for training
            sample_set =  len(self.kshot_set_test)
            niter =  int(sample_set/bsize)+1
        return niter

    def get_by_kylst(self,kys,sample_set):
        c3d_feat_lst = []
        aname_lst = []
        ln_lst = []
        for ky in kys:
            c3d_feat, aname = sample_set[ky]
            # c3d_feat = c3d_feat.swapaxes(0, 1)
            c3d_feat_lst.append(torch.from_numpy(np.asarray(c3d_feat)))
            aname_lst.append(aname)
            ln_lst.append(c3d_feat.shape[0])
        c3d_feat_lst = torch.nn.utils.rnn.pad_sequence(c3d_feat_lst).permute(1, 0, 2).to(self.device)
        return c3d_feat_lst,aname_lst,ln_lst


    def get_test_samples(self,aname,bidx,stream_mode=0):
        nsample_per_batch = self.kshot
        sample_set = self.test_samples
        if stream_mode==0: #return support set
            S_kys = self.kshot_train_action_samples[aname]
            c3d_feat_S, anames_S, lns_S = self.get_by_kylst(S_kys, sample_set)
            return c3d_feat_S, anames_S, lns_S
        elif stream_mode==1: #return support set
            tmp_Q_kys=self.kshot_test_action_samples[aname]
            Q_kys=tmp_Q_kys[bidx*nsample_per_batch:(bidx+1)*nsample_per_batch]
            if len(Q_kys) <nsample_per_batch:
                nmissing = nsample_per_batch-len(Q_kys)
                Q_kys.extend(random.sample(tmp_Q_kys,nmissing))
            c3d_feat_Q, anames_Q, lns_Q = self.get_by_kylst(Q_kys, sample_set)
            return c3d_feat_Q, anames_Q, lns_Q
        elif stream_mode == 2:  # return support set
            tmp_Q_kys=list(self.kshot_set_test.keys())
            Q_kys = tmp_Q_kys[bidx]
            Q_kys=[Q_kys]*self.kshot
            c3d_feat_Q, anames_Q, lns_Q = self.get_by_kylst(Q_kys, sample_set)
            return c3d_feat_Q, anames_Q, lns_Q

    def get_batch(self,bsize,uidx,stream_mode=0):
        random.seed(uidx)
        nsample_per_batch=self.kshot
        nclass_to_be_used=2
        if stream_mode==0:
            S_pos_kys = []
            S_neg_kys = []
            Q_kys = []
            sel_alist=random.sample(self.train_classes, self.nclass+1)
            poss_aname=sel_alist[0] #First one positive samples.
            kys_lst = self.train_action_samples[poss_aname]
            sel_kys = random.sample(kys_lst, nsample_per_batch * 2)
            S_pos_kys=sel_kys[:nsample_per_batch]
            Q_kys = sel_kys[nsample_per_batch:]
            #get negative ones
            for aidx,aname in enumerate(sel_alist[1:]):
                kys_lst=self.train_action_samples[aname]
                sel_kys = random.sample(kys_lst, 1)[0]
                S_neg_kys.append(sel_kys)
            sample_set=self.train_samples
        elif stream_mode==1: #data stream for training
            sample_set = self.test_samples
            batch_kys = list(sample_set.keys())[bsize * uidx:bsize * (uidx + 1)]
        elif stream_mode==2: #data stream for test
            sample_set = self.test_samples
            S_pos_kys = []
            S_neg_kys = []
            Q_kys = []
            sel_alist = self.kshot_class_set
            poss_aname = sel_alist[0]  # First one positive samples.
            kys_lst = self.kshot_train_action_samples[poss_aname]
            for kidx in range(self.kshot):
                sel_kys = random.sample(kys_lst, 2)
                S_pos_kys.append(sel_kys[0])
                Q_kys.append(sel_kys[1])

            for aidx,aname in enumerate(sel_alist[1:]):
                kys_lst=self.kshot_train_action_samples[aname]
                sel_kys = random.sample(kys_lst, 1)[0]
                S_neg_kys.append(sel_kys)
            if len(S_neg_kys)<self.kshot:
               nmissing=self.kshot-len(S_neg_kys)
               for mis_idx in range(nmissing):
                   aname=random.sample(sel_alist[1:],1)[0]
                   kys_lst = self.kshot_train_action_samples[aname]
                   sel_kys = random.sample(kys_lst, 1)[0]
                   S_neg_kys.append(sel_kys)



        c3d_feat_Q,anames_Q,lns_Q=self.get_by_kylst(Q_kys, sample_set)
        c3d_feat_S_pos,anames_S_pos,lns_S_pos=self.get_by_kylst(S_pos_kys, sample_set)
        c3d_feat_S_neg,anames_S_neg,lns_S_neg=self.get_by_kylst(S_neg_kys, sample_set)


        return c3d_feat_Q,anames_Q,lns_Q,Q_kys,c3d_feat_S_pos,anames_S_pos,lns_S_pos,c3d_feat_S_neg,anames_S_neg,lns_S_neg




import torch
import torch.nn as nn

class TARN(nn.Module):
    def __init__(self, input_size=4096,feats_gru_hidden_size=256,dml_gru_hidden_size=256,sim_type='EucCos'):
        super(TARN, self).__init__()
        self.feats_n_layers = 1
        self.sim_type=sim_type
        self.feats_gru = nn.GRU(input_size, feats_gru_hidden_size, self.feats_n_layers, batch_first=True,bidirectional=True)
        if self.sim_type in ['Mult','Subt']:
            self.d = feats_gru_hidden_size*2
        elif self.sim_type=='EucCos':
            self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            self.d=2

        self.W = torch.nn.Linear(feats_gru_hidden_size*2, feats_gru_hidden_size*2, bias=False)
        self.bias = nn.Parameter(torch.ones(feats_gru_hidden_size*2))

        self.dml_n_layers=1
        self.dml_gru = nn.GRU(self.d, dml_gru_hidden_size, self.dml_n_layers, batch_first=True)
        self.dml_fc = nn.Linear(dml_gru_hidden_size, 1)
        self.sft = nn.Softmax(dim=-1)

    def embed_gru(self,feats,feats_ln):
        F = self.feats_gru(feats)[0]
        F = [F[sidx, :s_ln, :] for sidx, s_ln in enumerate(feats_ln)]
        F_m = [f.unsqueeze(0) for f in F]
        return F_m,F

    def align_and_computelogits(self,Q_m,S_m):
        SQ = zip(S_m, Q_m)
        A_tmp_lst = []
        H_lst = []
        for S, Q in SQ:
            A_nomax = torch.matmul(self.W(S).add(self.bias.repeat(S.shape[1], 1)), torch.transpose(Q, -2, -1))
            A = self.sft(A_nomax)
            H = torch.matmul(torch.transpose(A, -2, -1), S)
            A_tmp_lst.append(A)
            H_lst.append(H)

        QH = zip(Q_m, H_lst)
        diff_QH_lst = []
        for q, h in QH:
            if self.sim_type == 'Mult':
                diff_qh = q * h
            elif self.sim_type == 'Subt':
                diff_qh = q - h
            elif self.sim_type == 'EucCos':
                diff_QH_c = self.cos(q, h)
                diff_QH_e = torch.norm(q - h, dim=2)
                diff_qh = torch.stack((diff_QH_c, diff_QH_e), -1)
            diff_QH_lst.append(diff_qh.squeeze(0))
            # print(diff_qh.shape,diff_qh.squeeze(0).shape,len(diff_QH_lst))
        diff_QH = torch.nn.utils.rnn.pad_sequence(diff_QH_lst).permute(1, 0, 2)
        q_kc = self.dml_fc(self.dml_gru(diff_QH)[1].squeeze())
        q_kc = torch.sigmoid(q_kc)
        return q_kc

    def forward(self, c3d_feat_Q,lns_Q,c3d_feat_S_pos,lns_S_pos,c3d_feat_S_neg,lns_S_neg,target_ones,target_zeros,uidx):
        Q_m,_ = self.embed_gru(c3d_feat_Q, lns_Q)
        S_pos_m,_ = self.embed_gru(c3d_feat_S_pos, lns_S_pos)
        S_neg_m,_ = self.embed_gru(c3d_feat_S_neg, lns_S_neg)
        q_kc_neg=self.align_and_computelogits(Q_m,S_pos_m)
        q_kc_pos=self.align_and_computelogits(Q_m,S_neg_m)

        return q_kc_pos,q_kc_neg

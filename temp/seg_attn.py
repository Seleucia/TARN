import torch
import torch.nn as nn

class TARN(nn.Module):
    def __init__(self, N=10, input_size=4096,feats_gru_hidden_size=256,dml_gru_hidden_size=256):
        super(TARN, self).__init__()
        self.feats_n_layers = 1
        self.feats_gru = nn.GRU(input_size, feats_gru_hidden_size, self.feats_n_layers, batch_first=True,bidirectional=True)

        self.d = feats_gru_hidden_size*2
        self.N = N
        self.W = torch.nn.Linear(self.d, self.d, bias=False)
        self.bias = nn.Parameter(torch.ones(self.d))

        self.dml_n_layers=1
        self.dml_gru = nn.GRU(self.d, dml_gru_hidden_size, self.dml_n_layers, batch_first=True)
        self.dml_fc = nn.Linear(dml_gru_hidden_size, 1)
        self.sft = nn.Softmax(dim=-1)

    def forward(self, feats_S, feats_Q):
        S=self.feats_gru(feats_S)[0]
        Q=self.feats_gru(feats_Q)[0]
        A_nomax = torch.matmul(self.W(S).add(self.bias.repeat(self.N, 1)), torch.transpose(Q, -2, -1))
        A = self.sft(A_nomax)
        H = torch.matmul(torch.transpose(A, -2, -1), S)
        diff_QH=Q-H
        q_kc = self.dml_fc(self.dml_gru(diff_QH)[1].squeeze())
        return A,q_kc



def CrossEntropy(yHat, y):
    if y == 1:
      return -torch.log(yHat)
    else:
      return -torch.log(1 - yHat)


kshot=5
nclass=5

bs = 10;
N = 11;
M = 13;
input_size=4096
feats_gru_hidden_size=256
dml_gru_hidden_size=256

S = torch.randn(bs, N, input_size)
Q = torch.randn(bs, M, input_size)

mdl_tarn=TARN(N,input_size,feats_gru_hidden_size,dml_gru_hidden_size)



for k in range(kshot):
    for nc in range(nclass):
        A,q_kc=mdl_tarn(S,Q)
        bc=CrossEntropy(q_kc, y=1)

print('q_kc', q_kc.shape)
print('A', A.shape)





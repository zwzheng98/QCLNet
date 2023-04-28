import torch.nn as nn
import torch

class Q2RM(nn.Module):
    def __init__(self, features, M):
        super(Q2RM, self).__init__()
        self.M = M
        self.features = features
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        bsz, qch, ha, wa = x.size()
        x =  x.view(bsz, 2, 2, int(qch/4), ha, wa).permute(0, 3, 4, 5, 1, 2).contiguous()
        bsz, ch, ha, wa, hb, wb = x.size()
        fea_U = x.view(bsz, ch, ha*wa, -1).mean(dim=2)
        feas = x.view(bsz, ch, ha, wa, -1)

        attention_vectors = self.softmax(fea_U)

        attention_vectors = attention_vectors.unsqueeze(2).unsqueeze(2)
        fea_v = (feas * attention_vectors).sum(dim=4)
        return fea_v

import torch
import torch.nn as nn
import numpy as np
from models.attention import Seq_Transformer



class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.lsoftmax = nn.LogSoftmax(1)
        self.device = device
        self.Wk = nn.ModuleList([nn.Linear(400, self.num_channels) for i in
                                 range(configs.features_len)])
        self.projection_head = nn.Sequential(
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Dropout(configs.dropout)
        )

        self.seq_transformer = Seq_Transformer(patch_size=128, dim=400, depth=1,
                                               heads=8, mlp_dim=200)

    def forward(self, features_aug1, features_aug2, features_aug3, tag=0):
        if tag == 0:
            z_aug1 = features_aug1
            z_aug1 = z_aug1.transpose(1, 2)

            z_aug2 = features_aug2
            seq_len = z_aug2.shape[2]
            z_aug2 = z_aug2.transpose(1, 2)

            z_aug3 = features_aug3
            # seq_len = z_aug3.shape[2]
            z_aug3 = z_aug3.transpose(1, 2)

            batch = z_aug1.shape[0]

            nce = 0  # average over timestep and batch
            encode_samples3 = torch.empty((seq_len, batch, self.num_channels)).float().to(self.device)

            for i in np.arange(0, seq_len):
                encode_samples3[i] = z_aug3[:, i, :].view(batch, self.num_channels)

            c_t = self.seq_transformer(z_aug2)

            pred = torch.empty((seq_len, batch, self.num_channels)).float().to(self.device)
            for i in np.arange(0, seq_len):
                linear = self.Wk[i]
                pred[i] = linear(c_t)

            for i in np.arange(0, seq_len):
                total = torch.mm(encode_samples3[i], torch.transpose(pred[i], 0, 1))
                nce += torch.sum(torch.diag(self.lsoftmax(total)))
            nce /= -1. * batch * seq_len

            return nce

        else:
            z = torch.cat([features_aug1, features_aug2, features_aug3], dim=-1).transpose(1, 2)
            c_t = self.seq_transformer(z)
            yt = self.projection_head(c_t)

            return self.lsoftmax(yt)
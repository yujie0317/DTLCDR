# %%
import torch.nn as nn
import torch
from torch.nn.modules import padding
from torch.nn.modules.conv import Conv1d
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from dgllife.model.gnn.gcn import GCN

class Conv1dReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,  stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.inc(x)

class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.inc(x)

class GCNReLU(nn.Module):
    def __init__(self, in_feats, hidden_feats,activation=None):
        super(GCNReLU, self).__init__()
        self.gnn = GCN(in_feats,hidden_feats,gnn_norm=None,activation=None,residual=None,batchnorm=None,dropout=None,allow_zero_in_degree=None)
        self.output_feats=hidden_feats[-1]
    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        batch_size = batch_graph.batch_size
        node_feats = self.gnn(batch_graph, node_feats)
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

# Inspiration: https://github.com/guaguabujianle/ML-DTI/blob/dev/network/ML_DTI.py
class BilinearPooling(nn.Module):
    def __init__(self, in_channels, out_channels, c_m, c_n):
        super().__init__()

        self.convA = nn.Conv1d(in_channels, c_m, kernel_size=1, stride=1, padding=0)
        self.convB = nn.Conv1d(in_channels, c_n, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(c_m, out_channels, bias=True)

    def forward(self, x):     
        A = self.convA(x) 
        B = self.convB(x)
        att_maps = F.softmax(B, dim=-1)
        global_descriptors = torch.bmm(A, att_maps.permute(0, 2, 1))
        global_descriptor = torch.mean(global_descriptors, dim=-1)
        out = self.linear(global_descriptor).unsqueeze(1)
        return out

class MutualAttentation(nn.Module):
    def __init__(self, in_channels, att_size, c_m, c_n):
        super().__init__()
        self.bipool = BilinearPooling(in_channels, in_channels, c_m, c_n)
        self.linearS = nn.Linear(in_channels, att_size)
        self.linearT = nn.Linear(in_channels, att_size)
    
    def forward(self, source, target):
        global_descriptor = self.bipool(source)
        target_org = target
        target = self.linearT(target.permute(0, 2, 1)).permute(0, 2, 1)
        global_descriptor = self.linearS(global_descriptor)
        att_maps = torch.bmm(global_descriptor, target)
        att_maps = torch.sigmoid(att_maps)
        out_target = torch.add(target_org, torch.mul(target_org, att_maps))
        out_target = F.relu(out_target)

        return out_target


class GCADTI(nn.Module):
    def __init__(self):
        super().__init__()
        self.prot_embed = nn.Embedding(26, 128, padding_idx=0)
        self.prot_conv1 = Conv1dReLU(128, 128, 3)
        self.bn1 = nn.BatchNorm1d(128)
        self.prot_conv2 = Conv1dReLU(128, 128, 6)
        self.bn2 = nn.BatchNorm1d(128)
        self.prot_conv3 = Conv1dReLU(128, 128, 9)
        self.bn3 = nn.BatchNorm1d(128)
        self.prot_pool = nn.AdaptiveMaxPool1d(1)

        self.drug_gcn = GCNReLU(75, [128, 128, 128])
        self.drug_pool = nn.AdaptiveMaxPool1d(1)

        self.prot_mut_att = MutualAttentation(128, 128, 128, 32)
        self.drug_mut_att = MutualAttentation(128, 128, 128, 32)

        self.linear1 = LinearReLU(256, 1024)
        self.drop1 = nn.Dropout(0.3)
        self.linear2 = LinearReLU(1024, 1024)
        self.drop2 = nn.Dropout(0.3)
        self.linear3 = LinearReLU(1024, 512)
        self.drop3 = nn.Dropout(0.3)
        self.out_layer = nn.Linear(512, 1)
    def forward(self, prot_x, drug_x):
        prot_x = self.prot_embed(prot_x).permute(0, 2, 1)
        prot_x = self.prot_conv1(prot_x)
        prot_x = self.bn1(prot_x)
        prot_x = self.prot_conv2(prot_x)
        prot_x = self.bn2(prot_x)
        prot_x = self.prot_conv3(prot_x)
        prot_x = self.bn3(prot_x)
        drug_x = self.drug_gcn(drug_x)
        prot_x_g = self.prot_mut_att(drug_x.permute(0, 2, 1), prot_x)
        drug_x_g = self.drug_mut_att(prot_x, drug_x.permute(0, 2, 1))
        prot_x = self.prot_pool(prot_x_g).squeeze(-1)
        drug_x = self.drug_pool(drug_x_g).squeeze(-1)
        x_cat = torch.cat([prot_x, drug_x], dim=-1)
        x = self.linear1(x_cat)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = self.linear3(x)
        x = self.drop3(x)
        x = self.out_layer(x)

        return prot_x,x

# %%

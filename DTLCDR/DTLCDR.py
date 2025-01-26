import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool,GCNConv,GATConv,GATv2Conv,GraphNorm
from performer import PerformerLM

class GINEncoder(torch.nn.Module):

    def __init__(self): 
        super().__init__()

        self.num_gc_layers = 5

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(self.num_gc_layers):
            if i == 0:
                block = nn.Sequential(nn.Linear(77, 128), nn.ReLU(),
                                nn.Linear(128, 128))
            else:
                block = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(128)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, drug):
        xs = []
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_max_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        xs = torch.cat(xs, 1)
        return x
    
class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU(),nn.Dropout(0.3)
        )

    def forward(self, x):
        
        return self.inc(x)   

    
class ConvPoolerShort(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, (1, dim))
    def forward(self, h):
        h = h[:,None,:,:]
        h = self.conv(h)
        h = h.view(h.shape[0], -1)
        return h

    
class DTLCDR(nn.Module):
    def __init__(self, drug_dti_dim, max_seq_len,gene2vec_path,exp_dim):
        super().__init__()
        self.drug_dti_dim = drug_dti_dim
        self.dropout = 0.2
        self.max_seq_len = max_seq_len
        self.gene2vec_path = gene2vec_path
        self.exp_dim=exp_dim
        self.cell_encoder_config = {'num_tokens': 7,
                                       'dim': 200,
                                       'depth': 6,
                                       'max_seq_len': self.max_seq_len,
                                       'heads': 10,
                                       'local_attn_heads': 0,
                                       'g2v_position_emb': True,
                                       'gene2vec_path': self.gene2vec_path,
                                       'ckpt_path': './panglao_pretrain.pth',
                                       'param_key': 'model_state_dict'}

    def _build(self):
        #drug graph 
        self.GNN_drug = GINEncoder()
        self.drug_emb = nn.Sequential(
            nn.Linear(128 * 5, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout))
        #drug dti
        self.drug_dti_linear1 = LinearReLU(self.drug_dti_dim,1024)
        self.drug_dti_linear2 = LinearReLU(1024,512)
        self.drug_dti_linear3 = LinearReLU(512,256)
        #drug descriptors
        self.drug_desc_linear1 = LinearReLU(196,1024)
        self.drug_desc_linear2 = LinearReLU(1024,512)
        self.drug_desc_linear3 = LinearReLU(512,256)
        # cell encoder
        self.cell_encoder = PerformerLM(**self.cell_encoder_config)
        ckpt = torch.load(self.cell_encoder_config["ckpt_path"])
        if self.cell_encoder_config["param_key"] != "":
            ckpt = ckpt[self.cell_encoder_config["param_key"]]
        self.cell_encoder.load_state_dict(ckpt)
        self.cell_encoder.to_out = ConvPoolerShort(self.cell_encoder_config["dim"])
        cell_encode_dim = self.cell_encoder_config["max_seq_len"]
        self.cell_emb = nn.Sequential(
            nn.Linear(cell_encode_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        #cell expression mlp
        self.cell_linear1 = LinearReLU(self.exp_dim,1024)
        self.cell_linear2 = LinearReLU(1024,256)
        self.cell_linear3 = LinearReLU(256,128)

        self.regression = nn.Sequential(
            nn.Linear(256*4+128, 1024),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 1)
        )

    def forward(self, drug_graph, drug_dti, drug_desc, cell_enc, cell_exp):

        drug_graph = self.GNN_drug(drug_graph)
        drug_graph = self.drug_emb(drug_graph)
        
        drug_dti = self.drug_dti_linear1(drug_dti)
        drug_dti = self.drug_dti_linear2(drug_dti)
        drug_dti = self.drug_dti_linear3(drug_dti)
        
        drug_desc = self.drug_desc_linear1(drug_desc)
        drug_desc = self.drug_desc_linear2(drug_desc)
        drug_desc = self.drug_desc_linear3(drug_desc)

        cell_enc = self.cell_encoder(cell_enc)
        cell_enc = self.cell_emb(cell_enc)
        
        cell_exp = self.cell_linear1(cell_exp)
        cell_exp = self.cell_linear2(cell_exp)
        cell_exp = self.cell_linear3(cell_exp)
        
        x = torch.cat([drug_graph, drug_dti, drug_desc, cell_enc, cell_exp], -1)
        x = self.regression(x)

        return x
    
class DTLCDR_cellenc(nn.Module):
    def __init__(self, drug_dti_dim, max_seq_len,gene2vec_path,exp_dim):
        super().__init__()
        self.drug_dti_dim = drug_dti_dim
        self.dropout = 0.2
        self.max_seq_len = max_seq_len
        self.gene2vec_path = gene2vec_path
        self.exp_dim=exp_dim
        self.cell_encoder_config = {'num_tokens': 7,
                                       'dim': 200,
                                       'depth': 6,
                                       'max_seq_len': self.max_seq_len,
                                       'heads': 10,
                                       'local_attn_heads': 0,
                                       'g2v_position_emb': True,
                                       'gene2vec_path': self.gene2vec_path,
                                       'ckpt_path': './panglao_pretrain.pth',
                                       'param_key': 'model_state_dict'}

    def _build(self):
        #drug graph 
        self.GNN_drug = GINEncoder()
        self.drug_emb = nn.Sequential(
            nn.Linear(128 * 5, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout))
        #drug dti
        self.drug_dti_linear1 = LinearReLU(self.drug_dti_dim,1024)
        self.drug_dti_linear2 = LinearReLU(1024,512)
        self.drug_dti_linear3 = LinearReLU(512,256)
        #drug descriptors
        self.drug_desc_linear1 = LinearReLU(196,1024)
        self.drug_desc_linear2 = LinearReLU(1024,512)
        self.drug_desc_linear3 = LinearReLU(512,256)
        # cell encoder
        self.cell_encoder = PerformerLM(**self.cell_encoder_config)
        ckpt = torch.load(self.cell_encoder_config["ckpt_path"])
        if self.cell_encoder_config["param_key"] != "":
            ckpt = ckpt[self.cell_encoder_config["param_key"]]
        self.cell_encoder.load_state_dict(ckpt)
        self.cell_encoder.to_out = ConvPoolerShort(self.cell_encoder_config["dim"])
        cell_encode_dim = self.cell_encoder_config["max_seq_len"]
        self.cell_emb = nn.Sequential(
            nn.Linear(cell_encode_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

        self.regression = nn.Sequential(
            nn.Linear(256*4, 1024),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 1)
        )

    def forward(self, drug_graph, drug_dti, drug_desc, cell_enc, cell_exp):

        drug_graph = self.GNN_drug(drug_graph)
        drug_graph = self.drug_emb(drug_graph)
        
        drug_dti = self.drug_dti_linear1(drug_dti)
        drug_dti = self.drug_dti_linear2(drug_dti)
        drug_dti = self.drug_dti_linear3(drug_dti)
        
        drug_desc = self.drug_desc_linear1(drug_desc)
        drug_desc = self.drug_desc_linear2(drug_desc)
        drug_desc = self.drug_desc_linear3(drug_desc)

        cell_enc = self.cell_encoder(cell_enc)
        cell_enc = self.cell_emb(cell_enc)
        
        x = torch.cat([drug_graph, drug_dti, drug_desc, cell_enc], -1)
        x = self.regression(x)

        return x
    
class DTLCDR_cellexp(nn.Module):
    def __init__(self, drug_dti_dim, max_seq_len,gene2vec_path,exp_dim):
        super().__init__()
        self.drug_dti_dim = drug_dti_dim
        self.dropout = 0.2
        self.exp_dim=exp_dim
    def _build(self):
        #drug graph 
        self.GNN_drug = GINEncoder()
        self.drug_emb = nn.Sequential(
            nn.Linear(128 * 5, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout))
        #drug dti
        self.drug_dti_linear1 = LinearReLU(self.drug_dti_dim,1024)
        self.drug_dti_linear2 = LinearReLU(1024,512)
        self.drug_dti_linear3 = LinearReLU(512,256)
        #drug descriptors
        self.drug_desc_linear1 = LinearReLU(196,1024)
        self.drug_desc_linear2 = LinearReLU(1024,512)
        self.drug_desc_linear3 = LinearReLU(512,256)
        #cell expression mlp
        self.cell_linear1 = LinearReLU(self.exp_dim,1024)
        self.cell_linear2 = LinearReLU(1024,256)
        self.cell_linear3 = LinearReLU(256,128)

        self.regression = nn.Sequential(
            nn.Linear(256*3+128, 1024),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 1)
        )

    def forward(self, drug_graph, drug_dti, drug_desc, cell_enc, cell_exp):

        drug_graph = self.GNN_drug(drug_graph)
        drug_graph = self.drug_emb(drug_graph)
        
        drug_dti = self.drug_dti_linear1(drug_dti)
        drug_dti = self.drug_dti_linear2(drug_dti)
        drug_dti = self.drug_dti_linear3(drug_dti)
        
        drug_desc = self.drug_desc_linear1(drug_desc)
        drug_desc = self.drug_desc_linear2(drug_desc)
        drug_desc = self.drug_desc_linear3(drug_desc)

        cell_exp = self.cell_linear1(cell_exp)
        cell_exp = self.cell_linear2(cell_exp)
        cell_exp = self.cell_linear3(cell_exp)
        
        x = torch.cat([ drug_dti, cell_exp, drug_graph, drug_desc], -1)
        x = self.regression(x)

        return  x

class DTLCDR_drugdti(nn.Module):
    def __init__(self, drug_dti_dim, max_seq_len,gene2vec_path,exp_dim):
        super().__init__()
        self.drug_dti_dim = drug_dti_dim
        self.dropout = 0.2
        self.max_seq_len = max_seq_len
        self.gene2vec_path = gene2vec_path
        self.exp_dim=exp_dim
        self.cell_encoder_config = {'num_tokens': 7,
                                       'dim': 200,
                                       'depth': 6,
                                       'max_seq_len': self.max_seq_len,
                                       'heads': 10,
                                       'local_attn_heads': 0,
                                       'g2v_position_emb': True,
                                       'gene2vec_path': self.gene2vec_path,
                                       'ckpt_path': './panglao_pretrain.pth',
                                       'param_key': 'model_state_dict'}

    def _build(self):
        #drug dti
        self.drug_dti_linear1 = LinearReLU(self.drug_dti_dim,1024)
        self.drug_dti_linear2 = LinearReLU(1024,512)
        self.drug_dti_linear3 = LinearReLU(512,256)
        # cell encoder
        self.cell_encoder = PerformerLM(**self.cell_encoder_config)
        ckpt = torch.load(self.cell_encoder_config["ckpt_path"])
        if self.cell_encoder_config["param_key"] != "":
            ckpt = ckpt[self.cell_encoder_config["param_key"]]
        self.cell_encoder.load_state_dict(ckpt)
        self.cell_encoder.to_out = ConvPoolerShort(self.cell_encoder_config["dim"])
        cell_encode_dim = self.cell_encoder_config["max_seq_len"]
        self.cell_emb = nn.Sequential(
            nn.Linear(cell_encode_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        #cell expression mlp
        self.cell_linear1 = LinearReLU(self.exp_dim,1024)
        self.cell_linear2 = LinearReLU(1024,256)
        self.cell_linear3 = LinearReLU(256,128)

        self.regression = nn.Sequential(
            nn.Linear(256*2+128, 1024),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 1)
        )

    def forward(self, drug_graph, drug_dti, drug_desc, cell_enc, cell_exp):
        
        drug_dti = self.drug_dti_linear1(drug_dti)
        drug_dti = self.drug_dti_linear2(drug_dti)
        drug_dti = self.drug_dti_linear3(drug_dti)

        cell_enc = self.cell_encoder(cell_enc)
        cell_enc = self.cell_emb(cell_enc)
        
        cell_exp = self.cell_linear1(cell_exp)
        cell_exp = self.cell_linear2(cell_exp)
        cell_exp = self.cell_linear3(cell_exp)
        
        x = torch.cat([drug_dti, cell_enc, cell_exp], -1)
        x = self.regression(x)

        return x
    
class DTLCDR_drugGIN(nn.Module):
    def __init__(self, drug_dti_dim, max_seq_len,gene2vec_path,exp_dim):
        super().__init__()
        self.drug_dti_dim = drug_dti_dim
        self.dropout = 0.2
        self.max_seq_len = max_seq_len
        self.gene2vec_path = gene2vec_path
        self.exp_dim=exp_dim
        self.cell_encoder_config = {'num_tokens': 7,
                                       'dim': 200,
                                       'depth': 6,
                                       'max_seq_len': self.max_seq_len,
                                       'heads': 10,
                                       'local_attn_heads': 0,
                                       'g2v_position_emb': True,
                                       'gene2vec_path': self.gene2vec_path,
                                       'ckpt_path': './panglao_pretrain.pth',
                                       'param_key': 'model_state_dict'}

    def _build(self):
        #drug graph 
        self.GNN_drug = GINEncoder()
        self.drug_emb = nn.Sequential(
            nn.Linear(128 * 5, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout))
        # cell encoder
        self.cell_encoder = PerformerLM(**self.cell_encoder_config)
        ckpt = torch.load(self.cell_encoder_config["ckpt_path"])
        if self.cell_encoder_config["param_key"] != "":
            ckpt = ckpt[self.cell_encoder_config["param_key"]]
        self.cell_encoder.load_state_dict(ckpt)
        self.cell_encoder.to_out = ConvPoolerShort(self.cell_encoder_config["dim"])
        cell_encode_dim = self.cell_encoder_config["max_seq_len"]
        self.cell_emb = nn.Sequential(
            nn.Linear(cell_encode_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        #cell expression mlp
        self.cell_linear1 = LinearReLU(3000,1024)
        self.cell_linear2 = LinearReLU(1024,256)
        self.cell_linear3 = LinearReLU(256,128)

        self.regression = nn.Sequential(
            nn.Linear(256*2+128, 1024),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 1)
        )

    def forward(self, drug_graph, drug_dti, drug_desc, cell_enc, cell_exp):

        drug_graph = self.GNN_drug(drug_graph)
        drug_graph = self.drug_emb(drug_graph)

        cell_enc = self.cell_encoder(cell_enc)
        cell_enc = self.cell_emb(cell_enc)
        
        cell_exp = self.cell_linear1(cell_exp)
        cell_exp = self.cell_linear2(cell_exp)
        cell_exp = self.cell_linear3(cell_exp)
        
        x = torch.cat([drug_graph, cell_enc, cell_exp], -1)
        x = self.regression(x)

        return x

class DTLCDR_drugdesc(nn.Module):
    def __init__(self, drug_dti_dim, max_seq_len,gene2vec_path,exp_dim):
        super().__init__()
        self.drug_dti_dim = drug_dti_dim
        self.dropout = 0.2
        self.max_seq_len = max_seq_len
        self.gene2vec_path = gene2vec_path
        self.exp_dim=exp_dim
        self.cell_encoder_config = {'num_tokens': 7,
                                       'dim': 200,
                                       'depth': 6,
                                       'max_seq_len': self.max_seq_len,
                                       'heads': 10,
                                       'local_attn_heads': 0,
                                       'g2v_position_emb': True,
                                       'gene2vec_path': self.gene2vec_path,
                                       'ckpt_path': './panglao_pretrain.pth',
                                       'param_key': 'model_state_dict'}

    def _build(self):
        #drug descriptors
        self.drug_desc_linear1 = LinearReLU(196,1024)
        self.drug_desc_linear2 = LinearReLU(1024,512)
        self.drug_desc_linear3 = LinearReLU(512,256)
        # cell encoder
        self.cell_encoder = PerformerLM(**self.cell_encoder_config)
        ckpt = torch.load(self.cell_encoder_config["ckpt_path"])
        if self.cell_encoder_config["param_key"] != "":
            ckpt = ckpt[self.cell_encoder_config["param_key"]]
        self.cell_encoder.load_state_dict(ckpt)
        self.cell_encoder.to_out = ConvPoolerShort(self.cell_encoder_config["dim"])
        cell_encode_dim = self.cell_encoder_config["max_seq_len"]
        self.cell_emb = nn.Sequential(
            nn.Linear(cell_encode_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        #cell expression mlp
        self.cell_linear1 = LinearReLU(3000,1024)
        self.cell_linear2 = LinearReLU(1024,256)
        self.cell_linear3 = LinearReLU(256,128)

        self.regression = nn.Sequential(
            nn.Linear(256*2+128, 1024),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 1)
        )

    def forward(self, drug_graph, drug_dti, drug_desc, cell_enc, cell_exp):
        
        drug_desc = self.drug_desc_linear1(drug_desc)
        drug_desc = self.drug_desc_linear2(drug_desc)
        drug_desc = self.drug_desc_linear3(drug_desc)

        cell_enc = self.cell_encoder(cell_enc)
        cell_enc = self.cell_emb(cell_enc)
        
        cell_exp = self.cell_linear1(cell_exp)
        cell_exp = self.cell_linear2(cell_exp)
        cell_exp = self.cell_linear3(cell_exp)
        
        x = torch.cat([drug_desc, cell_enc, cell_exp], -1)
        x = self.regression(x)

        return x    



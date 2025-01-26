# %%
import os
import sys, re, math, time
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch
import pandas as pd

import torch.utils.data as data
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import dgl

def graph_collate_func(x):
    d, p, y = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p),dtype=torch.long),torch.tensor(y,dtype=torch.float32)

VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target] 

class DTIDataset(data.Dataset):
    def __init__(self, df, max_drug_nodes=290):
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        smi = self.df.iloc[index]['smiles']
        v_d = self.fc(smiles=smi, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        v_p = self.df.iloc[index]['sequence']
        target = seqs2int(v_p)
        if len(target) < 1200:
            target = np.pad(target, (0, 1200 - len(target)))
        else:
            target = target[:1200]
        y = self.df.iloc[index]["label"]
        
        return v_d, target, y    


# %%


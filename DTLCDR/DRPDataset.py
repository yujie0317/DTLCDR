import rdkit.Chem as Chem
from rdkit.Chem import DataStructs, rdmolops
from rdkit.Chem import AllChem, Descriptors
from dgllife.utils import *
from torch_geometric.data import Data, Batch
from abc import ABC,abstractmethod
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

# Inspiration: https://github.com/violet-sto/TGSA/blob/master/smiles2graph.py
class smiles2graph():
    def __init__(self):
        super(smiles2graph, self).__init__()

    def atom_feature(self, atom):
        """
        Converts rdkit atom object to feature list of indices
        :param mol: rdkit atom object
        :return: list
        8 features are canonical, 2 features are from OGB
        """
        featurizer_funcs = [
            atom_type_one_hot,
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge,
            atom_num_radical_electrons,
            atom_hybridization_one_hot,
            atom_is_aromatic,
            atom_total_num_H_one_hot,
            atom_is_in_ring,
            atom_chirality_type_one_hot,
        ]
        atom_feature = np.concatenate([func(atom) for func in featurizer_funcs], axis=0)
        return atom_feature

    def bond_feature(self, bond):
        """
        Converts rdkit bond object to feature list of indices
        :param mol: rdkit bond object
        :return: list
        """
        featurizer_funcs = [bond_type_one_hot]
        bond_feature = np.concatenate([func(bond) for func in featurizer_funcs], axis=0)

        return bond_feature
    
    def __call__(self, data):
        mol = Chem.MolFromSmiles(data)
        """
        Converts SMILES string to graph Data object without remove salt
        :input: SMILES string (str)
        :return: pyg Data object
        """
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(self.atom_feature(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 4  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = self.bond_feature(bond)
                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            edge_index = np.array(edges_list, dtype=np.int64).T
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        )
        return graph
    
class DRPDataset(Dataset):
    def __init__(self, df, dti, desc, enc_num):
        self.df = df
        self.dti = dti
        self.desc = desc
        self.enc_num = enc_num
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        smi, label = data['smiles'], data['Label']
        drug_dti = np.array(self.dti.loc[smi])
        drug_desc = np.array(self.desc.loc[smi])
        feazer = smiles2graph()
        drug_graph = feazer(smi)
        cell_enc = np.array(data[2:2+self.enc_num]).astype(float)
        cell_exp = np.array(data[2+self.enc_num:]).astype(float)
        return drug_graph, torch.tensor(drug_dti, dtype=torch.float), torch.tensor(drug_desc, dtype=torch.float),torch.tensor(cell_enc, dtype=torch.long), torch.tensor(cell_exp, dtype=torch.float),torch.tensor(label, dtype=torch.float)

class BaseCollator(ABC):
    def __init__(self):
        super(BaseCollator, self).__init__()
    
    def __call__(self, data, **kwargs):
        raise NotImplementedError

    def _collate_single(self, data):
        if isinstance(data[0], Data):
            return Batch.from_data_list(data)
        elif torch.is_tensor(data[0]):
            return torch.stack([x.squeeze() for x in data])
        
class Collator(BaseCollator):
    def __init__(self):
        super(Collator, self).__init__()

    def __call__(self, x):
        batch = self._collate_single(x)
        return batch
    
class DRPCollator():
    def __init__(self):
        super(DRPCollator, self).__init__()
        self.Collator = Collator()

    def __call__(self, data):
        drug_graph, drug_dti, drug_desc, cell_enc, cell_exp, labels = map(list, zip(*data))
        return self.Collator(drug_graph), self.Collator(drug_dti), self.Collator(drug_desc), self.Collator(cell_enc), self.Collator(cell_exp), torch.tensor(labels)  
    


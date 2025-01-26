import numpy as np
import pandas as pd
import sys
import copy
import time
import pickle
import torch 
import os
import argparse
from torch.utils import data
import torch.nn.functional as F 
from torch.autograd import Variable
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold 
from prettytable import PrettyTable
from scipy.stats import pearsonr,spearmanr
from math import sqrt
from DRPDataset import *
from DTLCDR import *
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from rdkit.Chem import Descriptors
import rdkit
from rdkit.ML.Descriptors import MoleculeDescriptors

def arg_parse():
    parser = argparse.ArgumentParser(description="DTLCDR for CDR prediction")
    parser.add_argument('--model', type=str, default='DTLCDR', help='DTLCDR,DTLCDR_cellenc,DTLCDR_cellexp,DTLCDR_drugdti,DTLCDR_drugGIN,DTLCDR_drugdesc')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--split', type=str, default='warmstart', help='warmstart, cellcoldstart, drugcoldstart')
    parser.add_argument('--epoch', type=int, default=100, help='train_epoch')
    return parser.parse_args()

class process_cell():
    def __init__(self):

        self.cell_encoder = './process_data/exp_enc.csv'
        self.cell_expression = './process_data/exp.csv'
     
    def get_celldata(self, data):
        cellid = list(data['COSMIC_ID'])
        enc_data =  pd.read_csv(self.cell_encoder,index_col=0)
        exp_data =  pd.read_csv(self.cell_expression,index_col=0)
        encdata = enc_data.loc[cellid]
        expdata = exp_data.loc[cellid]

        return encdata, expdata
    
pred_dti_gdsc2=pd.read_csv('../GCADTI/pred_dti_gdsc2.csv',index_col=0)
pred_dti=[]
for i in range(len(pred_dti_gdsc2.smiles.unique())):
    pred_dti.append(pred_dti_gdsc2[i*1572:(i+1)*1572].label.tolist())
pred_dti=pd.DataFrame(pred_dti)
pred_dti.index=pred_dti_gdsc2.smiles.unique().tolist()

gene2vec_dim_200_iter_9=pd.read_csv('./gene2vec_dim_200_iter_9.txt',sep='\t| ',header=None)
gene2vec_dim_200_iter_9.index=gene2vec_dim_200_iter_9[0]
gene2vec_dim_200_iter_9=gene2vec_dim_200_iter_9.loc[:,1:]
gene=pd.read_csv('./process_data/exp_enc.csv',index_col=0)
col=gene.columns
np.save('gene2vec_595.npy',np.vstack([np.array(gene2vec_dim_200_iter_9.loc[col]),np.zeros([16906-595,200])]))

modeldict={'DTLCDR':DTLCDR,'DTLCDR_cellenc':DTLCDR_cellenc,'DTLCDR_cellexp':DTLCDR_cellexp,'DTLCDR_drugdti':DTLCDR_drugdti,\
          'DTLCDR_drugGIN':DTLCDR_drugGIN,'DTLCDR_drugdesc':DTLCDR_drugdesc}
class Model:
    def __init__(self,modeldir,model,kfold,device,epoch):
        self.model = modeldict[model](pred_dti.shape[1],596,'./gene2vec_595.npy',3000)
        self.model._build()
        self.device = torch.device(device)
        self.modeldir = modeldir
        self.kfold=kfold
        self.epoch=epoch
        self.record_fileval = os.path.join(self.modeldir, "valid_markdowntable.txt")
        self.record_filetest = os.path.join(self.modeldir, str(self.kfold)+'.txt')
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
        self.val_pkl_file = os.path.join(self.modeldir, "val_loss_curve_iter.pkl")
        self.val_loss_history = []
    def test(self,datagenerator,model,mode=None):
        y_label = []
        y_pred = []
        loss_s=0
        model.eval()
        
        for i, data in enumerate(datagenerator):
            
            drug_graph=data[0].to(self.device)
            drug_dti=data[1].to(self.device)
            drug_desc=data[2].to(self.device)
            cell_enc=data[3].to(self.device)
            cell_exp=data[4].to(self.device)
            label = data[5].to(self.device)

            score = model(drug_graph, drug_dti, drug_desc, cell_enc, cell_exp)
            
            loss_fct = torch.nn.MSELoss()
            score = torch.squeeze(score, 1)
            loss = loss_fct(score, label)
            self.val_loss_history.append(loss.item())
            #loss = loss_fct(score.squeeze(), label)
            logits =score.detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            loss_s = loss_s+loss
        loss_m = loss_s/(i+1)
        
        pcc = pearsonr(y_label, y_pred)[0]
        spm = spearmanr(y_label, y_pred)[0]
        mse = mean_squared_error(y_label, y_pred)
        r2 = r2_score(y_label, y_pred)
        mae = mean_absolute_error(y_label, y_pred)
        #cinx= concordance_index(y_label, y_pred)
        cinx=np.nan
        model.train()
        if mode == 'val':
            return loss_m,sqrt(mse),mse,pcc,spm,r2,mae,cinx
        elif mode == 'test':
            return loss_m,sqrt(mse),mse,pcc,spm,r2,mae,cinx
        elif mode == 'predict':
        
            return y_label, y_pred
        

    def train(self,trainset, valset, testset):
        lr = 1e-4
        decay = 0
        BATCH_SIZE = 64
        train_epoch = self.epoch
        self.model = self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 5])
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        loss_history = []
        collate_fn=DRPCollator()
        trainparams = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 4,
                  'drop_last': True,
                      'pin_memory':True,
                      "collate_fn":collate_fn}
        training_generator = DataLoader(trainset, **trainparams)
        valtestparams = {'batch_size': BATCH_SIZE,
                  'shuffle': False,
                  'num_workers': 4,
                  'drop_last': False,
                        'pin_memory':True,
                        "collate_fn":collate_fn}
        validation_generator = DataLoader(valset, **valtestparams)
        testing_generator = DataLoader(testset, **valtestparams)
        bestauroc = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ['# epoch',"loss",'rmse_val','mse_val','pcc_val','spm_val','r2_val','mae_val','cinx_val' ]
        test_metric_header = ['# epoch',"loss",'rmse_test','mse_test','pcc_test','spm_test','r2_test','mae_test','cinx_test' ]
        valtable = PrettyTable(valid_metric_header)
        testtable = PrettyTable(test_metric_header)
        float2str = lambda x: '%0.4f' % x
        print('--- Go for Training ---')
        self.model.train()
        writer = SummaryWriter(self.modeldir, comment='Drug_Transformer_MLP')
        t_start = time.time()
        iteration_loss = 0
        for epo in range(train_epoch):
            for i, data in enumerate(training_generator):

                drug_graph=data[0].to(self.device)
                drug_dti=data[1].to(self.device)
                drug_desc=data[2].to(self.device)
                cell_enc=data[3].to(self.device)
                cell_exp=data[4].to(self.device)
                label = data[5].to(self.device)

                score = self.model(drug_graph, drug_dti, drug_desc, cell_enc, cell_exp)
                loss_fct = torch.nn.MSELoss()
                score = torch.squeeze(score, 1)
             
                loss = loss_fct(score, label)
                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()
                if (i % 1000 == 0):
                    t_now = time.time()
                    print('Training at Epoch ' + str(epo + 1) +
                          ' iteration ' + str(i) + \
                          ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                          ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")

            with torch.no_grad():
                ### regression: MSE, Pearson Correlation, with p-value, Concordance Index
                loss_val,rmse_val,mse_val,pcc_val,spm_val,r2_val,mae_val,cinx_val = self.test(validation_generator, self.model,mode='val')
                vallst = ["epoch " + str(epo)] + list(map(float2str, [loss_val,rmse_val,mse_val,pcc_val,spm_val,r2_val,mae_val,cinx_val ]))
                valid_metric_record.append(vallst)
#                 loss_test,rmse_test,mse_test,pcc_test,spm_test,r2_test,mae_test,cinx_test = self.test(testing_generator, self.model,mode='val')
#                 testlst = ["epoch " + str(epo)] + list(map(float2str, [loss_test,rmse_test,mse_test,pcc_test,spm_test,r2_test,mae_test,cinx_test]))
#                 testtable.add_row(testlst)
#                 print(testlst)
                if  loss_val <= bestauroc:
                    model_max = copy.deepcopy(self.model)
                    bestauroc = loss_val
                    best_epo=epo
                    print('Validation at Epoch ' + str(epo + 1) +
                          ' , with loss:' + str(loss_val.item())[:7] +
                          ' , rmse: ' + str(rmse_val)[:7] +
                          ' , mse: ' + str(mse_val)[:7] +
                          ' , pcc: ' + str(pcc_val)[:7] +
                          ' , spm: ' + str(spm_val)[:7] +
                          ' , r2: ' + str(r2_val)[:7] +
                          ' , mae: ' + str(mae_val)[:7] +
                          ' , cinx: ' + str(cinx_val)[:7] 
                         )
                    writer.add_scalar('valid/roc_aucscore', pcc_val, epo)
                   
            valtable.add_row(vallst)
        
        with open(self.record_fileval, 'w') as fp1:
            fp1.write(valtable.get_string())
        with open(self.pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck) 
        with open(self.val_pkl_file, 'wb') as pckval:
            pickle.dump(self.val_loss_history, pckval)
        self.model = model_max
        print('--- Training Finished ---')
        writer.flush()
        writer.close()
        with torch.no_grad():
            loss_test,rmse_test,mse_test,pcc_test,spm_test,r2_test,mae_test,cinx_test = self.test(testing_generator, self.model,mode='val')
            testlst = ["epoch " + str(epo)] + list(map(float2str, [loss_test,rmse_test,mse_test,pcc_test,spm_test,r2_test,mae_test,cinx_test]))
            testtable.add_row(testlst)
        with open(self.record_filetest, 'w') as fp2:
            fp2.write(testtable.get_string())
 
    def predict(self,dataset):
        print('predicting...')
        self.model=self.model.to(self.device)
        collate_fn=DRPCollator()
        params = {'batch_size': 128,
                  'shuffle': False,
                  'num_workers': 2,
                  'drop_last': False,
                 "collate_fn":collate_fn}
        generator = DataLoader(dataset, **params)

        y_label, y_pred= self.test(generator, self.model,mode = 'predict')

        return y_label, y_pred

    def save_model(self,save_modelname):
        torch.save(self.model.state_dict(), self.modeldir + save_modelname)

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == device:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)
        
CDR_pairs=pd.read_csv('./process_data/GDSC2.csv',index_col=0)
CDR_pairs=CDR_pairs.reset_index(drop=True)
Process_cell=process_cell()

desc_list=pd.read_csv('desc_list.csv',index_col=0)['0'].tolist()
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
desc=[]
for i in pred_dti.index:
    mol=rdkit.Chem.MolFromSmiles(i)
    desc.append(list(calculator.CalcDescriptors(mol)))
desc=pd.DataFrame(desc)
desc=desc.dropna(axis=1)
from sklearn.preprocessing import MinMaxScaler
# 创建MinMaxScaler对象
scaler = MinMaxScaler()
# 训练集标准化
desc_scaled = scaler.fit_transform(desc)
desc_scaled=pd.DataFrame(desc_scaled)
desc_scaled.index=pred_dti.index

if __name__ == '__main__':
    args = arg_parse()
    args.device = 'cuda:{}'.format(args.device)
    if args.split == 'warmstart':
        kf = KFold(n_splits=10, shuffle=True, random_state=2024)
        kfold=1
        for train_index, test_index in kf.split(CDR_pairs):
            train=CDR_pairs.loc[train_index]
            test=CDR_pairs.loc[test_index]
            val, train = np.split(train.sample(frac = 1, random_state = 2024), [len(test)])
            train=train.reset_index(drop=True)
            test=test.reset_index(drop=True)
            val=val.reset_index(drop=True)
            print(len(train),len(val),len(test),len(train)/len(test))
            train_encdata, train_expdata = Process_cell.get_celldata(train)
            val_encdata, val_expdata = Process_cell.get_celldata(val)
            test_encdata, test_expdata = Process_cell.get_celldata(test)
            train_encdata.index = range(train_encdata.shape[0])
            val_encdata.index = range(val_encdata.shape[0])
            test_encdata.index = range(test_encdata.shape[0])
            train_encdata=train_encdata.astype(int)
            train_encdata[train_encdata>5]=5
            val_encdata=val_encdata.astype(int)
            val_encdata[val_encdata>5]=5
            test_encdata=test_encdata.astype(int)
            test_encdata[test_encdata>5]=5
            train_encdata['cat']=0
            val_encdata['cat']=0
            test_encdata['cat']=0

            train_expdata.index = range(train_expdata.shape[0])
            val_expdata.index = range(val_expdata.shape[0])
            test_expdata.index = range(test_expdata.shape[0])

            train=pd.concat([train[['smiles','Label']],train_encdata,train_expdata],axis=1)
            val=pd.concat([val[['smiles','Label']],val_encdata,val_expdata],axis=1)
            test=pd.concat([test[['smiles','Label']],test_encdata,test_expdata],axis=1)

            train_set = DRPDataset(train,pred_dti,desc_scaled,596)
            val_set = DRPDataset(val,pred_dti,desc_scaled,596)
            test_set = DRPDataset(test,pred_dti,desc_scaled,596)
            modeldir = './'+args.model+'_warmstart/'
            if not os.path.exists(modeldir):
                os.mkdir(modeldir)
            net = Model(modeldir=modeldir,model=args.model,kfold=kfold,device=args.device,epoch=args.epoch)
            net.train(train_set,val_set,test_set)
            net.save_model(args.model+'warmstart_kfold_'+str(kfold)+'.pt')
            with torch.no_grad():
                y_label, y_pred  = net.predict(test_set)
            test['pred']=y_pred
            test=test[['smiles','Label','pred']]
            test.to_csv(modeldir+str(kfold)+'.csv')
            kfold=kfold+1
            
    if args.split == 'cellcoldstart':
        kfold=1
        kf = KFold(n_splits=10, shuffle=True, random_state=2024)
        celldict=dict(zip([i for i in range(800)],CDR_pairs.COSMIC_ID.unique()))
        for train_index, test_index in kf.split(CDR_pairs.COSMIC_ID.unique()):
            train=CDR_pairs[CDR_pairs.COSMIC_ID.isin([celldict[i] for i in train_index])]
            test=CDR_pairs[CDR_pairs.COSMIC_ID.isin([celldict[i] for i in test_index])]
            val, train = np.split(train.sample(frac = 1, random_state = 2024), [len(test)])
            train=train.reset_index(drop=True)
            test=test.reset_index(drop=True)
            val=val.reset_index(drop=True)
            print(len(train),len(val),len(test),len(train)/len(test))
            train_encdata, train_expdata = Process_cell.get_celldata(train)
            val_encdata, val_expdata = Process_cell.get_celldata(val)
            test_encdata, test_expdata = Process_cell.get_celldata(test)
            train_encdata.index = range(train_encdata.shape[0])
            val_encdata.index = range(val_encdata.shape[0])
            test_encdata.index = range(test_encdata.shape[0])
            train_encdata=train_encdata.astype(int)
            train_encdata[train_encdata>5]=5
            val_encdata=val_encdata.astype(int)
            val_encdata[val_encdata>5]=5
            test_encdata=test_encdata.astype(int)
            test_encdata[test_encdata>5]=5
            train_encdata['cat']=0
            val_encdata['cat']=0
            test_encdata['cat']=0

            train_expdata.index = range(train_expdata.shape[0])
            val_expdata.index = range(val_expdata.shape[0])
            test_expdata.index = range(test_expdata.shape[0])

            train=pd.concat([train[['smiles','Label']],train_encdata,train_expdata],axis=1)
            val=pd.concat([val[['smiles','Label']],val_encdata,val_expdata],axis=1)
            test=pd.concat([test[['smiles','Label']],test_encdata,test_expdata],axis=1)

            train_set = DRPDataset(train,pred_dti,desc_scaled,596)
            val_set = DRPDataset(val,pred_dti,desc_scaled,596)
            test_set = DRPDataset(test,pred_dti,desc_scaled,596)
            modeldir = './'+args.model+'_cellcoldstart/'
            if not os.path.exists(modeldir):
                os.mkdir(modeldir)
            net = Model(modeldir=modeldir,model=args.model,kfold=kfold,device=args.device,epoch=args.epoch)
            net.train(train_set,val_set,test_set)
            net.save_model(args.model+'cellcoldstart_kfold_'+str(kfold)+'.pt')
            with torch.no_grad():
                y_label, y_pred  = net.predict(test_set)
            test['pred']=y_pred
            test=test[['smiles','Label','pred']]
            test.to_csv(modeldir+str(kfold)+'.csv')
            kfold=kfold+1
          
    if args.split == 'drugcoldstart':
        kfold=1
        kf = KFold(n_splits=10, shuffle=True, random_state=2024)
        drugdict=dict(zip([i for i in range(CDR_pairs.drug_name.unique().shape[0])],CDR_pairs.drug_name.unique()))
        for train_index, test_index in kf.split(CDR_pairs.drug_name.unique()):
            train=CDR_pairs[CDR_pairs.drug_name.isin([drugdict[i] for i in train_index])]
            test=CDR_pairs[CDR_pairs.drug_name.isin([drugdict[i] for i in test_index])]
            val, train = np.split(train.sample(frac = 1, random_state = 2024), [len(test)])
            train=train.reset_index(drop=True)
            test=test.reset_index(drop=True)
            val=val.reset_index(drop=True)
            print(len(train),len(val),len(test),len(train)/len(test))
            train_encdata, train_expdata = Process_cell.get_celldata(train)
            val_encdata, val_expdata = Process_cell.get_celldata(val)
            test_encdata, test_expdata = Process_cell.get_celldata(test)
            train_encdata.index = range(train_encdata.shape[0])
            val_encdata.index = range(val_encdata.shape[0])
            test_encdata.index = range(test_encdata.shape[0])
            train_encdata=train_encdata.astype(int)
            train_encdata[train_encdata>5]=5
            val_encdata=val_encdata.astype(int)
            val_encdata[val_encdata>5]=5
            test_encdata=test_encdata.astype(int)
            test_encdata[test_encdata>5]=5
            train_encdata['cat']=0
            val_encdata['cat']=0
            test_encdata['cat']=0

            train_expdata.index = range(train_expdata.shape[0])
            val_expdata.index = range(val_expdata.shape[0])
            test_expdata.index = range(test_expdata.shape[0])

            train=pd.concat([train[['smiles','Label']],train_encdata,train_expdata],axis=1)
            val=pd.concat([val[['smiles','Label']],val_encdata,val_expdata],axis=1)
            test=pd.concat([test[['smiles','Label']],test_encdata,test_expdata],axis=1)

            train_set = DRPDataset(train,pred_dti,desc_scaled,596)
            val_set = DRPDataset(val,pred_dti,desc_scaled,596)
            test_set = DRPDataset(test,pred_dti,desc_scaled,596)
            modeldir = './'+args.model+'_drugcoldstart/'
            if not os.path.exists(modeldir):
                os.mkdir(modeldir)
            net = Model(modeldir=modeldir,model=args.model,kfold=kfold,device=args.device,epoch=args.epoch)
            net.train(train_set,val_set,test_set)
            net.save_model(args.model+'drugcoldstart_kfold_'+str(kfold)+'.pt')
            with torch.no_grad():
                y_label, y_pred  = net.predict(test_set)
            test['pred']=y_pred
            test=test[['smiles','Label','pred']]
            test.to_csv(modeldir+str(kfold)+'.csv')
            kfold=kfold+1
            


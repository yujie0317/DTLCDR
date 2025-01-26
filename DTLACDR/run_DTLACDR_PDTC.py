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
from prettytable import PrettyTable
from DRPDataset import *
from DTLACDR import *
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,balanced_accuracy_score,\
        recall_score,precision_score,precision_recall_curve,confusion_matrix,average_precision_score
from rdkit.Chem import Descriptors
import rdkit
from rdkit.ML.Descriptors import MoleculeDescriptors

def arg_parse():
    parser = argparse.ArgumentParser(description="DTLACDR for clinical CDR prediction")
    parser.add_argument('--model', type=str, default='DTLACDR', help='DTLACDR,DTLACDR_cellenc,DTLACDR_cellexp,DTLACDR_drugdti,DTLACDR_drugGIN,DTLACDR_drugdesc,DTLCDR_cellexp')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--epoch', type=int, default=100, help='train_epoch')
    return parser.parse_args()

class process_cell():
    def __init__(self):

        self.cell_encoder = './process_data/PDTC_GDSC_enc.csv'
        self.cell_expression = './process_data/PDTC_GDSC_exp.csv'
     
    def get_celldata(self, data):
        cellid = list(data['COSMIC_ID'])
        enc_data =  pd.read_csv(self.cell_encoder,index_col=0)
        exp_data =  pd.read_csv(self.cell_expression,index_col=0)
        encdata = enc_data.loc[cellid]
        expdata = exp_data.loc[cellid]

        return encdata, expdata
    
pred_dti_gdsc2=pd.read_csv('../GCADTI/pred_dti_gdsc.csv',index_col=0)
pred_dti=[]
for i in range(len(pred_dti_gdsc2.smiles.unique())):
    pred_dti.append(pred_dti_gdsc2[i*1572:(i+1)*1572].label.tolist())
pred_dti=pd.DataFrame(pred_dti)
pred_dti.index=pred_dti_gdsc2.smiles.unique().tolist()

pred_dti_pdtc=pd.read_csv('../GCADTI/pred_dti_pdtc.csv',index_col=0)
pred_dti_test=[]
for i in range(len(pred_dti_pdtc.smiles.unique())):
    pred_dti_test.append(pred_dti_pdtc[i*1572:(i+1)*1572].label.tolist())
pred_dti_test=pd.DataFrame(pred_dti_test)
pred_dti_test.index=pred_dti_pdtc.smiles.unique().tolist()

gene2vec_dim_200_iter_9=pd.read_csv('../DTLCDR/gene2vec_dim_200_iter_9.txt',sep='\t| ',header=None)
gene2vec_dim_200_iter_9.index=gene2vec_dim_200_iter_9[0]
gene2vec_dim_200_iter_9=gene2vec_dim_200_iter_9.loc[:,1:]
gene=pd.read_csv('./process_data/PDTC_GDSC_enc.csv',index_col=0)
col=gene.columns
np.save('gene2vec_595.npy',np.vstack([np.array(gene2vec_dim_200_iter_9.loc[col]),np.zeros([16906-595,200])]))

modeldict={'DTLACDR':DTLACDR,'DTLACDR_cellexp':DTLACDR_cellexp,'DTLACDR_drugdti':DTLACDR_drugdti,\
          'DTLACDR_drugGIN':DTLACDR_drugGIN,'DTLACDR_drugdesc':DTLACDR_drugdesc, 'DTLCDR_cellexp':DTLCDR_cellexp }

class Model:
    def __init__(self,modeldir,model,device,epoch):
        self.model = modeldict[model](pred_dti.shape[1],596,'./gene2vec_595.npy',2732)
        self.random_layer = RandomLayer([2732,1], 256)
        self.domain_dmm = Discriminator(256,1)
        self.model._build()
        self.device = torch.device(device)
        self.epoch = epoch
        self.modeldir = modeldir
        self.record_fileval = os.path.join(self.modeldir, "valid_markdowntable.txt")
        self.record_filetest = os.path.join(self.modeldir, "test_markdowntable.txt")
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
        self.val_pkl_file = os.path.join(self.modeldir, "val_loss_curve_iter.pkl")
        self.val_loss_history = []

    def test(self,datagenerator,model,mode=None):
        y_label = []
        y_pred = []
        loss_s=0
        model.eval()
        if mode == 'predict':
            for i, data in enumerate(datagenerator):
                drug_graph=data[0].to(self.device)
                drug_dti=data[1].to(self.device)
                drug_desc=data[2].to(self.device)
                cell_enc=data[3].to(self.device)
                cell_exp=data[4].to(self.device)
                label = data[5].to(self.device)

                _, score = model(drug_graph, drug_dti, drug_desc, cell_enc, cell_exp)

                score = torch.squeeze(score, 1)
                logits =score.detach().cpu().numpy()
                label_ids = label.to('cpu').numpy()
                y_label = y_label + label_ids.flatten().tolist()
                y_pred = y_pred + logits.flatten().tolist()
            return y_label, y_pred
        else:
            for i, data in enumerate(datagenerator):
                drug_graph=data[0].to(self.device)
                drug_dti=data[1].to(self.device)
                drug_desc=data[2].to(self.device)
                cell_enc=data[3].to(self.device)
                cell_exp=data[4].to(self.device)
                label = data[5].to(self.device)

                _,score = model(drug_graph, drug_dti, drug_desc, cell_enc, cell_exp)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                score = torch.squeeze(score, 1)
                loss = loss_fct(score, label)
                #loss = loss_fct(score.squeeze(), label)
                logits =score.detach().cpu().numpy()
                label_ids = label.to('cpu').numpy()
                y_label = y_label + label_ids.flatten().tolist()
                y_pred = y_pred + logits.flatten().tolist()
                loss_s = loss_s+loss
            loss_m = loss_s/(i+1)
            auroc= roc_auc_score(y_label, y_pred)
            auprc = average_precision_score(y_label, y_pred)
            model.train()
            if mode == 'val':
                return loss_m,auroc,auprc
            elif mode == 'test':
                prec, recall, thresholds = precision_recall_curve(y_label, y_pred)
                f1 = 2 * prec * recall / (prec + recall + 0.00001)
                thred_optim = thresholds[np.argmax(f1)]
                y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
                cm1 = confusion_matrix(y_label, y_pred_s)
                accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
                sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
                specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
                return y_label, y_pred_s,accuracy_score(y_label, y_pred_s),auroc, auprc,sensitivity,specificity, f1_score(y_label, y_pred_s),\
                        recall_score(y_label, y_pred_s),balanced_accuracy_score(y_label, y_pred_s),precision_score(y_label, y_pred_s), loss_m

        

    def train(self,trainset, valset, testset):
        lr = 1e-4
        lr_da = 5e-5
        decay = 0
        BATCH_SIZE = 64
        train_epoch = 10
        self.model = self.model.to(self.device)
        self.random_layer.cuda(self.device)
        self.domain_dmm = self.domain_dmm.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        opt_da = torch.optim.Adam(self.domain_dmm.parameters(), lr=lr_da, weight_decay=decay)
        loss_history = []

        collate_fn=DRPCollator()
        trainparams = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 1,
                  'drop_last': True,
                      'pin_memory':False,
                      "collate_fn":collate_fn}
        source_generator = DataLoader(trainset, **trainparams)
        target_generator = DataLoader(testset, **trainparams)
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
        valtestparams = {'batch_size': BATCH_SIZE,
                  'shuffle': False,
                  'num_workers': 1,
                  'drop_last': False,
                        'pin_memory':False,
                        "collate_fn":collate_fn}
        validation_generator = DataLoader(valset, **valtestparams)
        testing_generator = DataLoader(testset, **valtestparams)
        bestauroc = 0

        valid_metric_record = []
        valid_metric_header = ['# epoch',"loss",
                                    "roc_auc_score",
                                    'average_precision_score']
        
        test_metric_header = ['# epoch',"loss", 'accuracy_score',
                                    "roc_auc_score",'average_precision_score','sensitivity','specificity',"f1_score",
                                    'recall_score',"balanced_accuracy_score",
                                    "precision_score"]
        valtable = PrettyTable(valid_metric_header)
        testtable = PrettyTable(test_metric_header)
        float2str = lambda x: '%0.4f' % x
        print('--- Go for Training ---')
        self.model.train()
#         total_loss_epoch = 0
#         model_loss_epoch = 0
#         da_loss_epoch = 0
#         epoch_lamb_da = 0
        writer = SummaryWriter(self.modeldir, comment='Drug_Transformer_MLP')
        t_start = time.time()
        iteration_loss = 0
        num_batches = len(multi_generator)
        for epo in range(train_epoch):
            for i, (data,data_t) in enumerate(multi_generator):
                drug_graph=data[0].to(self.device)
                drug_dti=data[1].to(self.device)
                drug_desc=data[2].to(self.device)
                cell_enc=data[3].to(self.device)
                cell_exp=data[4].to(self.device)
                label = data[5].to(self.device)
                
                drug_graph_t=data_t[0].to(self.device)
                drug_dti_t=data_t[1].to(self.device)
                drug_desc_t=data_t[2].to(self.device)
                cell_enc_t=data_t[3].to(self.device)
                cell_exp_t=data_t[4].to(self.device)
                label_t = data_t[5].to(self.device)
                
                f, score = self.model(drug_graph, drug_dti, drug_desc, cell_enc, cell_exp)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(torch.squeeze(score, 1), label)
                if epo >=2:
                    reverse_f = ReverseLayerF.apply(f, 1)
                    softmax_output = torch.nn.Softmax(dim=1)(score)
                    softmax_output = softmax_output.detach()
                    random_out = self.random_layer.forward([reverse_f, softmax_output])
                    adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                    f_t,score_t = self.model(drug_graph_t, drug_dti_t, drug_desc_t, cell_enc_t, cell_exp_t)
                    reverse_f_t = ReverseLayerF.apply(f_t, 1)
                    softmax_output_t = torch.nn.Softmax(dim=1)(score_t)
                    softmax_output_t = softmax_output_t.detach()
                    random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                    adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                    loss_cdan_src = loss_fct(torch.squeeze(adv_output_src_score, 1), torch.zeros(BATCH_SIZE).to(self.device))
                    loss_cdan_tgt = loss_fct(torch.squeeze(adv_output_tgt_score, 1), torch.ones(BATCH_SIZE).to(self.device))
                    da_loss = loss_cdan_src + loss_cdan_tgt
                    loss = loss+0.5*da_loss
                else:
                    loss = loss

                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                opt_da.zero_grad()
                loss.backward()
                opt.step()
                opt_da.step()
                if (i % 3000 == 0):
                    t_now = time.time()
                    print('Training at Epoch ' + str(epo + 1) +
                          ' iteration ' + str(i) + \
                          ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                          ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")

            with torch.no_grad():

                ### regression: MSE, Pearson Correlation, with p-value, Concordance Index
                loss_val,auroc_val ,auprc_val = self.test(validation_generator, self.model,mode='val')
                vallst = ["epoch " + str(epo)] + list(map(float2str, [loss_val,auroc_val ,auprc_val]))
                valid_metric_record.append(vallst)
                y_true,y_pred, \
                accuracyscore, roc_aucscore, auprc,sensitivity,specificity,\
                f1score, recallscore, balancedaccuracy_score,precisionscore,\
                loss_test = self.test(testing_generator, self.model,mode='test')
                testlst = ["epoch " + str(epo)] + list(map(float2str, [loss_test,accuracyscore, roc_aucscore, auprc,sensitivity,specificity,f1score, recallscore,
                                                                           balancedaccuracy_score,precisionscore]))
                testtable.add_row(testlst)
                print(roc_aucscore,auprc)
                if  auroc_val >= bestauroc:
                    model_max = copy.deepcopy(self.model)
                    bestauroc = auroc_val
                    best_epo=epo
                    print('Validation at Epoch ' + str(epo + 1) +
                          ' , with loss:' + str(loss_val.item())[:7] +
                          ' , roc_aucscore: ' + str(auroc_val)[:7] +
                        ' , prc_score: ' + str(auprc_val)[:7] 
                         )
                    writer.add_scalar('valid/roc_aucscore', auroc_val, epo)
                    writer.add_scalar("valid/prc_score", auprc_val, epo)
            valtable.add_row(vallst)
        
        with open(self.record_fileval, 'w') as fp1:
                fp1.write(valtable.get_string())
        with open(self.pkl_file, 'wb') as pck:
                pickle.dump(loss_history, pck)        
        self.model = model_max
        print('--- Training Finished ---')
        writer.flush()
        writer.close()
        with torch.no_grad():
            y_true,y_pred, \
            accuracyscore, roc_aucscore, auprc,sensitivity,specificity,\
            f1score, recallscore, balancedaccuracy_score,precisionscore,\
            loss_test = self.test(testing_generator, self.model,mode='test')
            testlst = ["epoch " + str(best_epo)] + list(map(float2str, [loss_test,accuracyscore, roc_aucscore, auprc,sensitivity,specificity,f1score, recallscore,
                                                                       balancedaccuracy_score,precisionscore]))
        testtable.add_row(testlst)
        with open(self.record_filetest, 'w') as fp2:
            fp2.write(testtable.get_string())

    def predict(self,dataset):
        print('predicting...')
        self.model=self.model.to(self.device)
        collate_fn=DRPCollator()
        params = {'batch_size': 128,
                  'shuffle': False,
                  'num_workers': 3,
                  'drop_last': False,
                 "collate_fn":collate_fn}
        generator = DataLoader(dataset, **params)

        y_label, y_pred= self.test(generator, self.model,mode = 'predict')

        return y_label, y_pred

    def save_model(self):
        torch.save(self.model.state_dict(), self.modeldir + '/model.pt')

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
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
        
CDR_pairs=pd.read_csv('./process_data/gdsc_aucflag.csv',index_col=0)
CDR_pairs=CDR_pairs.reset_index(drop=True)
Process_cell=process_cell()

test_pairs=pd.read_csv('./process_data/PDTC_aucflag.csv',index_col=0)
test_pairs=test_pairs.reset_index(drop=True)
desc_list=pd.read_csv('../DTLCDR/desc_list.csv',index_col=0)['0'].tolist()
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
desc=[]
for i in list(set(pred_dti.index)|set(pred_dti_test.index)):
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
desc_scaled.index=list(set(pred_dti.index)|set(pred_dti_test.index))


if __name__ == '__main__':
    args = arg_parse()
    args.device = 'cuda:{}'.format(args.device)
    train=CDR_pairs
    test=test_pairs
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
    test_set = DRPDataset(test,pred_dti_test,desc_scaled,596)

    modeldir = './'+args.model+'_warmstart/'
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    net = Model(modeldir=modeldir,model=args.model,device=args.device,epoch=args.epoch)
    net.train(train_set,val_set,test_set)
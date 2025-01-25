import numpy as np
import pandas as pd
import sys
import csv
import copy
import time 
import pickle
from sklearn.model_selection import train_test_split
from DTIDataset import *
from GCADTI import *
import argparse
import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,balanced_accuracy_score,\
        recall_score,precision_score,precision_recall_curve,confusion_matrix,average_precision_score
np.seterr(divide='ignore',invalid='ignore')

def arg_parse():
    parser = argparse.ArgumentParser(description="GCADTI for DTI prediction")
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--task', type=str, default='training', help='training, predict')
    return parser.parse_args()

IC_train=pd.read_csv('./IC_train.csv',index_col=0)
IC_test=pd.read_csv('./IC_test.csv',index_col=0)

IC_train ,IC_val = train_test_split(IC_train, random_state=1,test_size=0.1, shuffle=True)
val_set = DTIDataset(IC_val)
train_set = DTIDataset(IC_train)
test_set = DTIDataset(IC_test)

class Model:
    def __init__(self, modeldir, device):
        self.model = GCADTI()
        self.device = torch.device(device)
        self.modeldir = modeldir
        self.record_fileval = os.path.join(self.modeldir, "valid_markdowntable.txt")
        self.record_filetest = os.path.join(self.modeldir, "test_markdowntable.txt")
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")

    def test(self,datagenerator,model,mode=None):
        y_label = []
        y_pred = []
        loss_s=0
        model.eval()
        if mode == 'predict':
            for i, data in enumerate(datagenerator):
                drug, target, label = data
                drug, target, label = drug.to(self.device),target.to(self.device),label.float().to(self.device)
                _,score = model(target,drug)
                score = torch.squeeze(score, 1)
                logits =score.detach().cpu().numpy()
                label_ids = label.to('cpu').numpy()
                y_label = y_label + label_ids.flatten().tolist()
                y_pred = y_pred + logits.flatten().tolist()
            return y_label, y_pred
        else:
            for i, data in enumerate(datagenerator):
                drug, target, label = data
                drug, target, label = drug.to(self.device),target.to(self.device),label.float().to(self.device)
                _,score = model(target,drug)
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
                print(thred_optim)
                y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
                cm1 = confusion_matrix(y_label, y_pred_s)
                accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
                sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
                specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
                return y_label, y_pred_s,accuracy_score(y_label, y_pred_s),auroc, auprc,sensitivity,specificity, f1_score(y_label, y_pred_s),\
                        recall_score(y_label, y_pred_s),balanced_accuracy_score(y_label, y_pred_s),precision_score(y_label, y_pred_s), loss_m

        

    def train(self,trainset, valset, testset):
        lr = 1e-4
        decay = 0
        BATCH_SIZE = 128
        train_epoch = 50
        self.model = self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 5])
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        loss_history = []

        trainparams = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers':12,
                  'drop_last': True,
                       'pin_memory':True,
                'collate_fn': graph_collate_func}
        training_generator = DataLoader(trainset, **trainparams)
        valtestparams = {'batch_size': BATCH_SIZE,
                  'shuffle': False,
                  'num_workers':12,
                  'drop_last': False,
                         'pin_memory':True,
                'collate_fn': graph_collate_func}
        validation_generator = DataLoader(valset, **valtestparams)
        testing_generator = DataLoader(testset, **valtestparams)
        bestauroc = 0
        model_max = copy.deepcopy(self.model)

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
        writer = SummaryWriter(self.modeldir, comment='Drug_Transformer_MLP')
        t_start = time.time()
        iteration_loss = 0
        for epo in range(train_epoch):
            for i, data in enumerate(training_generator):
                drug, target, label = data
                drug, target, label = drug.to(self.device),target.to(self.device),label.float().to(self.device)
                _,score = self.model(target,drug)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                score = torch.squeeze(score, 1)
                loss = loss_fct(score, label)
                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()
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
        params = {'batch_size': 128,
                  'shuffle': False,
                  'num_workers':4,
                  'drop_last': False,
                'collate_fn': graph_collate_func}
        generator = DataLoader(dataset, **params)

        y_label, y_pred= self.test(generator, self.model,mode = 'predict')

        return y_label, y_pred

    def save_model(self):
        torch.save(self.model.state_dict(), self.modeldir + '/model.pt')

    def load_pretrained(self, path, device):
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


if __name__ == '__main__':
    args = arg_parse()
    args.device = 'cuda:{}'.format(args.device)
    if args.task == 'training':
        modeldir = './'
        net = Model(modeldir=modeldir,device=args.device)
        net.train(train_set,val_set,test_set)
        net.save_model()
    elif args.task == 'predict':
        modeldir = './'
        net = Model(modeldir=modeldir,device=args.device)
        net.load_pretrained('./model.pt', args.device)
        
        GDSC2=pd.read_csv('../DTLCDR/GDSC2.csv',index_col=0)
        prot=IC_test['sequence'].unique()
        pred_dti=[]
        for i in GDSC2['smiles'].unique():
            for j in prot:
                pred_dti.append([i,j])
        pred_dti=pd.DataFrame(pred_dti)
        pred_dti.columns=['smiles','sequence']
        pred_dti['label']=0 
        with torch.no_grad():
            pred_set=DTIDataset(pred_dti)
            y_label, y_pred  = net.predict(pred_set)
        pred_dti['label']=y_pred
        pred_dti.to_csv('pred_dti_gdsc2.csv')
        
        PDTC=pd.read_csv('../DTLACDR/PDTC_aucflag.csv',index_col=0)
        prot=IC_test['sequence'].unique()
        pred_dti=[]
        for i in PDTC['smiles'].unique():
            for j in prot:
                pred_dti.append([i,j])
        pred_dti=pd.DataFrame(pred_dti)
        pred_dti.columns=['smiles','sequence']
        pred_dti['label']=0 
        with torch.no_grad():
            pred_set=DTIDataset(pred_dti)
            y_label, y_pred  = net.predict(pred_set)
        pred_dti['label']=y_pred
        pred_dti.to_csv('pred_dti_pdtc.csv')
        
        GDSC=pd.read_csv('../DTLCDR/drug2smi.csv',index_col=0)
        prot=IC_test['sequence'].unique()
        pred_dti=[]
        for i in GDSC['smiles'].unique():
            for j in prot:
                pred_dti.append([i,j])
        pred_dti=pd.DataFrame(pred_dti)
        pred_dti.columns=['smiles','sequence']
        pred_dti['label']=0 
        with torch.no_grad():
            pred_set=DTIDataset(pred_dti)
            y_label, y_pred  = net.predict(pred_set)
        pred_dti['label']=y_pred
        pred_dti.to_csv('pred_dti_gdsc.csv')

        tcga=pd.read_csv('../DTLACDR/process_data/tcga_response.csv',index_col=0)
        prot=IC_test['sequence'].unique()
        pred_dti=[]
        for i in tcga['smiles'].unique():
            for j in prot:
                pred_dti.append([i,j])
        pred_dti=pd.DataFrame(pred_dti)
        pred_dti.columns=['smiles','sequence']
        pred_dti['label']=0 
        with torch.no_grad():
            pred_set=DTIDataset(pred_dti)
            y_label, y_pred  = net.predict(pred_set)
        pred_dti['label']=y_pred
        pred_dti.to_csv('pred_dti_tcga.csv')
        
        
        
        
    
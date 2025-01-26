import pandas as pd 
import numpy as np
import math

col=list(set(pd.read_csv('./PDTC_data/ExpressionModels.txt',sep='\t',index_col=0).T.columns)&set(pd.read_csv('../DTLCDR/process_data/exp.csv',index_col=0).columns))
pd.concat([pd.read_csv('./PDTC_data/ExpressionModels.txt',sep='\t',index_col=0).T[col],pd.read_csv('../DTLCDR/process_data/exp.csv',index_col=0)[col]]).to_csv('./process_data/PDTC_GDSC_exp.csv')

col=list(set(pd.read_csv('./PDTC_data/ExpressionModels.txt',sep='\t',index_col=0).T.columns)&set(pd.read_csv('../DTLCDR/process_data/exp_enc.csv',index_col=0).columns))
pd.concat([pd.read_csv('./PDTC_data/ExpressionModels.txt',sep='\t',index_col=0).T[col],pd.read_csv('../DTLCDR/process_data/exp_enc.csv',index_col=0)[col]]).to_csv('./process_data/PDTC_GDSC_enc.csv')

 
gdsc1_sensitivity_df=pd.read_csv('../DTLCDR/GDSC_data/GDSC1_fitted_dose_response_25Feb20.csv',index_col=0)
gdsc2_sensitivity_df=pd.read_csv('../DTLCDR/GDSC_data/GDSC2_fitted_dose_response_25Feb20.csv',index_col=0)
gdsc1_sensitivity_df=gdsc1_sensitivity_df[['COSMIC_ID', 'DRUG_NAME', 'AUC']]
gdsc1_grouped=gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
gdsc2_sensitivity_df=gdsc2_sensitivity_df[['COSMIC_ID', 'DRUG_NAME', 'AUC']]
gdsc2_grouped=gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
gdsc1_merged = pd.merge(gdsc1_sensitivity_df, gdsc1_grouped, on=['COSMIC_ID', 'DRUG_NAME'], how='left')
gdsc1_merged=gdsc1_merged.drop_duplicates(subset=['COSMIC_ID','DRUG_NAME','AUC_y'])
gdsc2_merged = pd.merge(gdsc2_sensitivity_df, gdsc2_grouped, on=['COSMIC_ID', 'DRUG_NAME'], how='left')
gdsc2_merged=gdsc2_merged.drop_duplicates(subset=['COSMIC_ID','DRUG_NAME','AUC_y'])
gdsc=pd.concat([gdsc2_merged, gdsc1_merged]).drop_duplicates(subset=['COSMIC_ID','DRUG_NAME'],keep='first')
gdsc=gdsc.reset_index(drop=True)
for i in gdsc.DRUG_NAME.unique():
    sub=gdsc[gdsc['DRUG_NAME']==i]
    threshold = np.median(sub.AUC_y)
    gdsc.loc[sub[sub.AUC_y<=threshold].index,'Label']=1
    gdsc.loc[sub[sub.AUC_y>threshold].index,'Label']=0
drug2smi=pd.read_csv('../DTLCDR/GDSC_data/drug2smi.csv',index_col=0)
drug2smi=dict(zip(drug2smi.DRUG_NAME,drug2smi.smiles))
gdsc=gdsc[gdsc.DRUG_NAME.isin(drug2smi.keys())]
gdsc['smiles']=[drug2smi[i] for i in gdsc['DRUG_NAME']]
gdsc.columns=['COSMIC_ID','drug','AUC','AUC_mean','Label','smiles']
gdsc['COSMIC_ID']=['DATA.'+str(i) for i in gdsc.COSMIC_ID]
gdsc[gdsc.COSMIC_ID.isin(pd.read_csv('./process_data/PDTC_GDSC_enc.csv',index_col=0).index)].to_csv('./process_data/gdsc_aucflag.csv')

DrugResponsesAUCModels=pd.read_csv('./PDTC_data/DrugResponsesAUCModels.txt',sep='\t')
Drug=pd.read_csv('./PDTC_data/PDTC_drugsmi.csv',index_col=0)
Drug=Drug[Drug['smi']!='0']
Drug=dict(zip(Drug['0'],Drug['smi']))
DrugResponsesAUCModels=DrugResponsesAUCModels[DrugResponsesAUCModels['Drug'].isin(Drug.keys())]
DrugResponsesAUCModels['smi']=[Drug[i] for i in DrugResponsesAUCModels['Drug']]
DrugResponsesAUCModels['log_ic']=[-np.log(i) for i in DrugResponsesAUCModels.iC50]
DrugResponsesAUCModels['label']=DrugResponsesAUCModels['log_ic']
DrugResponsesAUCModels=DrugResponsesAUCModels[['AUC','Drug','Model','smi']]
DrugResponsesAUCModels.columns=['AUC', 'drug', 'COSMIC_ID', 'smiles']
for i in DrugResponsesAUCModels.drug.unique():
    sub=DrugResponsesAUCModels[DrugResponsesAUCModels['drug']==i]
    threshold = np.median(sub.AUC)
    DrugResponsesAUCModels.loc[sub[sub.AUC>=threshold].index,'Label']=1
    DrugResponsesAUCModels.loc[sub[sub.AUC<threshold].index,'Label']=0
DrugResponsesAUCModels.to_csv('./process_data/PDTC_aucflag.csv')
import pandas as pd 
import numpy as np
GDSC2=pd.read_csv('./GDSC_data/GDSC2_fitted_dose_response_25Feb20.csv')
drug2smi=pd.read_csv('drug2smi.csv',index_col=0)
drug2smi=dict(zip(drug2smi.DRUG_NAME,drug2smi.smiles))
GDSC2=GDSC2[GDSC2.DRUG_NAME.isin(drug2smi)][['COSMIC_ID','DRUG_NAME','LN_IC50']]
exp=pd.read_csv('./GDSC_data/Cell_line_RMA_proc_basalExp.txt',sep='\t')
exp=exp.T
exp.columns=exp.iloc[0]
exp=exp[2:]
col=[]
a=0
for i in exp.columns:
    if i is np.nan:
        col.append(str(a))
        a=a+1
    else:
        col.append(i)
exp.columns=col
exp[exp.var().sort_values()[-3000:].index].to_csv('./process_data/exp.csv')
GDSC2['COSMIC_ID']=['DATA.'+str(i) for i in GDSC2.COSMIC_ID]
GDSC2=GDSC2[GDSC2.COSMIC_ID.isin(exp.index)]
GDSC2['smiles']=[drug2smi[i] for i in GDSC2.DRUG_NAME]
GDSC2=GDSC2.drop_duplicates(subset=['COSMIC_ID','LN_IC50','smiles'])
GDSC2['Label']=GDSC2['LN_IC50']
GDSC2.to_csv('./process_data/GDSC2.csv')

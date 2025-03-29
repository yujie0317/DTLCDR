# DTLCDR

The deep learning framework consists of three components: (1) GCADTI: a drug-target interaction (DTI) prediction model that is constructed to generate the complete drug-target profile as one of the drug features inputs for Cancer Drug Response (CDR) tasks; (2) DTLCDR: a CDR model for predicting drug response in cell lines; (3) DTLACDR: transfer the DTLCDR model to clinical prediction.


## Requirements

Create a virtual environment and install the requirements before running the code.

    conda create -n DTLCDR python==3.8
    conda activate DTLCDR
    pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install torch_geometric
    conda install cudatoolkit==11.1
    pip install --pre dgl==0.6.1 -f https://data.dgl.ai/wheels/repo.html -i  https://pypi.tuna.tsinghua.edu.cn/simple


# How to use:
## 1. GCADTI 
    cd ./GCADTI/
    run run_GCADTI.py to train the model
        python run_GCADTI.py --task training
    run run_GCADTI.py to predict dtis
        python run_GCADTI.py --task predict
            
## 2. DTLCDR
    cd ./DTLCDR/
    run process_GDSC2 to preprocess genetic profies and drug response data in GDSC2.
        python process_GDSC2.py 
    run run_DTLCDR.py to train the model
        python run_DTLCDR.py --model DTLCDR --device 0 --split warmstart --epoch 50
        
## 3. DTLACDR
    cd ./DTLACDR/
    run process_PDTC.py to preprocess genetic profies and drug response data in PDTC.
        python process_PDTC.py
    run run_DTLACDR_PDTC.py to test the model in PDTC
        python run_DTLACDR_PDTC.py --model DTLACDR_cellexp --device 1 --epoch 50
    run run_DTLACDR_TCGA.py to test the model in TCGA
        python run_DTLACDR_TCGA.py --model DTLACDR_cellexp --device 1 --epoch 50
    

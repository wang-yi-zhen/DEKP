import sys
import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import json
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import random
import pickle
import math
import os
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('/vocab.pkl')
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            # print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('UniKP/trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X

if __name__ == '__main__':
    # Dataset Load
    data_path = 'Dataset.csv'
    df = pd.read_csv(data_path, sep='\t')
    print(f"Load {len(df)} samples from dataset!")
    df.columns = ['ECNumber', 'Organism', 'Smiles', 'Substrate', 'Sequence', 'Type', 'Label', 'Unit', 'UniprotID', 'CID', 'Set']

    sequence = df['Sequence'].tolist()
    Smiles = df['Smiles'].tolist()
    CIDs = df['CID'].tolist()
    Label = df['Label'].tolist()
    ECNumber = df['ECNumber'].tolist()
    Organism = df['Organism'].tolist()
    Substrate = df['Substrate'].tolist()
    Type = df['Type'].tolist()
    smiles_input = smiles_to_vec(Smiles)
    smiles_dict = {s:data for (s, data) in zip (CIDs, smiles_input)}
    save_path = 'feature/trfm.pkl'
    with open(save_path, 'wb') as handle:
        pickle.dump(smiles_dict, handle)
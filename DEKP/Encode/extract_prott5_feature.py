import torch
import re
import gc
import json
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in tqdm(range(len(sequences_Example)), desc="Processing sequences"):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize

if __name__=='__main__':
    data_path = 'Dataset.csv'
    df = pd.read_csv(data_path, sep='\t')
    print(f"Load {len(df)} samples from dataset!")
    df.columns = ['ECNumber', 'Organism', 'Smiles', 'Substrate', 'Sequence', 'Type', 'Label', 'Unit', 'UniprotID', 'CID', 'Set']
    os.makedirs('feature', exist_ok=True)
    pdblist = df['UniprotID'].tolist()
    seqlist = df['Sequence'].tolist()

    fealist = Seq_to_vec(seqlist)
    fea_dict = {name:fea for (name, fea) in zip(pdblist, fealist)}
    with open(f'feature/t5.pkl', 'wb') as handle:
        pickle.dump(fea_dict, handle)
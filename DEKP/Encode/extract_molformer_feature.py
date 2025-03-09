import json
import numpy as np
import pandas as pd
import pickle
from argparse import Namespace
import yaml
import sys
sys.path.append('/media/dell/Data1/cheng/wangyizhen/molformer-main/molformer-main')
from tokenizer.tokenizer import MolTranBertTokenizer
from train_pubchem_light import LightningModule
import torch
from fast_transformers.masking import LengthMask as LM


def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size

def molformer_embed(model, smiles, tokenizer, batch_size=64):
    model.eval()
    embeddings = []
    for batch in batch_split(smiles, batch_size=batch_size):
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
        with torch.no_grad():
            token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
        # average pooling over tokens
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        embeddings.append(embedding.detach().cpu().numpy())
    return embeddings

if __name__=='__main__':
    # Dataset Load
    data_path = 'Dataset.csv'
    df = pd.read_csv(data_path, sep='\t')
    seqlist = df["Sequence"].tolist()
    Smiles = df['Smiles'].tolist()
    IDlist = df['UniprotID'].tolist()
    smiles_to_index = {s:ind for ind,s in enumerate(Smiles)}
    Smiles_ID_list = [ind for ind in range(len(Smiles))]
    df['CID'] = Smiles_ID_list
    smiles_list = df['Smiles'].tolist()
    with open('molformer/molformer-main/data/hparams.yaml', 'r') as f:
        config = Namespace(**yaml.safe_load(f))
    tokenizer = MolTranBertTokenizer('bert_vocab.txt')
    tokenizer.vocab
    ckpt = 'molformer/molformer-main/data/N-Step-Checkpoint_3_30000.ckpt'
    lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab)
    embeddings = molformer_embed(lm, smiles_list, tokenizer)
    mol_dict = {name:emd for name, emd in zip(Smiles_ID_list, embeddings)}
    with open('feature/molformer.pkl', 'wb') as f:
        pickle.save(mol_dict, f)

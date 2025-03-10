import sys
import os
import re
import time
import math
import json
import torch
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import torch_geometric
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from tokenizer import MolTranBertTokenizer
from torch_geometric.nn import TransformerConv, global_mean_pool, GCNConv, BatchNorm, NNConv
from torch_scatter import scatter_mean

def graph_collate_fn(batch):
    # create mini-batched graph
    graph_batch = torch_geometric.data.Batch.from_data_list([item[0] for item in batch])  # Batch graph data
    # concat batched samples in the first dimension
    token1 = torch.stack([item[1] for item in batch], dim=0)  # Stack tensors
    token2 = torch.stack([item[2] for item in batch], dim=0)  # Stack tensors
    fea  = torch.stack([item[3] for item in batch], dim=0)
    y = torch.stack([item[4] for item in batch], dim=0)
    
    #modify
    identifiers = [item[5] for item in batch]
    sequences = [item[6] for item in batch]
    ec_numbers = [item[7] for item in batch]
    types = [item[8] for item in batch]
    return graph_batch, token1, token2, fea, y, identifiers, sequences, ec_numbers, types

class KcatDataset(Dataset):
    def __init__(self, df, feature_list='molformer,t5,pst', seq_tokenzier="prot_t5_xl_uniref50", smi_tokenzier='bert_vocab.txt', max_protein_len=2500, max_smi_len=500):
        self.seq_tokenzier = T5Tokenizer.from_pretrained(seq_tokenzier, do_lower_case=False)
        self.smi_tokenzier = MolTranBertTokenizer('bert_vocab.txt')
        nneighbor = 32
        atom_type = "CA"
        self.y = []
        self.seq = []
        self.smi = []
        self.concat_fea = []
        self.graphlist = []
        self.identifiers = []
        self.sequences = []
        self.ec_numbers = []
        self.types = []
        
        with open('feature/molformer.pkl', 'rb') as handle:
            mol_dict = pickle.load(handle)
        with open('feature/trfm.pkl', 'rb') as handle:
            trfm_dict = pickle.load(handle)
        with open('feature/t5.pkl', 'rb') as handle:
            seq_dict = pickle.load(handle)
        with open('feature/pst.pkl', 'rb') as handle:
            struct_dict = pickle.load(handle)
        with open('feature/dssp.pkl', 'rb') as handle:
            dssp_dict = pickle.load(handle)
        with open('feature/pyg_graph.pkl', 'rb') as handle:
            graph_dict = pickle.load(handle)
        for (ind, data) in df.iterrows():
            name = data["UniprotID"]
            seq = data["Sequence"]
            rdkit_smiles = data["Smiles"]
            label = data["Label"]
            CID = int(data["CID"])
            feature_combined = []
            smi_fea = mol_dict[CID].squeeze() # 768
            if 'molformer' in feature_list:
                feature_combined.append(smi_fea)
            trfm_fea = trfm_dict[CID].squeeze() # 1024
            if 'trfm' in feature_list:
                feature_combined.append(trfm_fea)
            sequence = seq_dict[name].squeeze() # 1024
            if 't5' in feature_list:
                feature_combined.append(sequence)
            structure = struct_dict[name].squeeze() # 1280
            if 'pst' in feature_list:
                feature_combined.append(structure)
            dssp_fea = dssp_dict[name].mean(0).squeeze() # 13
            if 'dssp' in feature_list:
                feature_combined.append(dssp_fea)
            if ind == 1:
                #print(smi_fea.shape, trfm_fea.shape, sequence.shape, structure.shape, dssp_fea.shape)
                print(smi_fea.shape, trfm_fea.shape, sequence.shape)
            concat_fea = np.concatenate(feature_combined)
            graph = graph_dict[name]
            self.seq.append(seq)
            self.smi.append(rdkit_smiles)
            self.y.append(label)
            self.concat_fea.append(concat_fea)
            self.graphlist.append(graph)
            self.identifiers.append(name)
            self.sequences.append(seq)
            self.ec_numbers.append(data["ECNumber"])
            self.types.append(data["Type"])
            
        self.seq_vocab_size = self.seq_tokenzier.vocab_size
        self.smi_vocab_size = self.smi_tokenzier.vocab_size
        self.max_prolen = max_protein_len
        self.max_smi_len = max_smi_len
        # print(self.seq_vocab_size, self.smi_vocab_size)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        seq = [token for token in re.sub(r"[UZOB*]", "X", seq.rstrip('*'))]
        max_len = len(seq)
        encoded = self.seq_tokenzier.encode_plus(' '.join(seq), add_special_tokens=True, padding='max_length', return_token_type_ids=False, pad_to_max_length=True,truncation=True, max_length=self.max_prolen, return_tensors='pt')
        protein_token = encoded['input_ids'].flatten()
        # protein_attention_mask = encoded['attention_mask'].flatten().bool()
        smi = self.smi[idx]
        smi_encoded = self.smi_tokenzier.encode_plus(' '.join(smi), add_special_tokens=True, padding='max_length', return_token_type_ids=False, pad_to_max_length=True,truncation=True, max_length=self.max_smi_len, return_tensors='pt')
        smi_token = smi_encoded['input_ids'].flatten()
        # smi_attention_mask = smi_encoded['attention_mask'].flatten().bool()
        concat_fea = torch.tensor(self.concat_fea[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        identifier = self.identifiers[idx]
        sequence = self.sequences[idx]
        ec_number = self.ec_numbers[idx]
        type_value = self.types[idx]
        return self.graphlist[idx], protein_token, smi_token, concat_fea, label, identifier, sequence, ec_number, type_value

# For model evaluation
def evaluate_model(model, dataloader, device, criterion):
    epoch_loss = 0
    model.eval()
    total_pred, total_label = [], []
    with torch.no_grad():
        for ind, (graph, protein, smi, feature, label) in enumerate(dataloader):
            graph = graph.to(device)
            protein = protein.to(device)
            smi = smi.to(device)
            feature = feature.to(device)
            label = label.to(device)
            y_pred = model(graph, protein, smi, feature)
            # print(protein.shape, smi.shape, feature.shape, y_pred.shape)
            loss = criterion(y_pred, label)
            epoch_loss += loss.item()
            total_pred.extend(y_pred.cpu().numpy())
            total_label.extend(label.cpu().numpy())
    total_pred = np.array(total_pred)
    total_label = np.array(total_label)
    rmse = np.sqrt(mean_squared_error(total_label, total_pred))
    r2 = r2_score(total_label, total_pred)
    mae = mean_absolute_error(total_label, total_pred)
    pearson_corr, _ = pearsonr(total_label, total_pred)
    return epoch_loss/len(dataloader), rmse, r2, mae, pearson_corr, total_pred, total_label

class FeedForward(nn.Module):
    def __init__(self, hidden, out, dropout=0.1, residual=False):
        super().__init__()
        self.residual = residual
        self.linear1 = nn.Linear(hidden, 2*hidden)
        self.linear2 = nn.Linear(2*hidden,out,bias=False)
        self.norm = nn.LayerNorm(out)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x_update = self.linear2(self.dropout(self.linear1(x)))
        if self.residual:
            return self.norm(x + self.dropout(x_update))
        else:
            return x_update

class CNN(nn.Module):
    def __init__(self, hidden_dim, dropout, kernel_size=9):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        residue = x
        x = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        return x + residue
        
class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, hidden=32, num_layers=3, seq_len=1000, dropout=0.1, kernel_size=9):
        super().__init__()
        self.emd = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        self.encoder = nn.ModuleList([])
        for _ in range(num_layers):
            self.encoder.append(CNN(hidden, dropout, kernel_size=kernel_size))
    
            
    def forward(self, x):
        x = self.emd(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = torch.permute(x, (0,2,1)) # Batch, dim, length
        for f in self.encoder:
            x = f(x)
        return x

class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E

class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V

class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input_x, axis=1):
        input_size = input_x.size()
        trans_input = input_x.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input_x):
        x = torch.tanh(self.fc1(input_x))
        x = self.fc2(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		
        return attention

class GATLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GATLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        h_E = self.edge_update(h_V, edge_index, h_E)
        h_V = self.context(h_V, batch_id)
        return h_V, h_E

class GNNLayer(nn.Module):
    def __init__(self, num_hidden, conv=NNConv, dropout=0.2):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])
        self.conv1 = conv(num_hidden, num_hidden, nn=nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU()
                                ))
        self.conv2 = conv(num_hidden, num_hidden, nn=nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU()
                                ))
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)
    
    def forward(self, h_V, edge_index, h_E, batch_id):
        residual = h_V
        h_V = self.norm[0](residual + self.dropout(self.conv1(h_V, edge_index, edge_attr=h_E)))
        h_V = self.norm[1](residual + self.dropout(self.conv1(h_V, edge_index, edge_attr=h_E)))
        h_E = self.edge_update(h_V, edge_index, h_E)
        h_V = self.context(h_V, batch_id)
        return h_V, h_E

class GraphEncoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=16, num_layers=3, dropout=0.5, gnnconv='gat'):
        super(GraphEncoder, self).__init__()
        
        self.node_project = nn.Linear(node_in_dim, 64, bias=True)
        self.edge_project = nn.Linear(edge_in_dim, 64, bias=True)
        self.bn_node = nn.BatchNorm1d(64)
        self.bn_edge = nn.BatchNorm1d(64)
        
        self.W_v = nn.Linear(64, hidden_dim, bias=True)
        self.W_e = nn.Linear(64, hidden_dim, bias=True)
        if gnnconv == 'gat':
            conv = GATLayer(num_hidden=hidden_dim, dropout=dropout)
        self.layers = nn.ModuleList(conv for _ in range(num_layers))


    def forward(self, g):
        h_V, edge_index, h_E, batch_id = g.x, g.edge_index, g.edge_attr, g.batch
        h_V = self.W_v(self.bn_node(self.node_project(h_V)))
        h_E = self.W_e(self.bn_node(self.edge_project(h_E)))
        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
            # print(h_V.shape, h_E.shape)
        h_V = global_mean_pool(x=h_V,batch=batch_id).unsqueeze(1)
        # print(h_V.shape)
        return h_V

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=32, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout=dropout
        print(input_dim, input_dim//2, input_dim//4, output_dim)
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.layer_norm1 = nn.LayerNorm(input_dim//2)
        self.fc2 = nn.Linear(input_dim//2, input_dim//4)
        self.layer_norm2 = nn.LayerNorm(input_dim//4)
        self.fc3 = nn.Linear(input_dim//4, output_dim)
        self.layer_norm3 = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = F.dropout(self.relu(self.layer_norm1(self.fc1(x))),p=self.dropout)
        x = F.dropout(self.relu(self.layer_norm2(self.fc2(x))),p=self.dropout)
        x = F.dropout(self.relu(self.layer_norm3(self.fc3(x))),p=self.dropout)
        return x

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim=32, dropout=0.5):
        super(ResidualMLP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.layer_norm1 = nn.LayerNorm(input_dim // 2)
        self.res_fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.layer_norm2 = nn.LayerNorm(input_dim // 4)
        self.res_fc2 = nn.Linear(input_dim//2, input_dim // 4)
        self.fc3 = nn.Linear(input_dim // 4, output_dim)
        self.layer_norm3 = nn.LayerNorm(output_dim)
        self.res_fc3 = nn.Linear(input_dim // 4, output_dim)

        self.res_final = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        Residual_to_final = F.dropout(self.res_final(x), p=self.dropout)
        residual = x
        x = F.dropout(self.relu(self.layer_norm1(self.fc1(x))), p=self.dropout)
        x += F.dropout(self.res_fc1(residual), p=self.dropout)

        residual = x
        x = F.dropout(self.relu(self.layer_norm2(self.fc2(x))), p=self.dropout)
        x += F.dropout(self.res_fc2(residual), p=self.dropout)

        residual = x
        x = F.dropout(self.relu(self.layer_norm3(self.fc3(x))), p=self.dropout)
        x += F.dropout(self.res_fc3(residual), p=self.dropout)

        return x + Residual_to_final

class HighwayMLP(nn.Module):
    def __init__(self, input_dim, output_dim=32, dropout=0.5):
        super(HighwayMLP, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.layer_norm1 = nn.LayerNorm(input_dim // 2)
        self.res_fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.layer_norm2 = nn.LayerNorm(input_dim // 4)
        self.res_fc2 = nn.Linear(input_dim//2, input_dim // 4)
        self.fc3 = nn.Linear(input_dim // 4, output_dim)
        self.layer_norm3 = nn.LayerNorm(output_dim)
        self.res_fc3 = nn.Linear(input_dim // 4, output_dim)
        self.res_final = nn.Linear(input_dim, output_dim)
        self.transform_gate1 = nn.Linear(input_dim // 2, input_dim // 2)
        self.transform_gate2 = nn.Linear(input_dim // 4, input_dim // 4)
        self.transform_gate3 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        Residual_to_final = F.dropout(self.res_final(x), p=self.dropout)
        residual = F.dropout(self.res_fc1(x), p=self.dropout)
        x = F.dropout(self.relu(self.layer_norm1(self.fc1(x))), p=self.dropout)
        transform_gate = self.sigmoid(self.transform_gate1(x))
        x = transform_gate * x + (1 - transform_gate) * residual

        residual = F.dropout(self.res_fc2(x), p=self.dropout) 
        x = F.dropout(self.relu(self.layer_norm2(self.fc2(x))), p=self.dropout)
        transform_gate = self.sigmoid(self.transform_gate2(x))
        x = transform_gate * x + (1 - transform_gate) * residual

        residual = F.dropout(self.res_fc3(x), p=self.dropout)
        x = F.dropout(self.relu(self.layer_norm3(self.fc3(x))), p=self.dropout)
        transform_gate = self.sigmoid(self.transform_gate3(x))
        x = transform_gate * x + (1 - transform_gate) * residual

        return x + Residual_to_final
    
class MetaDecoder(nn.Module):
    def __init__(self, seq_vocab_size, smi_vocab_size, feature_dim_list=[1024], hidden=32, num_layers=3, protein_len=2590, smi_len=555, dropout=0.1, kernel_size=9):
        super(MetaDecoder, self).__init__()
       
        self.gnn_encoder = GraphEncoder(node_in_dim=42, edge_in_dim=92, hidden_dim=hidden, num_layers=num_layers, dropout=dropout, gnnconv='gat')
        self.seq_encoder = SeqEncoder(seq_vocab_size, hidden=hidden, num_layers=num_layers, seq_len=protein_len, dropout=dropout, kernel_size=kernel_size)
        self.smi_encoder = SeqEncoder(smi_vocab_size, hidden=hidden, num_layers=num_layers, seq_len=smi_len, dropout=dropout, kernel_size=kernel_size)       
        self.feature_dim_list = feature_dim_list
        self.feature_process = nn.ModuleList([ResidualMLP(fea_dim, hidden, dropout) for fea_dim in feature_dim_list])
        
        fuse_fea_dim = len(feature_dim_list) * hidden + hidden*3
        self.fc1 = nn.Linear(fuse_fea_dim, hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden//2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden//2, 1)

    def forward(self, graph, protein, smi, feature):
        protein_fea = self.seq_encoder(protein)
        protein_fea = F.normalize(protein_fea, p=2, dim=-1).mean(dim=2)
        smi_fea = self.smi_encoder(smi)
        smi_fea = F.normalize(smi_fea, p=2, dim=-1).mean(dim=2)
        splits = torch.split(feature, self.feature_dim_list, dim=-1)
        processed_features = []
        for i, mlp in enumerate(self.feature_process):
            fea = mlp(splits[i])
            processed_features.append(fea)
        concat_fea = torch.cat(processed_features, dim=-1)
        gnn_out = self.gnn_encoder(graph).squeeze().reshape(protein_fea.shape)
        fused_x = torch.concat([protein_fea, smi_fea, concat_fea, gnn_out], dim=1)
        self.latent = fused_x
        x = F.relu(self.dropout1(self.fc1(fused_x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        logit = self.fc(x)
        return logit.squeeze()
    
    def encode(self):
        return self.latent

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for training multi-modal Kcat predictor.')
    parser.add_argument('--epoch', '-e', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--data', '-d', type=str, default='Train_data.csv', help='Train data (.csv)')
    parser.add_argument('--outdir', '-o', type=str, default='result', help='Output directory')
    parser.add_argument('--model_dir', '-md', type=str, default='Model', help='Model directory')
    parser.add_argument('--name', '-n', type=str, default='Pretrained', help='Model name')
    parser.add_argument('--batch', '-b', type=int, default=256, help='Batch size')
    parser.add_argument('--hidden', type=int, default=32, help='Hidden size')
    parser.add_argument('--n_layer', '-l', type=int, default=3, help='Number of layers')
    parser.add_argument('--kernel', '-ks', type=int, default=9, help='CNN kernel size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU processor ID')  
    parser.add_argument('--seed', '-s', type=int, default=3407, help='random seed for reproducibility')
    parser.add_argument('--fea', '-f', type=str, default='trfm,t5,pst', help='Feature list separated by comma, including molformer,t5,trfm,pst,dssp.')
    parser.add_argument('--params', '-p', type=str, default='Model/Pretrained_model_trfm,t5.pkl', help='Pretrained weights.')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_arguments()
    print(args)
    data_path = args.data
    num_epochs = args.epoch
    out_dir = args.outdir
    model_dir = args.model_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("plot", exist_ok=True)
    name = args.name
    batch = args.batch
    hidden = args.hidden
    num_layers = args.n_layer
    kernel_size = args.kernel
    dropout = args.dropout
    lr = args.lr
    gpu = args.gpu
    seed = args.seed
    if seed != 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    feaD_dict = {'molformer':768, 't5':1024, 'trfm':1024, 'pst':1280, 'dssp':13}
    fea = args.fea
    feature_list = fea.split(',')
    feature_dim_list = [feaD_dict[k] for k in feature_list]

    df = pd.read_csv(data_path, sep='\t')
    df = df.dropna()
    print(f"Load {len(df)} samples from dataset!")
    df = df[df['Label'] != 1e-8]
    df = df[~df['Smiles'].str.contains('\.')]
    df['Set'] = np.random.choice(['train', 'valid', 'test'], size=len(df), p=[0.8, 0.1, 0.1])
    train_df = df[df['Set']=='train']
    valid_df = df[df['Set']=='valid']
    protein_len = [len(a) for a in df['Sequence']]
    smiles_len = [len(a) for a in df['Smiles']]
    max_prolen = np.array(protein_len).max()
    max_smilen = np.array(smiles_len).max()
    train_ds = KcatDataset(train_df, feature_list=feature_list, max_protein_len=max_prolen, max_smi_len=max_smilen)
    valid_ds = KcatDataset(valid_df, feature_list=feature_list, max_protein_len=max_prolen, max_smi_len=max_smilen)
    seq_vocab_size = train_ds.seq_vocab_size
    smi_vocab_size = train_ds.smi_vocab_size
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,num_workers=8, collate_fn=graph_collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch, shuffle=False, num_workers=8, collate_fn=graph_collate_fn)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = MetaDecoder(seq_vocab_size, smi_vocab_size, feature_dim_list=feature_dim_list, hidden=hidden, num_layers=num_layers, protein_len=max_prolen, smi_len=max_smilen, dropout=dropout, kernel_size=kernel_size).to(device)
    criterion = nn.MSELoss(reduction="mean") # nn.HuberLoss()# nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    # Model training and validating
    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print("Model Trainable Parameter: "+ str(params/1024/1024) + 'Mb' + "\n")
    max_patience = num_epochs
    patience = 0
    for epoch in range(num_epochs):
        start = time.perf_counter()
        model.train()
        epoch_loss = 0
        step = 0
        for ind, (graph, protein, smi, feature, label, identifier, sequence, ec_number, type_value) in enumerate(train_loader):
            graph = graph.to(device)
            protein = protein.to(device)
            smi = smi.to(device)
            feature = feature.to(device)
            label = label.to(device)
            y_pred = model(graph, protein, smi, feature)
            total_pred = np.array(y_pred.detach().cpu())
            total_label = np.array(label.detach().cpu())
            rmse = np.sqrt(mean_squared_error(total_label, total_pred))
            r2 = r2_score(total_label, total_pred)
            mae = mean_absolute_error(total_label, total_pred)
            pearson_corr, _ = pearsonr(total_label, total_pred)
            model.zero_grad()
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        end = time.perf_counter()
        print(f"Training {(end - start):.4f}s | Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")
        model.eval()
        val_loss, rmse, r2, mae, pearson_corr, _, _ = evaluate_model(model, valid_loader, device, criterion)
        print(f"Validation - Loss: {val_loss:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, Pearson: {pearson_corr:.4f}")
        torch.save(model.state_dict(), os.path.join(model_dir,f'{name}_model_{fea}.pkl'))

import torch
import re
import json
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

import argparse
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch
import sys
import numpy as np
import json

# 将项目根目录添加到 sys.path
sys.path.append('/file/path/PST/PST-main')
sys.path.append('/file/path/PST/PST-main/scripts')
from pst.esm2 import PST
from example_dataset import ExampleDataset

def cfg_parse_args():
    parser = argparse.ArgumentParser(
        description="Use PST to extract per-token representations \
        for pdb files stored in datadir/raw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="../data/",
        help="Path to the dataset, pdb files should be stored in datadir/raw/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pst_t33_so",
        help="Name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "--include-seq",
        action='store_true',
        help="Add sequence representation to the final representation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the data loader"
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default=None,
        help="How to aggregate protein representations across layers. \
        `None`: last layer; `mean`: mean pooling, `concat`: concatenation",
    )
    cfg = parser.parse_args()
    cfg.device = "cuda:1" if torch.cuda.is_available() else "cpu"
    return cfg

@torch.no_grad()
def compute_repr(data_loader, model, cfg):
    embeddings = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(cfg.device)
        out = model(data, return_repr=True, aggr=cfg.aggr)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        if cfg.include_seq:
            if "so" not in cfg.model:
                raise ValueError("Use models pretrained using struct only updates strategy!")
            data.edge_index = None
            out_seq = model(data, return_repr=True, aggr=cfg.aggr)
            out_seq = out_seq[data.idx_mask]
            out = (out + out_seq) * 0.5
        embeddings.extend(list(unbatch(out, batch)))
    return embeddings

if __name__=='__main__':
    data_path = 'Dataset.csv'
    df = pd.read_csv(data_path, sep='\t')
    print(f"Load {len(df)} samples from dataset!")
    df.columns = ['ECNumber', 'Organism', 'Smiles', 'Substrate', 'Sequence', 'Type', 'Label', 'Unit', 'UniprotID', 'CID', 'Set']
    os.makedirs('feature', exist_ok=True)
    os.makedirs('feature/pst', exist_ok=True)
    pdblist = df['UniprotID'].tolist()
    cfg = cfg_parse_args()

    pretrained_path = Path("/file/path/PST/pst_t33_so.pt")
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model, model_cfg = PST.from_pretrained_url(
            cfg.model, pretrained_path
        )
    except:
        model, model_cfg = PST.from_pretrained_url(
            cfg.model,
            pretrained_path,
            map_location=torch.device("cpu"),
        )
    model.eval()
    model.to(cfg.device)

    dataset = ExampleDataset(
        root=cfg.datadir,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    
    protein_repr_all = compute_repr(data_loader, model, cfg)
    raw_files = Path(cfg.datadir) / "raw"
    uniprot_ids = [f.stem for f in raw_files.glob("*.pdb")]
    assert len(protein_repr_all) == len(uniprot_ids), "Embeddings and UniprotID count mismatch!"
    embeddings_with_ids = {}
    for uniprot_id, protein_repr in zip(uniprot_ids, protein_repr_all):
        embeddings_with_ids[uniprot_id] = protein_repr.cpu().numpy()    # .flatten()

    save_path = 'feature/pst.pkl'
    with open(save_path, 'wb') as handle:
        pickle.dump(embeddings_with_ids, handle)
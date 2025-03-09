
import torch
import os
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import DSSP
import pandas as pd
import pickle
from tqdm import tqdm
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

maxASA = {'G':188,'A':198,'V':220,'I':233,'L':304,'F':272,'P':203,'M':262,'W':317,'C':201,
              'S':234,'T':215,'N':254,'Q':259,'Y':304,'H':258,'D':236,'E':262,'K':317,'R':319}

def pdb_split(line):
    atom_type = "CNOS$"
    aa_trans_DICT = {
        'ALA': 'A', 'CYS': 'C', 'CCS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'MSE': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', }
    aa_type = "ACDEFGHIKLMNPQRSTVWY$"
    Atom_order=int(line[6:11].strip())-1
    atom=line[11:16].strip()
    amino=line[16:21].strip()
    AA_order=int(line[22:28].strip())-1
    x=line[28:38].strip()
    y=line[38:46].strip()
    z=line[46:54].strip()
    atom_single_name=line.strip()[-1]
    atom_single_name_vec = np.zeros(len(atom_type))
    atom_single_name_vec[atom_type.find(atom_single_name)] = 1
    AA_single_name_vec = np.zeros(len(aa_type))
    AA_single_name_vec[   aa_type.find(   aa_trans_DICT[ amino ] )  ] = 1
    atom_feature_combine= np.concatenate(( atom_single_name_vec.reshape(1, -1)   , AA_single_name_vec.reshape(1, -1)),axis=1)
    return atom,amino,AA_order, Atom_order, float(x),float(y),float(z),atom_feature_combine

def process_dssp(dssp):
    aa_type = "ACDEFGHIKLMNPQRSTVWY$"
    SS_type = "HBEGITS-"
    dssp_feature = []
    for i in range(len(dssp)):
        SS_vec = np.zeros(8)
        SS=dssp.property_list[i][2]
        SS_vec[SS_type.find(SS)] = 1
        PHI = dssp.property_list[i][4]
        PSI = dssp.property_list[i][5]
        ASA = dssp.property_list[i][3]
        aa_name = dssp.property_list[i][1]
        angle = np.array([PHI, PSI])
        radian = angle * (np.pi / 180)
        sin_values = np.sin(radian)
        cos_values = np.cos(radian)
        try:
            ASA = min(float(ASA) / maxASA[aa_name], 1)
        except:
            ASA = 1.
        feature1= np.concatenate(   (np.array([PHI, PSI, ASA]), SS_vec))
        feature=np.concatenate((sin_values, cos_values, [ASA], SS_vec))
        feature = [float(a) if isinstance(a, float) or isinstance(a, int) else 1. for a in feature]
        dssp_feature.append(feature)
    dssp_feature = np.array(dssp_feature)
    return dssp_feature

if __name__=='__main__':
    pdb_dir='../data/pdbdataset/'
    data_path = 'dataset.csv'
    df = pd.read_csv(data_path, sep='\t')
    print(f"Load {len(df)} samples from dataset!")
    df.columns = ['ECNumber', 'Organism', 'Smiles', 'Substrate', 'Sequence', 'Type', 'Label', 'Unit', 'UniprotID', 'CID', 'Set']
    dssp_exe = './mkdssp/mkdssp'
    os.makedirs('feature', exist_ok=True)
    os.makedirs('feature/DSSP', exist_ok=True)
    pdblist = df['UniprotID'].tolist()
    done_list = []
    fea_dict = {}
    error_log_file = 'error_log.txt'
with open(error_log_file, 'w') as log_file:
        for pdb in tqdm(pdblist):
            if pdb not in done_list:
                pdb_path = os.path.join(pdb_dir, f'{pdb}.pdb')
                save_path = f'feature/DSSP/{pdb}.pkl'
                if not os.path.exists(pdb_path):
                    print(f"PDB file {pdb_path} not exist。")
                    log_file.write(f"PDB file {pdb_path} not exist。\n")
                    continue
                try:
                    p = PDBParser()
                    structure = p.get_structure("1", pdb_path)
                    model = structure[0]
                    dssp = DSSP(model, pdb_path, dssp=dssp_exe)
                    feature = process_dssp(dssp)
                    fea_dict[pdb] = feature
                    done_list.append(pdb)
                except Exception as e:
                    print(f"handling PDB file {pdb} , error：{e}")
                    log_file.write(f"handling PDB file {pdb} ,error：{e}\n")
                    continue
with open('feature/dssp.pkl', 'wb') as handle:
    pickle.dump(fea_dict, handle)
print(len(done_list))
import numpy as np
import pandas as pd
import sys
import os
import random
import re
import torch
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

atoms = ['N', 'CA', 'C', 'O', 'R', 'CB']
n_atoms = len(atoms)
atom_idx = {atom:atoms.index(atom) for atom in atoms}

def get_cb(n, ca, c):
    b = ca - n
    c = c - ca
    a = np.cross(b, c)
    cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + ca
    return cb

def calculate_distances(atom_coordinates, query_position):
    distances = np.linalg.norm(atom_coordinates - query_position, axis=1)
    return distances

def parse_pdb(pdb_file, pos=None, atom_type='CA', nneighbor=32, cal_cb=True):
    current_pos = -1000
    X = []
    fillna = np.array([0.,0.,0.]).astype(np.float32)
    current_aa = {} # N, CA, C, O, R
    with open(pdb_file, 'r') as pdb_f:
        for line in pdb_f:
            if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
                if current_aa != {}:
                    R_group = []
                    for atom in current_aa:
                        if atom not in ["N", "CA", "C", "O"]:
                            R_group.append(current_aa[atom])
                    if R_group == []:
                        try:
                            R_group = [current_aa["CA"]]
                        except:
                            R_group = fillna
                    R_group = np.array(R_group).mean(0)
                    try:
                        N_coord = current_aa["N"]
                    except:
                        N_coord = fillna
                    try:
                        CA_coord = current_aa["CA"]
                    except:
                        CA_coord = fillna
                    try:
                        C_coord = current_aa["C"]
                    except:
                        C_coord = fillna
                    try:
                        O_coord = current_aa["O"]
                    except:
                        O_coord = fillna
                    if len(np.array([N_coord, CA_coord, C_coord, O_coord, R_group]).shape) < 2:
                        X.append(np.zeros((5,3))) # For broken residue
                    else:
                        X.append([N_coord, CA_coord, C_coord, O_coord, R_group])
                    current_aa = {}
                if line[0:4].strip() != "TER":
                    current_pos = int(line[22:26].strip())

            if line[0:4].strip() == "ATOM":
                atom = line[13:16].strip()
                if atom != "H":
                    try:
                        xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                        current_aa[atom] = xyz
                    except:
                        current_aa[atom] = fillna
    X = np.array(X)
    if cal_cb:
        X = np.concatenate([X, get_cb(X[:,0], X[:,1], X[:,2])[:, None]], 1)
    if pos is not None:
        atom_ind = atom_idx[atom_type] # CA atom
        if pos >= X.shape[0]:
            pos = X.shape[0] - 1
        query_coord = X[pos,atom_ind]
        distances = calculate_distances(X[:,atom_ind,:], query_coord)
        closest_indices = np.argsort(distances)[:nneighbor]
        X = X[closest_indices]
    return X # array shape: [Length, 6, 3] N, CA, C, O, R, CB


def get_geo_feat(X, edge_index, D_count=1):
    pos_embeddings = _positional_encodings(edge_index)
    node_angles = _get_angle(X) # 12D
    node_dist, edge_dist = _get_distance(X, edge_index, D_count=D_count)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)

    node = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    edge = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)

    return node, edge


def _positional_encodings(edge_index, num_embeddings=16):
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE

def _get_angle(X, eps=1e-7):
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles # dim = 12

def _rbf(D, D_min=0., D_max=20., D_count=1):
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _get_distance(X, edge_index, D_count=1):
    atom_N = X[:,atom_idx['N']]  # [L, 3]
    atom_CA = X[:,atom_idx['CA']]
    atom_C = X[:,atom_idx['C']]
    atom_O = X[:,atom_idx['O']]
    atom_R = X[:,atom_idx['R']]
    atom_CB = X[:,atom_idx['CB']]

    node_list = ['N-CA', 'N-C', 'N-O', 'N-R', 'N-CB', 'CA-C', 'CA-O', 'CA-R', 'CA-CB', 'C-O', 'C-R', 'C-CB', 'O-R', 'O-CB', 'R-CB']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=D_count)
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1) # shape = [N, 15 * 16]
    atom_list = ["N", "CA", "C", "O", "R", "CB"]
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=D_count)
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1) # shape = [E, 36 * 16]
    return node_dist, edge_dist # 42D node features + 96D edge features when dim of rbf is set to 1

def _get_direction_orientation(X, edge_index): # N, CA, C, O, R, CB
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    Q = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)

    node_j, node_i = edge_index

    t = F.normalize(X[:, [0,2,3,4,5]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    node_direction = torch.matmul(t, Q).reshape(t.shape[0], -1) # [L, 4 * 3]

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, Q[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, Q[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3]
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    r = torch.matmul(Q[node_i].transpose(-1,-2), Q[node_j]) # [E, 3, 3]
    edge_orientation = _quaternions(r) # [E, 4]
    return node_direction, edge_direction, edge_orientation

def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q

def get_graph_fea(pdb_path, pos=None, nneighbor=20, radius=10, atom_type='CA', cal_cb=True, D_count=1):
    X = torch.tensor(parse_pdb(pdb_path, pos=pos, atom_type=atom_type, nneighbor=nneighbor, cal_cb=cal_cb)).float()
    query_atom = X[:, atom_idx[atom_type]]
    edge_index = radius_graph(query_atom, r=radius, loop=False, max_num_neighbors=nneighbor, num_workers = 16)
    node, edge = get_geo_feat(X, edge_index, D_count=D_count)
    # print(node.shape, edge.shape)
    return Data(x=node, edge_index=edge_index, edge_attr=edge, name=os.path.basename(pdb_path).split('.')[0])


if __name__=='__main__':
    pdb_dir='../data/Kmpdbdataset'
    df = pd.read_csv('Dataset.csv', sep='\t')
    pdblist = df['UniprotID'].tolist()
    os.makedirs('feature', exist_ok=True)
    done_list = []
    graph_dict = {}
    #modify 1 
    broken_files = []
    for pdb in tqdm(pdblist):
        if pdb not in done_list:
            try:
                pdb_path = os.path.join(pdb_dir, f'{pdb}.pdb')
                X = torch.tensor(parse_pdb(pdb_path)).float()
                g = get_graph_fea(pdb_path)
                graph_dict[pdb] = g
                done_list.append(pdb)
            except:
                print(f'{pdb} PDB file broken!')
                broken_files.append(pdb) 
    with open('feature/pyg_graph.pkl', 'wb') as handle:
        pickle.dump(graph_dict, handle)
    print(len(done_list))
    
    if broken_files:
        with open('broken_files.log', 'w') as log_file:
            log_file.write('fail：\n')
            for pdb_name in broken_files:
                log_file.write(f'{pdb_name}\n')
        print('done，see broken_files.log')
    else:
        print('success')

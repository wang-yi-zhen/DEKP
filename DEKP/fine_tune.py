from pretrain import *
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

def graph_collate_fn(batch):
    # create mini-batched graph
    graph_batch = torch_geometric.data.Batch.from_data_list([item[0] for item in batch])  # Batch graph data
    # concat batched samples in the first dimension
    token1 = torch.stack([item[1] for item in batch], dim=0)  # Stack tensors
    token2 = torch.stack([item[2] for item in batch], dim=0)  # Stack tensors
    fea  = torch.stack([item[3] for item in batch], dim=0)
    y = torch.stack([item[4] for item in batch], dim=0)
    identifiers = [item[5] for item in batch]
    sequences = [item[6] for item in batch]
    ec_numbers = [item[7] for item in batch]
    types = [item[8] for item in batch]
    return graph_batch, token1, token2, fea, y, identifiers, sequences, ec_numbers, types

class KcatDataset(Dataset):
    def __init__(self, df, feature_list='molformer,t5,pst', seq_tokenzier="prot_t5_xl_uniref50", smi_tokenzier='bert_vocab.txt', max_protein_len=2500, max_smi_len=500):
        self.seq_tokenzier = T5Tokenizer.from_pretrained(seq_tokenzier, do_lower_case=False)
        self.smi_tokenzier = MolTranBertTokenizer('bert_vocab.txt')
        self.y = []
        self.seq = []
        self.smi = []
        self.concat_fea = []
        self.graphlist = []
        self.identifiers = []
        self.sequences = []
        self.ec_numbers = []
        self.types = []        
        fea_list = feature_list if isinstance(feature_list, list) else feature_list.split(',')

        self.feature_dicts = {}
        for feature_name in fea_list:
            with open(f'feature/{feature_name}.pkl', 'rb') as handle:
                self.feature_dicts[feature_name] = pickle.load(handle)
        with open('feature/pyg_graph.pkl', 'rb') as handle:
            graph_dict = pickle.load(handle)

        for (ind, data) in df.iterrows():
            name = data["UniprotID"]
            seq = data["Sequence"]
            rdkit_smiles = data["Smiles"]
            label = data["Label"]
            CID = int(data["CID"])
            feature_combined = []

            for feature_name in fea_list:
                if feature_name == 'molformer':
                    smi_fea = self.feature_dicts[feature_name][CID].squeeze()
                    feature_combined.append(smi_fea)
                elif feature_name == 'trfm':
                    trfm_fea = self.feature_dicts[feature_name][CID].squeeze()
                    feature_combined.append(trfm_fea)
                elif feature_name == 't5':
                    sequence = self.feature_dicts[feature_name][name].squeeze()
                    feature_combined.append(sequence)
                elif feature_name == 'pst':
                    structure = self.feature_dicts[feature_name][name].squeeze()
                    feature_combined.append(structure)
                elif feature_name == 'dssp':
                    dssp_fea = self.feature_dicts[feature_name][name].mean(0).squeeze()
                    feature_combined.append(dssp_fea)

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

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        seq = [token for token in re.sub(r"[UZOB*]", "X", seq.rstrip('*'))]
        max_len = len(seq)
        encoded = self.seq_tokenzier.encode_plus(' '.join(seq), add_special_tokens=True, padding='max_length', return_token_type_ids=False, truncation=True, max_length=self.max_prolen, return_tensors='pt')
        protein_token = encoded['input_ids'].flatten()
        smi = self.smi[idx]
        smi_encoded = self.smi_tokenzier.encode_plus(' '.join(smi), add_special_tokens=True, padding='max_length', return_token_type_ids=False, truncation=True, max_length=self.max_smi_len, return_tensors='pt')
        smi_token = smi_encoded['input_ids'].flatten()
        concat_fea = torch.tensor(self.concat_fea[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        identifier = self.identifiers[idx]
        sequence = self.sequences[idx]
        ec_number = self.ec_numbers[idx]
        type_value = self.types[idx]
        return self.graphlist[idx], protein_token, smi_token, concat_fea, label, identifier, sequence, ec_number, type_value

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
        gnn_out = self.gnn_encoder(graph).squeeze().reshape(protein_fea.shape) # graph feature shape: (Batch, 1, hidden)
        fused_x = torch.concat([protein_fea, smi_fea, concat_fea, gnn_out], dim=1)
        self.latent = fused_x
        x = F.relu(self.dropout1(self.fc1(fused_x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        logit = self.fc(x) # shape: (Batch, 1)
        return logit.squeeze()
    
    def encode(self):
        return self.latent

def evaluate_model(model, dataloader, device, criterion):
    epoch_loss = 0
    model.eval()
    total_pred, total_label = [], []
    with torch.no_grad():
        for ind, (graph, protein, smi, feature, label, _, _, _, _) in enumerate(dataloader):
            graph = graph.to(device)
            protein = protein.to(device)
            smi = smi.to(device)
            feature = feature.to(device)
            label = label.to(device)
            y_pred = model(graph, protein, smi, feature)
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

def prepare_ds(loader, best_model, concat=False, only_ML=False):
    best_model.eval()
    train_X = []
    train_Y = []
    identifiers = []
    sequences = []
    ec_numbers = []
    types = []    
    with torch.no_grad():
        for ind, (graph, protein, smi, feature, label, identifier, sequence, ec_number, type_value) in enumerate(loader):
            graph = graph.to(device)
            protein = protein.to(device)
            smi = smi.to(device)
            feature = feature.to(device)
            label = label.to(device)
            y_pred = best_model(graph, protein, smi, feature)
            latent_repr = best_model.encode().cpu().detach().squeeze()
            embedding = []
            if only_ML:
                embedding.append(feature.cpu().detach())
            else:
                if concat:
                    embedding.append(feature.cpu().detach())
                    embedding.append(latent_repr.cpu().detach())
                else:
                    embedding.append(latent_repr.cpu().detach())
            embedding = np.concatenate(embedding, axis=-1)
            train_Y.extend(label.cpu().detach().tolist())
            train_X.extend(embedding.tolist())
            identifiers.extend(identifier)
            sequences.extend(sequence)
            ec_numbers.extend(ec_number)
            types.extend(type_value)
    return np.array(train_X), np.array(train_Y), identifiers, sequences, ec_numbers, types

def run_ML(ML_list, name, train_X, train_Y, test_X, test_Y, test_identifiers, test_sequences, test_ec_numbers, test_types):
    model_dict = {f: {} for f in ML_list}
    for algorithm in ML_list:
        if algorithm == 'ET':
            model = ExtraTreesRegressor()
        elif algorithm == 'RF':
            model = RandomForestRegressor()
        elif algorithm == 'GBR':
            model = GradientBoostingRegressor()
        elif algorithm == 'LR':
            model = LinearRegression()
        elif algorithm == 'Ridge':
            model = Ridge()
        elif algorithm == 'Lasso':
            model = Lasso()
        elif algorithm == 'SVR':
            model = SVR()
        elif algorithm == 'KNN':
            model = KNeighborsRegressor()
        elif algorithm == 'DT':
            model = DecisionTreeRegressor()
        elif algorithm == 'MLP':
            model = MLPRegressor()
        model.fit(train_X, train_Y)
        
        predictions = model.predict(test_X)
        rmse = np.sqrt(mean_squared_error(test_Y, predictions))
        r2 = r2_score(test_Y, predictions)
        mae = mean_absolute_error(test_Y, predictions)
        pearson_corr, _ = pearsonr(test_Y, predictions)
        print(f'{name} \t {algorithm} \t RMSE:{rmse:.4f} \t R2:{r2:.4f} \t  MAE:{mae:.4f} \t PCC:{pearson_corr:.4f}')
        
        test_df = pd.DataFrame({
            'Identifier': test_identifiers,
            'Sequence': test_sequences,
            'ECNumber': test_ec_numbers,
            'Type': test_types,
            'Label': test_Y,
            'Predict': predictions
        })
        os.makedirs(f'result/{name}', exist_ok=True)
        test_df.to_excel(f'result/{name}/{algorithm}.xlsx', index=False)
        
        model_dict[algorithm]['model'] = model
        model_dict[algorithm]['perf'] = [rmse, r2, mae, pearson_corr]
        model_dict[algorithm]['result'] = test_df
        
        if algorithm == 'ET':
            with open(f'{name}_ET_model.pkl', 'wb') as f:
                pickle.dump(model, f)        
    
    with open(f'result/{name}/Graph_ML_{name}.pkl', 'wb') as handle:
        pickle.dump(model_dict, handle)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for training multi-modal Kcat predictor.')
    parser.add_argument('--epoch', '-e', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--data', '-d', type=str, default='DatasetAdd.csv', help='Train data (.csv)')
    parser.add_argument('--outdir', '-o', type=str, default='result', help='Output directory')
    parser.add_argument('--model_dir', '-md', type=str, default='Model', help='Model directory')
    parser.add_argument('--name', '-n', type=str, default='Pretrained', help='Model name')
    parser.add_argument('--batch', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden', type=int, default=32, help='Hidden size')
    parser.add_argument('--n_layer', '-l', type=int, default=3, help='Number of layers')
    parser.add_argument('--kernel', '-ks', type=int, default=9, help='CNN kernel size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU processor ID')  
    parser.add_argument('--seed', '-s', type=int, default=3407, help='random seed for reproducibility')
    parser.add_argument('--fea', '-f', type=str, default='trfm,t5', help='Feature list separated by comma, including molformer,t5,trfm,pst,dssp.')
    parser.add_argument('--params', '-p', type=str, default='Model/Pretrained_model_199_trfm,t5.pkl', help='Pretrained weights.')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_arguments()
    print(args)
    data_path = args.data
    # data_path = 'Dataset.csv'
    num_epochs = args.epoch
    out_dir = args.outdir
    model_dir = args.model_dir
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
    fea = args.fea # 'molformer,t5,trfm,pst,dssp'
    feature_list = fea.split(',')
    feature_dim_list = [feaD_dict[k] for k in feature_list] # [768,1024,1024,1280,13]

    df = pd.read_csv(data_path, sep='\t')
    print(f"Load {len(df)} samples from dataset!")
    protein_len = [len(a) for a in df['Sequence']]
    smiles_len = [len(a) for a in df['Smiles']]
    max_prolen = np.array(protein_len).max()
    max_smilen = np.array(smiles_len).max()
    train = df[df['Set']=='train']
    valid = df[df['Set']=='valid']
    test = df[df['Set']=='test']
    print(f"{len(train)} | {len(valid)} | {len(test)}")

    train_ds = KcatDataset(train, feature_list=feature_list, max_protein_len=max_prolen, max_smi_len=max_smilen)
    valid_ds = KcatDataset(valid, feature_list=feature_list, max_protein_len=max_prolen, max_smi_len=max_smilen)
    test_ds = KcatDataset(test, feature_list=feature_list, max_protein_len=max_prolen, max_smi_len=max_smilen)

    #seq_vocab_size = 128 # test_ds.seq_vocab_size
    seq_vocab_size = 28
    smi_vocab_size = test_ds.smi_vocab_size
    best_rmse = 100
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,num_workers=8, collate_fn=graph_collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=8, collate_fn=graph_collate_fn, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=8, collate_fn=graph_collate_fn, drop_last=True)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = MetaDecoder(seq_vocab_size, smi_vocab_size, feature_dim_list=feature_dim_list, hidden=hidden, num_layers=num_layers, protein_len=max_prolen, smi_len=max_smilen, dropout=dropout, kernel_size=kernel_size).to(device)
    PATH = args.params # f'Model/Pretrained_model_trfm,t5.pkl'
    model.load_state_dict(torch.load(PATH, map_location=device))
    # print(model)
    criterion = nn.MSELoss(reduction="mean") # nn.HuberLoss()# nn.MSELoss()
    test_loss, rmse, r2, mae, pearson_corr, y_pred, y_true = evaluate_model(model, test_loader, device, criterion)
    print(f"Testing | Loss: {test_loss:.4f} | RMSE: {rmse:.4f} | r2: {r2:.4f} | mae: {mae:.4f} | PCC: {pearson_corr:.4f}")
    if rmse <= best_rmse:
        best_model = model
         
    #train_X, train_Y = prepare_ds(train_loader, best_model)
    #test_X, test_Y = prepare_ds(test_loader, best_model)
    train_X, train_Y, train_identifiers, train_sequences, train_ec_numbers, train_types = prepare_ds(train_loader, best_model)
    test_X, test_Y, test_identifiers, test_sequences, test_ec_numbers, test_types = prepare_ds(test_loader, best_model)  
    print(f"Using pretrained. {train_X.shape}, {train_Y.shape}, {test_X.shape}, {test_Y.shape}")
    # ML_list = ['ET', 'RF', 'GBR', 'SVR', 'LR', 'Ridge', 'Lasso', 'KNN', 'DT', 'MLP']
    ML_list = ['ET', 'RF', 'GBR', 'SVR', 'LR', 'Lasso', 'KNN', 'DT', 'MLP']
    run_ML(ML_list, 'fine_tuned', train_X, train_Y, test_X, test_Y, test_identifiers, test_sequences, test_ec_numbers, test_types)

    train_X, train_Y, train_identifiers, train_sequences, train_ec_numbers, train_types = prepare_ds(train_loader, best_model, concat=True)
    test_X, test_Y, test_identifiers, test_sequences, test_ec_numbers, test_types = prepare_ds(test_loader, best_model, concat=True)
    print(f"Using pretrained and raw features. {train_X.shape}, {train_Y.shape}, {test_X.shape}, {test_Y.shape}")
    ML_list = ['ET']# , 'RF', 'GBR', 'LR']
    run_ML(ML_list, 'combined', train_X, train_Y, test_X, test_Y, test_identifiers, test_sequences, test_ec_numbers, test_types)

    train_X, train_Y, train_identifiers, train_sequences, train_ec_numbers, train_types = prepare_ds(train_loader, best_model, only_ML=True)
    test_X, test_Y, test_identifiers, test_sequences, test_ec_numbers, test_types = prepare_ds(test_loader, best_model, only_ML=True)
    print(f"Using only chosen features. {train_X.shape}, {train_Y.shape}, {test_X.shape}, {test_Y.shape}")
    ML_list = ['ET']
    run_ML(ML_list, 'Chosen', train_X, train_Y, test_X, test_Y, test_identifiers, test_sequences, test_ec_numbers, test_types)

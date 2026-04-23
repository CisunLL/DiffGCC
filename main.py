import warnings
warnings.filterwarnings('ignore')

import torch
import dgl
from encoder import DualEncoder 
from torch.optim import Adam
from diffusion import DiffusionModule
from utils import get_logger, fix_seed, evaluate, normalize, augment, load_data, structure_denoising

if __name__ == "__main__":
    datasets = ['Computer'] 
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        hyperparams = {
            'Cora':       {'batch_size': 4096, 'epochs': 200, 'diff_weight':1., 'lam': 0.8, 'gam': 1.8, 'beta': 0.05, 'sim_thres': 0.05, 's': 0.6, 'tau': 0.1, 'hid_dims': [256, 64], 'lr': 1e-3, 'wd': 5e-5, 'pd1': 0.1, 'pd2': 0.2, 'pm1': 0.1, 'pm2': 0.2, 'heads': 2, 'dropout': 0.1},
            'Computer':   {'batch_size': 2048, 'epochs': 400, 'diff_weight': 3., 'lam': 1.1, 'gam': 1., 'beta': 0.05, 'sim_thres': 0.1, 's': 0.6, 'tau': 0.3, 'hid_dims': [256, 128], 'lr': 1e-3, 'wd': 1e-5, 'pd1': 0.6, 'pd2': 0.8, 'pm1': 0.0, 'pm2': 0.1, 'heads': 1, 'dropout': 0.0},
        }

        if dataset not in hyperparams:
             hyperparams[dataset] = hyperparams['Cora']

        G, A_full, X_full, Y = load_data(dataset)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        params = hyperparams[dataset]
        batch_size = params.get('batch_size', 4096)
        
        epochs = params['epochs']
        if G.num_nodes() > batch_size:
            epochs = int(epochs * 1.5)

        lam, gam, beta = params['lam'], params['gam'], params.get('beta', 0.1)
        sim_thres = params.get('sim_thres', 0.1)
        s, tau = params['s'], params['tau']
        hid_dims = params['hid_dims']
        lr, wd = params['lr'], params['wd']
        pd1, pd2, pm1, pm2 = params['pd1'], params['pd2'], params['pm1'], params['pm2']
        heads = params.get('heads', 1)
        dropout = params.get('dropout', 0.0)
        diff_dim = 512 
        diff_weight = params.get('diff_weight', 1)

        trans_layers = 1
        trans_heads = 1
        trans_dropout = 0.5 

        logger = get_logger(f'./{dataset}_Hetero_Diffusion.log')
        
        fix_seed(0)
        encoder = DualEncoder(
            in_dim=X_full.shape[1], 
            hid_dims=hid_dims, 
            heads=heads, 
            dropout=dropout,
            trans_num_layers=trans_layers,
            trans_heads=trans_heads,
            trans_dropout=trans_dropout
        ).to(device)
        diffusion_model = DiffusionModule(
            in_dim=hid_dims[-1], 
            hid_dim=diff_dim, 
            timesteps=1000, 
            device=device
        ).to(device)
        optimizer = Adam(
            list(encoder.parameters()) + list(diffusion_model.parameters()), 
            lr=lr, 
            weight_decay=wd
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.5)
        
        G = G.to(device) 
        num_nodes = G.num_nodes()

        for epoch in range(epochs):
            encoder.train()
            diffusion_model.train() 
            
            indices = torch.randperm(num_nodes).to(device)
            epoch_stats = {'loss': 0, 'ali': 0, 'nei': 0, 'hetero': 0, 'diff': 0}
            num_batches = (num_nodes + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_nodes)
                batch_nodes = indices[start_idx : end_idx]
                
                batch_g = dgl.node_subgraph(G, batch_nodes)
                X_batch = batch_g.ndata['feat']
                A_batch = batch_g.adj_external().to(device) 
                
                A_norm_for_attn = normalize(A_batch, add_self_loops=True) 

                encoder.eval()
                with torch.no_grad():
                    _, attn_weights = encoder.embed(A_norm_for_attn, X_batch, return_attn=True)
                encoder.train()
                
                attn_indices = A_norm_for_attn._indices()
                is_real_neighbor = attn_indices[0] != attn_indices[1]
                neighbor_indices = attn_indices[:, is_real_neighbor]
                neighbor_weights = attn_weights[is_real_neighbor]
                
                if neighbor_weights.sum() > 0:
                    neighbor_weights = neighbor_weights / neighbor_weights.mean()
                    neighbor_weights = neighbor_weights.clamp(max=5.0) 
                
                A_denoised = structure_denoising(A_batch, X_batch, threshold=sim_thres)

                A1, X1 = augment(A_denoised, X_batch, pd1, pm1)
                A2, X2 = augment(A_denoised, X_batch, pd2, pm2)
                
                A_norm1 = normalize(A1, add_self_loops=True)
                A_norm2 = normalize(A2, add_self_loops=True)
                

                Z1, Z2, Z1_loc, Z1_glob, Z2_loc, Z2_glob = encoder(A_norm1, A_norm2, X1, X2)
                
                if torch.isnan(Z1).any() or torch.isinf(Z1).any():
                    continue 

                S = Z1 @ Z2.T

                
                if neighbor_indices.shape[1] > 0:
                    s_neighbors = S[neighbor_indices[0], neighbor_indices[1]]
                    loss_nei = - (s_neighbors * neighbor_weights).mean()
                else:
                    loss_nei = torch.tensor(0.0).to(device)

                mask = torch.full(S.shape, True, device=device)
                if neighbor_indices.shape[1] > 0:
                    mask[neighbor_indices[0], neighbor_indices[1]] = False
                mask.fill_diagonal_(False)
                S_masked = torch.masked_select(S, mask)
                if S_masked.shape[0] > 0:
                    loss_spa = torch.sigmoid((S_masked - s) / tau).mean()
                else:
                    loss_spa = torch.tensor(0.0).to(device)

                loss_hetero_1 = - (Z1_loc * Z1_glob).sum(dim=1).mean()
                loss_hetero_2 = - (Z2_loc * Z2_glob).sum(dim=1).mean()
                loss_hetero = (loss_hetero_1 + loss_hetero_2) * 0.5
                loss_diff_1 = diffusion_model(Z1) 
                loss_diff = (loss_diff_1 + loss_diff_2) * 0.5 

                loss = loss_ali + lam * loss_nei + gam * loss_spa + beta * loss_hetero + diff_weight * loss_diff
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            
        print("\nOptimization Finished! Evaluating...")
        encoder.eval()
        
        A_norm_full = normalize(A_full.to(device), add_self_loops=True)
        with torch.no_grad():
            Z = encoder.embed(A_norm_full, X_full.to(device))
            Z = torch.nan_to_num(Z, nan=0.0)

        evaluate(Z.cpu().numpy(), Y.cpu().numpy(), logger)
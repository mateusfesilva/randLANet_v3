import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import open3d as o3d
from torch.utils.data import Dataset, DataLoader

# Importa as classes e funções necessárias
from network.RandLANet import Network
from dataset_aereo_full_test import datasetAereoFullTest 
from utils.config import ConfigSemanticKITTI as cfg
from utils.data_process import DataProcessing as DP

# Esta função prepara os dados geométricos para a RandLA-Net
def tf_map_test(batch_pc):
    features = batch_pc
    input_points, input_neighbors, input_pools, input_up_samples = [], [], [], []
    for i in range(cfg.num_layers):
        neighbour_idx = DP.knn_search(batch_pc[:, :, :3], batch_pc[:, :, :3], cfg.k_n)
        sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
        pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
        up_i = DP.knn_search(sub_points[:, :, :3], batch_pc[:, :, :3], 1)
        input_points.append(batch_pc)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        batch_pc = sub_points
    input_list = input_points + input_neighbors + input_pools + input_up_samples + [features]
    return input_list

# Função collate para agrupar os patches em um lote
def collate_fn_test_full(batch):
    patch_points = [item[0] for item in batch]
    patch_indices = [item[1] for item in batch]
    
    patch_points = np.stack(patch_points)
    flat_inputs = tf_map_test(patch_points)
    
    num_layers = cfg.num_layers
    inputs = {}
    inputs['xyz'] = [torch.from_numpy(i).float() for i in flat_inputs[:num_layers]]
    inputs['neigh_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[num_layers: 2 * num_layers]]
    inputs['sub_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[2 * num_layers:3 * num_layers]]
    inputs['interp_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[3 * num_layers:4 * num_layers]]
    inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()

    return inputs, patch_indices

# Classe auxiliar para gerar patches sob demanda.
class PatchDataset(Dataset):
    def __init__(self, points, tree, visit_counts, num_votes):
        self.points = points
        self.tree = tree
        self.visit_counts = visit_counts
        self.num_votes = num_votes

    def __len__(self):
        # A estimativa de tamanho é importante para o DataLoader
        num_patches = int(np.ceil(len(self.points) * self.num_votes / cfg.num_points))
        return num_patches

    def __getitem__(self, item):
        center_idx = np.argmin(self.visit_counts)
        center_point = self.points[center_idx, :3].reshape(1, -1)
        patch_indices = self.tree.query(center_point, k=cfg.num_points, return_distance=False)[0]
        patch_points = self.points[patch_indices].copy()
            
        # Normaliza o patch
        patch_tensor  = torch.from_numpy(patch_points)
        xyz_tensor = patch_tensor[:, :3]
        xyz_mean = torch.mean(xyz_tensor, dim=0, keepdim=True)
        patch_tensor[:, :3] = xyz_tensor - xyz_mean
        patch_points_normalized = patch_tensor.numpy()
        
        if self.points.shape[1] > 3:
            intensity_col = patch_points_normalized[:, 3]
            min_i, max_i = np.min(intensity_col), np.max(intensity_col)
            if max_i - min_i > 0:
                patch_points_normalized[:, 3] = (intensity_col - min_i) / (max_i - min_i)
        # 'yield' entrega um patch de cada vez para o DataLoader
        return patch_points_normalized, patch_indices
            
            
class Tester:
    def __init__(self, model, checkpoint_path):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Arquivo de checkpoint não encontrado em '{checkpoint_path}'")
        
        print(f"Carregando checkpoint de '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        print("Modelo carregado com sucesso.")

        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            print(f"Usando {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
            
    def test(self, test_dataset, result_dir, num_votes=10, batch_size=4):
        self.model.eval()
        save_path = Path(result_dir)
        save_path.mkdir(exist_ok=True)
        color_map = {0: [0.5, 0.5, 0.5], 1: [0, 1, 0], 2: [1, 0, 0]}

        with torch.no_grad():
            for full_points, full_tree, file_path in tqdm(test_dataset, desc="Processando nuvens"):
                num_points_full = len(full_points)
                probabilities = np.zeros((num_points_full, cfg.num_classes), dtype=np.float32)
                visit_counts = np.zeros(num_points_full, dtype=np.uint8)

                patch_dataset = PatchDataset(full_points, full_tree, visit_counts, num_votes)
                patch_loader = DataLoader(patch_dataset, batch_size=batch_size, collate_fn=collate_fn_test_full, num_workers=0)

                for inputs, patch_indices_batch in tqdm(patch_loader, desc=f"Arquivo {Path(file_path).name}", leave=False):
                    # Atualiza o contador de visitas
                    for patch_indices in patch_indices_batch:
                        visit_counts[patch_indices] += 1

                    # Move dados para a GPU
                    for key in inputs:
                        if isinstance(inputs[key], list):
                            for i in range(len(inputs[key])):
                                inputs[key][i] = inputs[key][i].to(self.device)
                        else:
                            inputs[key] = inputs[key].to(self.device)

                    end_points = self.model(inputs)

                    # --- AGREGAÇÃO CORRIGIDA PARA SOMA ---
                    logits_batch = end_points['logits'].transpose(1, 2)
                    probs_batch = torch.nn.functional.softmax(logits_batch, dim=2).cpu().numpy()

                    for b_idx, patch_indices in enumerate(patch_indices_batch):
                        # Simplesmente SOMA as novas probabilidades às acumuladas
                        probabilities[patch_indices] += probs_batch[b_idx]
                    # --- FIM DA CORREÇÃO ---

                final_preds = np.argmax(probabilities, axis=1).astype(np.uint8)
                self.save_results(full_points, final_preds, file_path, save_path, color_map)

        print(f"Teste finalizado. Resultados salvos em '{save_path}'.")

    def prepare_patch_inputs(self, patch_points):
        patch_points_expanded = np.expand_dims(patch_points, axis=0)
        
        features = patch_points_expanded
        input_points, input_neighbors, input_pools, input_up_samples = [], [], [], []
        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(features[:, :, :3], features[:, :, :3], cfg.k_n)
            sub_points = features[:, :features.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :features.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points[:, :, :3], features[:, :, :3], 1)
            input_points.append(features)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            features = sub_points
        
        flat_inputs = input_points + input_neighbors + input_pools + input_up_samples + [patch_points_expanded]
        
        inputs = {}
        num_layers = cfg.num_layers
        inputs['xyz'] = [torch.from_numpy(i).float() for i in flat_inputs[:num_layers]]
        inputs['neigh_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[num_layers: 2 * num_layers]]
        inputs['sub_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[2 * num_layers:3 * num_layers]]
        inputs['interp_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[3 * num_layers:4 * num_layers]]
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        return inputs

    def save_results(self, points, predictions, file_path, save_path, color_map):
        # (Este método permanece o mesmo)
        colors_list = [color_map.get(l, [1.0, 1.0, 1.0]) for l in predictions]
        colors_array = np.array(colors_list, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors_array)
        file_name = f"pred_{Path(file_path).stem}.ply"
        o3d.io.write_point_cloud(str(save_path / file_name), pcd)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Caminho para o checkpoint (.tar)')
    parser.add_argument('--test_path', required=True, help='Caminho para a pasta da sequência de teste (ex: .../sequences/99)')
    parser.add_argument('--result_dir', default='results/', help='Pasta para salvar as previsões .ply')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size para a inferência de patches')
    FLAGS = parser.parse_args()

    test_dataset = datasetAereoFullTest(test_path=FLAGS.test_path, config=cfg)
    model = Network(cfg)
    
    tester = Tester(model=model, checkpoint_path=FLAGS.checkpoint_path)
    tester.test(test_dataset=test_dataset, result_dir=FLAGS.result_dir, batch_size=FLAGS.batch_size)

if __name__ == '__main__':
    main()
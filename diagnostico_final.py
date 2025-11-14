import torch
import numpy as np
import argparse
from pathlib import Path

# Importe suas classes e configs
from network.RandLANet import Network
from dataset_aereo_full_test import datasetAereoFullTest 
from utils.config import ConfigSemanticKITTI as cfg
from utils.data_process import DataProcessing as DP

def print_batch_report(batch_name, inputs):
    """Função auxiliar para imprimir um relatório detalhado de um lote de dados."""
    print(f"\n--- RELATÓRIO DO LOTE: {batch_name} ---")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  - Chave '{key}':")
            print(f"    - Shape: {value.shape}")
            print(f"    - Tipo (dtype): {value.dtype}")
            if torch.is_floating_point(value):
                print(f"    - Min: {torch.min(value):.4f}, Max: {torch.max(value):.4f}, Média: {torch.mean(value):.4f}, Desv. Padrão: {torch.std(value):.4f}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"  - Chave '{key}' (Lista de tensores):")
            for i, tensor in enumerate(value):
                 print(f"    - Item [{i}] Shape: {tensor.shape}, Tipo: {tensor.dtype}")
    print("-" * (25 + len(batch_name)))

def run_inference_on_batch(model, inputs):
    """Executa a inferência em um lote e reporta as classes previstas."""
    device = next(model.parameters()).device
    for key in inputs:
        if isinstance(inputs[key], list):
            for i in range(len(inputs[key])):
                inputs[key][i] = inputs[key][i].to(device)
        else:
            inputs[key] = inputs[key].to(device)
    
    with torch.no_grad():
        end_points = model(inputs)
    
    preds = torch.argmax(end_points['logits'], dim=1).cpu().numpy().flatten()
    unique, counts = np.unique(preds, return_counts=True)
    
    print("\n--- RESULTADO DA INFERÊNCIA PARA ESTE LOTE ---")
    print("  Classes previstas:")
    if len(unique) == 0:
        print("    - Nenhuma previsão foi gerada.")
    else:
        for cls, count in zip(unique, counts):
            print(f"    - Classe {cls}: {count:,} pontos")
    print("-" * 45)

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

def prepare_patch_inputs(patch_points):
    """Função auxiliar que substitui a collate_fn para um único patch."""
    patch_points_expanded = np.expand_dims(patch_points, axis=0)
    flat_inputs = tf_map_test(patch_points_expanded)
    
    inputs = {}
    num_layers = cfg.num_layers
    inputs['xyz'] = [torch.from_numpy(i).float() for i in flat_inputs[:num_layers]]
    inputs['neigh_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[num_layers: 2 * num_layers]]
    inputs['sub_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[2 * num_layers:3 * num_layers]]
    inputs['interp_idx'] = [torch.from_numpy(i).long() for i in flat_inputs[3 * num_layers:4 * num_layers]]
    inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
    return inputs

def main():
    parser = argparse.ArgumentParser(description="Script de diagnóstico para o pipeline de teste.")
    parser.add_argument('--test_dir', required=True, help="Caminho para a pasta da sequência de TESTE.")
    parser.add_argument('--checkpoint_path', required=True, help='Caminho para o checkpoint.')
    FLAGS = parser.parse_args()

    # --- CARREGAR MODELO ---
    print(f"Carregando checkpoint de '{FLAGS.checkpoint_path}'...")
    model = Network(cfg)
    checkpoint = torch.load(FLAGS.checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print("Modelo carregado com sucesso.")

    # --- ANÁLISE DO DADO DE TESTE ---
    print("\n" + "="*50)
    print("Analisando uma amostra de dados de TESTE...")
    
    cfg_teste = cfg
    test_dataset = datasetAereoFullTest(test_path=FLAGS.test_dir, config=cfg_teste)
    
    if len(test_dataset) == 0:
        print(f"ERRO: Nenhum arquivo encontrado em: {FLAGS.test_dir}")
        return

    full_points, full_tree, _ = test_dataset[0]

    # --- SIMULAÇÃO DA PRIMEIRA ITERAÇÃO DO ROLLING PREDICT ---
    
    # 1. Recorta o patch
    center_idx = np.random.randint(0, len(full_points))
    center_point = full_points[center_idx, :3].reshape(1, -1)
    patch_indices = full_tree.query(center_point, k=cfg.num_points)[1][0]
    patch_points = full_points[patch_indices].copy()
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- ESTE É O BLOCO DE NORMALIZAÇÃO CRÍTICO ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("\n--- APLICANDO NORMALIZAÇÃO NO PATCH DE TESTE ---")
    
    # 2. Centraliza as coordenadas XYZ
    mean_xyz_before = np.mean(patch_points[:, :3], axis=0)
    patch_tensor = torch.from_numpy(patch_points)
    xyz_tensor = patch_tensor[:, :3]
    xyz_mean = torch.mean(xyz_tensor, dim=0, keepdim=True)
    patch_tensor[:, :3] = xyz_tensor - xyz_mean
    patch_points = patch_tensor.numpy()
    mean_xyz_after = np.mean(patch_points[:, :3], axis=0)
    print(f"Média XYZ antes: {mean_xyz_before} ---> Média XYZ depois: {mean_xyz_after}")
    print(patch_points[:, :3])

    # 3. Normaliza a Intensidade para [0, 1]
    if patch_points.shape[1] > 3:
        intensity_col = patch_points[:, 3]
        min_i, max_i = np.min(intensity_col), np.max(intensity_col)
        if max_i > min_i:
            patch_points[:, 3] = (intensity_col - min_i) / (max_i - min_i)
            print("Intensidade normalizada para o intervalo [0, 1].")
    
    print("--------------------------------------------------")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # 4. Prepara o lote para o modelo
    inputs = prepare_patch_inputs(patch_points)
    
    # 5. Imprime o relatório e a inferência
    print_batch_report("TESTE (1 Patch)", inputs)
    run_inference_on_batch(model, inputs)
    print("="*50)

if __name__ == "__main__":
    # Preencha a função tf_map_test com a sua implementação
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

    main()
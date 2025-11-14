import os
from pathlib import Path

class ConfigSemanticKITTI:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 4096 * 11  # Number of input points
    num_classes = 3  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter
    num_features = 4  # Number of input features (x, y, z, intensity)

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 5e-5  # initial learning rate
    # lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None
    
    def __init__(self, dataset_path=None):

        # Se um caminho for fornecido ao criar o objeto, use-o.
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            # Se nenhum caminho for fornecido, define um padrão ou lança um erro
            self.dataset_path = None
            print("AVISO: Nenhum 'dataset_path' fornecido na criação da configuração.")


class ConfigS3DIS:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 3  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


class ConfigSemantic3D:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 3  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 16  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 0.8

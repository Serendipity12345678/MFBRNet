import os
import torch


class Config:
    def __init__(self):
        dataset_dir = './data'
        self.dp = DataPath(dataset_dir)
        self.num_workers = 4

        self.CUDA = True
        self.device = torch.device('cuda' if self.CUDA else 'cpu')

        self.epochs = 180
        self.trainsize = 384
        self.batch_size = 16
        self.weight_decay = 4e-8
        self.learning_rate = 8.4e-5
        self.min_lr = 5e-7


class DataPath:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        ''' Train Dataset: CAMO-Train + COD10K-Train '''
        self.train_imgs = os.path.join(self.dataset_dir, 'TrainDataset', 'Imgs')
        self.train_masks = os.path.join(self.dataset_dir, 'TrainDataset', 'GT')
        self.train_edges = os.path.join(self.dataset_dir, 'TrainDataset', 'Edge')
        self.train_depth = os.path.join(self.dataset_dir, 'TrainDataset', 'Depth')

        ''' Test Dataset '''
        # CHAMELEON
        self.test_CHAMELEON_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'CHAMELEON', 'Imgs')
        self.test_CHAMELEON_masks = os.path.join(self.dataset_dir, 'TestDataset', 'CHAMELEON', 'GT')
        self.test_CHAMELEON_edges = os.path.join(self.dataset_dir, 'TestDataset', 'CHAMELEON', 'Edge')
        self.test_CHAMELEON_depth = os.path.join(self.dataset_dir, 'TestDataset', 'CHAMELEON', 'Depth')

        # CAMO-Test
        self.test_CAMO_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'CAMO', 'Imgs')
        self.test_CAMO_masks = os.path.join(self.dataset_dir, 'TestDataset', 'CAMO', 'GT')
        self.test_CAMO_edges = os.path.join(self.dataset_dir, 'TestDataset', 'CAMO', 'Edge')
        self.test_CAMO_depth = os.path.join(self.dataset_dir, 'TestDataset', 'CAMO', 'Depth')

        # COD10K-Test
        self.test_COD10K_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'COD10K', 'Imgs')
        self.test_COD10K_masks = os.path.join(self.dataset_dir, 'TestDataset', 'COD10K', 'GT')
        self.test_COD10K_edges = os.path.join(self.dataset_dir, 'TestDataset', 'COD10K', 'Edge')
        self.test_COD10K_depth = os.path.join(self.dataset_dir, 'TestDataset', 'COD10K', 'Depth')

        # NC4K
        self.test_NC4K_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'NC4K', 'Imgs')
        self.test_NC4K_masks = os.path.join(self.dataset_dir, 'TestDataset', 'NC4K', 'GT')
        self.test_NC4K_edges = os.path.join(self.dataset_dir, 'TestDataset', 'NC4K', 'Edge')
        self.test_NC4K_depth = os.path.join(self.dataset_dir, 'TestDataset', 'NC4K', 'Depth')

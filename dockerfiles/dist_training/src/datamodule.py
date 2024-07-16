import os
import h5py
import pandas as pd
import lightning as L

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import ISICDataset


class ISICDataModule(L.LightningDataModule):
    def __init__(self, data_path: str = "/data", batch_size: int = 48, num_workers: int = 2):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
       
    def setup(self, stage: str):
        path_train_hdf5 = os.path.join(self.data_path, "train-image.hdf5")
        path_train_meta = os.path.join(self.data_path, "train-metadata.csv")
        
        # load the HDF5 files
        hdf5_img = h5py.File(path_train_hdf5, 'r')
        # load the metadata
        meta_df = pd.read_csv(path_train_meta, low_memory=False)
        
        # split train/valid sets
        train_df, valid_df = train_test_split(meta_df, test_size=0.2)
        
        train_isic_ids = train_df['isic_id'].values
        valid_isic_ids = valid_df['isic_id'].values
        
        transformations = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_isic_dataset = ISICDataset(hdf5_img, train_isic_ids, train_df.target.values, transform=transformations)
        self.valid_isic_dataset = ISICDataset(hdf5_img, valid_isic_ids, valid_df.target.values, transform=transformations)
        
    def train_dataloader(self):
        return DataLoader(self.train_isic_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valid_isic_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


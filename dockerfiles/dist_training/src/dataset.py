from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, hdf5_file, isic_ids, targets=None, transform=None):
        self.hdf5_file = hdf5_file
        self.isic_ids = isic_ids
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]

        # convert byte array to PIL image
        image = Image.open(BytesIO(self.hdf5_file[isic_id][()]))
        
        if self.transform:
            image = self.transform(image)

        if self.targets is not None:
            target = self.targets[idx]
            return image, target
        else:
            return image

    def close(self):
        self.hdf5.close()


from torch.utils.data import Dataset, DataLoader
import os

import logging


class RanaDraytoniiDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.transform = transform

        # List all files in the directory
        logging.debug(f"Listing files in directory: {dir_path}")
        self.files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]

        if self.transform:
            audio_data = self.transform(audio_file)
            return audio_data
        else:
            return audio_file


def get_data_loader(directory, batch_size=32, shuffle=False):
    dataset = RanaDraytoniiDataset(directory)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

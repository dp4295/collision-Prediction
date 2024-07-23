from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):

         return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
            
        sample = {'input': torch.tensor(self.normalized_data[idx, :-1], dtype=torch.float32), 
              'label': torch.tensor(self.normalized_data[idx, :-1], dtype=torch.float32)}
        return sample

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()

        # Split dataset into train and test sets
        train_data, test_data = train_test_split(self.nav_dataset.normalized_data, test_size=0.2, random_state=42)

        # Create data loaders
        self.train_loader = data.DataLoader(dataset=DatasetWrapper(train_data), batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(dataset=DatasetWrapper(test_data), batch_size=batch_size, shuffle=False)


class DatasetWrapper(dataset.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'input': torch.tensor(self.data[idx, :-1], dtype=torch.float32),
                  'label': torch.tensor(self.data[idx, -1], dtype=torch.float32).view(1)}
        return sample


def main():
    batch_size = 64
    data_loaders = Data_Loaders(batch_size)

    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()

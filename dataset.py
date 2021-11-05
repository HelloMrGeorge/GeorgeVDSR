from torch.utils.data import Dataset
import torch

class TrainDataset(Dataset):
    def __init__(self, data_path, target_path):
        super().__init__()
        self.data = torch.load(data_path)
        self.target = torch.load(target_path)

    def __getitem__(self, index):
        return self.data[index].cuda(0), self.target[index].cuda(0)
        
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    d = TrainDataset()
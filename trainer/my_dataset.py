import sys
sys.path.extend(["sweep_config","trainer"])

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from sweep_config.config_generator import common_params

class myDataset(Dataset):
    def __init__(self, 
    name, 
    data_path, 
    train, 
    transforms, 
    download
    ):
        self.name = name
        self.data_path = data_path
        
        self.dataset = datasets.MNIST(
            root=self.data_path,
            train=train,
            transform=transforms,
            download=download
        )

        self.classes = [
            '0 - zero',
            '1 - one',
            '2 - two',
            '3 - three',
            '4 - four',
            '5 - five',
            '6 - six',
            '7 - seven',
            '8 - eight',
            '9 - nine',
        ]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        return X, y

    def __repr__(self):
        indent = 4
        template = []
        template.append(f"My Custom Dataset : {self.name}")
        template.append(f"Data path: {self.data_path}")
        template.append(f"Number of data: {self.__len__()}")
        template.append(f"Number of classes: {len(self.classes)}")
        X, y = self.__getitem__(0)
        template.append(f"Size of image: {X.shape}")
        template.append(f"Maxpoint of image: {X.max()}")
        template.append(f"Minpoint of image: {X.min()}")
        template.append(f"Item type of datapoint: {type(X.max().item())}")
        template.append(f"Type of label: {type(y)}")
        
        join_str = "\n" + " "*indent
        return join_str.join(template)
    
    def __str__(self):
        return self.__repr__()

if __name__ == "__main__":
    MD = myDataset(
        name=common_params["DATASET"]["value"],
        data_path=common_params["DATA_PATH"]["value"], 
        train=False, 
        transforms=transforms.ToTensor(), 
        download=True
    )
    print(MD)
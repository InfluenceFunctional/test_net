import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import tqdm

class build_dataset():
    def __init__(self,config):
        self.config = config
        self.data = self.generate_data()
        self.data_dimensions = self.get_dimension(self.data)


    def generate_data(self):
        np.random.seed(self.config.dataset_seed)
        data = {}
        data['samples'] = np.random.uniform(-1,1,size=(self.config.dataset_size, self.config.dataset_dimension))
        #data['values'] = np.sum(data['samples'],axis=1)
        #data['values'] = np.sum(data['samples'] ** 3, axis = 1)
        J_matrix = np.ones((self.config.dataset_dimension, self.config.dataset_dimension, self.config.dataset_dimension))
        for i in range(self.config.dataset_dimension):
            for j in range(self.config.dataset_dimension):
                for k in range(self.config.dataset_dimension):
                    J_matrix[i,j,k] = np.sin(i + j + k)

        print("Preparing synthetic dataset")
        inter = 3
        values = np.zeros(len(data['samples']))
        for i in tqdm.tqdm(range(len(data['samples']))):
            sample = data['samples'][i]
            for j in range(len(sample)):
                for k in range(max(0,j-inter),min(j+inter,len(sample))):
                    for l in range(max(0,k - inter), min(j+ inter, len(sample))):
                        values[i] += ((sample[j] + sample[k] * sample[l]) * J_matrix[j,k,l] /(np.abs(j- k - l) + 1)) ** 2 * np.log(j + 2)

        data['values'] = values
        data['values'] = (data['values'] - np.mean(data['values'])) / np.sqrt(np.var(data['values']))

        return data


    def get_dimension(self, data):
        dim = {
            'length' : len(data),
            'dimension' : data['samples'].shape[1],
            'mean' : np.mean(data['values']),
            'std dev' : np.sqrt(np.var(data['values'])),
        }

        return dim

    def reshuffle(self):
        samples = self.data['samples']
        values = self.data['values']
        samples,values = shuffle(samples, values, random_state = self.config.dataset_seed)
        self.data['samples'] = samples
        self.data['values'] = values

    def __getitem__(self, idx):
        return self.data['samples'][idx], self.data['values'][idx]

    def __len__(self):
        return len(self.data['samples'])

    def getFullDataset(self):
        return self.data


def build_dataloaders(dataset_builder, config):
    batch_size = config.batch_size
    dataset_builder.reshuffle()
    train_size = int(0.8 * len(dataset_builder))  # split data into training and test sets
    test_size = len(dataset_builder) - train_size

    train_dataset = []
    test_dataset = []

    for i in range(test_size, test_size + train_size):
        test_dataset.append(dataset_builder[i])
    for i in range(test_size):
        train_dataset.append(dataset_builder[i])

    tr = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    te = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return tr, te
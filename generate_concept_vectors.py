import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from os import listdir
from os.path import isfile, join

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--concept_dir", help="Directory where concept class training data files are stored")

args = parser.parse_args()


# Step 1: Read sentences from files A and B and create a list of dicts
def read_sentences_from_files(file_a_path, file_b_path):
    data = []
    
    with open(file_a_path, 'r') as file_a:
        sentences_a = file_a.readlines()
        data += [{'sentence': sentence.strip(), 'label': 0} for sentence in sentences_a]
    
    with open(file_b_path, 'r') as file_b:
        sentences_b = file_b.readlines()
        data += [{'sentence': sentence.strip(), 'label': 1} for sentence in sentences_b]
    
    return data


# Step 2: Create a custom PyTorch Dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {'sentence': sample['sentence'], 'label': sample['label']}


# Step 3: Create a PyTorch DataLoader for the custom dataset
def create_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



if __name__ == "__main__":

    trainfiles = [join(args.concept_dir, f) for f in listdir(args.concept_dir) if isfile(join(args.concept_dir, f))]

    # Step 1: Read sentences and assign labels
    data = read_sentences_from_files(trainfiles[0], trainfiles[1])

    # Step 2: Create a custom dataset
    dataset = CustomDataset(data)

    # Step 3: Create a data loader
    batch_size = 4
    data_loader = create_data_loader(dataset, batch_size)

    # You can now use the data_loader for training or other purposes
    for batch in data_loader:
        sentences = batch['sentence']
        labels = batch['label']
        print(sentences)
        print(labels)
        # Perform your training or data processing here

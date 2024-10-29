# Author: Nicola Francescon
#

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch.nn as nn

from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, resnet18, ResNet18_Weights

from wild_time_data import available_time_steps
from wild_time_data import load_dataset

import pickle
import random
import torch
import os
import time

def save_times(times_vector, labels, directory):
    times = pd.DataFrame([times_vector], columns=labels)
    with open(f'{directory}/Extraction times.pkl', 'wb') as f:
        pickle.dump(times, f)
    return

def feat_extract(folder_name, extractor = 'mobilenet_small'):
    '''
    Function to extract features from FMoW with a specified feature extractor.

    Args:
    folder_name (string): Directory to save the extracted features.
    extractor (string): One of 'mobilenet_small', 'resnet18', 'eurosat'
    '''
    random.seed(21100)
    final_length = 0
    data = []
    for i in available_time_steps("fmow"):
        data.append(load_dataset(dataset_name="fmow", split="train", time_step=i, data_dir=r"Dataset", transform=lambda x :x))
        final_length = final_length + data[i].size

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )]
    )

    if extractor == 'mobilenet_small':

        weights = MobileNet_V3_Small_Weights.DEFAULT
        feat_extract = mobilenet_v3_small(weights=weights)
    
    elif extractor == 'resnet18':

        weights = ResNet18_Weights.DEFAULT
        feat_extract = resnet18(weights=weights)
    
    elif extractor == 'eurosat':

        weights_path = 'features/eurosat/seco_resnet18_1m.ckpt'

        checkpoint = torch.load(weights_path)

        feat_extract = resnet18(num_classes=128)  # Adjust output dim to match your checkpoint

        model_dict = feat_extract.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}

        # Update the model's state_dict
        model_dict.update(pretrained_dict)

        # Load the new state_dict into the model
        feat_extract.load_state_dict(model_dict)

        feat_extract.eval()
        modules = list(feat_extract.children())[:-1]
        model = nn.Sequential(*modules)
        for p in model.parameters():
            p.requires_grad = False

    else:
        print('Feature extraction model not found')
        return

    if os.path.exists(folder_name):
        print('Features already extracted in the past')
        return
    
    feat_extract.eval()
    modules = list(feat_extract.children())[:-1]
    model = nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False
    t7 = time.perf_counter(), time.process_time()
    for year in available_time_steps("fmow"):
        print('Extracting year '+str(year+2002))
        features = []
        targets = []
        dataset = data[year]
        indices = list(range(len(dataset)))
        
        random.shuffle(indices)

        for idx in indices:
            img, labels = dataset[idx]
            img=transform(img)
            img = img.unsqueeze(0) 
            feat = model(img)
            feat = torch.flatten(feat)
            features.append(feat.squeeze().tolist())
            targets.append(labels.numpy())

        features = np.array(features)
        targets = np.array(targets)
        data_dict = {'features': features, 'targets': targets}

        with open(f'{folder_name}/fmow year_{year+2002}.pkl', 'wb') as f:
            pickle.dump(data_dict, f)

        print('Finished extracting year ' + str(year+2002))
    
    print('Ended feature extraction')
    t8 = time.perf_counter(), time.process_time()
    print(f" Total time for the feature extraction step: {t8[0] - t7[0]:.2f} seconds - {(t8[0] - t7[0])/final_length:.2f} seconds per sample")
    print(f" CPU time for the feature extraction step: {t8[1] - t7[1]:.2f} seconds - {(t8[1] - t7[1])/final_length:.2f} seconds per sample")
    save_times([t8[0]-t7[0], t8[1]-t7[1]], ['Total time for feature extraction', 'CPU time for feature extraction'], folder_name)
    return
import pickle

import os
os.chdir(r"C:\Users\nicol\OneDrive - Politecnico di Milano\Desktop\Magister\Tesi\FMOW\Single seed version\Thesis")

with open(f'saved_data/Original features/mobilenet_small/SLDA with Kalman model/Simulation times for 576 components.pkl', 'rb') as f:
        times = pickle.load(f)

print(f'Times for Kalman SLDA on features from MobileNet: {round(times.iloc[0,0],2)} seconds, {round(times.iloc[0,0]/60,2)} minutes, {round(times.iloc[0,0]/113291,2)} seconds per sample')

with open(f'saved_data/Original features/mobilenet_small/SLDA model/Simulation times for 576 components.pkl', 'rb') as f:
        times = pickle.load(f)

print(f'Times for SLDA on features from MobileNet: {round(times.iloc[0,0],2)} seconds, {round(times.iloc[0,0]/60,2)} minutes, {round(times.iloc[0,0]/113291,2)} seconds per sample')

with open(f'saved_data/Original features/mobilenet_small/Gaussian NB model/Simulation times for 576 components.pkl', 'rb') as f:
        times = pickle.load(f)

print(f'Times for Gaussian Naive Bayes on features from MobileNet: {round(times.iloc[0,0],2)} seconds, {round(times.iloc[0,0]/60,2)} minutes, {round(times.iloc[0,0]/113291,2)} seconds per sample')

with open(f'saved_data/Original features/mobilenet_small/Softmax regression model/Simulation times for 576 components.pkl', 'rb') as f:
        times = pickle.load(f)

print(f'Times for Softmax regression on features from MobileNet: {round(times.iloc[0,0],2)} seconds, {round(times.iloc[0,0]/60,2)} minutes, {round(times.iloc[0,0]/113291,2)} seconds per sample')

with open(f'features/initial_features/mobilenet_small/Extraction times.pkl', 'rb') as f:
        times = pickle.load(f)

print(f'Time to extract features through MobileNet v3 Small: {round(times.iloc[0,0],2)} seconds, {round(times.iloc[0,0]/60,2)} minutes, {round(times.iloc[0,0]/113291,2)} seconds per sample')

with open(f'features/initial_features/resnet18/Extraction times.pkl', 'rb') as f:
        times = pickle.load(f)

print(f'Time to extract features through ResNet18: {round(times.iloc[0,0],2)} seconds, {round(times.iloc[0,0]/60,2)} minutes, {round(times.iloc[0,0]/113291,2)} seconds per sample')


with open(f'saved_data/Original features/resnet18/SLDA with Kalman model/Simulation times for 512 components.pkl', 'rb') as f:
        times = pickle.load(f)

print(f'Times for Kalman SLDA on features from ResNet18: {round(times.iloc[0,0],2)} seconds, {round(times.iloc[0,0]/60,2)} minutes, {round(times.iloc[0,0]/113291,2)} seconds per sample')

with open(f'saved_data/Original features/resnet18/SLDA model/Simulation times for 512 components.pkl', 'rb') as f:
        times = pickle.load(f)

print(f'Times for SLDA on features from ResNet18: {round(times.iloc[0,0],2)} seconds, {round(times.iloc[0,0]/60,2)} minutes, {round(times.iloc[0,0]/113291,2)} seconds per sample')

with open(f'saved_data/Original features/resnet18/Gaussian NB model/Simulation times for 512 components.pkl', 'rb') as f:
        times = pickle.load(f)

print(f'Times for Gaussian Naive Bayes on features from ResNet18: {round(times.iloc[0,0],2)} seconds, {round(times.iloc[0,0]/60,2)} minutes, {round(times.iloc[0,0]/113291,2)} seconds per sample')

with open(f'saved_data/Original features/resnet18/Softmax regression model/Simulation times for 512 components.pkl', 'rb') as f:
        times = pickle.load(f)

print(f'Times for Softmax regression on features from ResNet18: {round(times.iloc[0,0],2)} seconds, {round(times.iloc[0,0]/60,2)} minutes, {round(times.iloc[0,0]/113291,2)} seconds per sample')

components_list = [30*i for i in range(1,18)]
models = ['Softmax regression', 'Gaussian NB', 'SLDA with Kalman', 'SLDA']

for comp in components_list:
        for model in models:
                with open(f'saved_data/RProj/mobilenet_small/{comp} components/{model} model/Simulation times for {comp} components.pkl', 'rb') as f:
                        times = pickle.load(f)
                with open(f'features/RProj/mobilenet_small/Dimensionality reduction times for {comp} components.pkl', 'rb') as f:
                        dim_red_times = pickle.load(f)
                with open(f'features/initial_features/mobilenet_small/Extraction times.pkl', 'rb') as f:
                        extraction_times = pickle.load(f)

                print(f'Times for {model} on {comp} features from MobileNet: {round(times.iloc[0,0]/113291 + dim_red_times.iloc[0,0]/113291 + extraction_times.iloc[0,0]/113291,2)} seconds per sample')

for comp in components_list:
        for model in models:
                with open(f'saved_data/RProj/resnet18/{comp} components/{model} model/Simulation times for {comp} components.pkl', 'rb') as f:
                        times = pickle.load(f)
                with open(f'features/RProj/resnet18/Dimensionality reduction times for {comp} components.pkl', 'rb') as f:
                        dim_red_times = pickle.load(f)
                with open(f'features/initial_features/resnet18/Extraction times.pkl', 'rb') as f:
                        extraction_times = pickle.load(f)

                print(f'Times for {model} on {comp} features from ResNet18: {round(times.iloc[0,0]/113291 + dim_red_times.iloc[0,0]/113291 + extraction_times.iloc[0,0]/113291,2)} seconds per sample')

for comp in components_list:
        for model in models:
                with open(f'saved_data/UMAP_batch/mobilenet_small/{comp} components/{model} model/Simulation times for {comp} components.pkl', 'rb') as f:
                        times = pickle.load(f)
                with open(f'features/UMAP_batch/mobilenet_small/Dimensionality reduction times for {comp} components.pkl', 'rb') as f:
                        dim_red_times = pickle.load(f)
                with open(f'features/initial_features/mobilenet_small/Extraction times.pkl', 'rb') as f:
                        extraction_times = pickle.load(f)

                print(f'Times for {model} on {comp} features from MobileNet: {round(times.iloc[0,0]/113291 + dim_red_times.iloc[0,0]/113291 + extraction_times.iloc[0,0]/113291,2)} seconds per sample')

for comp in components_list:
        for model in models:
                with open(f'saved_data/UMAP_batch/resnet18/{comp} components/{model} model/Simulation times for {comp} components.pkl', 'rb') as f:
                        times = pickle.load(f)
                with open(f'features/UMAP_batch/resnet18/Dimensionality reduction times for {comp} components.pkl', 'rb') as f:
                        dim_red_times = pickle.load(f)
                with open(f'features/initial_features/resnet18/Extraction times.pkl', 'rb') as f:
                        extraction_times = pickle.load(f)

                print(f'Times for {model} on {comp} features from ResNet18: {round(times.iloc[0,0]/113291 + dim_red_times.iloc[0,0]/113291 + extraction_times.iloc[0,0]/113291,2)} seconds per sample')
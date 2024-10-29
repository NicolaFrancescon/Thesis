import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

list_labels = ('airport', 'airport hangar', 'airport terminal', 'amusement park', 'aquaculture', 'archaeological site', 'barn', 
                   'border checkpoint', 'burial site', 'car dealership', 'construction site', 'crop field', 'dam', 'debris/rubble', 
                   'educational institution', 'electric substation', 'factory/powerplant', 'fire station', 'flooded road', 'fountain', 
                   'gas station', 'golf course', 'ground transportation station', 'helipad', 'hospital', 'impoverished settlement', 
                   'interchange', 'lake/pond', 'lighthouse', 'military facility', 'multi-unit residential', 'nuclear powerplant', 
                   'office building', 'oil/gas facility', 'park', 'parking lot/garage', 'place of worship', 'police station', 'port', 
                   'prison', 'race track', 'railway bridge', 'recreational facility', 'road bridge', 'runway', 'shipyard', 'shopping mall', 
                   'single-unit residential', 'smokestack', 'solar farm', 'space facility', 'stadium', 'storage tank', 'surface mine', 
                   'swimming pool', 'toll booth', 'tower', 'tunnel opening', 'waste disposal', 'water treatment facility', 'wind farm', 'zoo')
labels = {}
for i, name in enumerate(list_labels):
    labels[i] = name

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def compare_times_multiple_feat_extract (comparison_type='components', feature_extractor = None,
                                     dim_reduction = None, components=0, window_size=0):

    """
    Method to compare results produced by different runs on FMOW.
    
    Args:
        comparison_type (string): One of 'components', 'window_size'.
                                    It must specify the category for which a comparison is wanted.
                                    Once the category is chosen, the consequential argument MUST have multiple inputs in a list format.
        feature_extractor (string): One of 'mobilenet_small', 'resnet18'.
        dim_reduction (string): One of 'RProj', 'UMAP_batch', 'UMAP_streaming', 'GaussianRProj'.
        components (int): The embedding size for the reduced datasets.
        window_size (int): Required only for 'Streaming_UMAP' reduction method. The size of the moving window.
    """    
    convert = {
        'feature_extraction': feature_extractor,
        'dim_reduction': dim_reduction,
        'components': components,
        'window_size': window_size,
    }

    if comparison_type not in ['components', 'window_size']:
            print("Choose a proper comparison metric")
            return

    models = {}
    for i in range(len(convert[comparison_type])):
        for j in range(len(feature_extractor)):
            model_index = i * len(feature_extractor) + j
            models[f'model{model_index}'] = {
                'feature_extraction': feature_extractor[j],
                'dim_reduction': dim_reduction,
                'components': components[i] if comparison_type == 'components' else components,
                'window_size': window_size[i] if comparison_type == 'window_size' else window_size,
            }

    if not all(model['feature_extraction'] != None for model in models.values()):
        print('The model on the original images is not computed')
        return
    
    elif all(model['feature_extraction'] != None for model in models.values()) and not all(model['dim_reduction'] != None for model in models.values()):
        if not all(model['feature_extraction'] in ['mobilenet_small', 'resnet18'] for model in models.values()):
            print("Feature extractor not found")
            return
        print('Retrieving data from the directories...')
        for model in models.values():
            if model['dim_reduction'] == None:
                model['dim_reduction'] = 'Original features' 

    else: 
        if not all(model['feature_extraction'] in ['mobilenet_small', 'resnet18'] for model in models.values()):
            print("Feature extractor not found")
            return
        if not all(model['dim_reduction'] in ['RProj', 'UMAP_batch', 'UMAP_streaming', 'GaussianRProj'] for model in models.values()):
            print("Dimensionality reduction method not found")
            return
        print('Retrieving data from the directories...')

    fig_directories = []
    for model in models.values():
        if model['dim_reduction'] == 'Original features':
            directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}"
        elif model['dim_reduction'] == 'UMAP_streaming':
            directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components/{model['window_size']} window length"
        else: 
           directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components"
        fig_directories.append(directory)
    for directory in fig_directories:
        if not os.path.exists(directory):
            print('Requested folders are not available. Check if the run was executed and/or the informations provided are correct')
            print(directory)
            return
        
    data_directories = []
    for model in models.values():
        if model['dim_reduction'] == 'UMAP_streaming':
            directory = f"features/{model['dim_reduction']}/{model['feature_extraction']}/{model['window_size']} window length"
        else: 
           directory = f"features/{model['dim_reduction']}/{model['feature_extraction']}"
        data_directories.append(directory)
    for directory in data_directories:
        if not os.path.exists(directory):
            print('Requested data are not available. Check if the run was executed and/or the informations provided are correct')
            print(directory)
            return
        
    compare_times = []
    num_methods = len(feature_extractor) 
    current_row = [None] * num_methods
    i=0
    for j, directory in enumerate(data_directories):
        with open(f'{directory}/Dimensionality reduction times for {convert[comparison_type][i]} components.pkl', 'rb') as f:
            time = pickle.load(f).iloc[0, 0] / 60

        method_index = j % num_methods  
        current_row[method_index] = time

        if method_index == num_methods - 1:
            compare_times.append(current_row)
            i=i+1
            current_row = [None] * num_methods

    print('Information correctly recovered')

    if comparison_type == 'components':
        plt.figure(figsize=(10, 6))
        compare_times = pd.DataFrame(compare_times)
        for i, time in enumerate(compare_times.columns):
            sns.lineplot(x=components, y=compare_times[time], marker='o', label=f'Time for reduction with features from {feature_extractor[i]}')
        plt.xlabel('Number of components', fontsize=16)
        plt.ylabel('Dimensionality reduction times', fontsize=16)
        plt.title('Dimensionality reduction times across different embedding sizes', fontsize=18)
        plt.xticks(components, fontsize=14)
        plt.ylim([0, 4000/60])
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right', framealpha=1, fontsize=13)
        plt.grid(True)
        plt.tight_layout()
        filename = f"compare_multiple_dim_red_times_across_different_components_diff_features.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()
    
    if comparison_type == 'window_size':
        plt.figure(figsize=(10, 6))
        compare_times = pd.DataFrame(compare_times)
        for i, time in enumerate(compare_times.columns):
            sns.lineplot(x=window_size, y=compare_times[time], marker='o', label=f'Time for reduction with features from {feature_extractor[i]}')
        plt.ylabel('Dimensionality reduction times', fontsize=16)
        plt.xlabel('Window size of the embedding', fontsize=16)
        plt.title('Dimensionality reduction times across different window sizes of the embedding', fontsize=18)
        plt.xticks(window_size, fontsize=14)
        plt.ylim([0, 4000/60])
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right', framealpha=1, fontsize=13)
        plt.grid(True)
        plt.tight_layout()        
        filename = f"compare_multiple_dim_red_times_across_different_window_sizes_diff_features.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()

def compare_times_multiple_dim_reductions (comparison_type='components', feature_extractor = None,
                                     dim_reduction = None, components=0, window_size=0):

    """
    Method to compare results produced by different runs on FMOW.
    
    Args:
        comparison_type (string): One of 'components', 'window_size'.
                                    It must specify the category for which a comparison is wanted.
                                    Once the category is chosen, the consequential argument MUST have multiple inputs in a list format.
        feature_extractor (string): One of 'mobilenet_small', 'resnet18'.
        dim_reduction (string): One of 'RProj', 'UMAP_batch', 'UMAP_streaming', 'GaussianRProj'.
        components (int): The embedding size for the reduced datasets.
        window_size (int): Required only for 'Streaming_UMAP' reduction method. The size of the moving window.
    """    
    convert = {
        'feature_extraction': feature_extractor,
        'dim_reduction': dim_reduction,
        'components': components,
        'window_size': window_size,
    }

    if comparison_type not in ['components', 'window_size']:
            print("Choose a proper comparison metric")
            return

    models = {}
    for i in range(len(convert[comparison_type])):
        for j in range(len(dim_reduction)):
            model_index = i * len(dim_reduction) + j
            models[f'model{model_index}'] = {
                'feature_extraction': feature_extractor,
                'dim_reduction': dim_reduction[j],
                'components': components[i] if comparison_type == 'components' else components,
                'window_size': window_size[i] if comparison_type == 'window_size' else window_size,
            }

    if not all(model['feature_extraction'] != None for model in models.values()):
        print('The model on the original images is not computed')
        return
    
    elif all(model['feature_extraction'] != None for model in models.values()) and not all(model['dim_reduction'] != None for model in models.values()):
        if not all(model['feature_extraction'] in ['mobilenet_small', 'resnet18'] for model in models.values()):
            print("Feature extractor not found")
            return
        print('Retrieving data from the directories...')
        for model in models.values():
            if model['dim_reduction'] == None:
                model['dim_reduction'] = 'Original features' 

    else: 
        if not all(model['feature_extraction'] in ['mobilenet_small', 'resnet18'] for model in models.values()):
            print("Feature extractor not found")
            return
        if not all(model['dim_reduction'] in ['RProj', 'UMAP_batch', 'UMAP_streaming', 'GaussianRProj'] for model in models.values()):
            print("Dimensionality reduction method not found")
            return
        print('Retrieving data from the directories...')

    fig_directories = []
    for model in models.values():
        if model['dim_reduction'] == 'Original features':
            directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}"
        elif model['dim_reduction'] == 'UMAP_streaming':
            directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components/{model['window_size']} window length"
        else: 
           directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components"
        fig_directories.append(directory)
    for directory in fig_directories:
        if not os.path.exists(directory):
            print('Requested folders are not available. Check if the run was executed and/or the informations provided are correct')
            print(directory)
            return
        
    data_directories = []
    for model in models.values():
        if model['dim_reduction'] == 'UMAP_streaming':
            directory = f"features/{model['dim_reduction']}/{model['feature_extraction']}/{model['window_size']} window length"
        else: 
           directory = f"features/{model['dim_reduction']}/{model['feature_extraction']}"
        data_directories.append(directory)
    for directory in data_directories:
        if not os.path.exists(directory):
            print('Requested data are not available. Check if the run was executed and/or the informations provided are correct')
            print(directory)
            return
        
    compare_times = []
    num_methods = len(dim_reduction) 
    current_row = [None] * num_methods
    i=0
    for j, directory in enumerate(data_directories):
        with open(f'{directory}/Dimensionality reduction times for {convert[comparison_type][i]} components.pkl', 'rb') as f:
            time = pickle.load(f).iloc[0, 0] / 60

        method_index = j % num_methods  
        current_row[method_index] = time

        if method_index == num_methods - 1:
            compare_times.append(current_row)
            i=i+1
            current_row = [None] * num_methods

    print('Information correctly recovered')

    if comparison_type == 'components':
        plt.figure(figsize=(10, 6))
        compare_times = pd.DataFrame(compare_times)
        for i, time in enumerate(compare_times.columns):
            sns.lineplot(x=components, y=compare_times[time], marker='o', label=f'Time for reduction with {dim_reduction[i]}')
        plt.xlabel('Number of components', fontsize=16)
        plt.ylabel('Dimensionality reduction times', fontsize=16)
        plt.title('Dimensionality reduction times across different embedding sizes', fontsize=18)
        plt.xticks(components, fontsize=14)
        plt.ylim([0, 4000/60])
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right', framealpha=1, fontsize=13)
        plt.grid(True)
        plt.tight_layout()
        filename = f"compare_multiple_dim_red_times_across_different_components.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()
    
    if comparison_type == 'window_size':
        plt.figure(figsize=(10, 6))
        compare_times = pd.DataFrame(compare_times)
        for i, time in enumerate(compare_times.columns):
            sns.lineplot(x=window_size, y=compare_times[time], marker='o', label=f'Time for reduction with {dim_reduction[i]}')
        plt.ylabel('Dimensionality reduction times', fontsize=16)
        plt.xlabel('Window size of the embedding', fontsize=16)
        plt.title('Dimensionality reduction times across different window sizes of the embedding', fontsize=18)
        plt.xticks(window_size, fontsize=14)
        plt.ylim([0, 4000/60])
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right', framealpha=1, fontsize=13)
        plt.grid(True)
        plt.tight_layout()        
        filename = f"compare_multiple_dim_red_times_across_different_window_sizes.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()

def compare_times_multiple_methods (comparison_type='components', feature_extractor = None,
                                     dim_reduction = None, components=0, window_size=0, classification_method = None):

    """
    Method to compare results produced by different runs on FMOW.
    
    Args:
        comparison_type (string): One of 'components', 'window_size'.
                                    It must specify the category for which a comparison is wanted.
                                    Once the category is chosen, the consequential argument MUST have multiple inputs in a list format.
                                    Once the category is chosen, the consequential argument MUST have multiple inputs in a list format.
        feature_extractor (string): One of 'mobilenet_small', 'resnet18'.
        dim_reduction (string): One of 'RProj', 'UMAP_batch', 'UMAP_streaming', 'GaussianRProj'.
        components (int): The embedding size for the reduced datasets.
        window_size (int): Required only for 'Streaming_UMAP' reduction method. The size of the moving window.
        classification_method (string): One of 'Gaussian NB', 'Softmax regression', 'SLDA', 'SLDA with Kalman'.
    """    
    convert = {
        'feature_extraction': feature_extractor,
        'dim_reduction': dim_reduction,
        'components': components,
        'window_size': window_size,
        'classification_method': classification_method,
    }

    if comparison_type not in ['components', 'window_size']:
            print("Choose a proper comparison metric")
            return

    models = {}
    for i in range(len(convert[comparison_type])):
        for j in range(len(classification_method)):
            model_index = i * len(classification_method) + j
            models[f'model{model_index}'] = {
                'feature_extraction': feature_extractor,
                'dim_reduction': dim_reduction,
                'components': components[i] if comparison_type == 'components' else components,
                'window_size': window_size[i] if comparison_type == 'window_size' else window_size,
                'classification_method': classification_method[j],
            }

    if not all(model['feature_extraction'] != None for model in models.values()):
        print('The model on the original images is not computed')
        return
    
    elif all(model['feature_extraction'] != None for model in models.values()) and not all(model['dim_reduction'] != None for model in models.values()):
        if not all(model['feature_extraction'] in ['mobilenet_small', 'resnet18'] for model in models.values()):
            print("Feature extractor not found")
            return
        print('Retrieving data from the directories...')
        for model in models.values():
            if model['dim_reduction'] == None:
                model['dim_reduction'] = 'Original features' 

    else: 
        if not all(model['feature_extraction'] in ['mobilenet_small', 'resnet18'] for model in models.values()):
            print("Feature extractor not found")
            return
        if not all(model['dim_reduction'] in ['RProj', 'UMAP_batch', 'UMAP_streaming', 'GaussianRProj'] for model in models.values()):
            print("Dimensionality reduction method not found")
            return
        if not all(model['classification_method'] in ['Gaussian NB', 'Softmax regression', 'SLDA', 'SLDA with Kalman'] for model in models.values()):
            print("Classification method not found")
            return
        print('Retrieving data from the directories...')

    fig_directories = []
    for model in models.values():
        if model['dim_reduction'] == 'Original features':
            directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}/{model['classification_method']} model"
        elif model['dim_reduction'] == 'UMAP_streaming':
            directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components/{model['window_size']} window length/{model['classification_method']} model"
        else: 
           directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components/{model['classification_method']} model"
        fig_directories.append(directory)
    for directory in fig_directories:
        if not os.path.exists(directory):
            print('Requested folders are not available. Check if the run was executed and/or the informations provided are correct')
            print(directory)
            return
        
    data_directories = []
    for model in models.values():
        if model['dim_reduction'] == 'Original features':
            directory = f"saved_data/{model['dim_reduction']}/{model['feature_extraction']}/{model['classification_method']} model"
        elif model['dim_reduction'] == 'UMAP_streaming':
            directory = f"saved_data/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components/{model['window_size']} window length/{model['classification_method']} model"
        else: 
           directory = f"saved_data/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components/{model['classification_method']} model"
        data_directories.append(directory)
    for directory in data_directories:
        if not os.path.exists(directory):
            print('Requested data are not available. Check if the run was executed and/or the informations provided are correct')
            print(directory)
            return
        
    compare_times = []
    num_methods = len(classification_method) 
    current_row = [None] * num_methods
    i=0
    for j, directory in enumerate(data_directories):
        with open(f'{directory}/Simulation times for {convert[comparison_type][i]} components.pkl', 'rb') as f:
            time = pickle.load(f).iloc[0, 0] / 60

        method_index = j % num_methods  
        current_row[method_index] = time

        if method_index == num_methods - 1:
            compare_times.append(current_row)
            i=i+1
            current_row = [None] * num_methods

    baseline = []        
    if model['feature_extraction'] == 'mobilenet_small':
        for i, model in enumerate(models.values()):
            if i<num_methods:
                with open(f'saved_data/Original features/{model['feature_extraction']}/{model['classification_method']} model/Simulation times for 576 components.pkl', 'rb') as f:
                    baseline.append(pickle.load(f).iloc[0,0]/60)
    else:
        for i, model in enumerate(models.values()):
            if i<num_methods:
                with open(f'saved_data/Original features/{model['feature_extraction']}/{model['classification_method']} model/Simulation times for 512 components.pkl', 'rb') as f:
                    baseline.append(pickle.load(f).iloc[0,0]/60)
    baseline = pd.DataFrame(baseline)
    print('Information correctly recovered')

    if comparison_type == 'components':
        plt.figure(figsize=(10, 6))
        compare_times = pd.DataFrame(compare_times)
        for i, time in enumerate(compare_times.columns):
            sns.lineplot(x=components, y=compare_times[time], marker='o', label=f'Time for {classification_method[i]} model')
        plt.xlabel('Number of components', fontsize=16)
        plt.ylabel('Execution times', fontsize=16)
        plt.title('Execution times across different embedding sizes', fontsize=18)
        #for i in range(baseline.shape[0]):
        #    plt.axhline(baseline.iloc[i,0], color=default_colors[i], linestyle='--', label=f'Time with no dimensionality reduction for {classification_method[i]} model: {baseline.iloc[i,0]:.0f}')
        plt.xticks(components, fontsize=14)
        plt.ylim([0, 8500/60])
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right', framealpha=1, fontsize=13)
        plt.grid(True)
        plt.tight_layout()
        filename = f"compare_multiple_times_across_different_components.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()
    
    if comparison_type == 'window_size':
        plt.figure(figsize=(10, 6))
        compare_times = pd.DataFrame(compare_times)
        for i, time in enumerate(compare_times.columns):
            sns.lineplot(x=window_size, y=compare_times[time], marker='o', label=f'Time for {classification_method[i]} model')
        plt.ylabel('Execution times', fontsize=16)
        plt.xlabel('Window size of the embedding', fontsize=16)
        plt.title('Execution times across different window sizes of the embedding', fontsize=18)
        #for i in range(baseline.shape[0]):
        #    plt.axhline(baseline.iloc[i,0], color=default_colors[i], linestyle='--', label=f'Time with no dimensionality reduction for {classification_method[i]} model: {baseline.iloc[i,0]:.0f}')
        plt.xticks(window_size, fontsize=14)
        plt.ylim([0, 8500/60])
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right', framealpha=1, fontsize=13)
        plt.grid(True)
        plt.tight_layout()        
        filename = f"compare_multiple_times_across_different_window_sizes.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()

def compare_times (comparison_type='components', feature_extractor = None, 
                    dim_reduction = None, components=0, window_size=0, classification_method = None):

    """
    Method to compare results produced by different runs on FMOW.
    
    Args:
        comparison_type (string): One of 'components', 'window_size'.
                                    It must specify the category for which a comparison is wanted.
                                    Once the category is chosen, the consequential argument MUST have multiple inputs in a list format.
        feature_extractor (string): One of 'mobilenet_small', 'resnet18'.
        dim_reduction (string): One of 'RProj', 'UMAP_batch', 'UMAP_streaming', 'GaussianRProj'.
        components (int): The embedding size for the reduced datasets.
        window_size (int): Required only for 'Streaming_UMAP' reduction method. The size of the moving window.
        classification_method (string): One of 'Gaussian NB', 'Softmax regression', 'SLDA', 'SLDA with Kalman'.
    """    
    convert = {
        'feature_extraction': feature_extractor,
        'dim_reduction': dim_reduction,
        'components': components,
        'window_size': window_size,
        'classification_method': classification_method,
    }

    if comparison_type not in ['components', 'window_size']:
            print("Choose a proper comparison metric")
            return
    
    models = {}
    for i in range(len(convert[comparison_type])):
        models[f'model{i}'] = {
            'feature_extraction': feature_extractor[i] if comparison_type=='feature_extraction' else feature_extractor,
            'dim_reduction': dim_reduction[i] if comparison_type=='dim_reduction' else dim_reduction,
            'components': components[i] if comparison_type=='components' else components,
            'window_size': window_size[i] if comparison_type=='window_size' else window_size,
            'classification_method': classification_method[i] if comparison_type=='classification_method' else classification_method,
        }

    if not all(model['feature_extraction'] != None for model in models.values()):
        print('The model on the original images is not computed')
        return
    
    elif all(model['feature_extraction'] != None for model in models.values()) and not all(model['dim_reduction'] != None for model in models.values()):
        if not all(model['feature_extraction'] in ['mobilenet_small', 'resnet18'] for model in models.values()):
            print("Feature extractor not found")
            return
        print('Retrieving data from the directories...')
        for model in models.values():
            if model['dim_reduction'] == None:
                model['dim_reduction'] = 'Original features' 

    else: 
        if not all(model['feature_extraction'] in ['mobilenet_small', 'resnet18'] for model in models.values()):
            print("Feature extractor not found")
            return
        if not all(model['dim_reduction'] in ['RProj', 'UMAP_batch', 'UMAP_streaming', 'GaussianRProj'] for model in models.values()):
            print("Dimensionality reduction method not found")
            return
        if not all(model['classification_method'] in ['Gaussian NB', 'Softmax regression', 'SLDA', 'SLDA with Kalman'] for model in models.values()):
            print("Classification method not found")
            return
        print('Retrieving data from the directories...')

    fig_directories = []
    for model in models.values():
        if model['dim_reduction'] == 'Original features':
            directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}/{model['classification_method']} model"
        elif model['dim_reduction'] == 'UMAP_streaming':
            directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components/{model['window_size']} window length/{model['classification_method']} model"
        else: 
           directory = f"figures/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components/{model['classification_method']} model"
        fig_directories.append(directory)
    for directory in fig_directories:
        if not os.path.exists(directory):
            print('Requested images are not available. Check if the run was executed and/or the informations provided are correct')
            print(directory)
            return
        
    data_directories = []
    for model in models.values():
        if model['dim_reduction'] == 'Original features':
            directory = f"saved_data/{model['dim_reduction']}/{model['feature_extraction']}/{model['classification_method']} model"
        elif model['dim_reduction'] == 'UMAP_streaming':
            directory = f"saved_data/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components/{model['window_size']} window length/{model['classification_method']} model"
        else: 
           directory = f"saved_data/{model['dim_reduction']}/{model['feature_extraction']}/{model['components']} components/{model['classification_method']} model"
        data_directories.append(directory)
    for directory in data_directories:
        if not os.path.exists(directory):
            print('Requested data are not available. Check if the run was executed and/or the informations provided are correct')
            print(directory)
            return
        
    compare_times = []
    for i, directory in enumerate(data_directories):
        with open(f'{directory}/Simulation times for {components[i]} components.pkl', 'rb') as f:
            compare_times.append(pickle.load(f).iloc[0,0]/60)
    if model['feature_extraction'] == 'mobilenet_small':
        with open(f'saved_data/Original features/{model['feature_extraction']}/{model['classification_method']} model/Simulation times for 576 components.pkl', 'rb') as f:
            baseline = pickle.load(f).iloc[0,0]/60
    else:    
        with open(f'saved_data/Original features/{model['feature_extraction']}/{model['classification_method']} model/Simulation times for 512 components.pkl', 'rb') as f:
            baseline = pickle.load(f).iloc[0,0]/60

    print('Information correctly recovered')

    if comparison_type == 'components':
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=components, y = compare_times, marker='o', label=f'Time for {classification_method} model')
        plt.xlabel('Number of components', fontsize=16)
        plt.ylabel('Execution times', fontsize=16)
        plt.title('Execution times across different embedding sizes', fontsize=18)
        plt.axhline(baseline, color='grey', linestyle='--', label=f'Time with no dimensionality reduction: {baseline:.0f}')
        compare_times = pd.DataFrame(compare_times)
        plt.xticks(components, fontsize=14)
        plt.ylim([0, 8500/60])
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right', framealpha=1, fontsize=13)
        plt.grid(True)
        plt.tight_layout()
        filename = f"compare_times_across_different_components.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()
    
    if comparison_type == 'window_size':
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=components, y = compare_times, marker='o', label=f'Time for {classification_method} model')
        plt.xlabel('Number of components', fontsize=16)
        plt.ylabel('Execution times', fontsize=16)
        plt.title('Execution times across different embedding sizes', fontsize=18)
        plt.axhline(baseline, color='grey', linestyle='--', label=f'Time with no dimensionality reduction: {baseline:.0f}')
        compare_times = pd.DataFrame(compare_times)
        plt.xticks(components, fontsize=14)
        plt.ylim([0, 8500/60])
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right', framealpha=1, fontsize=13)
        plt.grid(True)
        plt.tight_layout()        
        filename = f"compare_times_across_different_window_sizes.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()
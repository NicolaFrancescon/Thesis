# Author: Nicola Francescon
#

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import umap
import joblib
import pickle
import torch
import os
from river.preprocessing import SparseRandomProjector
from river.stream import iter_pandas
import time
import gc

def save_times(times_vector, labels, directory, components, window_size = None):
    times = pd.DataFrame([times_vector], columns=labels)
    if window_size == None:
        with open(f'{directory}/Dimensionality reduction times for {components} components.pkl', 'wb') as f:
            pickle.dump(times, f)
    else: 
        with open(f'{directory}/Dimensionality reduction times for {components} components and sliding window length {window_size}.pkl', 'wb') as f:
            pickle.dump(times, f)
    return

total_length = 113291

def reduce_dimensionality (feature_extractor = 'mobilenet_small', dim_reduction = 'RProj', num_comp = 0, window_size=0):
    """
    Reduces the feature space by using a specified method and seed to the desired size.
    
    Args:
        feature_extractor (string): One of 'mobilenet_small', 'resnet18', 'eurosat'
        dim_reduction (string): One of 'RProj', 'UMAP_batch', 'UMAP_streaming'.
        seed (int): Seed for reproducibility.
        num_components (int): Dimension of the resulting embedding.
        window_size (int): Required only for 'Streaming_UMAP' method. The size of the moving window.
    """

    data_to_init = pd.DataFrame()
    label_to_init = []

    for k in range(6):
        filename = f'features/initial_features/{feature_extractor}/fmow year_{k+2002}.pkl'
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        features = pd.DataFrame(data['features'])
        targets = data['targets']

        data_to_init = pd.concat([data_to_init, features], ignore_index=True)
        label_to_init.extend(targets)

    dfs = []

    for year in range(6,16):
        filename = f'features/initial_features/{feature_extractor}/fmow year_{year+2002}.pkl'
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        features = data['features']
        targets = data['targets']

        df = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(features.shape[1])])
        df['Label'] = targets

        dfs.append(df)

    if dim_reduction == 'RProj':
        hyperparams = {
        'density': 1/3,
        'seed': 21100,
        }
        randomProjection(data_to_init, label_to_init, num_comp, dfs, feature_extractor, hyperparams)

    elif dim_reduction == 'UMAP_batch':
        hyperparams = {
            'n_neighbors': 20,
            'metric': 'cosine',
            'seed': 21100,
            'epochs': 200,
            'min_dist': 0.1,
            'spread': 1,
            'sample_rate': 10,
            'transform_queue_size': 20,
            'a': 65,
            'b': 5,
        }
        UMAP_batch(data_to_init, label_to_init, num_comp, dfs, feature_extractor, hyperparams)
    
    elif dim_reduction == 'UMAP_streaming':
        hyperparams = {
            'n_neighbors': int(window_size/100),
            'metric': 'cosine',
            'seed': 21100,
            'epochs': 200,
            'min_dist': 0.1,
            'spread': 1,
            'sample_rate': 10,
            'transform_queue_size': 20,
            'a': 65,
            'b': 5,
        }
        UMAP_streaming(data_to_init, label_to_init, num_comp, dfs, window_size, feature_extractor, hyperparams)

    else:
        return

def UMAP_batch (initialization_data, initialization_label, num_components, data, model, hyperparams=None):

    """
    Reduces the feature space using UMAP in incremental batch fashion and saves the new features in the correct folder.
    
    Args:
        initialization_data (pandas dataframe): Dataframe containing data to initialize the algorithm.
        initialization_label (pandas dataframe): Array containing the corresponding label.
        num_components (int): Dimension of the resulting embedding.
        data (pandas dataframe of pandas dataframes): Dataframe containing multiple dataframes, one for each batch of initialized data.
        model (string): One of 'mobilenet_small', 'eurosat', 'resnet18'
        hyperparams (vocabulary): Optional list of hyperparameters. In alternative, default hyperparameters are used.
    """
    if model not in ['mobilenet_small', 'resnet18', 'eurosat']:
        print('Model not found')
        return
    folder_name = f"features/UMAP_batch/{model}"
    if not os.path.exists(f'{folder_name}'):
        os.makedirs(folder_name)
    elif os.path.exists(f'{folder_name}/fmow init - DR{num_components}.pkl'):
        print('Dimensionality reduction already performed in the past')
        return

    if hyperparams['n_neighbors'] and hyperparams['metric']:
        knn = NearestNeighbors(n_neighbors=hyperparams['n_neighbors'], metric=hyperparams['metric'])
    elif hyperparams['n_neighbors']:
        knn = NearestNeighbors(n_neighbors=hyperparams['n_neighbors'])
    elif hyperparams['metric']:
        knn = NearestNeighbors(metric=hyperparams['metric'])
    else:
        knn = NearestNeighbors()

    t1 = time.perf_counter(), time.process_time()
    knn.fit(initialization_data)
    init_dists, init_indices = knn.kneighbors(initialization_data, return_distance=True)
    print(f"Reducing features from {initialization_data.shape[1]} to {num_components} using {knn.get_params()['n_neighbors']} nearest neighbors")

    mapper =  umap.UMAP(n_components = num_components, random_state = hyperparams['seed'], transform_seed = hyperparams['seed'],
                        n_neighbors = knn.get_params()['n_neighbors'], n_epochs = hyperparams['epochs'], min_dist=hyperparams['min_dist'], 
                        spread=hyperparams['spread'], metric=knn.get_params()['metric'], negative_sample_rate = hyperparams['sample_rate'], 
                        transform_queue_size = hyperparams['transform_queue_size'], force_approximation_algorithm=True, a=hyperparams['a'], b=hyperparams['b'], 
                        precomputed_knn = (init_indices, init_dists, 0))
    fit_umap = mapper.fit_transform(initialization_data)

    features_umap = np.array(fit_umap)
    dataframe_embed = {'features': features_umap, 'target':initialization_label}

    with open(f'{folder_name}/fmow init - DR{num_components}.pkl', 'wb') as f:
        pickle.dump(dataframe_embed, f)

    mapper._small_data=True

    for year in range(len(data)):
        print('Started dimensionality reduction on year '+str(year+2008))
        x=data[year].iloc[:, :-1]
        y=data[year].iloc[:, -1]

        x = torch.tensor(x.values)

        data_embedded = mapper.transform(x)

        knn = NearestNeighbors(n_neighbors = int(data[year].shape[0]*mapper._n_neighbors/initialization_data.shape[0]), metric=mapper.metric)
        knn.fit(x.numpy())
        knn_dists, knn_indices = knn.kneighbors(x.numpy(), return_distance=True)

        disconnected_index = knn_dists >= mapper._disconnection_distance
        knn_indices[disconnected_index] = -1
        knn_dists[disconnected_index] = np.inf
        graph, sigma, rho, graph_dists= umap.umap_.fuzzy_simplicial_set(
                    x.numpy(),
                    mapper._n_neighbors,
                    np.random.RandomState(mapper.random_state),
                    mapper.metric,
                    mapper._metric_kwds,
                    knn_indices,
                    knn_dists,
                    mapper.angular_rp_forest,
                    mapper.set_op_mix_ratio,
                    mapper.local_connectivity,
                    True,
                    mapper.verbose,
                    mapper.densmap or mapper.output_dens,
                )
        mapper.embedding_ = data_embedded
        mapper._raw_data = x.numpy()
        mapper.graph_ = graph
        mapper._knn_indices = knn_indices
        mapper._knn_dists = knn_dists
        mapper._sigmas = sigma
        mapper._rhos = rho
        mapper.graph_dists_ = graph_dists

        mapper._input_hash = joblib.hash(mapper._raw_data)
        features = data_embedded
        dataframe_embed = {'features': features, 'target':y}
        with open(f'{folder_name}/fmow year_{year+2008} - DR{num_components}.pkl', 'wb') as f:
            pickle.dump(dataframe_embed, f)
        print('Ended dimensionality reduction on year '+str(year+2008))
    t2 = time.perf_counter(), time.process_time()
    print(f" Total time for the dimensionality reduction step: {t2[0] - t1[0]:.2f} seconds - {(t2[0] - t1[0])/total_length:.2f} seconds per sample")
    print(f" CPU time for the dimensionality reduction step: {t2[1] - t1[1]:.2f} seconds - {(t2[1] - t1[1])/total_length:.2f} seconds per sample")
    save_times([t2[0]-t1[0], t2[1]-t1[1]], ['Total time for dimensionality reduction', 'CPU time for dimensionality reduction'], folder_name, num_components)
    return

def UMAP_streaming(initialization_data, initialization_label, num_components, data, embedding_window, model, hyperparams=None):

    """
    Reduces the feature space using UMAP in streaming fashion and saves the new features in the correct folder.
    
    Args:
        initialization_data (pandas dataframe): Dataframe containing data to initialize the algorithm.
        initialization_label (pandas dataframe): Array containing the corresponding label.
        num_components (int): Dimension of the resulting embedding.
        data (pandas dataframe of pandas dataframes): Dataframe containing multiple dataframes, one for each batch of initialized data.
        embedding_window (int): Number of samples to be included in the sliding window.
        model (string): One of "mobilenet_small", "resnet18", "eurosat".
        hyperparams (vocabulary): Optional list of hyperparameters. In alternative, default hyperparameters are used.
    """
    if model not in ['mobilenet_small', 'resnet18', 'eurosat']:
        print('Model not found')
        return
    folder_name = f"features/UMAP_streaming/{model}"  
    if not os.path.exists(f'{folder_name}'):
        os.makedirs(folder_name)
    elif os.path.exists(f'{folder_name}/fmow init - DR{num_components} + win{embedding_window}.pkl'):
        print('Dimensionality reduction already performed in the past')
        return

    if hyperparams['n_neighbors'] and hyperparams['metric']:
        knn = NearestNeighbors(n_neighbors=hyperparams['n_neighbors'], metric=hyperparams['metric'])
    elif hyperparams['n_neighbors']:
        knn = NearestNeighbors(n_neighbors=hyperparams['n_neighbors'])
    elif hyperparams['metric']:
        knn = NearestNeighbors(metric=hyperparams['metric'])
    else:
        knn = NearestNeighbors()

    t1 = time.perf_counter(), time.process_time()
    # Initializing Umap mapper with proper initialization data
    if(embedding_window<initialization_data.shape[0]):
        knn.fit(initialization_data[-embedding_window:])
        init_dists, init_indices = knn.kneighbors(initialization_data[-embedding_window:], return_distance=True)
        mapper =  umap.UMAP(n_components = num_components, random_state = hyperparams['seed'], transform_seed = hyperparams['seed'],
                        n_neighbors = knn.get_params()['n_neighbors'], n_epochs = hyperparams['epochs'], min_dist=hyperparams['min_dist'], 
                        spread=hyperparams['spread'], metric=knn.get_params()['metric'], negative_sample_rate = hyperparams['sample_rate'], 
                        transform_queue_size = hyperparams['transform_queue_size'], force_approximation_algorithm=True, a=hyperparams['a'], b=hyperparams['b'], 
                        precomputed_knn = (init_indices, init_dists, 0))
        fit_umap = mapper.fit_transform(initialization_data[-embedding_window:])
    else:
        knn.fit(initialization_data)
        init_dists, init_indices = knn.kneighbors(initialization_data, return_distance=True)
        mapper =  umap.UMAP(n_components = num_components, random_state = hyperparams['seed'], transform_seed = hyperparams['seed'],
                        n_neighbors = knn.get_params()['n_neighbors'], n_epochs = 200, min_dist=hyperparams['min_dist'], 
                        spread=hyperparams['spread'], metric=knn.get_params()['metric'], negative_sample_rate = hyperparams['sample_rate'], 
                        transform_queue_size = hyperparams['transform_queue_size'], force_approximation_algorithm=True, a=hyperparams['a'], b=hyperparams['b'], 
                        precomputed_knn = (init_indices, init_dists, 0))
        fit_umap = mapper.fit_transform(initialization_data)
    print(f"Reducing features from {initialization_data.shape[1]} to {num_components} using {knn.get_params()['n_neighbors']} nearest neighbors and a window of length {embedding_window}")

    # Saving initial fit
    features_umap = np.array(fit_umap)
    dataframe_embed = {'features': features_umap, 'target':initialization_label}
    with open(f'{folder_name}/fmow init - DR{num_components} + win{embedding_window}.pkl', 'wb') as f:
        pickle.dump(dataframe_embed, f)

    mapper._small_data = True
    mapper.n_epochs = hyperparams['epochs']

    for year in range(len(data)):
        print('Started dimensionality reduction on year '+str(year+2008))

        # Initialize data stream
        stream = iter_pandas(X=data[year].iloc[:, :-1], y=data[year].iloc[:, -1])
        features = []
        labels = []
        for x,y in stream:

            x = torch.tensor(list(x.values()))
            x = x.reshape(1, -1)

            # Compute the feature values for the new point in the existing embedding space
            data_embedded = mapper.transform(x).reshape(1, -1)

            # Update the embedding through a sliding window of fixed size
            new_embedding=np.concatenate((mapper.embedding_, data_embedded), axis=0)
            if(mapper.embedding_.shape[0]<embedding_window):
                mapper.embedding_ = new_embedding
                mapper._raw_data = np.concatenate((mapper._raw_data, x.numpy()), axis=0)
            else:
                mapper.embedding_ = new_embedding[1:]
                mapper._raw_data = np.concatenate((mapper._raw_data[1:], x.numpy()), axis=0)
            data_embedded = new_embedding[-1:].tolist()

            # Computing parameters for the new embedding
            knn = NearestNeighbors(n_neighbors=mapper._n_neighbors, metric=mapper.metric)
            knn.fit(mapper._raw_data)
            knn_dists, knn_indices = knn.kneighbors(mapper._raw_data, return_distance=True)

            disconnected_index = knn_dists >= mapper._disconnection_distance
            knn_indices[disconnected_index] = -1
            knn_dists[disconnected_index] = np.inf
            graph, sigma, rho, graph_dists= umap.umap_.fuzzy_simplicial_set(
                        mapper._raw_data,
                        mapper._n_neighbors,
                        np.random.RandomState(mapper.random_state),
                        mapper.metric,
                        mapper._metric_kwds,
                        knn_indices,
                        knn_dists,
                        mapper.angular_rp_forest,
                        mapper.set_op_mix_ratio,
                        mapper.local_connectivity,
                        True,
                        mapper.verbose,
                        mapper.densmap or mapper.output_dens,
                    )
            
            # Updating mapper parameters
            mapper.graph_ = graph
            mapper._knn_indices = knn_indices
            mapper._knn_dists = knn_dists
            mapper._sigmas = sigma
            mapper._rhos = rho
            mapper.graph_dists_ = graph_dists
            mapper._input_hash = joblib.hash(mapper._raw_data)
            
            features.append(data_embedded)
            labels.append(y)

            del knn, graph, sigma, rho, graph_dists, knn_indices, knn_dists
            gc.collect()

        # Saving the new fit for the entire batch (year)
        features = np.array(features)
        dataframe_embed = {'features': features, 'target':labels}
        with open(f'{folder_name}/fmow year_{year+2008} - DR{num_components} + win{embedding_window}.pkl', 'wb') as f:
            pickle.dump(dataframe_embed, f)
        print('Ended dimensionality reduction on year '+str(year+2008))
    t2 = time.perf_counter(), time.process_time()
    print(f" Total time for the dimensionality reduction step: {t2[0] - t1[0]:.2f} seconds - {(t2[0] - t1[0])/total_length:.2f} seconds per sample")
    print(f" CPU time for the dimensionality reduction step: {t2[1] - t1[1]:.2f} seconds - {(t2[1] - t1[1])/total_length:.2f} seconds per sample")
    save_times([t2[0]-t1[0], t2[1]-t1[1]], ['Total time for dimensionality reduction', 'CPU time for dimensionality reduction'], folder_name, num_components, embedding_window)
    return

def randomProjection (initialization_data, initialization_label, num_components, data, extractor, hyperparams=None):

    """
    Reduces the feature space using Random Projection and saves the new features in the correct folder.
    
    Args:
        initialization_data (pandas dataframe): Dataframe containing data to initialize the algorithm.
        initialization_label (pandas dataframe): Array containing the corresponding label.
        num_components (int): Dimension of the resulting embedding.
        data (pandas dataframe of pandas dataframes): Dataframe containing multiple dataframes, one for each batch of initialized data.
        extractor (string): One of "mobilenet_small", "resnet18", "eurosat"
        hyperparams (vocabulary): Optional list of hyperparameters. In alternative, default hyperparameters are used.
    """
    if extractor not in ['mobilenet_small', 'resnet18', 'eurosat']:
        print('Model not found')
        return
    folder_name = f"features/RProj/{extractor}"
    if not os.path.exists(f'{folder_name}'):
        os.makedirs(folder_name)
    elif os.path.exists(f'{folder_name}/fmow init - DR{num_components}.pkl'):
        print('Dimensionality reduction already performed in the past')
        return

    print(f"Reducing features from {initialization_data.shape[1]} to {num_components} using density {hyperparams['density']:.2f}")

    mapper = SparseRandomProjector(n_components=num_components, density=hyperparams['density'], seed=hyperparams['seed'])
    embedded_data = []
    label = []
    stream = iter_pandas(X=initialization_data, y=pd.DataFrame(initialization_label))
    for x, y in stream:
        embedded_data.append(np.array(list(mapper.transform_one(x).values())))
        label.extend(y.values())

    features_RP = np.array(embedded_data)
    dataframe_embed = {'features': features_RP, 'target':label}

    with open(f'{folder_name}/fmow init - DR{num_components}.pkl', 'wb') as f:
        pickle.dump(dataframe_embed, f)
    t1 = time.perf_counter(), time.process_time()
    for year in range(len(data)):
        print('Started dimensionality reduction on year '+str(year+2008))
        
        data_embedded = []
        label = []

        stream = iter_pandas(X=data[year].iloc[:, :-1], y=data[year].iloc[:, -1])
        
        for x,y in stream:
            data_embedded.append(np.array(list(mapper.transform_one(x).values())))
            label.append(y)

        dataframe_embed = {'features': data_embedded, 'target': label}
        with open(f'{folder_name}/fmow year_{year+2008} - DR{num_components}.pkl', 'wb') as f:
            pickle.dump(dataframe_embed, f)
        print('Ended dimensionality reduction on year '+str(year+2008))
    t2 = time.perf_counter(), time.process_time()
    print(f" Total time for the dimensionality reduction step: {t2[0] - t1[0]:.2f} seconds - {(t2[0] - t1[0])/total_length:.2f} seconds per sample")
    print(f" CPU time for the dimensionality reduction step: {t2[1] - t1[1]:.2f} seconds - {(t2[1] - t1[1])/total_length:.2f} seconds per sample")
    save_times([t2[0]-t1[0], t2[1]-t1[1]], ['Total time for dimensionality reduction', 'CPU time for dimensionality reduction'], folder_name, num_components)
    return
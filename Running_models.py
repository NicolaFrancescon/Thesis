import numpy as np
import pandas as pd
import pickle
import os
from Streaming_functions import SLDA, SLDA_init, SLDA_with_Kalman, streaming_train_with_scaling
from Useful_functions import print_metrics, print_cm, print_means
from river.stream import iter_pandas
from river.naive_bayes import GaussianNB
from river.linear_model import SoftmaxRegression
from river.metrics import Accuracy, BalancedAccuracy, ConfusionMatrix
from river.metrics.base import Metrics
from river.utils import Rolling
import time

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

def save_times(times_vector, labels, directory, components, window_size = None):
    times = pd.DataFrame([times_vector], columns=labels)
    if window_size == None:
        with open(f'{directory}/Simulation times for {components} components.pkl', 'wb') as f:
            pickle.dump(times, f)
    else: 
        with open(f'{directory}/Simulation times for {components} components and sliding window length {window_size}.pkl', 'wb') as f:
            pickle.dump(times, f)
    return
def run_model (feature_extractor = None, dim_reduction = None, components=0, window_size=None, classification_method = None, scaler = None):

    """
    Method to execute any classification model on FMOW.
    
    Args:
        feature_extractor (string): One of 'mobilenet_small', 'resnet18', 'eurosat'.
        dim_reduction (string): One of 'RProj', 'UMAP_batch', 'UMAP_streaming'.
        components (int): The embedding size for the reduced datasets.
        window_size (int): Required only for 'Streaming_UMAP' reduction method. The size of the moving window.
        classification_method (string): One of 'Gaussian NB', 'Softmax regression', 'SLDA', 'SLDA with Kalman', 'SQDA', 'SQDA with Kalman'
        scaler (river scaler): Scaler to scale the data before feeding the model.
    """

    if feature_extractor==None:
        print('The model on the original images is too heavy')
        return
    
    elif feature_extractor!=None and dim_reduction==None:
        if feature_extractor not in ['mobilenet_small', 'resnet18', 'eurosat']:
            print("Feature extractor not found")
            return
        data_to_init = pd.DataFrame()
        label_to_init = []

        print('Retrieving the original datasets...')
        for k in range(6):
            filename = f'features/initial_features/{feature_extractor}/fmow year_{k+2002}.pkl'
            if not os.path.exists(filename):
                print('Requested file is not available')
                print(filename)
                return
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
        dim_reduction = 'Original features'
        print('Datasets correctly recovered')
        

    elif feature_extractor!=None and dim_reduction!=None:

        if feature_extractor not in ['mobilenet_small', 'resnet18', 'eurosat']:
            print("Feature extractor not found")
            return
        if dim_reduction not in ['RProj', 'UMAP_batch', 'UMAP_streaming']:
            print("Reduction technique not found")
            return
        print('Retrieving the reduced datasets...')

        if dim_reduction in ['RProj', 'UMAP_batch']:
            filename = f'features/{dim_reduction}/{feature_extractor}/fmow init - DR{components}.pkl'
            if not os.path.exists(filename):
                print('Requested file is not available')
                print(filename)
                return
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            features = pd.DataFrame(data['features'])
            label_to_init = data['target']
            data_to_init = pd.DataFrame(features)

            dfs = []

            for year in range(6,16):
                filename = f'features/{dim_reduction}/{feature_extractor}/fmow year_{year + 2002} - DR{components}.pkl'
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

                features = data['features']
                features = np.reshape(features,(-1, components))
                targets = data['target']

                df = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(features.shape[1])])
                df['Label'] = targets

                dfs.append(df)

        else:
            filename = f'features/{dim_reduction}/{feature_extractor}/fmow init - DR{components} + win{window_size}.pkl'
            if not os.path.exists(filename):
                print('Requested file is not available')
                print(filename)
                return
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            features = pd.DataFrame(data['features'])
            label_to_init = data['target']
            data_to_init = pd.DataFrame(features)

            dfs = []

            for year in range(6,16):
                filename = f'features/{dim_reduction}/{feature_extractor}/fmow year_{year + 2002} - DR{components} + win{window_size}.pkl'
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

                features = data['features']
                features = np.reshape(features,(-1, components))
                targets = data['target']

                df = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(features.shape[1])])
                df['Label'] = targets

                dfs.append(df)
        print('Datasets correctly recovered')
    
    # Create directories to save relevant data and avoid run duplication
    if dim_reduction != 'Original features':
        fig_directory = f"figures/{dim_reduction}/{feature_extractor}/{components} components/{classification_method} model"
    elif dim_reduction == 'UMAP_streaming':
        fig_directory = f"figures/{dim_reduction}/{feature_extractor}/{components} components/{window_size} window length/{classification_method} model"
    else: 
        fig_directory = f"figures/{dim_reduction}/{feature_extractor}/{classification_method} model"
    if not os.path.exists(fig_directory):
        os.makedirs(fig_directory)
    else:
        print('Simulation already executed')
        return

    if dim_reduction != 'Original features':
        data_directory = f"saved_data/{dim_reduction}/{feature_extractor}/{components} components/{classification_method} model"
    elif dim_reduction == 'UMAP_streaming':
        data_directory = f"saved_data/{dim_reduction}/{feature_extractor}/{components} components/{window_size} window length/{classification_method} model"
    else:
        data_directory = f"saved_data/{dim_reduction}/{feature_extractor}/{classification_method} model"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    else:
        print('Simulation already executed')
        return

    t1 = time.perf_counter(), time.process_time()

    if classification_method in ['Gaussian NB', 'Softmax regression']:
            if classification_method == 'Gaussian NB':
                model = GaussianNB()
            else:
                model = SoftmaxRegression()

            metric = Metrics(metrics=[Accuracy(),BalancedAccuracy(),
                        Rolling(Metrics(metrics=[Accuracy(),BalancedAccuracy()]),window_size=5000)])
            sizes = []
            res = []
            list_cm = []
            cm = ConfusionMatrix()
            for i in range(len(dfs)):
                stream = iter_pandas(X=dfs[i].iloc[:, :-1], y=dfs[i].iloc[:, -1])

                (result, cm, model, list_cm, scaler) = streaming_train_with_scaling(stream, model, metric, cm, list_cm, scaler)
                res.extend(result)
                sizes.append(dfs[i].shape[0])
                print(f'Ended training on year {i+2008}')

    elif classification_method in ['SLDA', 'SLDA with Kalman']:
        data_to_init_normalized = pd.DataFrame(columns=range(features.shape[1]))
        for index, x in data_to_init.iterrows():
                scaler.learn_one(x)
                x = scaler.transform_one(x)
                data_to_init_normalized.loc[len(data_to_init_normalized)]=list(x.values())

        data_to_init = pd.DataFrame(data_to_init_normalized.values)

        metric = Metrics(metrics=[Accuracy(),BalancedAccuracy(),
                        Rolling(Metrics(metrics=[Accuracy(),BalancedAccuracy()]),window_size=5000)])
        sizes = []
        res = []
        list_cm =[]
        cm = ConfusionMatrix()
        mu, cov, counts, idx = SLDA_init(data_to_init, label_to_init, len(list_labels))
        print('Ended model initialization')
        if classification_method == 'SLDA':
            for i in range(len(dfs)):
                stream = iter_pandas(X=dfs[i].iloc[:, :-1], y=dfs[i].iloc[:, -1])
                (result, cm, list_cm, mu, cov, counts, idx, scaler) = SLDA(stream, metric, cm, list_cm, mu, cov, counts, idx, scaler)
                res.extend(result)
                sizes.append(dfs[i].shape[0])
                print(f'Ended training on year {i+2008}')
        else: 
            p0 = [0] * len(list_labels)
            x0 = [0] * len(list_labels)
            for i in range(len(dfs)):
                stream = iter_pandas(X=dfs[i].iloc[:, :-1], y=dfs[i].iloc[:, -1])
                (result, cm, list_cm, mu, cov, counts, idx, scaler, p0, x0) = SLDA_with_Kalman(stream, metric, cm, list_cm, mu, cov, counts, idx, scaler, p_0=p0, x_0=x0)
                res.extend(result)
                sizes.append(dfs[i].shape[0])
                print(f'Ended training on year {i+2008}')

    else:
        print('Classification method not available')
        return

    print(f"Size of the different batches: {sizes}")
    cum_sizes = np.cumsum(sizes)
    print(f"Cumulative size of the different batches: {cum_sizes}")

    t2 = time.perf_counter(), time.process_time()
    print(f" Total time for the classification step: {t2[0] - t1[0]:.2f} seconds - {(t2[0] - t1[0])/cum_sizes[-1]:.2f} seconds per sample")
    print(f" CPU time for the classification step: {t2[1] - t1[1]:.2f} seconds - {(t2[1] - t1[1])/cum_sizes[-1]:.2f} seconds per sample")
    save_times([t2[0]-t1[0], t2[1]-t1[1]], ['Total time to classify the stream', 'CPU time to classify the stream'], data_directory, components, window_size)

    # Plot the obtained metrics
    print_base_metrics(res, cum_sizes, fig_directory)

    print_rolling_metrics(res, cum_sizes, fig_directory)

    # Plot the average accuracy and balanced accuracy
    years = list(range(2008, 2018))
    (mean_acc, full_mean) = print_means(cum_sizes, res, years[:len(sizes)], fig_directory, "accuracy", 0)
    print(f'Accuracy over each batch: {[round(mean, 4) for mean in mean_acc]}')
    print(f'Final accuracy is: {full_mean:.4f}')

    (mean_acc, full_mean) = print_means(cum_sizes, res, years[:len(sizes)], fig_directory, "balanced accuracy", 1)
    print(f'Balanced accuracy over each batch: {[round(mean, 4) for mean in mean_acc]}')
    print(f'Final balanced accuracy is: {full_mean:.4f}')

    # Plot the relevant confusion matrices
    print_cm(cm, labels, fig_directory, 'Full confusion matrix')
    print_cm(list_cm[-1], labels, fig_directory, 'Confusion matrix of last year')
    print_cm(list_cm[0], labels, fig_directory, 'Confusion matrix of first year')

    # Save the relevant data
    with open(f'{data_directory}/List of confusion matrices.pkl', 'wb') as f:
        pickle.dump(list_cm, f)
    with open(f'{data_directory}/Final confusion matrix.pkl', 'wb') as f:
        pickle.dump(cm, f)
    with open(f'{data_directory}/List of metrics.pkl', 'wb') as f:
        pickle.dump(res, f)
    print('Plots and data correctly saved')
    return

def print_base_metrics (metrics_vector, batch_sizes, years, directory):
    min_length = min(len(sublist) for sublist in metrics_vector)
    truncated_list = [sublist[:min_length-1] for sublist in metrics_vector]
    res_array = np.array(truncated_list)

    plot_label=['Accuracy','Balanced accuracy']
    res_array = pd.DataFrame(res_array, columns=plot_label)

    print_metrics(res_array, plot_label, batch_sizes, years, directory)

def print_rolling_metrics (metrics_vector, batch_sizes, years, directory):
    rolling_metrics = []

    for sublist in metrics_vector:
        last_element = sublist[-1]
        if isinstance(last_element, list):
            rolling_metrics.append(last_element)

    plot_label=['Rolling Accuracy', 'Rolling Balanced Accuracy']
    rolling_metrics = pd.DataFrame(rolling_metrics, columns=plot_label)

    print_metrics(rolling_metrics, plot_label, batch_sizes, years, directory)
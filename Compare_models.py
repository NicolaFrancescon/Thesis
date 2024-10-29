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

def compare_models (comparison_type='components', feature_extractor = None, 
                    dim_reduction = None, components=0, window_size=0, classification_method = None):

    """
    Method to compare results produced by different runs on FMOW.
    
    Args:
        comparison_type (string): One of 'feature_extraction', 'dim_reduction', 'components', 'window_size', 'classification_method'.
                                    It must specify the category for which a comparison is wanted.
                                    Once the category is chosen, the consequential argument MUST have multiple inputs in a list format.
        seed (int): Seed for reproducibility.
        feature_extractor (string): One of 'mobilenet_small', 'resnet18', 'eurosat'.
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

    if comparison_type not in ['feature_extraction', 'dim_reduction', 'components', 'window_size', 'classification_method']:
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
        if not all(model['feature_extraction'] in ['mobilenet_small', 'resnet18', 'eurosat'] for model in models.values()):
            print("Feature extractor not found")
            return
        print('Retrieving data from the directories...')
        for model in models.values():
            if model['dim_reduction'] == None:
                model['dim_reduction'] = 'Original features' 

    else: 
        if not all(model['feature_extraction'] in ['mobilenet_small', 'resnet18', 'eurosat'] for model in models.values()):
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
        
    compare_metrics = []
    compare_cm = []
    compare_cm_list = []
    for directory in data_directories:
        with open(f'{directory}/List of metrics.pkl', 'rb') as f:
            compare_metrics.append(pickle.load(f))
        with open(f'{directory}/Final confusion matrix.pkl', 'rb') as f:
            compare_cm.append(pickle.load(f))
        with open(f'{directory}/List of confusion matrices.pkl', 'rb') as f:
            compare_cm_list.append(pickle.load(f))
    with open(f"saved_data/Original features/{model['feature_extraction']}/{model['classification_method']} model/List of metrics.pkl", 'rb') as f:
        baseline = pickle.load(f)

    print('Information correctly recovered')
    if comparison_type == 'components':
        final_metrics = get_final_metrics(compare_metrics)
        baseline_metrics = get_baseline_metrics(baseline)
        plt.figure(figsize=(10, 6))
        for metric in final_metrics.columns[:2]:
            sns.lineplot(x=components, y=final_metrics[metric], marker='o', label=metric)
        plt.xlabel('Number of components', fontsize=16)
        plt.ylabel('Final model performance', fontsize=16)
        plt.xticks(components, fontsize=14)
        plt.title('Variation of the metrics across different embedding sizes', fontsize=18)
        plt.axhline(baseline_metrics['Accuracy'].mean(), color=default_colors[0], linestyle='--', label=f'Baseline accuracy with no dimensionality reduction: {baseline_metrics['Accuracy'].mean():.2f}')
        plt.axhline(baseline_metrics['Balanced accuracy'].mean(), color=default_colors[1], linestyle='--', 
                    label=f'Baseline balanced accuracy with no dimensionality reduction: {baseline_metrics['Balanced accuracy'].mean():.2f}')
        plt.ylim([-0.04, 0.31])
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right', framealpha=1, fontsize=13)
        plt.grid(True)
        plt.tight_layout()
        filename = f"compare_performance_across_different_components.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()
    
    if comparison_type == 'window_size':
        final_metrics = get_final_metrics(compare_metrics)
        baseline_metrics = get_baseline_metrics(baseline)
        plt.figure(figsize=(10, 6))
        for metric in final_metrics.columns[:2]:
            sns.lineplot(x=window_size, y=final_metrics[metric], marker='o', label=metric)
        plt.xlabel('Window size of the embedding', fontsize=16)
        plt.ylabel('Final model performance', fontsize=16)
        plt.xticks(window_size, fontsize=14)
        plt.title('Variation of the metrics across different window sizes of the embedding', fontsize=18)
        plt.axhline(baseline_metrics['Accuracy'].mean(), color=default_colors[0], linestyle='--', label=f'Baseline accuracy with no dimensionality reduction: {baseline_metrics['Accuracy'].mean():.2f}')
        plt.axhline(baseline_metrics['Balanced accuracy'].mean(), color=default_colors[1], linestyle='--', 
                    label=f'Baseline balanced accuracy with no dimensionality reduction: {baseline_metrics['Balanced accuracy'].mean():.2f}')
        plt.ylim([-0.02, 0.31])
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right', framealpha=1, fontsize=13)
        plt.grid(True)
        plt.tight_layout()
        filename = f"compare_performance_across_different_window_sizes.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()

    elif comparison_type in ['feature_extraction', 'dim_reduction', 'classification_method']:
        model_metrics = rearrange_metrics(compare_metrics)
        num_metrics = 2
        num_models = len(model_metrics)

        fig, axes = plt.subplots(1, num_models, figsize=(15, 8), sharey=True)

        palette = sns.color_palette("muted", num_metrics)

        for i, (model, ax) in enumerate(zip(model_metrics, axes)):
            for j, metric in enumerate(model.columns[:num_metrics]):
                color = palette[j]
                line_style = '-'
                sns.lineplot(x=range(len(model[metric])), y=model[metric],
                            label=f'{metric}',
                            color=color, linestyle=line_style, ax=ax)
            ax.axhline(model['Accuracy'].mean(), color=palette[0], linestyle='dotted', 
                        label=f'Overall accuracy: {model["Accuracy"].mean():.2f}')
            ax.axhline(model['Balanced accuracy'].mean(), color=palette[1], linestyle='dotted', 
                        label=f'Overall balanced accuracy: {model["Balanced accuracy"].mean():.2f}')
            ax.set_xlabel('Number of observed samples', fontsize=16)
            ax.set_ylabel('Model performance over time', fontsize=16)
            ax.set_title(f'Model using {list(models.values())[i][comparison_type]}', fontsize=18)
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True)
            ax.set_ylim([-0.01, model[['Accuracy', 'Balanced accuracy']].max().max()+0.03])
            ax.legend(loc='lower right', bbox_to_anchor=(1, 0),framealpha=1, fontsize=13)

        y_min, y_max = axes[0].get_ylim()
        for ax in axes:
            ax.set_ylim(-0.02, y_max)

        if comparison_type in ['feature_extraction', 'dim_reduction']:
            title = f'using different {" ".join(comparison_type.split("_"))} methods'
        elif comparison_type == 'classification_method':
            title = 'using different classification methods'

        plt.suptitle(f'Variation of the metrics {title}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if comparison_type in ['feature_extraction', 'dim_reduction']:
            filename = f"compare_performance_across_different_{comparison_type}_methods.eps"
        elif comparison_type=='classification_method':
            filename = f"compare_performance_across_different_classification_methods.eps"
        for dir in fig_directories:
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath, format='eps', dpi=1200)
        plt.show()

        plot_grid_confusion_matrices(compare_cm, labels, models, comparison_type, fig_directories, title=f"Confusion Matrices Comparison {title}", color='GnBu')

def get_final_metrics(metrics_vector):
    final_metrics = pd.DataFrame()

    for metric_data in metrics_vector:
        min_length = min(len(sublist) for sublist in metric_data)
        truncated_list = [sublist[:min_length-1] for sublist in metric_data]
        res_array = np.array(truncated_list)
        if res_array.shape[1]>3:
            plot_label=['Accuracy','Balanced accuracy', '_', '-']
        else:
            plot_label=['Accuracy','Balanced accuracy']
        res_array = pd.DataFrame(res_array, columns=plot_label)
        final_metrics = pd.concat([final_metrics, pd.DataFrame([res_array.mean(axis=0)])], ignore_index=True)
    return final_metrics

def get_baseline_metrics(baseline_metrics):
    final_metrics = pd.DataFrame()

    min_length = min(len(sublist) for sublist in baseline_metrics)
    truncated_list = [sublist[:min_length-1] for sublist in baseline_metrics]
    res_array = np.array(truncated_list)
    if res_array.shape[1]>3:
        plot_label=['Accuracy','Balanced accuracy', '_', '-']
    else:
        plot_label=['Accuracy','Balanced accuracy']
    res_array = pd.DataFrame(res_array, columns=plot_label)
    final_metrics = pd.concat([final_metrics, pd.DataFrame([res_array.mean(axis=0)])], ignore_index=True)
    return final_metrics

def rearrange_metrics(metrics_vector):
    final_metrics = []

    for metric_data in metrics_vector:
        min_length = min(len(sublist) for sublist in metric_data)
        truncated_list = [sublist[:min_length-1] for sublist in metric_data]
        res_array = np.array(truncated_list)
        if res_array.shape[1]>3:
            plot_label=['Accuracy','Balanced accuracy', '_', '-']
        else:
            plot_label=['Accuracy','Balanced accuracy']
        res_array = pd.DataFrame(res_array, columns=plot_label)
        final_metrics.append(res_array)
    return final_metrics

def plot_grid_confusion_matrices(cm_list, labels, models, comparison_type, fig_directories, title="Confusion Matrices Comparison", color='GnBu'):
    num_matrices = len(cm_list)
    cols = len(models)
    rows = (num_matrices + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20 * cols, 15 * rows), sharey=True)
    axes = axes.flatten()

    num_labels = len(labels)
    label_names = [labels[label_num] for label_num in range(num_labels)]

    for i, (ax, cm, model) in enumerate(zip(axes, cm_list, models.values())):
        cm_matrix = [[int(cm[true_label][pred_label]) for true_label in range(num_labels)] for pred_label in range(num_labels)]
        cm_matrix = np.array(cm_matrix).T
        row_sums = cm_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_matrix_perc = np.round(cm_matrix / row_sums, 2)
        cm_df = pd.DataFrame(cm_matrix_perc, index=labels, columns=labels)
        sns.heatmap(cm_df, annot=False, fmt='.2f', cmap=color, ax=ax, linewidths=.5, cbar=False)
        ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=16)
        ax.set_yticklabels(label_names, rotation=0, fontsize=16)
        ax.set_xlabel('Predicted Labels', fontsize=18)
        if i==0:
            ax.set_ylabel('True Labels', fontsize=18)
        ax.set_title(model[comparison_type], fontsize=20)
    
    for ax in axes[num_matrices:]:
        ax.remove()
    
    plt.suptitle(title, fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if comparison_type in ['feature_extraction', 'dim_reduction']:
        filename = f"compare_confusion_matrices_across_different_{comparison_type}_methods.eps"
    elif comparison_type=='classification_method':
        filename = f"compare_confusion_matrices_using_different_classification_methods.eps"
    for dir in fig_directories:
        filepath = os.path.join(dir, filename)
        plt.savefig(filepath, format='eps', dpi=1200)
    plt.show()

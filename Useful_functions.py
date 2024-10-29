import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def print_cm(cm, labels, directory=None, title="Full confusion matrix", color='GnBu', threshold=0.00):
    """
    Plot a confusion matrix with true labels on the y-axis.

    Args:
        cm (river confusion matrix): A confusion matrix to be plot.
        labels (list): List of class labels.
        title (string): Title of the plot.
        color (string): Plot color from the cmap colors.
        threshold (int): Threshold to display the value in the cm with annot=True.
       
    Returns:

    """

    num_labels = len(labels)

    cm_matrix = [[int(cm[true_label][pred_label]) for true_label in range(num_labels)] for pred_label in range(num_labels)]
    cm_matrix = np.array(cm_matrix).T
    row_sums = cm_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_matrix_perc = np.round(cm_matrix / row_sums, 2)
    label_names = [labels[label_num] for label_num in range(num_labels)]


    annot_matrix = np.empty_like(cm_matrix_perc, dtype=object)
    for i in range(cm_matrix_perc.shape[0]):
        for j in range(cm_matrix_perc.shape[1]):
            if cm_matrix_perc[i, j] >= threshold:
                annot_matrix[i, j] = f"{cm_matrix_perc[i, j]:.2f}"
            else:
                annot_matrix[i, j] = ""

    fig, ax = plt.subplots(figsize=(25, 25))

    sns.heatmap(cm_matrix_perc, annot=False, fmt='', xticklabels=label_names, yticklabels=label_names, cmap=color, 
                cbar=True, cbar_kws={'orientation': 'horizontal', 'pad': 0.2}, ax=ax, linewidths=.5)

    ax.set_xlabel('Predicted labels', fontsize=26)
    ax.set_ylabel('True labels', fontsize=26)
    ax.set_title('Confusion matrix', fontsize=28)
    ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=18)
    ax.set_yticklabels(label_names, rotation=0, fontsize=18)
    
    plt.subplots_adjust(bottom=0.2)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    
    if directory:
        filename = f"{title}.eps"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, format='eps', dpi=600)

def print_means(sizes, accuracy, years, directory=None, title="of the chosen metric", idx=0):
    """
    Plot the mean accuracy over time.

    Args:
        sizes (list): List of cumulative observed samples across the years.
        accuracy (list): List of recorded metrics.
        years (list): List of observed years.
        title (string): Chosen metric to be included in the title.
        idx (int): 0 for accuracy, 1 for balanced accuracy, 3 for Cohen Kappa. 
       
    Returns:
        mean_acc (list): Mean chosen metric for each year of observation.
        full_mean (list): Mean chosen metric across the full set of samples.

    """
    min_length = min(len(sublist) for sublist in accuracy)
    truncated_list = [sublist[:min_length-1] for sublist in accuracy]
    res_array = np.array(truncated_list)
    cum_sizes_aux = [0]
    cum_sizes_aux.extend(sizes)
    mean_acc = []
    acc = res_array[:, idx]
    for i in range(len(sizes)):
        mean_acc.append(np.mean(acc[cum_sizes_aux[i]:cum_sizes_aux[i+1]]))
    mean_acc = np.array(mean_acc)
    full_mean = [mean_acc[i] * (cum_sizes_aux[i+1] - cum_sizes_aux[i]) for i in range(len(sizes))]
    full_mean = sum(full_mean) / sizes[-1]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=years, y=mean_acc, marker='o')
    plt.xlabel('Year', fontsize=16)
    plt.ylabel(title.capitalize() if title != "of the chosen metric" else 'Metric', fontsize=16)
    plt.title('Mean ' + title + ' over years', fontsize=18)
    plt.axhline(full_mean, color='grey', linestyle='--', label=f'Overall Mean: {full_mean:.2f}')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', fontsize=13, framealpha=1)
    plt.grid(True)
    plt.tight_layout()

    if directory:
        filename = f"{title}.eps"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, format='eps', dpi=1200)

    return mean_acc, full_mean

def print_metrics(metrics, labels, sizes, years, directory=None):
    """
    Plot the given metrics over the increasing number of learned samples.

    Args:
        metrics (list of lists): A list of lists where each sublist contains metric values.
        labels (list): List of labels for each metric.
        sizes (list): List of cumulative observed samples across the years.

    Returns:

    """
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("muted", len(metrics))
    sns.lineplot(data=metrics)
    for x in sizes:
        plt.axvline(x=x, color='gray', linestyle='--')
    plt.xlabel('Observed samples', fontsize=16)
    plt.ylabel('Metric', fontsize=16)
    plt.ylim([0, 0.34])
    plt.title('Metrics over Observed Samples', fontsize=18)
    plt.axhline(metrics['Accuracy'].mean(), color=palette[0], linestyle='dotted', 
                        label=f'Overall accuracy: {metrics["Accuracy"].mean():.2f}')
    plt.axhline(metrics['Balanced accuracy'].mean(), color=palette[1], linestyle='dotted', 
                        label=f'Overall balanced accuracy: {metrics["Balanced accuracy"].mean():.2f}')
    plt.xticks(sizes, labels=years, fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', fontsize=13, framealpha=1)
    plt.grid(True)
    plt.tight_layout()

    if directory:
        filename = f"{labels}.eps"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, format='eps', dpi=1200)

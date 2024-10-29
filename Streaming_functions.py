from river import metrics
import numpy as np
import torch
from sklearn.covariance import OAS

def streaming_train_with_scaling(stream, model, metric, cm, list_part, scaler):
    """
    Compute the streaming learning of a given stream.
    
    Args:
        stream (iter_pandas output): A stream containing features and true labels.
        model (river model): A river model to be used for the fit.
        metric (river metric): The metrics to be computed over time.
        cm (river confusion matrix): The confusion matrix to be updated.
        list_part (list): An empty list.
        scaler (river preprocessing): A scaler.
        
    Returns:
        tuple: The metrics history over the samples, the final confusion matrix, the trained model and a list containing the partial confusion matrices.
    """
    res = []
    y_pred = []
    y_true = []
    partial_cm = metrics.ConfusionMatrix()

    for x, y in stream:
        scaler.learn_one(x)
        x = scaler.transform_one(x)
        y_p = model.predict_one(x)
        if y_p is not None:
            metric.update(y_true=y, y_pred=y_p)
            y_pred.append(y_p)
            y_true.append(y)
            res.append(metric.get())
        model.learn_one(x, y)
    for y, y_p in zip(y_true, y_pred):
        cm.update(y_true=y, y_pred=y_p)
        partial_cm.update(y_true=y, y_pred=y_p)
    list_part.append(partial_cm)

    return res, cm, model, list_part, scaler

def SLDA_init(data, label, num_classes):
    """
    Initialize the environment for a Streaming LDA algorithm.
    
    Args:
        data (Pandas dataframe): A dataframe containing the features to initialize the model.
        label (list): The correct labels for each observation.
        num_classes (int): The number of distinct classes to be initialized.
        
    Returns:
        tuple: The initial mean vector, the initial covariance matrix, the initial counts and the initial iteration.
    """
    num_features = data.shape[1]
    mu = np.zeros((num_features, num_classes))
    count = np.zeros(num_classes)
    for k in np.unique(label):
        indexes = [i for i, x in enumerate(label) if x == k]
        mu[:, k] = data.iloc[indexes,:].mean(axis=0)
        count[k] = np.sum(np.array(label) == k)
    idx = data.shape[0]
    cov_estimator = OAS(assume_centered=True)
    cov_estimator.fit(data - mu[:,label].T)
    cov = cov_estimator.covariance_
    return mu, cov, count, idx

def SLDA(stream, metric, cm, list_part, mu, cov, counts, idx, scaler, eps=1e-5):
    """
    Compute the streaming LDA of a given stream.
    
    Args:
        stream (iter_pandas output): A stream containing features and true labels.
        metric (river metric): The metrics to be computed over time.
        cm (river confusion matrix): The confusion matrix to be updated.
        list_part (list): A list containing confusion matrices.
        mu (): The mean vector to be updated, of size [num_features * num_classes].
        cov (): The global covariance matrix to be updated, of size [num_features * num_features].
        counts (list): The label count.
        idx (int): The current iteration of the model.
        scaler (river scaler): The scaler used to preprocess initial data.
        
    Returns:
        tuple: The metrics history over the samples, the final confusion matrix, a list containing the partial confusion matrices,
          the updated mean vector, the updated covariance matrix, the updated count, the updated iteration and the updated feature scaler.
    """
    res = []
    y_pred = []
    y_true = []
    partial_cm = metrics.ConfusionMatrix()

    for x, y in stream:
        inv_cov = np.linalg.inv((1-eps)*cov + eps*np.identity(cov.shape[0]))
        inv_tensor = torch.tensor(inv_cov)
        mu_tensor = torch.tensor(mu)

        scaler.learn_one(x)
        x = scaler.transform_one(x)
        x = torch.tensor(np.array(list(x.values())).reshape(1, -1))

        W = torch.matmul(inv_tensor, mu_tensor)
        c =  0.5 * torch.sum(mu_tensor* W, dim=0)
        scores = torch.matmul(x.double(), W) - c
        y_p = torch.argmax(scores).item()
        if y_p is not None:
            metric.update(y_true=y, y_pred=y_p)
            y_pred.append(y_p)
            y_true.append(y)
            res.append(metric.get())

        counts = counts.astype(float)
        mu = mu.astype(float)
        x = x.to(dtype=torch.float64)

        dt = (idx * torch.matmul((x - mu[:, y]), (x - mu[:, y]).transpose(1,0)))/(idx + 1)
        mu[:, y] = (np.add(counts[y] * mu[:, y], x))/(counts[y] + 1)
        counts[y] = counts[y] + 1
        cov = (np.add(idx * cov, dt))/(idx + 1)
        idx = idx + 1

    for y, y_p in zip(y_true, y_pred):
        cm.update(y_true=y, y_pred=y_p)
        partial_cm.update(y_true=y, y_pred=y_p)
    list_part.append(partial_cm)

    return res, cm, list_part, mu, cov, counts, idx, scaler

def SLDA_with_Kalman(stream, metric, cm, list_part, mu, cov, counts, idx, scaler, p_0, x_0, eps=1e-5, r=1000, q=1):
    """
    Compute the streaming LDA of a given stream with Kalman adaptation.
    
    Args:
        stream (iter_pandas output): A stream containing features and true labels.
        metric (river metric): The metrics to be computed over time.
        cm (river confusion matrix): The confusion matrix to be updated.
        list_part (list): A list containing confusion matrices.
        mu (): The mean vector to be updated, of size [num_features * num_classes].
        cov (): The global covariance matrix to be updated, of size [num_features * num_features].
        counts (list): The label count.
        idx (int): The current iteration of the model.
        scaler (river scaler): The scaler used to preprocess initial data.
        
    Returns:
        tuple: The metrics history over the samples, the final confusion matrix, a list containing the partial confusion matrices,
          the updated mean vector, the updated covariance matrix, the updated count, the updated iteration and the updated feature scaler.
    """
    res = []
    y_pred = []
    y_true = []
    partial_cm = metrics.ConfusionMatrix()
    for x, y in stream:
        inv_cov = np.linalg.inv((1-eps)*cov + eps*np.identity(cov.shape[0]))
        inv_tensor = torch.tensor(inv_cov)
        mu_tensor = torch.tensor(mu)
        scaler.learn_one(x)
        x = scaler.transform_one(x)
        x = torch.tensor(np.array(list(x.values())).reshape(1, -1))

        W = torch.matmul(inv_tensor, mu_tensor)
        c =  0.5 * torch.sum(mu_tensor* W, dim=0)
        scores = torch.matmul(x.double(), W) - c
        y_p = torch.argmax(scores).item()
        if y_p is not None:
            metric.update(y_true=y, y_pred=y_p)
            y_pred.append(y_p)
            y_true.append(y)
            res.append(metric.get())

        if y!=y_p:
            value = 7
        else:
            value = 1
        
        x_1 = x_0
        p_1 = p_0
        k = p_0[y] / (p_0[y] + r)
        x_1[y] = x_0[y] + k * (value - x_0[y])
        p_1[y] = p_0[y] * (1.0 - k) + q

        dt = (idx * torch.matmul((x - mu[:, y]), (x - mu[:, y]).transpose(1,0)))/(idx + 1)
        if counts[y] + x_1[y] != 0:
            mu[:, y] = (np.add(counts[y] * mu[:, y], x_1[y]*x))/(counts[y] + x_1[y])
        else:
            mu[:, y] = (np.add(counts[y] * mu[:, y], x))/(counts[y] + 1)
        counts[y] = counts[y] + 1
        if idx + x_1[y] != 0:
            cov = (np.add(idx * cov, x_1[y]*dt))/(idx + x_1[y])
        else:
            cov = (np.add(idx * cov, dt))/(idx + 1)
        idx = idx + 1
        p_0 = p_1
        x_0 = x_1

    for y, y_p in zip(y_true, y_pred):
        cm.update(y_true=y, y_pred=y_p)
        partial_cm.update(y_true=y, y_pred=y_p)
    list_part.append(partial_cm)

    return res, cm, list_part, mu, cov, counts, idx, scaler, p_1, x_1
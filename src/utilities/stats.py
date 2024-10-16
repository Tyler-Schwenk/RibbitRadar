import numpy as np
from scipy import stats

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy by comparing predicted and true labels.
    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
    Returns:
        float: Accuracy score.
    """
    return np.mean(np.array(y_true) == np.array(y_pred))

def precision_recall_curve(y_true, y_scores):
    """
    Calculate precision-recall curve.
    Args:
        y_true (np.array): True binary labels.
        y_scores (np.array): Predicted probabilities or scores.
    Returns:
        tuple: Arrays for precision, recall, and thresholds.
    """
    thresholds = np.linspace(0, 1, num=100)  # Define thresholds
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = y_scores >= threshold
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls), thresholds

def roc_curve(y_true, y_scores):
    """
    Calculate ROC curve.
    Args:
        y_true (np.array): True binary labels.
        y_scores (np.array): Predicted probabilities or scores.
    Returns:
        tuple: Arrays for false positive rate, true positive rate, and thresholds.
    """
    thresholds = np.linspace(0, 1, num=100)
    fpr = []
    tpr = []

    for threshold in thresholds:
        y_pred = y_scores >= threshold
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0

        fpr.append(fpr_value)
        tpr.append(tpr_value)

    return np.array(fpr), np.array(tpr), thresholds

def roc_auc_score(fpr, tpr):
    """
    Calculate the AUC (Area Under the ROC Curve).
    Args:
        fpr (np.array): False positive rate.
        tpr (np.array): True positive rate.
    Returns:
        float: AUC score.
    """
    return np.trapz(tpr, fpr)  # Trapezoidal rule

def average_precision_score(y_true, y_scores):
    """
    Calculate Average Precision (AP).
    Args:
        y_true (np.array): True binary labels.
        y_scores (np.array): Predicted probabilities or scores.
    Returns:
        float: Average precision score.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    return np.trapz(precisions, recalls)  # Use trapezoidal rule to approximate the area under the precision-recall curve



def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):
        # Average precision
        avg_precision = average_precision_score(target[:, k], output[:, k])

        # AUC
        fpr, tpr, _ = roc_curve(target[:, k], output[:, k])
        auc = roc_auc_score(fpr, tpr)

        # Precisions, recalls
        precisions, recalls, thresholds = precision_recall_curve(target[:, k], output[:, k])

        save_every_steps = 1000  # Sample statistics to reduce size
        dict = {
            "precisions": precisions[0::save_every_steps],
            "recalls": recalls[0::save_every_steps],
            "AP": avg_precision,
            "fpr": fpr[0::save_every_steps],
            "fnr": 1.0 - tpr[0::save_every_steps],
            "auc": auc,
            # note acc is not class-wise, this is just to keep consistent with other metrics
            "acc": acc,
        }
        stats.append(dict)

    return stats

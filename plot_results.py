import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14, cmap=None):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=False, fmt="d", cmap=cmap)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig('static/model_output/cnf_matrix.png', bbox_inches='tight', dpi=300)
    return fig


def plot_results_x(labels, predicted):
    class_names = list(set(labels))
    cnf_matrix = confusion_matrix(labels, predicted)
    cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    class_names.sort()
    _ = print_confusion_matrix(cm, class_names, figsize=(16, 13), fontsize=8, cmap='Greens')


def plot_results(model, features, target):
    y_pred = model.predict(features)
    plot_results_x(target, y_pred)
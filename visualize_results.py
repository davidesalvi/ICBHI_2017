import os
import torch
import random
import GPUtil
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
import torch.nn as nn
from sklearn.metrics import classification_report, balanced_accuracy_score

def plot_confusion_matrix(y_true, y_pred, label_dict, normalize=False):

    labels = list(label_dict.keys())
    label_indices = list(label_dict.values())

    cm = confusion_matrix(y_true, y_pred, labels=label_indices)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.replace(0, np.nan, inplace=True)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.replace(0, np.nan, inplace=True)

    fsize = 25

    plt.figure()
    plt.imshow(cm_df, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    for i, j in np.ndindex(cm.shape):
        value = cm_df.iloc[i, j]
        text = f"{value:.1f}" if pd.notna(value) else "NaN"
        plt.text(j, i, text, ha="center", va="center",
                 color="white" if pd.notna(value) and value > np.nanmax(cm) / 2.0 else "black")

    plt.ylabel('True Label', fontsize=fsize)
    plt.xlabel('Predicted Label', fontsize=fsize)
    plt.tight_layout()
    plt.show()


def visualize_results():

    result_dir = '/nas/home/dsalvi/ICBHI_2017/results'
    feature_sets = ['MelSpec', 'LogSpec', 'MFCC', 'LFCC']
    classification_types = ['multi', 'binary']
    classification_types = ['multi']

    for feature in feature_sets:
        for classification in classification_types:

            if classification == 'binary':
                label_dict = {'Healthy': 0, 'Unhealthy': 1}
            else:
                label_dict = {'Healthy': 0, 'COPD': 1, 'URTI': 2, 'Asthma': 3, 'LRTI': 4, 'Bronchiectasis': 5,
                              'Pneumonia': 6, 'Bronchiolitis': 7}

            model_name = f"RESNET_{classification}_{feature}.txt"
            results = pd.read_csv(os.path.join(result_dir, model_name), sep=' ', header=None)
            results.columns = ['filename', 'pred', 'label']

            report = classification_report(results['label'], results['pred'])
            print(report)

            print("Balanced Accuracy: ", balanced_accuracy_score(results['label'], results['pred']))

            plot_confusion_matrix(results['label'], results['pred'], normalize=True, label_dict=label_dict)


            y_true = results[0].values
            y_pred = results[1].values
            plot_confusion_matrix(y_true, y_pred, normalize=True)
            plt.savefig(f"{model_name.split('.')[0]}.png")
            plt.close()





if __name__ == '__main__':
    visualize_results()

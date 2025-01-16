import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, confusion_matrix


def compute_eer_auc(labels, pred):

    fpr, tpr, thres = roc_curve(labels, pred)
    rocauc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    return eer, rocauc


def plot_roc_curve(labels, pred, legend=None):

    fpr, tpr, thres = roc_curve(labels, pred)
    rocauc = auc(fpr, tpr)

    if legend:
        plt.plot(fpr, tpr, lw=3, label=legend + ' - AUC = %0.2f' % rocauc)
    else:
        plt.plot(fpr, tpr, lw=3, label='AUC = %0.2f' % rocauc)
    plt.plot([0, 1], [0, 1], color='black', lw=3, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.legend(loc="lower right", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)


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

    fsize = 20

    plt.figure(figsize=(6,6))
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

    result_dir = './results'

    model_types = ['RESNET', 'LCNN']
    feature_types = ['MelSpec', 'LogSpec', 'MFCC', 'LFCC']
    win_lens = [5.0, 10.0]

    classification_type = 'binary'

    for model in model_types:
        for win_len in win_lens:

            plt.figure(figsize=(8,8))
            plt.title(f"{model} - {win_len}sec")

            for feature in feature_types:

                model_name = f"{model}_{classification_type}_{feature}_{win_len}sec.txt"
                results = pd.read_csv(os.path.join(result_dir, model_name), sep=' ', header=None)
                results.columns = ['filename', 'pred', 'label']

                plot_roc_curve(results['label'], results['pred'], legend=feature)

                label_dict = {'Healthy': 0, 'Unhealthy': 1}
                plot_confusion_matrix(results['label'], np.round(results['pred']), normalize=True, label_dict=label_dict)

                eer, rocauc = compute_eer_auc(results['label'], results['pred'])
                print(f"{model}_{classification_type}_{feature}_{win_len}sec - EER: {eer:.2f} - ROC-AUC: {rocauc:.2f}")

            plt.show()


    classification_type = 'multi'

    for model in model_types:
        for feature in feature_types:
            for win_len in win_lens:

                label_dict = {'Healthy': 0, 'COPD': 1, 'URTI': 2, 'Asthma': 3, 'LRTI': 4, 'Bronchiectasis': 5,
                              'Pneumonia': 6, 'Bronchiolitis': 7}

                model_name = f"{model}_{classification_type}_{feature}_{win_len}sec.txt"
                results = pd.read_csv(os.path.join(result_dir, model_name), sep=' ', header=None)
                results.columns = ['filename', 'pred', 'label']

                bal_acc = accuracy_score(results['label'], results['pred'])
                print(f"{model}_{classification_type}_{feature}_{win_len}sec - Bal. Acc.: {bal_acc:.2f}")

                plot_confusion_matrix(results['label'], results['pred'], normalize=True, label_dict=label_dict)


if __name__ == '__main__':

    visualize_results()

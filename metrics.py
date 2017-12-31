import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score,accuracy_score, average_precision_score,precision_recall_curve, \
    classification_report,roc_auc_score,roc_curve, recall_score


def pr(truths,scores,filepath):
    average_precision = average_precision_score(truths, scores)
    precision, recall, _ = precision_recall_curve(truths, scores)
    print("precision: ")
    print(average_precision)
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(filepath)


#roc_auc_score + curve
def roc(truths,scores,filepath):
    roc_auc = roc_auc_score(truths,scores)
    print("roc_auc: ")
    print(roc_auc)
    fpr,tpr,_ = roc_curve(truths,scores)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(filepath)

def gen_metrics(truths,preds,scores):

    pr_filepath = 'pr1.png'
    roc_filepath = 'roc1.png'

    #accuracy
    accuracy = accuracy_score(truths, preds)
    print("accuracy: ")
    print(accuracy)

    #recall
    rscore = recall_score(truths,preds,average='binary')
    print("rscore: ")
    print(rscore)

    #precision_recall
    pr(truths,scores,pr_filepath)

    #roc
    roc(truths,scores,roc_filepath)

import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def class_metrics(y_true, y_pred, rfc):

    cm = confusion_matrix(y_true, y_pred).astype('float')
    row_sums = cm.sum(axis=1, keepdims=True)
    cm /= row_sums

    print("Confusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=rfc.classes_)

    disp.plot()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy*100:.1f}")
    print(f"Precision: {precision*100:.1f}")
    print(f"Recall: {recall*100:.1f}")
    print(f"F1-score: {f1*100:.1f}")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4), sharex=True, sharey=True)
    ax1.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax2.step(recall, precision, where='post')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall curve: AP={average_precision:.2f}')
    fig.tight_layout()
    plt.show()

    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
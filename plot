import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming y_true, y_scores_model1, y_scores_model2, y_scores_model3 are defined

# Compute ROC curve and ROC area for each model
fpr1, tpr1, _ = roc_curve(y_true, y_scores_model1)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, _ = roc_curve(y_true, y_scores_model2)
roc_auc2 = auc(fpr2, tpr2)

fpr3, tpr3, _ = roc_curve(y_true, y_scores_model3)
roc_auc3 = auc(fpr3, tpr3)

# Plotting
plt.figure()
plt.plot(fpr1, tpr1, color='blue', lw=2, label='Model 1 (AUC = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='green', lw=2, label='Model 2 (AUC = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='red', lw=2, label='Model 3 (AUC = %0.2f)' % roc_auc3)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


roc_auc_plot_path = '/mnt/data/roc_auc_comparison_plot.png'
plt.savefig(roc_auc_plot_path, format='png')

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





def calculate_lift(y_true, y_scores, percentile):
    # Calculate the number of instances to consider
    top_k = int(len(y_scores) * percentile / 100)
    
    # Get indices of the top k scores
    indices = np.argsort(y_scores)[::-1][:top_k]
    
    # Calculate the number of positive instances in the top k
    positives_top_k = np.sum(y_true[indices])
    
    # Calculate the expected positives in a random sample of the same size
    expected_positives = np.sum(y_true) * top_k / len(y_true)
    
    # Avoid division by zero
    if expected_positives == 0:
        return 0

    # Calculate lift
    lift = positives_top_k / expected_positives
    return lift

# Generate lift values for each model
percentiles = np.arange(5, 0, -0.2)  # From 5% to 0%, in steps of 0.2%
lift_values_model1 = [calculate_lift(y_true, y_scores_model1, p) for p in percentiles]
lift_values_model2 = [calculate_lift(y_true, y_scores_model2, p) for p in percentiles]
lift_values_model3 = [calculate_lift(y_true, y_scores_model3, p) for p in percentiles]



# Re-plotting with the x-axis not inverted (0% at left, 5% at right)
plt.figure(figsize=(10, 6))
plt.plot(percentiles, lift_values_model1, marker='o', color='blue', label='Model 1')
plt.plot(percentiles, lift_values_model2, marker='o', color='green', label='Model 2')
plt.plot(percentiles, lift_values_model3, marker='o', color='red', label='Model 3')

plt.xlabel('Top Percentile')
plt.ylabel('Lift')
plt.title('Lift Curve Comparison')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the corrected lift curve plot
corrected_lift_curve_plot_path = '/mnt/data/corrected_lift_curve_comparison_plot.png'
plt.savefig(corrected_lift_curve_plot_path, format='png')

corrected_lift_curve_plot_path


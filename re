python
Copy code
# ... [previous code for training and prediction]

    # Get the indices of the top 10% predictions
    top_10_percent_indices = np.argsort(y_pred_prob)[-int(0.1 * len(y_pred_prob)):]

    # Number of actual positives in the top 10%
    actual_positives = y_test[top_10_percent_indices].sum()

    # Expected positives in a random 10% sample
    expected_positives = y_test.sum() * 0.1

    # Calculate lift
    lift = actual_positives / expected_positives
    lift_list.append(lift)
    print(f"Lift for fold {len(lift_list)}: {lift:.4f}")

# ... [rest of the code for average lift]
This version simplifies the steps to compute the lift by directly extracting the top 10% indices and using them to compute the actual number of positives. The lift is then straightforwardly calculated as the ratio of actual to expected positives.






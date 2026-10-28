
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_models(trained_models, X_train, y_train, X_test, y_test):

    results = []

    for name, model in trained_models.items():

        # Training predictions
        y_train_pred = model.predict(X_train)

        # Testing predictions
        y_test_pred = model.predict(X_test)

        # Training and testing accuracy
        training_accuracy = accuracy_score(y_train, y_train_pred)
        testing_accuracy = accuracy_score(y_test, y_test_pred)

        # Testing evaluation metrics
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)

        results.append({
            "Model": name,
            "Training Accuracy": training_accuracy,
            "Testing Accuracy": testing_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    results_df = pd.DataFrame(results)

    return results_df


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score , accuracy_score
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

cross = '2'
model = 'LDA'
modal = 'Text'
c = None
pca = 0.35
kernel = 'linear'
inner_cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=None)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

# Load data
data = pd.read_csv(f'E:\\{modal}.csv')
self_esteem = data.iloc[:, -1]
low_self_esteem = (self_esteem < 31).astype(int)
ids = data.iloc[:, 0]
if modal == 'Voice':
    features1 = data.iloc[:, 1:89]
    features2 = data.iloc[:, 89:177]
    features3 = data.iloc[:, 177:-1]
    features2.columns = [col.split('.2')[0] for col in features3.columns]
    features3.columns = [col.split('.2')[0] for col in features3.columns]
if modal == 'Text':
    features1 = data.iloc[:, 1:257]
    features2 = data.iloc[:, 257:513]
    features3 = data.iloc[:, 513:-1]
    features2.columns = [col.split('.2')[0] for col in features3.columns]
    features3.columns = [col.split('.2')[0] for col in features3.columns]

# Initialize a DataFrame to store results from all experiments
overall_results = pd.DataFrame(
    columns=['Experiment', 'Average Accuracy', 'Average Precision', 'Average Recall', 'Average F1'])
if model == 'LR':
    selection = 'LASSO'
if model == 'SVC':
    selection = 'SVC'
if model == 'NBC':
    selection = 'NBC'
if model == 'LDA':
    selection = 'LDA'

# Run 30 experiments
for study_num in range(30):
    print(f"\n study {study_num + 1} / 30")
    results_list = []
    total_accuracy = []
    total_precision = []
    total_recall = []
    total_f1 = []
    total_fold_number = 1
    acc_outer = []

    # Perform 10-fold cross-validation to split training and test sets
    for fold_number, (train_index, test_index) in enumerate(outer_cv.split(features3, low_self_esteem)):
        # training state using the feature set of task3
        X_train = features3.iloc[train_index]
        y_train = low_self_esteem.iloc[train_index]
        # testing state using the feature set of task1 or task2
        if cross == '1':
            X_test = features1.iloc[test_index]
            y_test = low_self_esteem.iloc[test_index]
        if cross == '2':
            X_test = features2.iloc[test_index]
            y_test = low_self_esteem.iloc[test_index]
        ids_test = ids.iloc[test_index]

        if model == 'SVC':
            classifier = SVC(C=c, kernel=kernel, class_weight='balanced', probability=True)
        if model == 'LR':
            classifier = LogisticRegression(C=c, penalty='l2', class_weight='balanced',
                                            max_iter=1000, n_jobs=-1)
        if model == 'NBC':
            classifier = GaussianNB()

        if model == 'LDA':
            classifier = LinearDiscriminantAnalysis()

        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca)),
            ('model', classifier)
        ])
        best_model = final_pipeline.fit(X_train, y_train)

        # Evaluate performance on the training set
        y_train_pred = best_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, pos_label=1, zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, pos_label=1, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, pos_label=1, zero_division=0)

        # Evaluate on the test set
        y_test_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)

        # Print results
        print(
            f"Training - Accuracy: {train_accuracy:2f}, Precision: {train_precision:2f}, Recall: {train_recall:2f}, F1 Score: {train_f1:2f}")
        print(
            f"Testing - Accuracy: {test_accuracy:2f}, Precision: {test_precision:2f}, Recall: {test_recall:2f}, F1 Score: {test_f1:2f}")

        # Store results for this fold
        result_df = pd.DataFrame({
            'Fold': [total_fold_number],
            'ID': [ids_test.tolist()],
            'True Label': [y_test.tolist()],
            'Predicted Label': [y_test_pred.tolist()],
            'ACC': [(y_test == y_test_pred).astype(int).tolist()]
        })
        results_list.append(result_df)

        # Accumulate fold results
        total_accuracy.append(test_accuracy)
        total_precision.append(test_precision)
        total_recall.append(test_recall)
        total_f1.append(test_f1)
        total_fold_number += 1

    # Print average metrics across folds
    print(f"Average Accuracy: {np.mean(total_accuracy) * 100:.2f}%")
    print(f"Average Precision: {np.mean(total_precision) * 100:.2f}%")
    print(f"Average Recall: {np.mean(total_recall) * 100:.2f}%")
    print(f"Average F1: {np.mean(total_f1) * 100:.2f}%")

    # Record overall results for this experiment
    summary_df = pd.DataFrame({
        'Experiment': [study_num + 1],
        'Average Accuracy': [np.mean(total_accuracy)],
        'Average Precision': [np.mean(total_precision)],
        'Average Recall': [np.mean(total_recall)],
        'Average F1': [np.mean(total_f1)]
    })
    overall_results = pd.concat([overall_results, summary_df], ignore_index=True)

# Save summary results from all 30 experiments
overall_df = pd.DataFrame(overall_results)
overall_df.to_excel(f'E:\\.xlsx', index=False)
print(classifier.get_params())

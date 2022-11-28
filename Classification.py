import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Import data
label_table = pd.read_csv('Datasets/ElNino/ElNino_ERSST_label.csv')
label = label_table.x.to_numpy()

# FAE representation
FAE_reps_all = pred(model, x)[1]
FAE_reps_all_identity = pred(model, x)[1]
AE_reps_all = pred_AE(x)[1]
# FPCA representation
fd_all = representation.grid.FDataGrid(x.numpy(), tpts_fd)
basis_fd_all = fd_all.to_basis(bss_fpca)
fpc_scores_all = fpca_basis.transform(basis_fd_all)

TrainLabel = label[train_no]
FAE_reps_train = FAE_reps_all[train_no].detach().numpy()
FAE_reps_train_identity = FAE_reps_all_identity[train_no].detach().numpy()
AE_reps_train = AE_reps_all[train_no].detach().numpy()
fpc_scores_train = fpc_scores_all[train_no]
if split.rate == 1:
    TestLabel=label
    FAE_reps_train = FAE_reps_all.detach().numpy()
    FAE_reps_train_identity = FAE_reps_all_identity.detach().numpy()
    AE_reps_train = AE_reps_all.detach().numpy()
    fpc_scores_train = fpc_scores_all
else:
    TestLabel = label[[i for i in range(len(x)) if i not in train_no]]
    FAE_reps_test = FAE_reps_all[[i for i in range(len(x)) if i not in train_no]].detach().numpy()
    FAE_reps_test_identity = FAE_reps_all_identity[[i for i in range(len(x)) if i not in train_no]].detach().numpy()
    AE_reps_test = AE_reps_all[[i for i in range(len(x)) if i not in train_no]].detach().numpy()
    fpc_scores_test = fpc_scores_all[[i for i in range(len(x)) if i not in train_no]]


# Create classifiers (logistic regression) & train the model with the training set
FAE_classifier = LogisticRegression(solver='liblinear', random_state=0, multi_class='auto').fit(FAE_reps_train, TrainLabel)
FAE_identity_classifier = LogisticRegression(solver='liblinear', random_state=0, multi_class='auto').fit(FAE_reps_train_identity, TrainLabel)
AE_classifier = LogisticRegression(solver='liblinear', random_state=0, multi_class='auto').fit(AE_reps_train, TrainLabel)
FPCA_classifier = LogisticRegression(solver='liblinear', random_state=0, multi_class='auto').fit(fpc_scores_train, TrainLabel)

# Evaluate the classifier with the test set
# FAE_classifier.predict_proba(FAE_reps_test)
FAE_classifier.predict(FAE_reps_test)
FAE_identity_classifier.predict(FAE_reps_test_identity)
AE_classifier.predict(AE_reps_test)
FPCA_classifier.predict(fpc_scores_test)

# Classification accuracy on the test set
FAE_classifier.score(FAE_reps_test, TestLabel)
FAE_identity_classifier.score(FAE_reps_test_identity, TestLabel)
AE_classifier.score(AE_reps_test, TestLabel)
FPCA_classifier.score(fpc_scores_test, TestLabel)

# Classification accuracy on the training set
FAE_classifier.score(FAE_reps_train, TrainLabel)
FAE_identity_classifier.score(FAE_reps_train_identity, TrainLabel)
AE_classifier.score(AE_reps_train, TrainLabel)
FPCA_classifier.score(fpc_scores_train, TrainLabel)
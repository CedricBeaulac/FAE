import sklearn
from sklearn.cluster import KMeans

# Functions for evaluation
def most_frequent(label_list):
    counter = 0
    for i in range(0, optimal_n_cluster):
        no = np.count_nonzero(label_list == i)
        if no > counter:
            counter = no
            num = i
    return num

def acc(label_list):
    most_cluster = most_frequent(label_list)
    most_no = np.count_nonzero(label_list == most_cluster)
    prob = most_no/len(label_list)
    return prob

# Import data
label_table = pd.read_csv('Datasets/ElNino/ElNino_ERSST_label.csv')
label = label_table.x.to_numpy()

TrainLabel = label[train_no]
if split.rate == 1:
    TestLabel=label
else:
    TestLabel = label[[i for i in range(len(x)) if i not in train_no]]

# FAE representation
FAE_reps_all = pred(model, x)[1]
FAE_reps_all_identity = pred(model, x)[1]
AE_reps_all = pred_AE(x)[1]
# FPCA representation
fd_all = representation.grid.FDataGrid(x.numpy(), tpts_fd)
basis_fd_all = fd_all.to_basis(bss_fpca)
fpc_scores_all = fpca_basis.transform(basis_fd_all)


optimal_n_cluster = 4
kmeans_par = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 0}
kmeans_labels_FAE = KMeans(n_clusters=optimal_n_cluster, **kmeans_par).fit_predict(FAE_reps_all.detach().numpy())
kmeans_labels_FAE_identity = KMeans(n_clusters=optimal_n_cluster, **kmeans_par).fit_predict(FAE_reps_all_identity.detach().numpy())
kmeans_labels_FPCA = KMeans(n_clusters=optimal_n_cluster, **kmeans_par).fit_predict(fpc_scores_all)
kmeans_labels_AE = KMeans(n_clusters=optimal_n_cluster, **kmeans_par).fit_predict(AE_reps_all.detach().numpy())
for i in range(optimal_n_cluster):
    no_FAE = np.count_nonzero(kmeans_labels_FAE == i)
    no_FAE_identity = np.count_nonzero(kmeans_labels_FAE_identity == i)
    no_FPCA = np.count_nonzero(kmeans_labels_FPCA == i)
    no_AE = np.count_nonzero(kmeans_labels_AE == i)
    print( no_FAE,"(FAE) &", no_FAE_identity, "(FAE_linear) &",
           no_FPCA, "(FPCA) &", no_AE, "(AE) employee(s) for Cluster", i + 1,  )
print("-------------------------------------------------")


acc_list_FAE=[]
acc_list_FAE_identity =[]
acc_list_FPCA=[]
acc_list_AE = []
label_list_FAE=[]
label_list_FAE_identity =[]
label_list_FPCA=[]
label_list_AE = []
for i in range(1, optimal_n_cluster+1):
    acc_list_FAE.append(acc(kmeans_labels_FAE[(69*(i-1)):(69*i)]))
    acc_list_FAE_identity.append(acc(kmeans_labels_FAE_identity[(69 * (i - 1)):(69 * i)]))
    acc_list_FPCA.append(acc(kmeans_labels_FPCA[69 *(i-1):69 * i]))
    acc_list_AE.append(acc(kmeans_labels_AE[(69 * (i - 1)):(69 * i)]))

    label_list_FAE.append(most_frequent(kmeans_labels_FAE[(69*(i-1)):(69*i)]))
    label_list_FAE_identity.append(most_frequent(kmeans_labels_FAE_identity[(69 * (i - 1)):(69 * i)]))
    label_list_FPCA.append(most_frequent(kmeans_labels_FPCA[69 *(i-1):69 * i]))
    label_list_AE.append(most_frequent(kmeans_labels_AE[(69 * (i - 1)):(69 * i)]))

acc_list_FAE
mean(acc_list_FAE)
std(acc_list_FAE)

acc_list_FAE_identity
mean(acc_list_FAE_identity)
std(acc_list_FAE_identity)

acc_list_FPCA
mean(acc_list_FPCA)
std(acc_list_FPCA)

acc_list_AE
mean(acc_list_AE)
std(acc_list_AE)

raw = x.numpy()
FAE_all = pred(model, x)[0].detach().numpy()
AE_all = pred_AE(x)[0].detach().numpy()

fd_all = representation.grid.FDataGrid(x.numpy(), tpts_fd)
basis_fd_all = fd_all.to_basis(bss_fpca)
# fpca_basis = fpca_basis.fit(basis_fd_train)
fpc_scores_all = fpca_basis.transform(basis_fd_all)
FPCA_all = fpca_basis.inverse_transform(fpc_scores_all)._evaluate(tpts_fd)[:,:,0]


raw_color = ["b", "g", "r","y"] # 0-b, 1-g, 2-r, 3-y
# AE_color = ["g", "b", "r","y"] #1-2-0(2)-3
# FPCA_color = ["r", "y", "b", "g"] #2-3-0-1
# FAE_color = ["y", "b", "g", "r"] #1-2-3-0
AE_color = [None] * 4
FPCA_color = [None] * 4
FAE_color = [None] * 4
for i in range(len(raw_color)):
    AE_color[label_list_AE[i]] = raw_color[i]
    FPCA_color[label_list_FPCA[i]] = raw_color[i]
    FAE_color[label_list_FAE[i]] = raw_color[i]

plt.figure(1, figsize=(20, 20))
plt.subplot(221)
for m in range(0, len(raw)):
# for m in id_plt:
    plt.plot(tpts, raw[m], raw_color[label[m]-1])
plt.title("Original Labelling")
plt.subplot(222)
for m in range(0, len(AE_all)):
# for m in id_plt:
    plt.plot(tpts, AE_all[m], AE_color[kmeans_labels_AE[m]])
plt.title("AE-representation Labelling")
plt.subplot(223)
for m in range(0, len(FPCA_all)):
# for m in id_plt:
    plt.plot(tpts, FPCA_all[m], FPCA_color[kmeans_labels_FPCA[m]])
plt.title("FPCA-representation Labelling")
plt.subplot(224)
for m in range(0, len(FAE_all)):
# for m in id_plt:
    plt.plot(tpts, FAE_all[m], FAE_color[kmeans_labels_FAE[m]])
plt.title("FAE-representation Labelling")
plt.show()

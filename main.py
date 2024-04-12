import os, time
from tqdm import tqdm
import pandas as pd
import numpy as np # type: ignore
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression, r_regression, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix #, roc_curve, roc_auc_score, recall_score, precision_score,
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier #,GradientBoostingClassifier
from xgboost import XGBClassifier

from helpers import get_ip_address, has_write_permission, measure_performance

import warnings
warnings.filterwarnings("ignore") # 경고 메시지 무시

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RANDOM_SEED = 42

# ----------------------------------------------------------------------------------------------------------------

class data_loader:
    def __init__(self, X_path, sample_annotation_file):
        super().__init__()
        
        self.target_label_name ='Population code'

        self.X = np.load(X_path)

        self.sample_annotation_df = pd.read_csv(sample_annotation_file, sep='\t')
        self.y = self.sample_annotation_df[self.target_label_name]
        logging.info(f"[progress] Read data done. X.shape: {self.X.shape}, y.shape: {self.y.shape}")

        self.drop_notusing_sample()
        self.y_encoded = self.encode_y()
        self.train_indices, self.test_indices = self.split_dataset()

        assert self.X.shape[0] == self.y.shape[0]
        assert self.X.shape[0] == self.y_encoded.shape[0]

    def drop_notusing_sample(self):
        indices_to_drop = self.sample_annotation_df[self.sample_annotation_df[self.target_label_name] == 'IBS,MSL'].index

        if not indices_to_drop.empty:
            self.sample_annotation_df = self.sample_annotation_df.drop(indices_to_drop)
            self.y = self.y.drop(indices_to_drop)
            self.X = np.delete(self.X, indices_to_drop, axis=0)

        logging.info(f"[progress] Dropped {len(indices_to_drop)} samples from the dataset. X.shape: {self.X.shape}, y.shape: {self.y.shape}")

    def encode_y(self):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(self.y)
        return(y_encoded)

    def split_dataset(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = RANDOM_SEED)
        train_indices, test_indices = next(sss.split(self.X, self.y))
        return(train_indices, test_indices)


    def get_data(self):
        return self.X, self.y_encoded, self.train_indices, self.test_indices
        
    def get_combined_df(self):
        df = pd.DataFrame(self.X)
        new_columns = ['com' + str(i) for i in range(1, len(df.columns) + 1)]
        df.columns = new_columns
        df['country_encoded'] = self.y_encoded

        return df
        
@measure_performance
def train_ML(X_train, y_train, X_test, method = "SVM"):
    if method == "SVM":
        params = {'C': 0.1, 'gamma': 0.01, 'kernel': 'linear'} 
        svm_model = SVC(**params, random_state = RANDOM_SEED)
        svm_model.fit(X_train, y_train)    

        y_pred = svm_model.predict(X_test)
        return y_pred

        # #XGB
        # xgboost_params = {
        #     # 'max_depth': 3,
        #     'learning_rate': 0.1,
        #     # 'n_estimators': 100,
        #     'gamma': 0, # default
        #     'subsample': 1, # default
        #     # 'random_state': random_seed
        # }
        
        # xgboost_model = XGBClassifier(**xgboost_params)
        # xgboost_model.fit(self.X_train, self.y_train)

        # #DT
        # decision_tree_model = DecisionTreeClassifier(random_state=42)
        # decision_tree_model.fit(self.X_train, self.y_trainn)

        # #pred
        # SVM_pred = svm_model.predict(self.X_test)
        # XGB_pred = xgboost_model.predict(self.X_test)
        # DT_pred = decision_tree_model.predict(self.X_test)

        # if self.svm == True :
        #     return SVM_pred
        # else:
        #     return SVM_pred, XGB_pred, DT_pred
    
def evaluate_performance(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return [accuracy, f1_micro, f1_macro, f1_weighted, conf_matrix]

def uni_feature_selection(X, y, score_func, n):
    print(f"input array shape : {X.shape}, {y.shape}. n = {n}")
    X_selected = SelectKBest(score_func, k = n).fit_transform(X, y)
    print(f"output array shape : {X_selected.shape}")

    return(X_selected)

@measure_performance
def select_feature(X, y, method, n, df = None, rf_selection=False): 
    if method == "rf":
        #select feature by random forest method
        if rf_selection == True:
            skf = StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state = RANDOM_SEED)
            lst_results = []
            i = 1
            for train_idx, test_idx in skf.split(X,y):
                X_train = X[train_idx, :-1]
                X_test = X[test_idx, :-1]
                y_train = y[train_idx]
                y_test = y[test_idx]

                idx1 = list(X_train)
                idx2 = list(X_test)
                
                clf = RandomForestClassifier(n_estimators=1000)
                start_time = time.time()
                clf.fit(X_train, y_train)
                end_time = time.time()
                training_time = end_time - start_time
                
                with open(f'clf_rf{i}_anal.pickle_auc', 'wb') as f:
                    pickle.dump(clf, f)

                pred = clf.predict(X_test)
                pred_proba = clf.predict_proba(X_test)    
                accuracy = accuracy_score(y_test, pred) 

                lst_results.append([i, 'Random Forest', idx1, idx2, accuracy,  training_time])
                print("Random Forest_{}".format(i))
                i += 1 
                # loop

                # df_results : Check AUC
                df_results = pd.DataFrame(data=lst_results, columns=['iter', 'method', 
                                                                    'train_idx', 'test_idx',
                                                                    'accuracy', 'training_time'])
            else:
                pass
            # ended loop and reload
        for i in range(1,11):
            with open (f'clf_rf{i}_anal.pickle_auc', 'rb') as f:
                globals()[f'clf_rf{i}'] = pickle.load(f)
        
        sum = []
        for i in range(1,11):
            sum.append(globals()[f'clf_rf{i}'].feature_importances_)

        np.array(sum).mean(axis=0)

        fi_rf = pd.DataFrame(np.array(sum).mean(axis=0))
        # fi_rf.index = [f'com{i}' for i in range(5105448)]
        fi_rf.index = [f'com{i}' for i in range(9190)]
        fi_rf_rank = fi_rf.copy()

        for col in fi_rf_rank.columns[:]:
            fi_rf_rank[col] = fi_rf_rank[col].rank(ascending=False)
        list(fi_rf_rank.sort_values(by=0).T)
        sort_list = list(fi_rf_rank.sort_values(by=0).T)
        
        #sort_list

        initial_num_features = 128
        # final_num_features = 4194304  # max_num
        final_num_features = 8192  # test_max_num

        dfs = []
        num_features = initial_num_features
        while num_features <= final_num_features:
            selected_features = sort_list[:num_features]

            selected_df = selected_features.copy()
            dfs.append(selected_df)

            num_features *= 2

        X_selected = [] # all_list

        var_names = [] # list_num1...16 

        for i, df in enumerate(dfs):
            var_name = f"data_{i+1}"
            globals()[var_name] = df
            var_names.append(var_name)
            X_selected.append(np.array(globals()[var_name]))

        print("Name of list:", var_names)
        """
        1. X_selected를 Numpy.array로 바꾸는 것을 어떻게 해야할지 고민입니다. 
        feature selection을 한 각 데이터들은 numpy로 바꾸었는데 한번에 가져오려고하니까 리스트 말고는 생각이 안나고 
        
        2. 이를 해결해보려고했더니 밑의 
        logging.info(f"[progress] '{feature_select_method}' feature selection selected {n_select} variants. X_selected.shape = {X_selected.shape}. perf_metrics_selection: {perf_metric_select}")
        이 코드에서 Dataframe으로 하신건지 궁금하고, 제가 X_selected를 list로 하다보니 저 부분에서 오류가 발생합니다. 

        3. 마지막으로 이렇게 loop 만든 형태가 위에서 SVM에 넣었을 때 잘 돌아가는지 교수님도 loop로 하신건지 궁금해서 여쭤봅니다.
        """
    else:
        np.random.seed(1004)
        num_snps_before = X.shape[1]

        if method in ["random", "variance"]:
            if method == "random":
                boolean_mask = np.zeros(num_snps_before, dtype=bool)
                selected_indices = np.random.choice(num_snps_before, n, replace=False)
                boolean_mask[selected_indices] = True

            elif method == "variance":
                threshold = 0.1
                batch_process = False
                # from sklearn.feature_selection import VarianceThreshold
                # selector = VarianceThreshold(threshold = threshold)
                # genotype_array_filtered = selector.fit_transform(genotype_array)
                # boolean_mask = selector.get_support()
                # print(f"This filter will return {genotype_array_filtered.shape} /", genotype_array.shape[1], f"variants (")
                # print(f"This filter will retain {boolean_mask.sum()} /", genotype_array.shape[1], f"variants (", boolean_mask.sum()/genotype_array.shape[1] * 100,"%)")

                if batch_process:
                    batch_size=100000
                    n_samples, n_snps = X.shape
                    variances = np.zeros((n_snps, feature_dim))
                
                    for start in tqdm(range(0, n_snps, batch_size)):
                        end = min(start + batch_size, n_snps)
                        batch_var = np.var(genotype_array_onehot[:, start:end, :], axis=0)
                        variances[start:end, :] = batch_var
                else:
                    variances = np.var(genotype_array_onehot, axis=0)
                
                boolean_mask = (variances > threshold).any(axis=1)
                print(f"Variance filter (threshold = {threshold}) will retain {boolean_mask.sum()} /", genotype_array_onehot.shape[1], f"variants (", boolean_mask.sum()/genotype_array_onehot.shape[1] * 100,"%)")

                # sns.histplot(variances.reshape(-1), bins=100, kde=True)  # 'bins' controls the number of bins, 'kde' adds a Kernel Density Estimate plot
                # plt.title('Histogram of Data')
                # plt.xlabel('Variances')
                # plt.ylabel('Frequency')
                # plt.show()
                raise(NotImplemented)

            X_selected = X[:, boolean_mask]
            num_snps_after = X_selected.shape[1]

            assert boolean_mask.sum() == num_snps_after

            
        elif method in ["chi2", "f_classif", "mutual_info_classif"]:
            if method == "chi2":
                score_fun = chi2
            elif method == "f_classif":
                score_fun = f_classif
            elif method == "mutual_info_classif":
                score_fun = mutual_info_classif
            else:
                raise

            X_selected = uni_feature_selection(X = snp_dataset.genotype_array, 
                                               y = sample_annotation_df["Population code"],
                                               score_func = score_fun,
                                               n = n)

        elif method == "recursive_feature_selection":
            print(f"input array shape : {X.shape}, {y.shape}. n = {n}")
            knn = KNeighborsClassifier(n_neighbors = n)
            sfs = SequentialFeatureSelector(knn, n_features_to_select = n)
            sfs.fit(X, y)

            #sfs.get_support()
            X_selected = sfs.transform(X)
            print(f"output array shape : {X_selected.shape}")

    return(X_selected)

### 각 Task별 조건으로 해서 1은 ab만 2는 a만 이런식으로해서.
### 시각화는 따로

########################################################

# class Feature_important:

#     def __init__(self):
#         pass

#     def Feature_selection(self):

#         StratifiedShuffleSplit
#         RandomForestClassifier
#         # list로 추출 -> 저장(파일)
#         return sort_list

#     def random_sample(self):

#         # list 생성
#         return list


# class train(self):
    
#     def model(self, svm = True):
#         # seed
#         random_seed =42
#         random_state = random_seed
#         SVM_parameter = {'C':, 'gamma':, 'kernel' = }
#         XGB_parameter = {
#             # 'max_depth': 3,
#             'learning_rate': 0.1,
#             # 'n_estimators': 100,
#             'gamma': 0, # default
#             'subsample': 1, # default
#             # 'random_state': random_seed
#         }

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
def draw_PCA(X, y):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label, alpha=0.5)
    plt.title(f'PCA of {snp_dataset.genotype_array.shape[1]} SNPs')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc=1, prop={'size': 5})

    plt.show()

def draw_tSNE(X, y):
    tsne = TSNE(n_components=2, verbose=1)
    tsne_result = tsne.fit_transform(X)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 2)
    labels_unique = np.unique(y)
    colors = cm.viridis(np.linspace(0, 1, len(labels_unique)))  # Using viridis colormap

    for label, color in zip(labels_unique, colors):
        indices = np.where(y == label)
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=label, alpha=0.3,
                    #color=color,
                    ) 

    plt.title(f't-SNE of {snp_dataset.genotype_array.shape[1]} SNPs')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(loc=1, prop={'size': 5})

    plt.show()


def main():
    ## ----- setup environment
    data_locations = {
        '223.195.111.48': '/project/datacamp/team11/data',
        '147.47.44.229': '/home/jinhyun/data/1kGP',
    }

    chr_list = [str(x) for x in range(1,23)]
    gt_dict = {"0|0" :0, "0|1" : 1, "1|0" : 2, "1|1" : 3 } # genotype dict for converting string-> inteter 

    raw_data_path = data_locations.get(get_ip_address(), '/not_found')
    sample_annotation_file = os.path.join(raw_data_path, "igsr-1000 genomes 30x on grch38.tsv")
    preprocess_path = os.path.join(raw_data_path, "preprocessed")

    assert os.path.exists(preprocess_path), f"Data path not exists: {raw_data_path} OR IP setting is incorrect: {get_ip_address()}"
    assert os.path.isfile(sample_annotation_file), f"File not exists : {sample_annotation_file}"
    # assert has_write_permission(preprocess_path), f"You do not have write permission for {preprocess_path}"


    ## ----- data loader 
    # merged_support3_variance_0.1 # Real_data
    # merged_support3_variance_0.1 # Test_data
    target_feature = "merged_support3_variance_0.2499999"
    dataset = data_loader(os.path.join(preprocess_path, target_feature + "_matrix.npy"), 
                       sample_annotation_file)
    X, y, train_indices, test_indices = dataset.get_data()
    # df = dataset.get_combined_df()

    result_combined = []
    for feature_select_method in ["rf"]:#["random", "rf", "variance", "chi2", "f_classif", "mutual_info_classif"]:
        for n_select in [128, 256]:
            X_selected, perf_metric_select = select_feature(X = X, y = y, method = feature_select_method, n = n_select)
            logging.info(f"[progress] '{feature_select_method}' feature selection selected {n_select} variants. X_selected.shape = {X_selected.shape}. perf_metrics_selection: {perf_metric_select}")

            X_train, X_test = X_selected[train_indices], X_selected[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            logging.info(f"[progress] Start ML training")
            logging.info(f" - X_train.shape = {X_train.shape} X_test.shape = {X_test.shape} ")

            y_pred, perf_metric_train = train_ML(method = "SVM", X_train = X_train, y_train = y_train, X_test = X_test)
            logging.info(f"[progress] Train done with perf_metrics_train: {perf_metric_train}")

            metrics = evaluate_performance(y_test, y_pred)
            logging.info(f"[progress] Accuracy: {metrics[0]*100:.4f}")

    #         result = [] # combine perf_metric_select, perf_metric_train, results
    #         result_combined.append(result)

    # results_df = pd.DataFrame(result_combined)
    # results_df.to_excel("result.xlsx")


if __name__ == "__main__":
    main()

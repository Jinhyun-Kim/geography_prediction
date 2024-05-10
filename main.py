import os, time
from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support #, roc_curve, roc_auc_score, recall_score, precision_score,
from sklearn.preprocessing import LabelEncoder#, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier #, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
# from umap import UMAP

from xgboost import XGBClassifier

from helpers import get_ip_address, has_write_permission, measure_performance

import warnings
warnings.filterwarnings("ignore") # 경고 메시지 무시

import logging

# Set up logging
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.INFO)  # Set the minimum logging level

log_file_handler = logging.FileHandler('logs.txt')
log_file_handler.setLevel(logging.INFO)  # Set the logging level for the file
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(log_file_handler)

log_console_handler = logging.StreamHandler()
log_console_handler.setLevel(logging.INFO)
log_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(log_console_handler)

RANDOM_SEED = 42
RANDOM_SEED_DATASET = 42

class data_loader:
    def __init__(self, X_path, sample_annotation_file):
        super().__init__()
        
        self.target_label_name ='Population code'
        self.notusing_lables = ['IBS,MSL', # only 1 sample
                                'GBR', # accuracy 11%
                                'ASW', 'ACB', # accuracy ~ 60%
                                'GIH', # acuracy < 80%
                                'CHB', 'STU', 'ITU',  # accuracy < 90%
                                ]
        data_split = [0.6, 0.2, 0.2] #train, val, test

        self.X = np.load(X_path)

        self.sample_annotation_df = pd.read_csv(sample_annotation_file, sep='\t')
        self.y = self.sample_annotation_df[self.target_label_name]
        logging.info(f"[progress] Read data done. X.shape: {self.X.shape}, y.shape: {self.y.shape}")

        self.drop_notusing_sample(notusing_list= self.notusing_lables)
        self.y_encoded, self.label_mapping = self.encode_y()
        self.train_indices, self.val_indices, self.test_indices = self.split_dataset(val_size = data_split[1], test_size = data_split[2])

        logging.info(f" - Data_split: train_set (n= {len(self.train_indices)}), val_set (n= {len(self.val_indices)}), test_set (n= {len(self.test_indices)})")

        # print("class distribution: ", self.y.value_counts())

        assert self.X.shape[0] == self.y.shape[0]
        assert self.X.shape[0] == self.y_encoded.shape[0]
        assert self.test_index_coverage(self.train_indices, self.val_indices, self.test_indices, self.X.shape[0])


    def test_index_coverage(self, train_indices, val_indices, test_indices, total_length):
        combined_indices = np.concatenate((train_indices, val_indices, test_indices))
        unique_indices = np.unique(combined_indices)

        # Check if the concatenated indices cover all possible indices
        expected_indices = np.arange(total_length)

        if np.array_equal(np.sort(unique_indices), expected_indices):
            return True
        else:
            missing_indices = np.setdiff1d(expected_indices, unique_indices)
            extra_indices = np.setdiff1d(unique_indices, expected_indices)
            print(f"Missing indices: {missing_indices}")
            print(f"Extra indices: {extra_indices}")
            return False

    def drop_notusing_sample(self, notusing_list):
        indices_to_drop = self.sample_annotation_df[self.sample_annotation_df[self.target_label_name].isin(notusing_list)].index

        if not indices_to_drop.empty:
            self.sample_annotation_df = self.sample_annotation_df.drop(indices_to_drop)
            self.y = self.y.drop(indices_to_drop)
            self.X = np.delete(self.X, indices_to_drop, axis=0)

        logging.info(f"[progress] Dropped {len(indices_to_drop)} samples from the dataset. X.shape: {self.X.shape}, y.shape: {self.y.shape}")

    def encode_y(self):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(self.y)
        label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

        return y_encoded, label_mapping

    def split_dataset(self, val_size=0.15, test_size=0.15):

        sss = StratifiedShuffleSplit(n_splits=1, test_size = test_size, random_state = RANDOM_SEED)
        train_val_idx, test_indices = next(sss.split(self.X, self.y_encoded))

        adjusted_val_size = val_size / (1 - test_size)

        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=RANDOM_SEED)
        train_idx, val_idx = next(sss_val.split(self.X[train_val_idx], self.y_encoded[train_val_idx]))

        train_indices = train_val_idx[train_idx]
        val_indices = train_val_idx[val_idx]

        return train_indices, val_indices, test_indices


    def get_data(self):
        return (self.X, np.array(self.y), self.y_encoded), (self.train_indices, self.val_indices, self.test_indices), self.label_mapping
        
    def get_combined_df(self):
        df = pd.DataFrame(self.X)
        new_columns = ['com' + str(i) for i in range(1, len(df.columns) + 1)]
        df.columns = new_columns
        df['country_encoded'] = self.y_encoded

        return df
        
@measure_performance
def train_ML(X_train, y_train, X_val, y_val, X_test, params, method = "SVM"):
    if method == "SVM":
        # model = SVC(**params, random_state = RANDOM_SEED)

        # classes = np.unique(y_train)
        # class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        # class_weight_dict = dict(zip(classes, class_weights))
        # params = {**params, 'class_weight': [class_weight_dict]}

        model = SVC(random_state=RANDOM_SEED)
        grid_search = GridSearchCV(estimator=model, param_grid = params, cv=5, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)  # Fits the model on the training data
        model = grid_search.best_estimator_

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        return (y_pred_train, y_pred_val, y_pred_test, model.get_params())

    elif method == "LinearSVM":
        model = LinearSVC(penalty='l1', dual= "auto", **params, random_state=RANDOM_SEED)

    elif method == "XGB":
        model = XGBClassifier(**params, 
                              objective='multi:softmax',  # Use softmax for multi-class classification
                              num_class=len(np.unique(y_train)),  # Number of classes
                              use_label_encoder=False,
                              eval_metric='mlogloss',  # Metric used for multiclass classification
                              random_state = RANDOM_SEED)
        
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=eval_set, verbose=True)
        
        # if hasattr(model, 'best_iteration'): # this is not necessary for sklearn interface
        #     # print(f"XGB best iteration: {model.best_iteration}")
        #     y_pred_train = model.predict(X_train, iteration_range=(0, model.best_iteration+1))
        #     y_pred_val = model.predict(X_val, iteration_range=(0, model.best_iteration+1))
        #     y_pred_test = model.predict(X_test, iteration_range=(0, model.best_iteration+1))
        # else:
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        return (y_pred_train, y_pred_val, y_pred_test, model.get_params())

    elif method == "DT":
        model = DecisionTreeClassifier(random_state = RANDOM_SEED)

    elif method == "RF":
        model = RandomForestClassifier(**params, random_state=RANDOM_SEED)

    elif method == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)

    else:
        raise ValueError(f"Unsupported method: {method}")

    model.fit(X_train, y_train)    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    return (y_pred_train, y_pred_val, y_pred_test, model.get_params())

def evaluate_performance(y_test, y_pred, label_mapping, save_file_previx):
    accuracy = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    conf_matrix = confusion_matrix(y_test, y_pred)

    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    class_names = [label_mapping.get(i, f"Class_{i}") for i in range(len(np.unique(y_test)))]
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Creating a dictionary for class-specific metrics
    class_metrics = {
        f"class_{class_name}_accuracy": acc for class_name, precision, recall, f1, acc in zip(class_names, precisions, recalls, f1_scores, class_accuracies)
    }

    plt.figure(figsize=(8, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # plt.savefig(f"{save_file_previx}_confusion_matrix.pdf", dpi = 300)
    plt.show()

    metrics = {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_matrix,
        **class_metrics
    }

    return metrics


@measure_performance
def select_feature(X, y, method, n_list, train_idx, val_idx, cache_file_prefix = None, from_cache = False): 
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    X_selected_list = []

    if method == "xgb":
        # params = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 2, 'gamma': 1, "subsample": 0.8, "colsample_bytree": 0.8, 'reg_lambda': 2, 'reg_alpha': 0.5} # inital trial
        params = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1, 'reg_alpha': 0} # current best

        if not (from_cache and os.path.exists(f"{cache_file_prefix}_basic_feature_importance_mean.npy")):
            model = XGBClassifier(**params, 
                                objective='multi:softmax',  # Use softmax for multi-class classification
                                num_class=len(np.unique(y)),  # Number of classes
                                use_label_encoder=False,
                                eval_metric='mlogloss',  # Metric used for multiclass classification
                                random_state = RANDOM_SEED)
            
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, 
                        early_stopping_rounds=10, 
                        eval_set=eval_set, 
                        verbose=True)
            logging.info(f" - {method} model train done for feature selection")

            # basic tree based feature importance
            feature_importances_impurity = model.feature_importances_
            if cache_file_prefix is not None:
                np.save(f"{cache_file_prefix}_basic_feature_importance_mean.npy", feature_importances_impurity)
            
            # permutation feature importance
            # perm_importance_results = permutation_importance(model, X_val, y_val, n_repeats=5, n_jobs = 3, random_state=RANDOM_SEED)
            # perm_feature_importances = perm_importance_results.importances_mean
            # if cache_file_prefix is not None:
            #     np.save(f"{cache_file_prefix}_perm_feature_importance_mean.npy", perm_feature_importances)
            #     np.save(f"{cache_file_prefix}_perm_feature_importance_std.npy", perm_importance_results.importances_std)
        else:
            feature_importances_impurity = np.load(f"{cache_file_prefix}_basic_feature_importance_mean.npy")
            # perm_feature_importances = np.load(f"{cache_file_prefix}_perm_feature_importance_mean.npy")

        feature_importance_use = feature_importances_impurity 
        # feature_importance_use = perm_feature_importances

        for n in n_list:
            selected_indices = np.argsort(feature_importance_use)[-n:][::-1]
            X_selected = X[:, selected_indices]
            X_selected_list.append(X_selected)


    elif method == "rf":
        if len(X.shape) >= 3:
            raise NotImplemented

        params = {'n_estimators': 2000, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
        
        if not (from_cache and os.path.exists(f"{cache_file_prefix}_basic_feature_importance_mean.npy")):
            model = RandomForestClassifier(**params, random_state = RANDOM_SEED)
            model.fit(X_train, y_train)
            logging.info(f" - {method} model train done for feature selection")

            # basic tree based feature importance
            feature_importances_impurity = model.feature_importances_
            if cache_file_prefix is not None:
                np.save(f"{cache_file_prefix}_basic_feature_importance_mean.npy", feature_importances_impurity)

            # permutation feature importance
            # perm_importance_results = permutation_importance(model, X_val, y_val, n_repeats=5, n_jobs = 3, random_state=RANDOM_SEED)
            # perm_feature_importances = perm_importance_results.importances_mean
            # if cache_file_prefix is not None:
            #     np.save(f"{cache_file_prefix}_perm_feature_importance_mean.npy", perm_feature_importances)
            #     np.save(f"{cache_file_prefix}_perm_feature_importance_std.npy", perm_importance_results.importances_std)
        else:
            feature_importances_impurity = np.load(f"{cache_file_prefix}_basic_feature_importance_mean.npy")
            # perm_feature_importances = np.load(f"{cache_file_prefix}_perm_feature_importance_mean.npy")

        feature_importance_use = feature_importances_impurity 
        # feature_importance_use = perm_feature_importances

        for n in n_list:
            selected_indices = np.argsort(feature_importance_use)[-n:][::-1]
            X_selected = X[:, selected_indices]
            X_selected_list.append(X_selected)

    elif method == "svm":
        params = {'C': 0.1, 'kernel': 'linear'}
        model = SVC(**params, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        logging.info(f" - {method} model train done for feature selection")

        perm_importance_results = permutation_importance(model, X_val, y_val, n_repeats=5, n_jobs = 3, random_state=RANDOM_SEED)
        perm_feature_importances = perm_importance_results.importances_mean
        if cache_file_prefix is not None:
            np.save(f"{cache_file_prefix}_perm_feature_importance_mean.npy", perm_feature_importances)
            np.save(f"{cache_file_prefix}_perm_feature_importance_std.npy", perm_importance_results.importances_std)

        feature_importance_use = perm_feature_importances

        for n in n_list:
            selected_indices = np.argsort(feature_importance_use)[-n:][::-1]
            X_selected = X[:, selected_indices]
            X_selected_list.append(X_selected)

    else:
        num_snps_before = X.shape[1]
        for n in n_list:
            if method in ["random", "variance"]:
                if method == "random":
                    rng = np.random.default_rng(seed = RANDOM_SEED)
                    boolean_mask = np.zeros(num_snps_before, dtype=bool)
                    selected_indices = rng.choice(num_snps_before, n, replace=False)
                    boolean_mask[selected_indices] = True

                elif method == "variance":
                    batch_process = False

                    if batch_process:
                        raise NotImplemented
                        batch_size=100000
                        n_samples, n_snps = X.shape
                        variances = np.zeros((n_snps, feature_dim))
                    
                        for start in tqdm(range(0, n_snps, batch_size)):
                            end = min(start + batch_size, n_snps)
                            batch_var = np.var(genotype_array_onehot[:, start:end, :], axis=0)
                            variances[start:end, :] = batch_var
                    else:
                        variances = np.var(X, axis=0)
                    
                    selected_indices = np.argsort(variances)[-n:]
                    boolean_mask = np.zeros(num_snps_before, dtype=bool)
                    boolean_mask[selected_indices] = True

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

                X_selected = SelectKBest(score_fun, k = n).fit_transform(X, y)

            elif method == "recursive_feature_selection":
                knn = KNeighborsClassifier(n_neighbors = n)
                sfs = SequentialFeatureSelector(knn, n_features_to_select = n)
                sfs.fit(X, y)

                #sfs.get_support()
                X_selected = sfs.transform(X)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            X_selected_list.append(X_selected)

    return(X_selected_list)


def feature_transform(X_train, X_val, X_test, n, method='PCA'):
    if method == 'PCA':
        transformer = PCA(n_components=n, random_state = RANDOM_SEED)
    elif method == 'KernelPCA':
        transformer = KernelPCA(n_components=n, kernel='rbf')  # kernel can be changed based on requirement
    # elif method == 'UMAP':
    #     transformer = UMAP(n_components=n)
    else:
        raise ValueError(f"Unsupported method: {method}")

    transformer.fit(X_train)
    
    X_train_transformed = transformer.transform(X_train)
    X_val_transformed = transformer.transform(X_val)
    X_test_transformed = transformer.transform(X_test)

    return X_train_transformed, X_val_transformed, X_test_transformed

def draw_PCA(X, y, file_name):
    pca = PCA(n_components=2, random_state = RANDOM_SEED)
    pca_result = pca.fit_transform(X)

    plt.figure(figsize=(8, 8))

    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label, alpha=0.5)
    plt.title(f'PCA of {X.shape[1]} SNPs')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    #plt.legend(loc='best', prop={'size': 10})
    plt.legend(loc=1, prop={'size': 5})

    plt.savefig(f"{file_name}_PCA.pdf", dpi = 300)
    plt.show()  # Optionally show the plot

def draw_tSNE(X, y, file_name):
    tsne = TSNE(n_components=2, verbose=1)
    tsne_result = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))

    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=label, alpha=0.5)

    ## use other color codes
    # labels_unique = np.unique(y)
    # colors = cm.viridis(np.linspace(0, 1, len(labels_unique)))  # Using viridis colormap
    # for label, color in zip(labels_unique, colors):
    #     indices = np.where(y == label)
    #     plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=label, alpha=0.3,
    #                 color=color,
    #                 ) 

    plt.title(f't-SNE of {X.shape[1]} SNPs')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # plt.legend(loc='best', prop={'size': 10})
    plt.legend(loc=1, prop={'size': 5})

    plt.savefig(f"{file_name}_tSNE.pdf", dpi = 300)
    plt.show()  # Optionally show the plot

def select_label(y, y_original, target_label):
    target_label_encoded = np.unique(y[y_original == target_label])[0]
    y_binary = (y == target_label_encoded).astype(int)
    y_original_binary = np.where(y_original == target_label, "target", 'others')

    label_mapping = {0 : "others", 1: "target"}
    
    return y_binary, y_original_binary, label_mapping


def get_data_path(save_data_path):
    data_locations = {
        '223.195.111.48': '/project/datacamp/team11/data',
        '147.47.44.229': '/home/jinhyun/data/1kGP',
        '147.47.44.93': '/home/jinhyun/data/1kGP',

    }

    raw_data_path = data_locations.get(get_ip_address(), '/not_found')
    sample_annotation_file = os.path.join(raw_data_path, "igsr-1000 genomes 30x on grch38.tsv")
    preprocess_path = os.path.join(raw_data_path, "preprocessed")

    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    assert os.path.exists(preprocess_path), f"Data path not exists: {raw_data_path} OR IP setting is incorrect: {get_ip_address()}"
    assert os.path.isfile(sample_annotation_file), f"File not exists : {sample_annotation_file}"
    # assert has_write_permission(preprocess_path), f"You do not have write permission for {preprocess_path}"

    return preprocess_path, sample_annotation_file
    

def main():
    ### arguments -----
    # target_feature = "merged_support3_variance_0.1" # Real_data
    target_feature = "merged_support3_variance_0.1_random_1M"
    # target_feature = "merged_support3_variance_0.1_random_1M_xgb_8192"
    # target_feature = "merged_random_1k" # Test_data

    target_feature_suffix = "_matrix.npy"
    # target_feature_suffix = "_matrix_onehot.npy"

    save_data_path = "./results"

    select_methods = ["rf"]# ["random", "xgb", "rf", "variance", "chi2", "f_classif"] # Extra-trees # "mutual_info_classif"
    select_feature_from_cache = True
    
    # n_select_start = 128
    # n_select_max_power = int(np.floor(np.log2(X.shape[1])))
    # n_select_start_power = int(np.ceil(np.log2(n_select_start)))  
    # for power in range(n_select_start_power, n_select_max_power + 2):
    # n_select = 2**power if (power <= n_select_max_power) else X.shape[1]
    n_select_list = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] #5105448
    # n_select_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # n_select_list = [1048576] #5105448

    n_dim_reduce_list = [128, 256, 512, 1024, None]  ## list should always contain None to perform whole feature training after selection

    ML_models = ["SVM"] #["SVM", "XGB", "RF", "DT", "KNN"]

    hyper_params = {
        "SVM": [
                {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']}  # grid search params
                # # {'C': 0.001, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # # {'C': 0.01, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # # {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # # {'C': 0.2, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # # {'C': 0.4, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # # {'C': 0.6, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # # {'C': 0.8, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # # {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # # {'C': 1, 'kernel': 'rbf', 'gamma': 'auto'},       # RBF kernel, automatic gamma
                # # {'C': 10, 'kernel': 'rbf', 'gamma': 0.01},        # RBF with specific low gamma
                # # {'C': 50, 'kernel': 'rbf', 'gamma': 0.1},         # RBF with higher C and moderate gamma
                # # {'C': 100, 'kernel': 'rbf', 'gamma': 1},          # RBF with high C and gamma
                # # {'C': 0.5, 'kernel': 'sigmoid', 'gamma': 'auto'}, # Sigmoid kernel, low C
                # # {'C': 2, 'kernel': 'sigmoid', 'gamma': 'scale'}   # Sigmoid kernel with scaled gamma
                ],
        "RF": [
            # {'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}, # default
            {'n_estimators': 2000, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}, # current best for 1048576 features
            # {'n_estimators': 4000, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}, # current best for 1048576 features
        ],

        "XGB": [
            # {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3, 'gamma': 0, 'subsample': 1, 'colsample_bytree': 1, 'reg_lambda': 1, 'reg_alpha': 0}, #default
            {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1, 'reg_alpha': 0}, # current best for 1048576 features
        ]
    }

    ### code start -----
    feature_data_path, sample_annotation_file = get_data_path(save_data_path)

    dataset = data_loader(os.path.join(feature_data_path, target_feature + target_feature_suffix), 
                          sample_annotation_file)
    (X, y_original, y), (train_indices, val_indices, test_indices), label_mapping = dataset.get_data()


    
    result_combined = []

    for feature_select_method in select_methods:
        current_loop = {"random_seed": RANDOM_SEED, "select_method": feature_select_method}
        feature_importance_cache_file_prefix = f"{X.shape[1]}_seed{RANDOM_SEED}_{feature_select_method}"

    # y_backup, y_original_backup = y, y_original
    # feature_select_method = "xgb"
    # for class_target in ['BEB', 'CDX', 'CEU', 'CHS', 'CLM', 'ESN', 'FIN', 'GWD', 'IBS', 'JPT', 'KHV', 'LWK', 'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'TSI', 'YRI']:
    #     y, y_original, label_mapping = select_label(y_backup, y_original_backup, target_label = class_target)
    #     current_loop = {"random_seed": RANDOM_SEED, "class_target": class_target}
    #     feature_importance_cache_file_prefix = f"{X.shape[1]}_seed{RANDOM_SEED}_{feature_select_method}_cls_{class_target}"


        logging.info(f"*************** current loop: {current_loop} ***************")

        try:
            X_selected_list, perf_metric_select = select_feature(X = X, y = y, method = feature_select_method, n_list = n_select_list, train_idx = train_indices, val_idx = val_indices, cache_file_prefix = feature_importance_cache_file_prefix, from_cache = select_feature_from_cache) 
        except Exception as e:
            logging.error(f"An unexpected error occurred while select_feature of {current_loop}. {e.__class__.__name__}: {str(e)}")
            continue 

        for X_selected, n_select in zip(X_selected_list, n_select_list):
            current_loop["select_n"] = n_select

            logging.info(f" - '{feature_select_method}' feature selection selected {n_select} variants. X_selected.shape = {X_selected.shape}. perf_metrics_selection: {perf_metric_select}")

            # if feature_select_method == "random":
            #     try:
            #         save_file_prefix = os.path.join(save_data_path, f"{feature_select_method}_{n_select}")
            #         draw_PCA(X = X_selected, y = y_original, file_name=save_file_prefix)
            #         # draw_tSNE(X = X_selected, y = y_original, file_name=save_file_prefix)
            #     except Exception as e:
            #         logging.error(f"An unexpected error occurred while draw_PCA or draw_tSNE of {current_loop}. {e.__class__.__name__}: {str(e)}")

            if len(X_selected.shape) == 3: # boolean encoding of SNP status
                X_selected = X_selected.reshape(X_selected.shape[0], -1) #flatten last feature dims
            X_train, X_val, X_test = X_selected[train_indices], X_selected[val_indices], X_selected[test_indices]
            y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
            
            for n_dim_reduced in n_dim_reduce_list:
                if (n_dim_reduced is None): # use whole feature
                    current_loop["n_dim_reduced"] = n_select
                    X_train_reduced, X_val_reduced, X_test_reduced = X_train, X_val, X_test 
                    logging.info(f" - Using whole features for training: X_train.shape = {X_train_reduced.shape}")
                else:
                    if (n_dim_reduced < n_select):
                        current_loop["n_dim_reduced"] = n_dim_reduced
                        try:
                            X_train_reduced, X_val_reduced, X_test_reduced = feature_transform(X_train, X_val, X_test, n = n_dim_reduced)
                            logging.info(f" - Reduced to {n_dim_reduced} features using PCA: X_train_reduced.shape = {X_train_reduced.shape}")
                        except Exception as e:
                            logging.error(f"An unexpected error occurred while feature_transform of {current_loop}. {e.__class__.__name__}: {str(e)}")
                            continue
                    else:
                        continue
                    
            
                for train_model in ML_models:
                    for hyper_param_index, current_hyper_param in enumerate(hyper_params[train_model]):
                        current_loop["train_model"] = train_model

                        logging.info(f" - Start {train_model} training: X_train.shape = {X_train_reduced.shape} X_test.shape = {X_test_reduced.shape} with hyper_param {current_hyper_param}")

                        try:
                            (y_pred_train, y_pred_val, y_pred_test, train_params), perf_metric_train = train_ML(method = train_model, 
                                                                                X_train = X_train_reduced, y_train = y_train, 
                                                                                X_val = X_val_reduced, y_val = y_val, 
                                                                                X_test = X_test_reduced,
                                                                                params = current_hyper_param) 
                        except Exception as e:
                            logging.error(f"An unexpected error occurred while train_ML of {current_loop}. {e.__class__.__name__}: {str(e)}")
                            continue 
                        eval_metrics_train = evaluate_performance(y_train, y_pred_train, label_mapping, os.path.join(save_data_path, f"{feature_select_method}_{n_select}_{train_model}_{hyper_param_index}_train"))
                        eval_metrics_val = evaluate_performance(y_val, y_pred_val, label_mapping, os.path.join(save_data_path, f"{feature_select_method}_{n_select}_{train_model}_{hyper_param_index}_val"))
                        eval_metrics_test = evaluate_performance(y_test, y_pred_test, label_mapping, os.path.join(save_data_path, f"{feature_select_method}_{n_select}_{train_model}_{hyper_param_index}_test"))
                        logging.info(f' - Train done with Accuracy: {eval_metrics_test["accuracy"]*100:.4f}%, perf_metrics_train: {perf_metric_train}')


                        merged_metrics = {**current_loop,
                                        "hyper_params" : str(current_hyper_param),
                                        "model_params" : str(train_params),
                                        **{f"select_{k}": v for k, v in perf_metric_select.items()},
                                        **{f"train_{k}": v for k, v in perf_metric_train.items()},
                                        **{f"testset_{k}": v for k, v in eval_metrics_test.items() if k != 'confusion_matrix'},
                                        **{f"valset_{k}": v for k, v in eval_metrics_val.items() if k != 'confusion_matrix'},
                                        **{f"trainset_{k}": v for k, v in eval_metrics_train.items() if k != 'confusion_matrix'},
                                        }
                        result_combined.append(merged_metrics)

                        ## update the dataframe
                        results_df = pd.DataFrame(result_combined)
                        results_df.to_excel(os.path.join(save_data_path, "results.xlsx"), index = False)



if __name__ == "__main__":
    main()

import os, time
from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support #, roc_curve, roc_auc_score, recall_score, precision_score,
from sklearn.preprocessing import LabelEncoder#, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier #, ExtraTreesClassifier,GradientBoostingClassifier

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

        self.X = np.load(X_path)

        self.sample_annotation_df = pd.read_csv(sample_annotation_file, sep='\t')
        self.y = self.sample_annotation_df[self.target_label_name]
        logging.info(f"[progress] Read data done. X.shape: {self.X.shape}, y.shape: {self.y.shape}")

        self.drop_notusing_sample()
        self.y_encoded, self.label_mapping = self.encode_y()
        self.train_indices, self.val_indices, self.test_indices = self.split_dataset(val_size=0.2, test_size=0.2)

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
        model = SVC(**params, random_state = RANDOM_SEED)
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
        # **class_metrics
    }

    return metrics

@measure_performance
def select_feature(X, y, method, n, train_idx, val_idx): 
    if method == "xgb":
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        raise NotImplemented # not validated

        xgb_model = XGBClassifier(
            n_estimators=1000,  # Start with a large number and rely on early stopping. default 100
            learning_rate=0.1, # default 0.1
            gamma=0.1,  # Adjust gamma, might require tuning. default 0
            subsample=0.8,  # Subsample ratio of the training instance. default 1
            colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree. default 1
            objective='multi:softmax',  # Use softmax for multi-class classification
            num_class=len(np.unique(y)),  # Number of classes
            use_label_encoder=False,
            eval_metric='mlogloss',  # Metric used for multiclass classification
            random_state=RANDOM_SEED
        )

        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,  # Stop if the validation metric does not improve in 50 rounds
            verbose=False  # Set to True if you want to see the progress
        )
        
        # Get feature importances and select the top 'n' features
        fi = xgb_model.feature_importances_
        fi_series = pd.Series(fi)
        print(fi_series)
        selected_indices = fi_series.nlargest(n).index
        X_selected = X[:, selected_indices]
        print(X_selected.shape)

        validation_score = xgb_model.best_score
        print(validation_score)
        pass

    elif method == "rf":
        if len(X.shape) >= 3:
            raise NotImplemented
        X_train = X[train_idx]
        y_train = y[train_idx]

        clf = RandomForestClassifier(n_estimators=1000, random_state = RANDOM_SEED)
        clf.fit(X_train, y_train)

        fi = clf.feature_importances_
        fi_series = pd.Series(fi)
        selected_indices = fi_series.nlargest(n).index
        
        X_selected = X[:, selected_indices]

    else:
        num_snps_before = X.shape[1]

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

    return(X_selected)

def feature_transform(X_train, X_val, X_test, n, method='PCA'):
    if method == 'PCA':
        transformer = PCA(n_components=n)
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
    pca = PCA(n_components=2)
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


def get_data_path(save_data_path):
    data_locations = {
        '223.195.111.48': '/project/datacamp/team11/data',
        '147.47.44.229': '/home/jinhyun/data/1kGP',
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
    target_feature = "merged_support3_variance_0.1" # Real_data
    # target_feature = "merged_random_1k" # Test_data
    target_feature_suffix = "_matrix.npy"
    # target_feature_suffix = "_matrix_onehot.npy"

    save_data_path = "./results"

    n_select_start = 128
    select_methods = ["random"]#["random", "xgb", "rf", "variance", "chi2", "f_classif", "mutual_info_classif"] # Extra-trees

    n_dim_reduce_list = [128, 1024, None]  ## list should always contain None to perform whole feature training after selection

    ML_models = ["XGB"] #["SVM", "XGB", "RF", "DT", "KNN"]
    hyper_params = {
        "SVM": [{'C': 0.1, 'kernel': 'linear'},    # previous best params
                {'C': 0.0001, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 0.0002, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 0.0004, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 0.0006, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 0.0008, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 0.001, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 0.002, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 0.004, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 0.008, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 0.01, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 0.1, 'kernel': 'linear'},  # Linear kernel with low regularization
                {'C': 1, 'kernel': 'linear'},   # Linear kernel with more regularization
                {'C': 10, 'kernel': 'linear'},   # Higher regularization
                # {'C': 0.001, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # {'C': 0.01, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # {'C': 0.2, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # {'C': 0.4, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # {'C': 0.6, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # {'C': 0.8, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'},    # RBF kernel with low regularization
                # {'C': 1, 'kernel': 'rbf', 'gamma': 'auto'},       # RBF kernel, automatic gamma
                # {'C': 10, 'kernel': 'rbf', 'gamma': 0.01},        # RBF with specific low gamma
                # {'C': 50, 'kernel': 'rbf', 'gamma': 0.1},         # RBF with higher C and moderate gamma
                # {'C': 100, 'kernel': 'rbf', 'gamma': 1},          # RBF with high C and gamma
                # {'C': 0.5, 'kernel': 'sigmoid', 'gamma': 'auto'}, # Sigmoid kernel, low C
                # {'C': 2, 'kernel': 'sigmoid', 'gamma': 'scale'}   # Sigmoid kernel with scaled gamma
                ],
        "RF": [
            {'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}, # default
            {'n_estimators': 1000, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 100, 'max_features': 'log2', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 100, 'max_features': 'log2', 'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 10},
            {'n_estimators': 100, 'max_features': 'log2', 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 10},
        ],

        "XGB": [
            # {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3, 'gamma': 0, "subsample": 1, "colsample_bytree": 1, 'reg_lambda': 1, 'reg_alpha': 0}, #default
            {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3, 'gamma': 0, "subsample": 1, "colsample_bytree": 1, 'reg_lambda': 1, 'reg_alpha': 0},
            {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3, 'gamma': 1, "subsample": 1, "colsample_bytree": 1, 'reg_lambda': 1, 'reg_alpha': 0},
            {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3, 'gamma': 1, "subsample": 0.8, "colsample_bytree": 0.8, 'reg_lambda': 1, 'reg_alpha': 0},
            {'learning_rate': 0.01, 'n_estimators': 1000, 'max_depth': 2, 'gamma': 1, "subsample": 0.8, "colsample_bytree": 0.8, 'reg_lambda': 1, 'reg_alpha': 0},
            {'learning_rate': 0.01, 'n_estimators': 1000, 'max_depth': 2, 'gamma': 1, "subsample": 0.8, "colsample_bytree": 0.8, 'reg_lambda': 2, 'reg_alpha': 0.5},
        ]
    }

    ### code start -----
    feature_data_path, sample_annotation_file = get_data_path(save_data_path)

    dataset = data_loader(os.path.join(feature_data_path, target_feature + target_feature_suffix), 
                          sample_annotation_file)
    (X, y_original, y), (train_indices, val_indices, test_indices), label_mapping = dataset.get_data()

    n_select_max_power = int(np.floor(np.log2(X.shape[1])))
    n_select_start_power = int(np.ceil(np.log2(n_select_start)))  
    
    result_combined = []
    for feature_select_method in select_methods:
        # for power in range(n_select_start_power, n_select_max_power + 2):
        #     n_select = 2**power if (power <= n_select_max_power) else X.shape[1]
        for n_select in [8192, 131072]:

            current_loop = {"randon_seed": RANDOM_SEED, "select_method": feature_select_method, "select_n": n_select}

            logging.info(f"*************** current loop: {current_loop} ***************")

            try:
                X_selected, perf_metric_select = select_feature(X = X, y = y, method = feature_select_method, n = n_select, train_idx = train_indices, val_idx = val_indices) 
            except Exception as e:
                logging.error(f"An unexpected error occurred while select_feature of {current_loop}. {e.__class__.__name__}: {str(e)}")
                continue 

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

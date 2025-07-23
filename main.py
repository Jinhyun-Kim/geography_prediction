import os, time, random
from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support #, roc_curve, roc_auc_score, recall_score, precision_score,
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer#, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier #, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
# from umap import UMAP

from deap import base, creator, tools, algorithms

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

        base_model = SVC(random_state=RANDOM_SEED)
        grid_search = GridSearchCV(estimator=base_model, param_grid = params, cv=5, scoring='accuracy', verbose=1, refit=True)
        grid_search.fit(X_train, y_train)  # Fits the model on the training data

        y_pred_train = grid_search.predict(X_train)
        y_pred_val = grid_search.predict(X_val)
        y_pred_test = grid_search.predict(X_test)

        return (y_pred_train, y_pred_val, y_pred_test, grid_search.best_params_)

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

def evaluate_individual(individual, X_train, y_train, X_val, y_val):
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)


    selected_features = [i for i in individual]
    X_selected = X_train_val[:, selected_features]

    skf = StratifiedKFold(n_splits=5, shuffle=True) #, random_state = RANDOM_SEED
    accuracy_scores = []
    for train_index, val_index in skf.split(X_selected, y_train_val):
        X_train_fold, X_val_fold = X_selected[train_index], X_selected[val_index]
        y_train_fold, y_val_fold = y_train_val[train_index], y_train_val[val_index]

        model = SVC(C=0.1, kernel='linear') #, random_state=RANDOM_SEED
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_pred_fold)
        accuracy_scores.append(accuracy)

    avg_accuracy = np.mean(accuracy_scores)
    return avg_accuracy,

def cx_unique(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    
    temp1 = ind1[:cxpoint] + [x for x in reversed(ind2) if x not in ind1[:cxpoint]]
    temp2 = ind2[:cxpoint] + [x for x in reversed(ind1) if x not in ind2[:cxpoint]]
    
    ind1[:] = temp1[:len(ind1)]
    ind2[:] = temp2[:len(ind2)]

    assert(len(np.unique(ind1)) == len(ind1))
    assert(len(np.unique(ind2)) == len(ind2))
    
    return ind1, ind2

def mut_unique(individual, low, up, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            new_val = random.randint(low, up)
            while new_val in individual:
                new_val = random.randint(low, up)
            individual[i] = new_val
    assert len(np.unique(individual)) == len(individual)
    return individual,

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

        params = {'n_estimators': 500, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
        
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

    elif method == "ga":
        for n in n_list:
            pop_size = 500
            num_generation = 100
            crossover_prob = 0.5
            mutation_prob = 0.2
            mutation_prob_ind = 0.05

            num_features = X.shape[1]

            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()
            toolbox.register("indices", random.sample, range(num_features), n)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", evaluate_individual, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            toolbox.register("mate", cx_unique)
            toolbox.register("mutate", mut_unique, low=0, up=num_features-1, indpb=mutation_prob_ind)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Parallel processing setup
            pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)

            population = toolbox.population(n = pop_size)
            
            # Add stats and hall of fame
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            hof = tools.HallOfFame(3)

            no_improvement_count = 0
            best_fitness = 0
            
            for gen in range(num_generation):
                # # Dynamic parameter adjustments
                # if gen > 0 and gen % 10 == 0:
                #     mutation_prob *= 0.9  # Decrease mutation probability over generations

                fits = toolbox.map(toolbox.evaluate, population)
                for fit, ind in zip(fits, population):
                    ind.fitness.values = fit
                # hof.update(population)
        
                offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
                offspring = list(map(toolbox.clone, offspring))
                
                fits = toolbox.map(toolbox.evaluate, offspring)
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                
                # population[:] = toolbox.select(population + offspring, k=pop_size - len(hof)) + list(hof.items)  # natural selection with Elitism
                population = toolbox.select(population + offspring, k=pop_size)
                # hof.update(population)
                
                record = stats.compile(population)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{timestamp} - Generation {gen}: Avg={record['avg']:.4f}, Min={record['min']:.4f}, Max={record['max']:.4f}")

                # Early stopping if no improvement in maximum fitness
                if record['max'] > best_fitness:
                    best_fitness = record['max']
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= 10:  # Stop if no improvement in 10 generations
                    print(f"Early stopping at generation {gen} due to no improvement")
                    break
            
            hof.update(population)
            best_ind = hof[0]
            selected_features = [i for i in best_ind]
            X_selected = X[:, selected_features]
            X_selected_list.append(X_selected)

    else:
        num_snps_before = X.shape[1]
        for n in n_list:
            if method in ["random", "variance", "fst", "af"]:
                if method == "random":
                    rng = np.random.default_rng(seed = RANDOM_SEED)
                    boolean_mask = np.zeros(num_snps_before, dtype=bool)
                    selected_indices = rng.choice(num_snps_before, n, replace=False)
                    boolean_mask[selected_indices] = True

                elif method == "variance":
                    batch_process = num_snps_before > 1000000

                    if batch_process:
                        batch_size = 1000000
                        n_samples, n_snps = X.shape
                        variances = np.zeros((n_snps, )) #np.zeros((n_snps, feature_dim))

                        for start in tqdm(range(0, n_snps, batch_size)):
                            end = min(start + batch_size, n_snps)
                            batch_var = np.var(X[:, start:end], axis=0) # np.var(X[:, start:end, :], axis=0)
                            variances[start:end] = batch_var #variances[start:end, :] = batch_var
                    else:
                        variances = np.var(X, axis=0)

                    # np.save(f"{num_snps_before}_seed{RANDOM_SEED}_{method}.npy", variances)

                    selected_indices = np.argsort(variances)[-n:]
                    boolean_mask = np.zeros(num_snps_before, dtype=bool)
                    boolean_mask[selected_indices] = True
                elif method == "fst":
                    unique_pops = np.unique(y)
                    num_variants = X.shape[1]

                    total_mean_freq = np.nanmean(X, axis=0) / 2  # 전체 대립 유전자 빈도 계산

                    # 집단 간 및 집단 내 분산 초기화
                    ss_between = np.zeros(num_variants)
                    ss_within = np.zeros(num_variants)

                    for pop in unique_pops:
                        pop_indices = np.where(y == pop)[0]
                        pop_data = X[pop_indices, :]

                        pop_mean_freq = np.nanmean(pop_data, axis=0) / 2  # 집단별 평균 대립 유전자 빈도 계산
                        ss_between += len(pop_indices) * (pop_mean_freq - total_mean_freq) ** 2  # 집단 간 분산 계산
                        ss_within += np.nansum(((pop_data / 2) - pop_mean_freq) ** 2, axis=0)  # 집단 내 분산 계산

                    # 자유도 계산
                    df_between = len(unique_pops) - 1
                    df_within = len(y) - len(unique_pops)

                    # 평균 제곱 계산
                    ms_between = ss_between / df_between
                    ms_within = ss_within / df_within

                    fst = np.nan_to_num(ms_between / (ms_between + ms_within))
                    selected_indices = np.argsort(fst)[-n:]
                    boolean_mask = np.zeros(num_snps_before, dtype=bool)
                    boolean_mask[selected_indices] = True
                elif method == "af":
                    allele_counts = (X & 1) + ((X >> 1) & 1)
                    allele_freqs = np.nanmean(allele_counts, axis=0) / 2.0
                    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)
                    selected_indices = np.argsort(mafs)[-n:]

                    boolean_mask = np.zeros(num_snps_before, dtype=bool)
                    boolean_mask[selected_indices] = True

                X_selected = X[:, boolean_mask]
                num_snps_after = X_selected.shape[1]

                assert boolean_mask.sum() == num_snps_after

            elif method == "mrmr":
                X_discrete = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform').fit_transform(X)
                y_ = np.array(y).ravel()
                relevance = mutual_info_classif(X_discrete, y_, discrete_features=True)

                selected = []
                remaining = list(range(X.shape[1]))

                while len(selected) < n and remaining:
                    max_score = -np.inf
                    best_feature = None
                    for i in remaining:
                        redundancy = np.mean([
                            mutual_info_classif(X_discrete[:, [i]], X_discrete[:, j].ravel(), discrete_features=True)[0]
                            for j in selected
                        ]) if selected else 0

                        score = relevance[i] - redundancy
                        if score > max_score:
                            max_score = score
                            best_feature = i

                    if best_feature is not None:
                        selected.append(best_feature)
                        remaining.remove(best_feature)
                X_selected = X[:, selected]
                print(f"Selected {X_selected.shape} features using mRMR method.")
                
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

    plt.figure(figsize=(6, 6))

    # Get distinct colors from tab20 and tab10 colormaps
    colors = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('Dark2').colors) + list(plt.get_cmap('Set1').colors) + list(plt.get_cmap('Set2').colors) + list(plt.get_cmap('Set3').colors)
    if len(np.unique(y)) > len(colors):
        raise ValueError("Not enough distinct colors available for the number of classes.")

    color_map = {label: colors[i] for i, label in enumerate(np.unique(y))}

    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label, 
                    color=color_map[label], 
                    alpha=0.5)
    
    plt.title(f'PCA of {X.shape[1]} SNPs')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='best', prop={'size': 5})

    plt.savefig(f"{file_name}_PCA.pdf", dpi=300)
    plt.show()

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
        '223.195.111.31': '/nfs_share/students/jinhyun/1kGP',
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
    

def select_and_train(target_feature, save_result_file_name = "results.xlsx"):
    ### arguments -----
    logging.info(f"Start select_and_train function with args: {target_feature}, {save_result_file_name}")

    target_feature_suffix = "_matrix.npy"
    # target_feature_suffix = "_matrix_onehot.npy"

    save_data_path = "./results"

    pre_selection_methods = ["variance", "random", "fst", "af"] #"chi2", "f_classif"
    n_pre_select_list = [1000000] #[2000000, 4000000, 8000000, 16000000, 32000000]#
    n_pre_select_goal = 1000000

    select_methods = ["random", "xgb", "rf", "variance", "chi2", "f_classif", "fst", "af"] # Extra-trees # "mutual_info_classif"
    select_feature_from_cache = False
    n_select_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    # n_select_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] # Mimimum SNPs
    # n_select_list = [100, 1000, 10000, 100000, 1000000] # PCA

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
            {'n_estimators': 500, 'max_features': 'sqrt', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}, # current best for 1048576 features
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

    for pre_feature_select_method in pre_selection_methods:
        try:
            X_pre_selected_list, perf_metric_preselect = select_feature(X = X, y = y, method = pre_feature_select_method, n_list = n_pre_select_list, train_idx = train_indices, val_idx = val_indices) 
        except Exception as e:
            logging.error(f"An unexpected error occurred while pre_select_feature of {pre_feature_select_method}. {e.__class__.__name__}: {str(e)}")
            continue 

        for X_pre_selected, n_pre_select in zip(X_pre_selected_list, n_pre_select_list):
            if n_pre_select > n_pre_select_goal:
                try:
                    X_pre_selected_final_list, _ = select_feature(X = X_pre_selected, y = y, method = "random", n_list = [n_pre_select_goal], train_idx = train_indices, val_idx = val_indices) 
                except Exception as e:
                    logging.error(f"An unexpected error occurred while random selection after pre_select_feature. {e.__class__.__name__}: {str(e)}")
                    continue 
                X_pre_selected_final = X_pre_selected_final_list[0]
                logging.info(f" - Further selecting feature by random from {n_pre_select} to {n_pre_select_goal} variants. X_pre_selected.shape = {X_pre_selected.shape}. X_pre_selected_final.shape = {X_pre_selected_final.shape}")
            else:
                X_pre_selected_final = X_pre_selected
            logging.info(f" - '{pre_feature_select_method}' feature selection selected {min(n_pre_select, n_pre_select_goal)} variants. X_pre_selected_final.shape = {X_pre_selected_final.shape}. perf_metrics_selection: {perf_metric_preselect}")
        
            for feature_select_method in select_methods:
                current_loop = {"random_seed": RANDOM_SEED, "pre_select_method": pre_feature_select_method, "n_pre_select": n_pre_select, "n_pre_select_goal": n_pre_select_goal, "select_method": feature_select_method}
                feature_importance_cache_file_prefix = f"{X.shape[1]}_seed{RANDOM_SEED}_{pre_feature_select_method}_{n_pre_select}_{n_pre_select_goal}_{feature_select_method}"

    # for _ in [1]:
    #         perf_metric_preselect = {}
    #         X_pre_selected_final = X
    #         y_backup, y_original_backup = y, y_original
    #         feature_select_method = "xgb"
    #         for class_target in ['BEB', 'CDX', 'CEU', 'CHS', 'CLM', 'ESN', 'FIN', 'GWD', 'IBS', 'JPT', 'KHV', 'LWK', 'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'TSI', 'YRI']:
    #             y, y_original, label_mapping = select_label(y_backup, y_original_backup, target_label = class_target)
    #             current_loop = {"random_seed": RANDOM_SEED, "select_method": feature_select_method, "class_target": class_target}
    #             feature_importance_cache_file_prefix = f"{X.shape[1]}_seed{RANDOM_SEED}_{feature_select_method}_cls_{class_target}"

                logging.info(f"*************** current loop: {current_loop} ***************")

                try:
                    X_selected_list, perf_metric_select = select_feature(X = X_pre_selected_final, y = y, method = feature_select_method, n_list = n_select_list, train_idx = train_indices, val_idx = val_indices, cache_file_prefix = feature_importance_cache_file_prefix, from_cache = select_feature_from_cache) 
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
                                                **{f"preselect_{k}": v for k, v in perf_metric_preselect.items()},
                                                **{f"select_{k}": v for k, v in perf_metric_select.items()},
                                                **{f"train_{k}": v for k, v in perf_metric_train.items()},
                                                **{f"testset_{k}": v for k, v in eval_metrics_test.items() if k != 'confusion_matrix'},
                                                **{f"valset_{k}": v for k, v in eval_metrics_val.items() if k != 'confusion_matrix'},
                                                **{f"trainset_{k}": v for k, v in eval_metrics_train.items() if k != 'confusion_matrix'},
                                                }
                                result_combined.append(merged_metrics)

                                ## update the dataframe
                                results_df = pd.DataFrame(result_combined)
                                results_df.to_excel(os.path.join(save_data_path, save_result_file_name), index = False)

def main():
    global RANDOM_SEED
    input_feature_list =  [
        "merged_support3",
        # "merged_support3_variance_1M_seed_42_xgb_8192",
        # "merged_support3_random_1k_seed_42",
    ]
    seed_list = [42, 919, 1204, 624, 306]

    # for feature_file in input_feature_list:
    #     try:
    #         data = np.load(f"/home/jinhyun/data/1kGP/preprocessed/{feature_file}_matrix.npy")
    #         print("data check:", feature_file, data.shape)
    #     except:
    #         print("[warning] file not found: ", feature_file)
    # return

    for feature_file in input_feature_list:
        for seed in seed_list:
            RANDOM_SEED = seed
            save_result_file_name = f"{feature_file}_seed_{RANDOM_SEED}_results.xlsx"
            select_and_train(target_feature = feature_file, save_result_file_name = save_result_file_name)

if __name__ == "__main__":
    main()

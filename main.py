import pandas as pd
import numpy as np # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix #, roc_curve, roc_auc_score, recall_score, precision_score,
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler

import glob
import pickle
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier #,GradientBoostingClassifier
from xgboost import XGBClassifier
# from sklearn.linear_model import Ridge

import joblib 
from helpers import measure_performance


import os, logging
from tqdm import tqdm
import time
from helpers import get_ip_address, has_write_permission, save_preprocessed_data, read_preprocessed_data

# !pip install memory_profiler

import warnings
# 경고 메시지 무시
warnings.filterwarnings("ignore")

random_seed = 42
# ----------------------------------------------------------------------------------------------------------------

class data_loader:
    
    def __init__(self, X_path, sample_annotation_file, data_path, selection=False):
        super().__init__()
        
        self.X = pd.DataFrame(np.load(X_path))
        self.sample_annotatopn_df = pd.read_csv(sample_annotation_file, sep='\t')
        self.df = pd.DataFrame(np.load(data_path))
        self.selection = selection
        
        
    def preprocesser(self):

        # X

        ## Random Sampling
        new_columns = ['com' + str(i) for i in range(1, len(self.X.columns) + 1)]
        self.X.columns = new_columns
        R_df = self.X.copy()
        
        ## Feature selection
        S_df = self.df.T.copy()
        S_df.columns = S_df.iloc[0]
        S_df = S_df[1:]
        
        common_columns = set(self.X) & set(S_df)
        select_X = self.X[list(common_columns)]
        
        # Y

        remove_row = self.sample_annotatopn_df[self.sample_annotatopn_df['Population code']=='IBS,MSL']
        
        New_sample_annotation_df_ = self.sample_annotatopn_df.drop(remove_row.index)
        select_X = select_X.drop(remove_row.index)
        R_df = R_df.drop(remove_row.index)

        
        R_df['Y'] = New_sample_annotation_df_['Population code']
        select_X['Y'] = New_sample_annotation_df_['Population code']

        label_encoder = LabelEncoder()
        
        # Random Sampling
        R_df['country_encoded'] = label_encoder.fit_transform(R_df['Y'])
        R_df = R_df.drop(['Y'], axis=1)  
        # Feature selection
        select_X['country_encoded'] = label_encoder.fit_transform(select_X['Y'])
        select_X = select_X.drop(['Y'], axis=1)
       
        if self.selection == False :
            return R_df
        else:
            return select_X
        
@measure_performance
class Train:
    
    def __init__(self, df, svm=True):
        super().__init__()
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df.iloc[:,:-1], df['country_encoded'], test_size=0.2, random_state=42)
        self.svm = svm

    def model(self):
        #SVM
        previous_best_params = {'C': 0.1, 'gamma': 0.01, 'kernel': 'linear'} 
        svm_model = SVC(**previous_best_params,random_state=42)
        svm_model.fit(self.X_train, self.y_train)    

        #XGB
        xgboost_params = {
            # 'max_depth': 3,
            'learning_rate': 0.1,
            # 'n_estimators': 100,
            'gamma': 0, # default
            'subsample': 1, # default
            # 'random_state': random_seed
        }
        
        xgboost_model = XGBClassifier(**xgboost_params)
        xgboost_model.fit(self.X_train, self.y_train)

        #DT
        decision_tree_model = DecisionTreeClassifier(random_state=42)
        decision_tree_model.fit(self.X_train, self.y_trainn)

        #pred
        SVM_pred = svm_model.predict(self.X_test)
        XGB_pred = xgboost_model.predict(self.X_test)
        DT_pred = decision_tree_model.predict(self.X_test)

        if self.svm == True :
            return SVM_pred
        else:
            return SVM_pred, XGB_pred, DT_pred
    
    def visualization(self):
        pass


        # accuracy = accuracy_score(self.y_test, y_pred)
        # f1_micro = f1_score(self.y_test, y_pred, average='micro')
        # f1_macro = f1_score(self.y_test, y_pred, average='macro')
        # f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
        # confusion_matrix = confusion_matrix(self.y_test, y_pred)
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
        #             xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.title('Confusion Matrix')
        # plt.show()
        # return accuracy, f1_micro, f1_macro, f1_weighted, confusion_matrix


def select_feature(X, y, method, n):
    # RandomForestClassifier
    if method == "rf":
        #selcect feature by random forest method
        skf = StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state=42)
        y = self.X['country_encoded']

        lst_results = []

        i = 1
        for train_idx, test_idx in skf.split(X,y):
            self.X_train = self.X.iloc[train_idx, :-1]
            self.X_test = self.X.iloc[test_idx, :-1]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            idx1 = list(self.X_train.index)
            idx2 = list(self.X_test.index)
            
            clf = RandomForestClassifier(n_estimators=1000)
            start_time = time.time()
            clf.fit(self.X_train, y_train)
            end_time = time.time()
            training_time = end_time - start_time
            
            with open(f'clf_rf{i}_anal.pickle_auc', 'wb') as f:
                pickle.dump(clf, f)
            
            pred = clf.predict(self.X_test)
            pred_proba = clf.predict_proba(self.X_test)    
            accuracy = accuracy_score(y_test, pred) 
            lst_results.append([i, 'Random Forest', idx1, idx2, accuracy,  training_time])
            print("Random Forest_{}".format(i))
            df_results = pd.DataFrame(data=lst_results, columns=['iter', 'method', 
                                                                 'train_idx', 'test_idx',
                                                                 'accuracy', 'training_time'])
            i+=1

            for i in range(1,11):
                with open (f'clf_rf{i}_anal.pickle_auc', 'rb') as f:
                    globals()[f'clf_rf{i}'] = pickle.load(f)
        
            sum = []
            for i in range(1,11):
                sum.append(globals()[f'clf_rf{i}'].feature_importances_)

            np.array(sum).mean(axis=0)

            fi_rf = pd.DataFrame(np.array(sum).mean(axis=0))
            fi_rf.index = [f'com{i}' for i in range(5105448)]

            fi_rf_rank = fi_rf.copy()
            for col in fi_rf_rank.columns[:]:
                fi_rf_rank[col] = fi_rf_rank[col].rank(ascending=False)
            list(fi_rf_rank.sort_values(by=0).T)
            sort_list = list(fi_rf_rank.sort_values(by=0).T)
            #sort_list
            initial_num_features = 128
            final_num_features = 4194304  # max_num

            dfs = []
            num_features = initial_num_features
            while num_features <= final_num_features:
                selected_features = sort_list[:num_features]

                selected_df = selected_features.copy()
                dfs.append(selected_df)

                num_features *= 2

            for i, df in enumerate(dfs):
                print(f"DataFrame {i+1}: {df}")

            for i, df in enumerate(dfs):
                file_name = f"data_{i+1}.npy"
                np.save(file_name, df)

    else if method == "random":
        # select feature randomly
        pass
        # return(X_select)

def main():
    X, y = data_loader("merged_support3_variance_0.1", sample_annotation_file)

    result_combined = []
    for feature_select_method in ["random", "rf", "variance", "chi2", "f_classif", "mutual_info_classif"]:
        for n_select in [128, 256]:
            X_select, perf_metric_select = select_feature(X = X, y = y, method = feature_select_method, n = n_select)
            X_train, X_test, y_train, y_test = split_dataset(X_select, y, seed = random_seed)
            y_pred, perf_metric_train = train(method = "SVM", X = X_train, y = y_train, X_test = X_test)
            metrics = measure_performance(y_test, y_pred)
            result = [] # combine perf_metric_select, perf_metric_train, results
            result_combined.append(result)

    results_df = pd.DataFrame(result_combined)
    results_df.to_excel("result.xlsx")


if __name__ == "__main__":
    main()

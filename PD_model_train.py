import pandas as pd
import numpy as np
import warnings
import random
import category_encoders as ce
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, recall_score, precision_score, average_precision_score, confusion_matrix
import pickle
import joblib
import json

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.options.display.float_format = '{:.5f}'.format


class WoeEncode:
    def __init__(self, df):

        # self.data_path = data_path
        # self.df = pd.read_csv(self.data_path, sep = ";")
        self.df = df
        self.df['has_paid'] = self.df['has_paid'].astype(int)
        self.df_nna = self.df[self.df['default'].notna()]

        # self.df_na = self.df[self.df['default'].isna()]

        self.df_num_cat = self.num_cat()
        self.df_nna_bin, self.num_vars = self.num_bin()

        self.df_nna_bin_woe, self.woe_cols = self.WOE()
        self.lookup_table = self.lookup()
        # self.df_na_bin_woe = self.na_bin_WOE()

    def num_cat(self):
        nan_col_list = self.df_nna.columns[self.df_nna.isnull().any()]
        columns = self.df_nna.columns
        unique_vals = [self.df_nna[col].unique() for col in columns]
        unique_counts = [len(vals) for vals in unique_vals]

        df_num_cat = pd.DataFrame({
            'Columns': columns,
            'Unique Val': unique_vals,
            'Unique Val Count': unique_counts
        })

        df_num_cat['Variable Type'] = np.where(df_num_cat['Unique Val Count'] < 15, 'Categorical', 'Numerical')
        df_num_cat.loc[df_num_cat['Columns'].isin(
            ['merchant_category', 'merchant_group', 'name_in_email']), 'Variable Type'] = 'Categorical'
        df_num_cat.loc[df_num_cat['Columns'].isin(['uuid', 'default']), 'Variable Type'] = '/'
        df_num_cat['Has NA'] = np.where(df_num_cat['Columns'].isin(nan_col_list), 'NA', 'OK')
        return df_num_cat

    def num_bin(self):
        df_nna_bin = self.df_nna.copy()
        num_vars = list(self.df_num_cat['Columns'].loc[self.df_num_cat['Variable Type'] == 'Numerical'])
        for i in num_vars:
            df_nna_bin[i] = pd.cut(df_nna_bin[i], bins=20, precision=0)
        df_nna_bin[self.df_nna.columns[2:]] = df_nna_bin[self.df_nna.columns[2:]].astype(str)
        return df_nna_bin, num_vars

    def WOE(self):
        data_set = self.df_nna_bin
        data_targets = data_set['default']
        data_features = data_set.drop(['default', 'uuid'], axis=1)
        columns = [col for col in data_features.columns]
        woe_encoder = ce.WOEEncoder(cols=columns)
        woe_encoded_data = woe_encoder.fit_transform(data_features[columns], data_targets).add_suffix('_woe')
        return data_set.join(woe_encoded_data), list(woe_encoded_data.columns)

    def lookup(self):
        col_name, lower_bound, upper_bound, orig_value = [], [], [], []
        for i in self.num_vars:
            for j in self.df_nna_bin[i].unique():
                if j != 'nan':
                    col_name.append(i)
                    bounds = j.strip('()[]').split(',')
                    lower_bound.append(float(bounds[0]))
                    upper_bound.append(float(bounds[1]))
                    orig_value.append(j)
        lookup_table = pd.DataFrame(
            {'col_name': col_name, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'orig_value': orig_value})
        return lookup_table

    def map_range(self, col_name, value):
        if pd.isna(value):
            return 'nan'

        sub_table = self.lookup_table.loc[self.lookup_table['col_name'] == col_name]
        try:
            return str(sub_table["orig_value"][
                           (sub_table["lower_bound"] <= value) & (sub_table["upper_bound"] > value)].values[0])
        except:
            return 'nan'

    def na_bin_WOE(self, df_na):

        df_na_bin = df_na.copy()

        for i in self.num_vars:
            df_na_bin[i] = df_na_bin[i].apply(lambda x: self.map_range(i, x))

        df_na_bin[df_na.columns[2:]] = df_na_bin[self.df_nna.columns[2:]].astype(str)
        df_nna_bin_cols = list(self.df_nna_bin.columns)[2:]

        woe_mapping = pd.DataFrame({'Cols': df_nna_bin_cols, 'Cols_WOE': self.woe_cols})
        df_na_bin_woe = df_na_bin.copy()

        for i in range(0, len(woe_mapping)):
            df_map = self.df_nna_bin_woe[[str(woe_mapping.Cols[i]), str(woe_mapping.Cols_WOE[i])]].drop_duplicates()
            df_na_bin_woe = pd.merge(df_na_bin_woe, df_map, how="left", on=[str(woe_mapping.Cols[i])])

        return df_na_bin_woe


class TrainValTest:
    def __init__(self, path, val_prop, test_prop):
        """
        Initialize the TrainVal class by splitting data into train, validation, and test sets.

        Parameters:
        - path (str): Path to the CSV file.
        - val_prop (float): Proportion of data for the validation set.
        - test_prop (float): Proportion of data for the test set.
        """

        _df = pd.read_csv(path, sep=";")
        _df = _df[~_df['default'].isna()]

        _val_size = int(val_prop * len(_df))
        _test_size = int(test_prop * len(_df))

        _pop_index = list(_df.index)
        _val_index = random.sample(_pop_index, _val_size)

        _pop_index = list(set(_pop_index) - set(_val_index))
        _test_index = random.sample(_pop_index, _test_size)

        _train_index = list(set(_pop_index) - set(_test_index))

        _train = _df[_df.index.isin(_train_index)].reset_index(drop=True)
        _test = _df[_df.index.isin(_test_index)].reset_index(drop=True)
        _val = _df[_df.index.isin(_val_index)].reset_index(drop=True)

        _train_y = _train['default'].copy()

        _val_y = _val['default'].copy()
        _val['default'] = np.nan

        _test_y = _test['default'].copy()
        _test['default'] = np.nan

        self.train = _train
        self.train_y = _train_y

        self.val = _val
        self.val_y = _val_y

        self.test = _test
        self.test_y = _test_y


class Model:
    def __init__(self, dataset_path, train_prop, test_prop, grid):

        self.data = TrainValTest("./dataset.csv", 0.1, 0.1)
        self.obj = WoeEncode(data.train)

        _val_woe_x = self.obj.na_bin_WOE(self.data.val)
        _test_woe_x = self.obj.na_bin_WOE(self.data.test)

        self.train_data = self.data.train
        self.train_X = self.obj.df_nna_bin_woe[self.obj.woe_cols]
        self.train_y = self.data.train_y

        self.val_data = self.data.val
        self.val_X = _val_woe_x[self.obj.woe_cols]
        self.val_y = self.data.val_y.values

        self.test_data = self.data.test
        self.test_X = _test_woe_x[self.obj.woe_cols]
        self.test_y = self.data.test_y

        self.param_grid = grid

        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')  # ,
        self.grid_search = GridSearchCV(estimator=xgb_model, param_grid=self.param_grid, cv=5,
                                        scoring='roc_auc')  # , roc_auc'
        self.grid_search.fit(self.train_X.values, self.train_y.values)
        print("Model is trained")
        print("Best parameters found: ", grid_search.best_params_)

    def test_set_performance(self):

        y_pred_prob = self.grid_search.best_estimator_.predict_proba(self.test_X.values)[:, 1]
        y_pred = self.grid_search.best_estimator_.predict(self.test_X.values)

        y_true = self.test_y.values.tolist()

        auc_roc = roc_auc_score(y_true, y_pred_prob)
        rec = recall_score(y_true, y_pred)
        prc = precision_score(y_true, y_pred)
        auc_pr = average_precision_score(y_true, y_pred_prob, )

        print('AUC-ROC:', auc_roc)
        print('Recall:', rec)
        print('Precision:', prc)
        print('AUC-PR:', auc_pr)

        print(confusion_matrix(y_true, y_pred))

    def predict(self, loans_dict):

        if type(loans_dict) == str:
            with open(loans_dict, 'r') as json_file:
                loans = json.load(json_file)
        else:
            loans = loans_dict

        self.loan_frame = pd.DataFrame(loans)

        loans_woe_x = self.obj.na_bin_WOE(self.loan_frame)
        loans_X = loans_woe_x[self.obj.woe_cols].values

        y_pred_prob = self.grid_search.best_estimator_.predict_proba(loans_X)[:, 1]

        res = pd.DataFrame(
            {'Load_ID': list(loans['uuid'].values()),
             'Probability_of_Default': list(y_pred_prob), })
        return res

    def generate_loans(self, test_size=10, file_name="./loan.json"):
        random_loans = random.sample(range(1, len(self.data.test)), test_size)
        test_dict = self.data.test.iloc[random_loans].to_dict()
        with open(file_name, 'w') as json_file:
            json.dump(test_dict, json_file, indent=4)


if __name__ == '__main__':
    param_grid = {
        'learning_rate': [0.01],
        'max_depth': [8, 9, 10],
        'subsample': [0.8, 0.7, 0.6],
        'colsample_bytree': [0.9, 0.8],
        'scale_pos_weight': [70]  # ~ ratio of of Count(Default=0) / count(Default=1) , due to our unbalanced data
        # 'reg_alpha': [0.1, 0.5],
        # 'reg_lambda': [0.1, 0.5],
    }

    mod = Model("./dataset.csv", 0.1, 0.1, param_grid)
    joblib.dump(mod, './mod.pkl')
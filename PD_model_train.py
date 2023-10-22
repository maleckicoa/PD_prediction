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
        _df = _df[~_df['default'].isna()]  # .head(1000)

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


class WoeEncode:
    def __init__(self, df):

        """
        The WoeEncode class receives a dataset to encode.
        The class should be initialized with a train set.
        The class creates an object which holds the encoded train set,
        as well as methods to encode the validation and test set.

        Parameters:
        - df (pd.DataFrame): the training dataset
        """

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
        """
        The num_cat method separates the dataset into categorical and numerical features.
        This is done because the numerical features must be binned before encoding.
        The method returns a summary table which classifies the features into either numerical or categorical

        Parameters:
        -none
        """
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
        """
        The method num_bin utilizes the df_num_cat object and returns a dataframe object,
        whose numerical features have been binned into 20 bins each.
        The method also returns a list of all numerical columns/features.

        Parameters:
        -none
        """
        df_nna_bin = self.df_nna.copy()
        num_vars = list(self.df_num_cat['Columns'].loc[self.df_num_cat['Variable Type'] == 'Numerical'])
        for i in num_vars:
            df_nna_bin[i] = pd.cut(df_nna_bin[i], bins=20, precision=0)
        df_nna_bin[self.df_nna.columns[2:]] = df_nna_bin[self.df_nna.columns[2:]].astype(str)
        return df_nna_bin, num_vars

    def WOE(self):
        """
        The WOE class, encodes the train dataset, whose numerical features have previously been binned

        Parameters:
        -none
        """
        data_set = self.df_nna_bin
        data_targets = data_set['default']
        data_features = data_set.drop(['default', 'uuid'], axis=1)
        columns = [col for col in data_features.columns]
        woe_encoder = ce.WOEEncoder(cols=columns)
        woe_encoded_data = woe_encoder.fit_transform(data_features[columns], data_targets).add_suffix('_woe')
        return data_set.join(woe_encoded_data), list(woe_encoded_data.columns)

    def lookup(self):

        """
        The lookup method serves to define a lookup table which is a part of the mapping table,
        later used to WOE encode the validation and train dataset.

        Parameters:
        -none
        """
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
        """
        map_range method handles the cases of outliers when performing the WOE encoding of the
        validation/test dataset. The outliers are values which do not fall within the defined
        ranges of the lookup table.

        This occurs because the outliers belong to the test dataset, but the lookup table was created
        upon the WOE encoding of the train dataset.

        The map_range method upgrades the lookup table to handle the outlier values.

        Parameters:
        -col_name (str): names of the numerical columns/features
        -value (str): upper or lower range value of the train dataset feature

        """
        if pd.isna(value):
            return 'nan'

        sub_table = self.lookup_table.loc[self.lookup_table['col_name'] == col_name]
        try:
            return str(sub_table["orig_value"][
                           (sub_table["lower_bound"] <= value) & (sub_table["upper_bound"] > value)].values[0])
        except:
            return 'nan'

    def na_bin_WOE(self, df_na):
        """
        The na_bin_WOE method encodes the valudation/test dataset.
        The method basically maps the original feature values into their WOE values using a mapping table.

        Parameters:
        -df_na (pd. Dataframe):the validation/test dataset
        """

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


class Model:
    def __init__(self, dataset_path, val_prop, test_prop, grid):

        """
        The class Model initializes the TrainValTest and the WoeEncode data preparation classes,
        and creates the trains the XGB model using grid search cross-validation.

        Parameters:
        - dataset_path(str): Loacation of the complete dataset
        - val_prop (float): Proportion of data for the validation set
        - test_prop (float): Proportion of data for the test set
        - grid (dict): Dictionary of hyper parameters for training the model

        """

        self.data = TrainValTest(dataset_path, val_prop, test_prop)
        self.obj = WoeEncode(self.data.train)

        _val_woe_x = self.obj.na_bin_WOE(self.data.val)
        _test_woe_x = self.obj.na_bin_WOE(self.data.test)

        self.train_data = self.data.train
        self.train_X = self.obj.df_nna_bin_woe[self.obj.woe_cols]
        self.train_y = self.data.train_y

        self.val_data = self.data.val
        self.val_X = _val_woe_x[self.obj.woe_cols]
        self.val_y = self.data.val_y

        self.test_data = self.data.test
        self.test_X = _test_woe_x[self.obj.woe_cols]
        self.test_y = self.data.test_y

        self.param_grid = grid

        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')  # ,
        self.grid_search = GridSearchCV(estimator=xgb_model, param_grid=self.param_grid, cv=5,
                                        scoring='roc_auc')  # , roc_auc'
        self.grid_search.fit(self.train_X.values, self.train_y.values)
        print("Model is trained")
        print("Best parameters found: ", self.grid_search.best_params_)

    def val_set_performance(self):

        """
        This methods runs the XGB classifer on the validation set and returns the performance
        results together with the confusion matrix.

        The performance resuls can be used to furter optimize the model hyper-parameters.

        Parameters:
        -none
        """

        y_pred_prob = self.grid_search.best_estimator_.predict_proba(self.val_X.values)[:, 1]
        y_pred = self.grid_search.best_estimator_.predict(self.val_X.values)

        y_true = self.val_y.values.tolist()

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
        """
        The predict method is applied within the dockerized application.
        The method takes in one or more loans (either as a JSON file or as a curl post request)
        and returns a dictionary of loan IDs and their probabilities of default

        Parameters:
        - loan_dict(dict/json):

        """
        appended_data = []
        for i in range(0, len(loans_dict)):
            df = pd.DataFrame([loans_dict[i].dict()])
            appended_data.append(df)

        dff = pd.concat(appended_data)
        self.loan_frame = dff.reset_index(drop=True)

        loans_woe_x = self.obj.na_bin_WOE(self.loan_frame)
        loans_X = loans_woe_x[self.obj.woe_cols].values

        y_pred_prob = self.grid_search.best_estimator_.predict_proba(loans_X)[:, 1]

        loan_id = list(self.loan_frame['uuid'])
        probability_of_default = list(y_pred_prob)
        probability_of_default = [str(x) for x in probability_of_default]

        res_dict = {key: value for key, value in zip(loan_id, probability_of_default)}
        return res_dict

    def predict_ipynb(self, loans_dict):
        """
        The predict_ipynb method is applied within the Jupyter Notebook.
        The method takes in one or more loans (via a path string to JSON file)
        and returns a dictionary of loan IDs and their probabilities of default

        Parameters:
        - loan_dict(str): path location of a JSON file where the loans are defined
        """
        if type(loans_dict) == str:
            f = open(loans_dict)
            data = json.load(f)
        else:
            data = loans_dict.copy()

        self.loan_frame = pd.DataFrame(data)

        loans_woe_x = self.obj.na_bin_WOE(self.loan_frame)
        loans_X = loans_woe_x[self.obj.woe_cols].values

        y_pred_prob = self.grid_search.best_estimator_.predict_proba(loans_X)[:, 1]

        loan_id = list(self.loan_frame['uuid'])
        probability_of_default = list(y_pred_prob)
        probability_of_default = [str(x) for x in probability_of_default]

        res_dict = {key: value for key, value in zip(loan_id, probability_of_default)}

        return res_dict

    def generate_loans(self, test_size=10, file_name="./loan.json"):

        """
        The generate_loans method generates a JSON file holding a dictionary of loans,
        for which the user can calculate the probability of default by passing
        this JSON file as in input to the predict/predich_ipynb method.

        The JSON file is made up of random loans from the test dataset, the idea is to
        allow the user to quickly generate test loans.

        Note that the user can also manualy specify a customized loan and pass it to
        the prediction method within a JSON file.

        Parameters:
        -test_size (int): number of test loans within the JSON file
        -file name (str): string path to the JSON file

        """
        random_loans = random.sample(range(1, len(self.data.test)), test_size)
        test_dict = self.data.test.iloc[random_loans]  # .to_dict()
        # print(test_dict)

        with open(file_name, 'w') as file:
            test_dict.to_json(file, orient='records', indent=4)


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

    # Instantiate & train the Model object
    mod_dump = Model("./dataset.csv", 0.1, 0.1, param_grid)

    # Serialize the trained Model object - done to avoid retraining of the model at every application start
    joblib.dump(mod_dump, './mod.pkl')
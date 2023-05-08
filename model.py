import pickle
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, recall_score, precision_score, average_precision_score
import random


def map_range(lookup_table, col_name, value):

    if np.isnan(value) == True:
        fff='nan'

    elif (type(value) == float or type(value) == int):
        sub_table = lookup_table.loc[lookup_table['col_name']==col_name]
        try:
            fff = sub_table["orig_value"][(sub_table["lower_bound"] <= value) & (sub_table["upper_bound"] > value)]
            fff =str(fff.values[0])
        except:
            fff='nan'
    else:
        fff='nan'

    return fff


def na_bin_WOE_test(df_na, df_nna, df_nna_bin, df_nna_bin_woe,  num_vars, woe_cols, lookup_table):

    #lookup_table = lookup(df_nna)
    #df_nna_bin, num_vars = nna_bin(df_nna)
    #df_nna_bin_woe, woe_cols = WOE(df_nna)

    df_na_bin = df_na.copy()

    for i in num_vars:
        df_na_bin[i] = df_na_bin[i].apply(lambda x: map_range(lookup_table,i,x) )
    df_na_bin[df_na.columns[2:]] = df_na_bin[df_nna.columns[2:]].astype(str)

    ##########################

    df_nna_bin_cols= list(df_nna_bin.columns)[2:]
    woe_mapping = pd.DataFrame({'Cols': df_nna_bin_cols,'Cols_WOE': woe_cols})

    df_na_bin_woe = df_na_bin.copy()

    for i in range(0,len(woe_mapping)):
        df_map = df_nna_bin_woe[[str(woe_mapping.Cols[i]),str(woe_mapping.Cols_WOE[i])]].drop_duplicates()

        df_na_bin_woe = pd.merge(df_na_bin_woe, df_map, how="left", on=[str(woe_mapping.Cols[i])])
    return df_na_bin_woe




# Deserialization
with open("df_nna.pickle", "rb") as infile:
    df_nna = pickle.load(infile)

with open("df_nna_bin.pickle", "rb") as infile:
    df_nna_bin = pickle.load(infile)

with open("df_nna_bin_woe.pickle", "rb") as infile:
    df_nna_bin_woe = pickle.load(infile)

with open("woe_cols.pickle", "rb") as infile:
    woe_cols = pickle.load(infile)

with open("lookup_table.pickle", "rb") as infile:
    lookup_table = pickle.load(infile)

with open("num_vars.pickle", "rb") as infile:
    num_vars = pickle.load(infile)


with open("grid_search.pickle", "rb") as infile:
    grid_search = pickle.load(infile)


def test_result(d):
    df_na_bin_woe = na_bin_WOE_test(d, df_nna, df_nna_bin, df_nna_bin_woe,  num_vars, woe_cols, lookup_table)
    X_test = df_na_bin_woe[woe_cols].values
    uuid = df_na_bin_woe.uuid.values.tolist()
    try:
        y_pred_prob = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
        test_set_result = pd.DataFrame({'UUID': uuid,'PD': y_pred_prob}).to_dict('list')
    except:
        test_set_result = "Insuffucient training data to provide probability of default OR Incorrect predictor variable specification"


    return test_set_result

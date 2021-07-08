# %%
"""
# [Example] Training a Scorecard model using California Housing Dataset
"""

# %%
"""
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Task-Defination" data-toc-modified-id="Task-Defination-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Task Defination</a></span></li><li><span><a href="#Data-Preparation" data-toc-modified-id="Data-Preparation-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Preparation</a></span><ul class="toc-item"><li><span><a href="#Download-sample-data" data-toc-modified-id="Download-sample-data-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Download sample data</a></span></li><li><span><a href="#Define-feature-and-target" data-toc-modified-id="Define-feature-and-target-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Define feature and target</a></span></li></ul></li><li><span><a href="#Scorecard-Model" data-toc-modified-id="Scorecard-Model-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Scorecard Model</a></span><ul class="toc-item"><li><span><a href="#Feature-selection-before-feature-discretization" data-toc-modified-id="Feature-selection-before-feature-discretization-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Feature selection before feature discretization</a></span><ul class="toc-item"><li><span><a href="#Predictability" data-toc-modified-id="Predictability-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Predictability</a></span></li><li><span><a href="#Colinearity" data-toc-modified-id="Colinearity-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Colinearity</a></span></li></ul></li><li><span><a href="#Feature-Discretization" data-toc-modified-id="Feature-Discretization-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Feature Discretization</a></span></li><li><span><a href="#Feature-selection-after-feature-discretization" data-toc-modified-id="Feature-selection-after-feature-discretization-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Feature selection after feature discretization</a></span><ul class="toc-item"><li><span><a href="#Predictability" data-toc-modified-id="Predictability-3.3.1"><span class="toc-item-num">3.3.1&nbsp;&nbsp;</span>Predictability</a></span></li><li><span><a href="#Colinearity" data-toc-modified-id="Colinearity-3.3.2"><span class="toc-item-num">3.3.2&nbsp;&nbsp;</span>Colinearity</a></span></li></ul></li><li><span><a href="#Feature-Engineering-(manually-adjusting-feature-intervals)" data-toc-modified-id="Feature-Engineering-(manually-adjusting-feature-intervals)-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Feature Engineering (manually adjusting feature intervals)</a></span><ul class="toc-item"><li><span><a href="#Latitude" data-toc-modified-id="Latitude-3.4.1"><span class="toc-item-num">3.4.1&nbsp;&nbsp;</span>Latitude</a></span></li><li><span><a href="#HouseAge" data-toc-modified-id="HouseAge-3.4.2"><span class="toc-item-num">3.4.2&nbsp;&nbsp;</span>HouseAge</a></span></li><li><span><a href="#Population" data-toc-modified-id="Population-3.4.3"><span class="toc-item-num">3.4.3&nbsp;&nbsp;</span>Population</a></span></li><li><span><a href="#Longitude" data-toc-modified-id="Longitude-3.4.4"><span class="toc-item-num">3.4.4&nbsp;&nbsp;</span>Longitude</a></span></li><li><span><a href="#AveRooms" data-toc-modified-id="AveRooms-3.4.5"><span class="toc-item-num">3.4.5&nbsp;&nbsp;</span>AveRooms</a></span></li></ul></li><li><span><a href="#Feature-Encoding-with-Weight-of-Evidence-(WOE)" data-toc-modified-id="Feature-Encoding-with-Weight-of-Evidence-(WOE)-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Feature Encoding with Weight of Evidence (WOE)</a></span></li><li><span><a href="#Double-check-predictability-and-colinearity" data-toc-modified-id="Double-check-predictability-and-colinearity-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Double check predictability and colinearity</a></span><ul class="toc-item"><li><span><a href="#predictability" data-toc-modified-id="predictability-3.6.1"><span class="toc-item-num">3.6.1&nbsp;&nbsp;</span>predictability</a></span></li><li><span><a href="#colinearity" data-toc-modified-id="colinearity-3.6.2"><span class="toc-item-num">3.6.2&nbsp;&nbsp;</span>colinearity</a></span></li></ul></li><li><span><a href="#Model-Training" data-toc-modified-id="Model-Training-3.7"><span class="toc-item-num">3.7&nbsp;&nbsp;</span>Model Training</a></span></li><li><span><a href="#Model-Evaluation" data-toc-modified-id="Model-Evaluation-3.8"><span class="toc-item-num">3.8&nbsp;&nbsp;</span>Model Evaluation</a></span><ul class="toc-item"><li><span><a href="#Train" data-toc-modified-id="Train-3.8.1"><span class="toc-item-num">3.8.1&nbsp;&nbsp;</span>Train</a></span><ul class="toc-item"><li><span><a href="#Classification-performance-on-differet-levels-of-model-scores" data-toc-modified-id="Classification-performance-on-differet-levels-of-model-scores-3.8.1.1"><span class="toc-item-num">3.8.1.1&nbsp;&nbsp;</span>Classification performance on differet levels of model scores</a></span></li><li><span><a href="#Performance-plots" data-toc-modified-id="Performance-plots-3.8.1.2"><span class="toc-item-num">3.8.1.2&nbsp;&nbsp;</span>Performance plots</a></span></li></ul></li><li><span><a href="#Validation" data-toc-modified-id="Validation-3.8.2"><span class="toc-item-num">3.8.2&nbsp;&nbsp;</span>Validation</a></span><ul class="toc-item"><li><span><a href="#Classification-performance-on-differet-levels-of-model-scores" data-toc-modified-id="Classification-performance-on-differet-levels-of-model-scores-3.8.2.1"><span class="toc-item-num">3.8.2.1&nbsp;&nbsp;</span>Classification performance on differet levels of model scores</a></span></li><li><span><a href="#Performance-plots" data-toc-modified-id="Performance-plots-3.8.2.2"><span class="toc-item-num">3.8.2.2&nbsp;&nbsp;</span>Performance plots</a></span></li></ul></li></ul></li><li><span><a href="#Model-interpretation" data-toc-modified-id="Model-interpretation-3.9"><span class="toc-item-num">3.9&nbsp;&nbsp;</span>Model interpretation</a></span></li></ul></li></ul></div>
"""

# %%
"""
## Task Defination
"""

# %%
"""
Build a Scorecard model to rate house values using the California Housing Dataset. The higher the score, the more valuable the house is (the more probable that the house belong to the Top10% Expensive houses).
"""


# %%
from sys import meta_path
from scipy.stats import stats
from scorecardbundle.feature_discretization import ChiMerge as cm
from scorecardbundle.feature_discretization import FeatureIntervalAdjustment as fia
from scorecardbundle.feature_encoding import WOE as woe
from scorecardbundle.feature_selection import FeatureSelection as fs
from scorecardbundle.model_training import LogisticRegressionScoreCard as lrsc
from scorecardbundle.model_evaluation import ModelEvaluation as me
from scorecardbundle.model_interpretation import ScorecardExplainer as mise

# from temp.feature_discretization import ChiMerge as cm
# from temp.feature_discretization import FeatureIntervalAdjustment as fia
# from temp.feature_encoding import WOE as woe
# from temp.feature_selection import FeatureSelection as fs
# from temp.model_training import LogisticRegressionScoreCard as lrsc
# from temp.model_evaluation import ModelEvaluation as me
# from temp.model_interpretation import ScorecardExplainer as mise

# from importlib import reload
# reload(cm)
# reload(fia)
# reload(woe)
# reload(fs)
# reload(lrsc)
# reload(me)
# reload(mise)

# %%
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

# %%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime
import copy, os, pathlib, re

plt.style.use('seaborn-colorblind')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Enable display of Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Enable display of negative sign '-'

plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 150

# Font settings
font_text = {'family':'SimHei', 
        'weight':'normal',
         'size':11,
        } # Font setting for normal texts

font_title = {'family':'SimHei',
        'weight':'bold',
         'size':14,
        } # Font setting for title

# Thousands seperator
from matplotlib.ticker import FuncFormatter 
def format_thousands(x,pos):
    return '{:,.0f}'.format(x,pos)
formatter_thousands = FuncFormatter(format_thousands)


# %%
"""
## Get Parameters 
"""
# # ...
# import sys,argparse
# from IPython.display import HTML
# CONFIG_FILE = '.config_ipynb'
# if os.path.isfile(CONFIG_FILE):
#     with open(CONFIG_FILE) as f:
#         sys.argv = f.read().split()
# else:
#     sys.argv = ['jupyter_args.py', 'input_file', '--int_param', '12']

# parser = argparse.ArgumentParser()
# parser.add_argument("input_file",help="Input image, directory, or npy.")
# parser.add_argument("--int_param", type=int, default=4, help="an optional integer parameter.")
# args = parser.parse_args()
# p = args.int_param
# print(args.input_file,p)


# %%
"""
## Data Preparation
"""

# %%
"""
### Download sample data
"""

# %%
# from sklearn.datasets import fetch_california_housing
# dataset = fetch_california_housing() # Download dataset
# __file__ = 'TrainingaScorecardmodelusingAutoLoansDataset.ipynb'
# pPath = str(pathlib.Path(__file__).parent.absolute())
pPath = 'c:/Users/Alex/Documents/GitHub/FICO-Internship/hw2/'
# print(pPath)
dataset = pd.read_csv(os.path.join(pPath, 'AutoLoans.csv'), thousands=',')


# %%
print(dataset.keys())
print(dataset)

# %%
# Features data
# X0 = pd.DataFrame(dataset['data'],columns=dataset['feature_names'])

X_excluded_columns = ['target','loanID','loanDateOpened',  'loanPerformance','loanPerformNum']
sample_weight_col = 'sampwt'

# target = 1 ONLY when loanPerformance = "Paid as agreed", loanPerformNum = 7

X0 = dataset[dataset.columns[~dataset.columns.isin(X_excluded_columns)]]
print(X0.columns)
print('shape:',X0.shape)
print(X0.head())


# %%
X0.isna().sum()

# %%
X0.info()
# exit()

# %%
"""
### Define feature and target
"""

# %%
# features = [i for i in X0.columns if i != sample_weight_col]
features = X0.columns
X_all, y_all = X0[features], dataset['target']

# %%
print(X_all.shape)
print(y_all.shape)
y_all.value_counts(normalize=True)


delim = '~'

NO_INFO_STR = 'No Info'
NO_INFO_NUM = -1e-3
NO_INFO_INTERVAL = str(np.NINF)+delim+str(float(NO_INFO_NUM))

# %%
"""
### Encode categorical variables, Preprocessing missing values
"""
def list_intersect(a_list, b_list):
    return sorted(set(a_list).intersection(set(b_list)))

def ordinal_encoder_create_dict(X, ordinal_features):
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    feat = list_intersect(X.columns,ordinal_features)
    orig = X.loc[:,feat].replace({np.nan: None})
    encoder.fit(orig)
    ret = dict()
    for f in feat:
        ret[f] = dict()
        for arr in encoder.categories_: # arr is sorted, maps to 0, 1, 2, ..., len(arr)-1
            if set(arr) == set(orig[f]):
                i = 0
                for v in arr:
                    if NO_INFO_STR == v:
                        ret[f][v], ret[f][NO_INFO_NUM] = NO_INFO_NUM, v
                    else:
                        ret[f][v], ret[f][i] = i, v
                    i += 1
                break
    return ret

def toggle_ordinal_encoder(X, ordinal_encode_dict, ordinal_features):
    feat = list_intersect(X.columns,ordinal_features)
    typs = set(X.loc[:,feat].dtypes)
    for f in feat:
        X[f] = X[f].map(ordinal_encode_dict[f])
    if np.dtype('O') in typs:
        for f in feat:
            X.loc[:,f] = pd.to_numeric(X.loc[:,f])
    # print(X.loc[:,feat].dtypes)

# print(X_all.dtypes)

ordinal_features = list(X_all.select_dtypes(include='object').columns) # get ordinal_features before dtypes are messed up
numeric_features = list(X_all.select_dtypes(exclude='object').columns) # get ordinal_features before dtypes are messed up
# print(ordinal_features)
# print(numeric_features)
# exit()

# print(X_all.loc[X_all.appOcc.isna(), :])
# print(X_all.isna().sum())

# %%
# preprocessing
def toggle_missing_encoder(X, features, missing_dict):
    feat = list_intersect(X.columns,features)
    X.loc[:,feat] = X.loc[:,feat].replace(missing_dict)


# ordinal_missing_dict = {np.nan: NO_INFO_STR}
numeric_missing_dict = {NO_INFO_STR:NO_INFO_STR, NO_INFO_STR:NO_INFO_NUM} # np.nan: NO_INFO_NUM, 

# print(X_all.loc[X_all.appOcc.isna(), :])
# print(X_all.isna().sum())

toggle_missing_encoder(X_all, ordinal_features, {np.nan: NO_INFO_STR}) # only do once for ordinal
toggle_missing_encoder(X_all, numeric_features, {np.nan: NO_INFO_NUM}) # np.nan: NO_INFO_NUM <- only do once for numeric

# print(X_all.loc[X_all.appOcc == NO_INFO_STR, :])
# print(X_all.isna().sum())
# exit()

# %%
# print(X_all[['cbMosAvg','cbMosDlq','cbUtilizn','cbMosInq']])
ordinal_encode_dict = ordinal_encoder_create_dict(X_all, ordinal_features)

def print_ordinal_encodings(ordinal_encode_dict):
    for i in ordinal_encode_dict:
        print(i,'\n\t',ordinal_encode_dict[i])

# print_ordinal_encodings(ordinal_encode_dict)

toggle_ordinal_encoder(X_all, ordinal_encode_dict, ordinal_features)
# print(X_all.loc[X_all.appOcc == ordinal_encode_dict['appOcc'][NO_INFO_STR], :])
# print(X_all.isna().sum())

# toggle_ordinal_encoder(X_all, ordinal_encode_dict, ordinal_features)  # test
# print(X_all.loc[X_all.appOcc == NO_INFO_STR, :])
# exit()



# %%

def separate_sample_weight(X):
    if sample_weight_col in X0.columns:
        return X.loc[:,[i for i in features if i != sample_weight_col]], X.loc[:,sample_weight_col]
    else:
        return X, None


DEFAULT_RANDOM_STATE = 42
from sklearn.model_selection import train_test_split
X, X_val, y, y_val = train_test_split(X_all, y_all, test_size=0.4, stratify=y_all, random_state=DEFAULT_RANDOM_STATE)
X, sampwt = separate_sample_weight(X)
X_val, sampwt_val = separate_sample_weight(X_val)
print(X.shape)
print(y.shape)
print(sampwt)
print(X_val.shape)
print(y_val.shape)
print(sampwt_val)
# exit()


# %%
y.value_counts(normalize=True)

# %%
y_val.value_counts(normalize=True)

# %%
"""
## Scorecard Model
"""

# %%
"""
### Feature selection before feature discretization
"""

# %%
"""
#### Predictability
Evaluate the predictive power of features with information value (IV) and Chi2-test
"""

# %%
# IV
trans_woe_raw = woe.WOE_Encoder(output_dataframe=True)
result_woe_raw = trans_woe_raw.fit_transform(X,y)
# print(result_woe_raw)

# %%
# Chi2
res_chi2 = chi2(MinMaxScaler().fit_transform(X),y)
# print(res_chi2)

# %%
# IV result
res_iv = pd.DataFrame.from_dict(trans_woe_raw.iv_, orient='index').sort_values(0,ascending=False).reset_index()
res_iv.columns = ['feature','IV']
# Chi2 result
res_chi2 = pd.DataFrame(np.stack(res_chi2,axis=1),columns=['Chi2_stat','Chi2_pvalue'])
res_chi2['feature'] = X.columns
# Merge
res_predictability = res_iv.merge(res_chi2,on='feature')
res_predictability.sort_values('Chi2_stat',ascending=False)

# %%
threshold_iv = 0.02
threshold_chi2 = 0.05
mask_iv = res_predictability.IV > threshold_iv
mask_chi2 = res_predictability.Chi2_pvalue <= threshold_chi2
print(f'There are {X.shape[1]} features in total. {sum(mask_iv)} features have IV that is larger than 0.02, while only {sum(mask_chi2)} features passed the Chi2 test')

def remove_features(orig, remove):
    return sorted(set(orig)-set(remove))

belowthreshold_iv = list(res_predictability.loc[res_predictability.IV <= threshold_iv, 'feature'])
print('Features with IV that less than 0.02 (and are thus dropped)', belowthreshold_iv)

selected_features = remove_features([i for i in features if i != sample_weight_col], belowthreshold_iv)
print(selected_features)

# %%
"""
In this case, no feature is dropped since all their IVs are larger than 0.02.
"""

# %%
"""
#### Colinearity
Identify highly-correlated feature pairs where the feature with lower IV is dropped. 
In this example, a pair of features are highly-correlated if their Pearson Correlation Coefficient is larger than 0.7
"""

# %%
fs.selection_with_iv_corr(trans_woe_raw, result_woe_raw,threshold_corr=0.7)

# %%
features_to_drop_auto,features_to_drop_manual,corr_auto,corr_manual = fs.identify_colinear_features(result_woe_raw,trans_woe_raw.iv_,threshold_corr=0.7)
print('The features with lower IVs in highly correlated pairs: ',features_to_drop_auto)
print('The features with equal IVs in highly correlated pairs: ',features_to_drop_manual)

# %%
corr_auto # highly correlated features (with unequal IVs)

# %%
corr_manual # highly correlated features (with equal IVs)

# %%
"""
Based on the analysis results above, 'MedInc', 'AveBedrms', and 'AveOccup' are dropped.
"""

# %%
# selected_features = remove_features(selected_features, ['MedInc', 'AveBedrms', 'AveOccup'])
selected_features = remove_features(selected_features, features_to_drop_auto)
# selected_features = remove_features(selected_features, features_to_drop_manual[1:])
print('auto dropping:',features_to_drop_auto)
# print('manual dropping:',features_to_drop_manual[1:])
print('selected:',selected_features)
# print(X[selected_features])

# %%
"""
### Feature Discretization
"""

# %%
# trans_cm = cm.ChiMerge(max_intervals=10, min_intervals=2, decimal=3, output_dataframe=True) # Given the meanings of these features, round them up to 3 decimals                                    ##################

# perform only on numeric features
# print_ordinal_encodings(ordinal_encode_dict)

result_cm = X.loc[:,selected_features].copy()

# fine binning   -> 'initial_intervals'
# coarse binning -> 'max_intervals'
fine_bins = 100
coarse_bins_max = 6
coarse_bins_min = 2
bin_decimal = 1 # 1 is highest precision present in dataset (dealLoanToVal); any higher has no effect

# asdf

chimerge_desc = '_bins-'+str(fine_bins)+'-'+str(coarse_bins_max)+'-'+str(coarse_bins_min)+'_decimal'+str(bin_decimal)

trans_cm = cm.ChiMerge(max_intervals=coarse_bins_max, min_intervals=coarse_bins_min, initial_intervals=fine_bins,
                        decimal=bin_decimal, output_dataframe=True)

temp = list_intersect(selected_features,numeric_features)
trans_cm.fit(X.loc[:,temp], y)

for col in temp:
    if NO_INFO_NUM in X_all.loc[:,col].values and NO_INFO_NUM not in trans_cm.boundaries_[col]:
        trans_cm.boundaries_[col] = np.array([NO_INFO_NUM] + list(trans_cm.boundaries_[col]))

# print(result_cm.loc[:,temp])
result_cm.loc[:,temp] = trans_cm.transform(X.loc[:,temp]).set_index(result_cm.index)
# print(result_cm.loc[:,temp])


# %%
def bin_dict(bins0):
    ret = dict()
    binmax = np.max(bins0)
    for b in bins0:
        s = cm.assign_interval_str(np.array([b]),bins0)[0] if b == binmax else [i for i in cm.assign_interval_str(np.array([b,np.inf]),bins0) if '~inf' not in i][0]
        ret[b], ret[s] = s, b
    return ret

interval_encode_dict = dict()
for col in selected_features:
    if col in numeric_features:
        bins0 = sorted([i for i in list(trans_cm.boundaries_[col]) if i != np.inf])
        if NO_INFO_NUM in X_all.loc[:,col]:
            bins0 = [NO_INFO_NUM] + bins0
    elif col in ordinal_features:
        bins0 = sorted([i for i in list(ordinal_encode_dict[col]) if isinstance(i, int) or isinstance(i, float)]) # already includes NO_INFO_NUM
    # print(col, bins0)
    print(col, cm.assign_interval_str(np.array(bins0),bins0))
    result_cm.loc[:,col] = cm.assign_interval_str(X.loc[:,col].values,bins0) # pass new interval boundaries
    interval_encode_dict[col] = bin_dict(bins0)


# print(interval_encode_dict)
# interval_encode_dict


# %%
# trans_cm.boundaries_ # show boundaries (for numeric_features)
# exit()

# %%
feature_doc = pd.DataFrame({
    'feature':selected_features
})
feature_doc['num_intervals'] = feature_doc['feature'].map(result_cm.nunique().to_dict())
feature_doc['min_interval_size'] = [fia.feature_stat(result_cm[col].values,y.values)['sample_size'].min() for col in feature_doc['feature']]
feature_doc

# %%
"""
### Feature selection after feature discretization
"""

# %%
"""
#### Predictability
"""

# %%
# IV
trans_woe_tem = woe.WOE_Encoder(output_dataframe=True)
result_woe_tem = trans_woe_tem.fit_transform(result_cm,y) # 0:05:54.309297
# Chi2
res_chi2_tem = chi2(OrdinalEncoder().fit_transform(result_cm),y)

# IV result 
res_iv_af = pd.DataFrame.from_dict(trans_woe_tem.iv_, orient='index').sort_values(0,ascending=False).reset_index()
res_iv_af.columns = ['feature','IV']
# Chi2 result
res_chi2_af = pd.DataFrame(np.stack(res_chi2_tem,axis=1),columns=['Chi2_stat','Chi2_pvalue'])
res_chi2_af['feature'] = selected_features
# Merge
res_predictability_af = res_iv_af.merge(res_chi2_af,on='feature')
res_predictability_af.sort_values('Chi2_stat',ascending=False)

# %%
mask_iv = res_predictability_af.IV > threshold_iv
mask_chi2 = res_predictability_af.Chi2_pvalue <= threshold_chi2
print(f'There are {len(selected_features)} features in total. {sum(mask_iv)} features have IV that is larger than 0.02, while {sum(mask_chi2)} features passed the Chi2 test')

# %%
"""
Drop features
"""
features_to_drop_iv = list(res_predictability_af.loc[res_predictability_af.IV <= threshold_iv,'feature'])
print('Dropping features with IVs below threshold: ',features_to_drop_iv)
selected_features = remove_features(selected_features, features_to_drop_iv)

result_cm = result_cm[selected_features]
result_woe_tem = result_woe_tem[selected_features]

[interval_encode_dict.pop(key) for key in features_to_drop_iv]
[trans_woe_tem.iv_.pop(key) for key in features_to_drop_iv]
[trans_woe_tem.result_dict_.pop(key) for key in features_to_drop_iv]
[trans_cm.boundaries_.pop(key) for key in features_to_drop_iv if key in trans_cm.boundaries_]

trans_woe_tem.columns_ = np.array(sorted(trans_woe_tem.iv_.keys()))
trans_cm.columns_ = np.array(sorted(trans_cm.boundaries_.keys()))


# %%
"""
#### Colinearity
"""

# %%
fs.selection_with_iv_corr(trans_woe_tem, result_woe_tem,threshold_corr=0.7)

# %%
# Have a look at the correlation table
fs.unstacked_corr_table(result_woe_tem,trans_woe_tem.iv_)

# %%
"""
In case of automatically discretized values, there are no highly-correlated features.
"""

# %%
"""
### Feature Engineering (manually adjusting feature intervals)
Analyze the sample distribution and event rate distribution for each feature, and adjust the feature intervals so that the feature's predictability is intuitive to human (high explainability). Of course the feature need to maintain a reasonable predicbility (e.g. has a IV larger than 0.02) 
"""

# %%
manual_coarse_bins = dict()                                        # coarse binning automatically performed by ChiMerge
# manual_coarse_bins['AveRooms'] = [5.96,6.426,6.95,7.41]          # example of manual coarse binning
# manual_coarse_bins['HouseAge'] = [24,36,45]
# manual_coarse_bins['Latitude'] = [34.1,34.47,37.59]
# manual_coarse_bins['Longitude'] = [-121.59,-118.37]
# manual_coarse_bins['Population'] = [420,694,877,1274,2812]


# %%
for col in selected_features:
    print(col)
    if col in numeric_features:
        bins0 = sorted([i for i in list(trans_cm.boundaries_[col]) if i != np.inf])
        if NO_INFO_NUM in X_all.loc[:,col]:
            bins0 = [NO_INFO_NUM] + bins0
    elif col in ordinal_features:
        bins0 = sorted([i for i in list(ordinal_encode_dict[col]) if isinstance(i, int) or isinstance(i, float)])
    if col in manual_coarse_bins:
        bins0 = sorted(manual_coarse_bins[col])
    print(bins0)
    # continue
    if col not in manual_coarse_bins:
        continue
        fia.plot_event_dist(result_cm[col],y
                        ,title=f'Feature distribution of {col}'
                        ,x_label=col
                        ,y_label=''
                        ,x_rotation=90
                        ,save=False
                        ,file_name=col)
    else:
        new_x = cm.assign_interval_str(X[col].values,bins0) # pass new interval boundaries             (coarse binning)
        woe.woe_vector(new_x, y.values)
        result_cm[col] = new_x # reasonable explainability and predictability. Select.
        continue
        fia.plot_event_dist(result_cm[col],y
                        ,title=f'Feature distribution of {col}'
                        ,x_label=col
                        ,y_label=''
                        ,x_rotation=90
                        ,save=False
                        ,file_name=col
                        ,table_vpos=-0.6)
        

# exit()
# %%
"""
### Feature Encoding with Weight of Evidence (WOE)
"""

# %%
print(selected_features)

# %%
trans_woe = woe.WOE_Encoder(output_dataframe=True)
result_woe = trans_woe.fit_transform(result_cm[selected_features], y) # WOE is fast. This only takes less then 1 seconds
result_woe.head()

# %%
result_cm.head()

# %%
trans_woe.iv_ # the information value (iv) for each feature

# %%
"""
### Double check predictability and colinearity
"""

# %%
"""
#### predictability
"""

# %%
# IV
trans_woe_tem = woe.WOE_Encoder(output_dataframe=True)
result_woe_tem = trans_woe_tem.fit_transform(result_cm[selected_features],y) # 0:05:54.309297
res_iv_selected = pd.DataFrame.from_dict(trans_woe_tem.iv_, orient='index').sort_values(0,ascending=False).reset_index()
res_iv_selected.columns = ['feature','IV']
res_iv_selected

# %%
"""
All features' IV are above 0.02
"""

# %%
"""
#### colinearity
"""

# %%
fs.selection_with_iv_corr(trans_woe, result_woe)

# %%
fs.unstacked_corr_table(result_woe,trans_woe.iv_)

# %%
"""
There are no highly-correlated features
"""

# %%
cmap = sns.light_palette("steelblue", as_cmap=True) # mono color
result_woe.corr().style.background_gradient(cmap=cmap)

# %%
corr_matrix = result_woe.corr()
# plt.figure(figsize=(3,3))
# sns.heatmap(corr_matrix, cmap = 'seismic', center=0, xticklabels=result_woe.columns, yticklabels=result_woe.columns, square=True, annot = True, fmt='.3f', annot_kws={"fontsize":5})
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.show()

# %%
"""
### Model Training
"""

# %%
# Weight the positive class so that the model cares more about the classification performace of the positive class.
def compute_class_weight(labels):
    '''Compute weight for each class and return a dictionary. 
    This is for the class_weight parameter in classifiers'''
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    return dict(zip(np.unique(labels), class_weights))

weights = compute_class_weight(y.values)
pos_weight = weights[1]/weights[0]
# weights,pos_weight

# %%
DEFAULT_N_JOBS = 5
# Search for the optimal parameters set for Logistic Regression
param_grid = {                                                         ##################
# 'C': np.arange(1e-1,1,1e-1)/10,
'C': np.arange(1e-2,1,1e-2)/10,
# 'penalty':['l2','l1'],   
# 'penalty':['l2','none'],  # when solver = newton-cg, lbfgs (default), sag, saga
# 'penalty':['l1','l2','none'],   
'penalty':['l2'],  
  
# 'solver':['lbfgs'],

# 'class_weight':[weights,None] # same as None if 50:50 split between y = 0/1.
'class_weight':[weights]
}

max_iter = 1000

print(param_grid)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

cl = LogisticRegression(max_iter=max_iter, random_state=DEFAULT_RANDOM_STATE*3)
grid_search = GridSearchCV(cl, param_grid, cv=KFold(n_splits=10, shuffle=True, random_state=DEFAULT_RANDOM_STATE*5), scoring='roc_auc',verbose=True,n_jobs=DEFAULT_N_JOBS) # n_jobs=-1                                   ##################
grid_search.fit(result_woe, y, sample_weight=sampwt)

print('Best parameters:',grid_search.best_params_,
      '\n \n Best score',grid_search.best_score_,
      '\n \n Best model:',grid_search.best_estimator_)

# static params
grid_search.best_params_['max_iter'] = max_iter
# print(vars(grid_search.best_estimator_))


# %%
# https://scorecard-bundle.bubu.blue/API/4.model.html   # LogisticRegressionScoreCard function documentation


model = lrsc.LogisticRegressionScoreCard(trans_woe, PDO=-20, basePoints=100, verbose=True, random_state=DEFAULT_RANDOM_STATE*3, n_jobs=DEFAULT_N_JOBS,
                                        output_path=os.path.join(pPath, 'scorecards/'),
                                        **grid_search.best_params_)
model.fit(result_woe, y, sample_weight=sampwt)
# print(vars(model.lr_))


## %% EXAMPLE
# # Users can use `baseOdds` parameter to set base odds. 
# # Default is None, where base odds will be calculate using the number of positive class divided by the number of negative class in y
# # Assuming Users want base odds to be 1:60 (positive:negative)
# model = lrsc.LogisticRegressionScoreCard(... , baseOdds=1/60)
# model.fit(result_woe, y, sample_weight=sampwt)

# %%
"""
Access the Scorecard rule table by attribute `woe_df_`. This is the Scorecard model.
"""

# %%
model.AB_

# %%
# print_ordinal_encodings(ordinal_encode_dict)
model.woe_df_


# %%
"""
Scorecard should be applied on the **original feature values** (before discretization and WOE encoding).
"""

# %%
"""
Users can manually adjust the Scorecard rules (as shown below, or output excel files to local position, edit it in excel and load it), and use `load_scorecard` parameter of predict() to load the adjusted rule table. See details in the documentation of `load_scorecard`.
Assuming we want to change the highest score for `AveRooms` from 92 to 91.
"""

# %%
sc_table = model.woe_df_.copy()
# sc_table['score'][(sc_table.feature=='AveRooms') & (sc_table.value=='7.41~inf')] = 91                                   ##################  example of a manual adjustment
# sc_table


# %%
result = model.predict(X[selected_features], load_scorecard=sc_table) # Scorecard should be applied on the original feature values
result_val = model.predict(X_val[selected_features], load_scorecard=sc_table) # Scorecard should be applied on the original feature values
# result.head() # if model object's verbose parameter is set to False, predict will only return Total scores
result
# exit()

# %%
# OR if we load rules from file:
# sc_table = pd.read_excel('rules.xlsx')
# model = lrsc.LogisticRegressionScoreCard(woe_transformer=None, verbose=True)
# result = model.predict(X[selected_features], load_scorecard=sc_table) # Scorecard should be applied on the original feature values
# result_val = model.predict(X_val[selected_features], load_scorecard=sc_table) # Scorecard should be applied on the original feature values


# %%
result['TotalScore'].hist(bins=10)

# %%
result_val['TotalScore'].hist(bins=10)

# %%
"""
### Model Evaluation
"""
# %%
"""
#### Train
"""

# %%
"""
##### Performance plots
"""

# %%
evaluation = me.BinaryTargets(y, result['TotalScore'])
print(evaluation.ks_stat())
# # evaluation.plot_ks()
# # evaluation.plot_roc()
# # evaluation.plot_precision_recall()
evaluation.plot_all()
# asdf
# %%
"""
#### Validation
"""


# %%
"""
##### Performance plots
"""

# %%
evaluation = me.BinaryTargets(y_val, result_val['TotalScore'])
print(evaluation.ks_stat())
evaluation.plot_all()
# asdf
# %%
"""
### Model interpretation
Interprete the result for an instance by identifying the important features to the result
"""

# %%
list(sc_table.feature.unique())

# %%
# Features that contribute 80%+ of total score
imp_fs = mise.important_features(result_val
                   ,feature_names=list(sc_table.feature.unique())
                   ,col_totalscore='TotalScore'
                   ,threshold_method=0.8, bins=None)
result_val['important_features'] = imp_fs

# %%
# Features with top n highest score
n = 2
imp_fs = mise.important_features(result_val
                   ,feature_names=list(sc_table.feature.unique())
                   ,col_totalscore='TotalScore'
                   ,threshold_method=n, bins=None)
result_val['top'+str(n)+'_features'] = imp_fs

# %%
result_val


# %%
"""
#### Classification performance on different levels of model scores
"""
# Evaluate the classification performance on differet levels of model scores (y_pred_proba). Useful for setting classification threshold based on requirements of precision and recall.

# %%
"""
#### Training samples, training thresholds
"""
me.pref_table(y,result['TotalScore'].values,thresholds=result['TotalScore'].quantile(np.arange(1,10)/10).values)

# %%
"""
#### Validation samples, training thresholds
"""
me.pref_table(y_val,result_val['TotalScore'].values,thresholds=result['TotalScore'].quantile(np.arange(1,10)/10).values)


# %%
"""
EXAMPLE
Based on the classification performance table, choose 152 as the threshold (precision 20%, recall 79%)
"""
# # %%
# result_val['y_pred'] = result_val['TotalScore'].map(lambda x: 1 if x>152 else 0)                                   ##################
# print(np.mean(result_val['y_pred']))
# result_val


# %%
"""
### Determine best TotalScore threshold(s) with classification reports
"""

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

def thresholds_stats(res, y_true, keyword, weights=None):
    # print('Use sample weights:',not isinstance(weights, type(None)))
    threshold_scores = {key:[] for key in ['threshold','accuracy']}

    for i in range(int(np.min(res['TotalScore'])), int(np.max(res['TotalScore']))):
        y_pred = res['TotalScore'].map(lambda x: 1 if x >= i else 0)
        class_report = classification_report(y_true, y_pred, sample_weight=weights, output_dict=True, zero_division=0)
        # break
        threshold_scores['threshold'].append(i)

        # for i in class_report:
        for i in ['accuracy','1','macro avg','weighted avg']: # '0',     -- macro same as weighted if weights=None and 50:50 y=0/1 ratio (this is the case for AutoLoans)
            if i == 'accuracy':
                threshold_scores[i].append(class_report[i])
                continue

            # for j in class_report[i]:
            for j in ['precision','recall','f1-score','support']:
                k = i+': '+j
                if k not in threshold_scores:
                    threshold_scores[k] = []
                threshold_scores[k].append(class_report[i][j])
        
        tn_fp_fn_tp = {key:val/sup for key, val, sup in zip(['tn', 'fp', 'fn', 'tp'], confusion_matrix(y_true, y_pred, sample_weight=weights).ravel(), [class_report['0']['support'], class_report['1']['support'], class_report['0']['support'], class_report['1']['support']])}
        for i in ['fp', 'tp', 'tn', 'fn']:
            if i not in threshold_scores:
                threshold_scores[i] = []
            threshold_scores[i].append(tn_fp_fn_tp[i])

    threshold_scores = pd.DataFrame.from_dict(threshold_scores)

    statsdict = dict()
    statsdict[keyword+'-roc_auc'] = roc_auc_score(y_true, res['TotalScore'])
    statsdict[keyword+'-average_precision'] = average_precision_score(y_true, res['TotalScore'])
    statsdict[keyword+'-best_accuracy'] = max(threshold_scores['accuracy'])
    statsdict[keyword+'-best_accuracy_threshold'] = min(threshold_scores.loc[threshold_scores['accuracy'] == statsdict[keyword+'-best_accuracy'],'threshold'].values)

    # return threshold_scores.sort_values(by='accuracy', ascending=False)
    return threshold_scores, statsdict

# asdf

# print('For training samples:')
df, statsdict = thresholds_stats(result, y, 'train')
df.to_excel(os.path.join(pPath, 'thresholds','train'+chimerge_desc+'.xlsx'), index=False)

# print('For validation samples:')
df, statsdict_val = thresholds_stats(result_val, y_val, 'val')
df.to_excel(os.path.join(pPath, 'thresholds','val'+chimerge_desc+'.xlsx'), index=False)

# exit()


# %%
"""
### Record stats for given binning specs - roc auc, avg prec, etc.
"""
# chimerge_desc = '_bins-'+str(fine_bins)+'-'+str(coarse_bins_max)+'-'+str(coarse_bins_min)+'_decimal'+str(bin_decimal)
fn = 'stats based on binning specs'
chimerge_params = ['__m__', '__confidence_level__', '__max_intervals__', '__min_intervals__','__initial_intervals__', '__decimal__']

columns = [i.split('__')[1] for i in chimerge_params] + list(statsdict.keys()) + list(statsdict_val.keys())
data = np.array([[vars(trans_cm)[i] for i in chimerge_params] + [statsdict[i] for i in statsdict] + [statsdict_val[i] for i in statsdict_val]], dtype=object)
temp = pd.DataFrame(data, columns=columns)

statsdf = pd.read_excel(os.path.join(pPath, fn+'.xlsx')) if fn+'.xlsx' in os.listdir(pPath) else pd.DataFrame(columns=columns)
statsdf = pd.concat([statsdf, temp]).drop_duplicates(ignore_index=True)
statsdf.to_excel(os.path.join(pPath, fn+'.xlsx'), index=False)


# %%
"""
### Process, Export Scorecard table
"""

lohi = sc_table.loc[:,'value'].str.split(delim, n = 1, expand = True)
sc_table['lo'] = [float(i) for i in lohi[0]]
sc_table['hi'] = [float(i) for i in lohi[1]]

for i in selected_features:
    temp = sc_table.loc[sc_table.feature == i,:].copy()
    tempindex = temp.index
    if i in ordinal_features:
        temp = temp.sort_values(by ='score', ascending=False)
    elif i in numeric_features:
        temp = temp.sort_values(by ='lo', ascending=False)
    temp.index = tempindex
    sc_table.loc[sc_table.feature == i,:] = temp
    
    if i in ordinal_features:
        sc_table.loc[sc_table.feature == i, 'value'] = sc_table.loc[sc_table.feature == i, 'value'].replace(interval_encode_dict[i]).replace(ordinal_encode_dict[i])

sc_table.loc[:,'value'] = sc_table.loc[:,'value'].replace({NO_INFO_INTERVAL: NO_INFO_STR, NO_INFO_NUM: NO_INFO_STR})
sc_table.loc[:,'value'] = sc_table.loc[:,'value'].str.replace(str(float(NO_INFO_NUM)), str(np.NINF))

for row in sc_table.itertuples():
  # Pandas(Index=0, feature='appAge', value='66.0~inf', woe=-0.074107972149601, beta=0.6426741005533283, score=9.0, lo=66.0, hi=inf)
  # print(row)
  if row.feature in numeric_features:
    if row.lo == np.NINF and NO_INFO_NUM not in interval_encode_dict[row.feature]: # row.value != NO_INFO_STR
      sc_table.loc[sc_table.index == row.Index,'value'] = str(row.hi) + ' or less'
    elif row.lo == NO_INFO_NUM and NO_INFO_NUM in interval_encode_dict[row.feature]:
      if row.hi == 0:
        sc_table.loc[sc_table.index == row.Index,'value'] = str(0.0)
      else:
        sc_table.loc[sc_table.index == row.Index,'value'] = str(row.hi) + ' or less'
    elif row.hi == np.inf:
      sc_table.loc[sc_table.index == row.Index,'value'] = 'More than ' + str(row.lo)

for row in sc_table.itertuples():
  if row.feature in numeric_features and row.lo == 0:
    sc_table.loc[sc_table.index == row.Index,'value'] = str(0.0) + delim + str(row.hi)

sc_table.loc[:,'value'] = sc_table.loc[:,'value'].str.replace(delim, ' <= ')

del sc_table['lo'], sc_table['hi']

sc_table.to_excel(os.path.join(pPath, 'scorecards','scorecard_AutoLoans'+chimerge_desc+'.xlsx'), index=False)
sc_table
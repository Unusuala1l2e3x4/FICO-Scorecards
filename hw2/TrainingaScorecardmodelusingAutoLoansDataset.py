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
pPath = 'c:/Users/Alex/Documents/GitHub/FICO-Internship/hw2'
# print(pPath)
dataset = pd.read_csv(os.path.join(pPath, 'AutoLoans.csv'), thousands=',')


# with open('dataset_california_housing.pkl','wb') as f:
#     pickle.dump(dataset,f,4)

# %%
# with open('dataset_california_housing.pkl','rb') as f:
#     dataset = pickle.load(f)

# %%
print(dataset.keys())
print(dataset)

# %%
# Features data
# X0 = pd.DataFrame(dataset['data'],columns=dataset['feature_names'])

X_excluded_columns = ['target','loanID','loanDateOpened',  'loanPerformance','loanPerformNum']

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
features = list(X0.columns)
X_all, y_all = X0[features], dataset['target']

NO_INFO_STR = 'No Info'
NO_INFO_INT = -999
NO_INFO_INTERVAL = '-inf~'+str(float(NO_INFO_INT))

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
                        ret[f][v], ret[f][NO_INFO_INT] = NO_INFO_INT, v
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
numeric_missing_dict = {NO_INFO_STR:NO_INFO_STR, NO_INFO_STR:NO_INFO_INT} # np.nan: -999, 

# print(X_all.loc[X_all.appOcc.isna(), :])
# print(X_all.isna().sum())

toggle_missing_encoder(X_all, ordinal_features, {np.nan: NO_INFO_STR}) # only do once for ordinal
toggle_missing_encoder(X_all, numeric_features, {np.nan: NO_INFO_INT}) # np.nan: -999 <- only do once for numeric

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

# toggle_ordinal_encoder(X_all, ordinal_encode_dict, ordinal_features)
# print(X_all.loc[X_all.appOcc == NO_INFO_STR, :])
# exit()





# do after ##########################################################################################
# toggle_missing_encoder(X_all, numeric_features, numeric_missing_dict)
# toggle_ordinal_encoder(X_all, ordinal_encode_dict, ordinal_features)



# %%
# with open('data_ch.pkl','wb') as f:
#     pickle.dump([X_all,y_all],f,4)
# with open('data_ch.pkl','rb') as f:
#     X_all,y_all = pickle.load(f)
features = sorted(X_all.columns)

# %%
print(X_all.shape)
print(y_all.shape)
y_all.value_counts(normalize=True)


# %%
DEFAULT_RANDOM_STATE = 42
from sklearn.model_selection import train_test_split
X, X_val, y, y_val = train_test_split(X_all, y_all, test_size=0.4, stratify=y_all, random_state=DEFAULT_RANDOM_STATE)
print(X.shape)
print(y.shape)
print(X_val.shape)
print(y_val.shape)
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
mask_iv = res_predictability.IV>0.02
mask_chi2 = res_predictability.Chi2_pvalue<=0.05
print(f'There are {X.shape[1]} features in total. {sum(mask_iv)} features have IV that is larger than 0.02, while only {sum(mask_chi2)} features passed the Chi2 test')

def remove_features(orig, remove):
    return sorted(set(orig)-set(remove))

selected_features = remove_features(features, list(res_predictability.loc[res_predictability.IV <= 0.02, 'feature']))
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
print_ordinal_encodings(ordinal_encode_dict)


result_cm = X.loc[:,selected_features].copy()

# fine binning   -> 'initial_intervals'
# coarse binning -> 'max_intervals'
trans_cm = cm.ChiMerge(max_intervals=6, min_intervals=2,
                        initial_intervals=100,
                        decimal=3, output_dataframe=True)

temp = list_intersect(selected_features,numeric_features)
# print(result_cm.loc[:,temp])
result_cm.loc[:,temp] = trans_cm.fit_transform(X.loc[:,temp], y).set_index(result_cm.index)
print(result_cm.loc[:,temp])

#
for feat in temp:
    result_cm.loc[result_cm.loc[:,feat].str.startswith('-inf~') & X.loc[:,feat]!=NO_INFO_INT,feat] = result_cm.loc[result_cm.loc[:,feat].str.startswith('-inf~') & X.loc[:,feat]!=NO_INFO_INT,feat].str.replace('-inf',str(float(NO_INFO_INT)))
    result_cm.loc[X.loc[:,feat]==NO_INFO_INT,feat] = NO_INFO_INTERVAL
    # assert list(X.loc[X.loc[:,feat]==NO_INFO_INT,feat].index) == list(result_cm.loc[result_cm.loc[:,feat].str.startswith('-inf~'+str(NO_INFO_INT)),feat].index)
    # assert list(X.loc[X.loc[:,feat]==NO_INFO_INT,feat].index) == list(result_cm.loc[X.loc[:,feat]==NO_INFO_INT,feat].index)
    # if list(X.loc[X.loc[:,feat]==NO_INFO_INT,feat].index) != list(result_cm.loc[result_cm.loc[:,feat].str.startswith('-inf~'+str(NO_INFO_INT)),feat].index):
    #     print('---',feat)
    #     print(X.loc[X.loc[:,feat]==NO_INFO_INT,feat])
    #     print(result_cm.loc[X.loc[:,feat]==NO_INFO_INT,feat])

# exit()

temp = list_intersect(selected_features,ordinal_features)
# print(result_cm.loc[:,temp])

for col in temp:
    print(col)
    bins0 = [i for i in list(ordinal_encode_dict[col]) if isinstance(i, int) or isinstance(i, float)]
    result_cm.loc[:,col] = cm.assign_interval_str(X.loc[:,col].values,bins0) # pass new interval boundaries



print(X.loc[:,selected_features])
print(result_cm)

# print(X.loc[:,selected_features].dtypes)
# print(result_cm.dtypes)
# exit()


# %%
trans_cm.boundaries_ # show boundaries for all features (numeric_features)

# print(trans_cm.boundaries_)
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
mask_iv = res_predictability_af.IV>0.02
mask_chi2 = res_predictability_af.Chi2_pvalue<=0.05
print(f'There are {len(selected_features)} features in total. {sum(mask_iv)} features have IV that is larger than 0.02, while {sum(mask_chi2)} features passed the Chi2 test')

# %%
"""
Thus no feature is dropped
"""

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
coarse_bins = dict()                                        # coarse binning automatically performed by ChiMerge
# coarse_bins['AveRooms'] = [5.96,6.426,6.95,7.41]          # example of manual coarse binning
# coarse_bins['HouseAge'] = [24,36,45]
# coarse_bins['Latitude'] = [34.1,34.47,37.59]
# coarse_bins['Longitude'] = [-121.59,-118.37]
# coarse_bins['Population'] = [420,694,877,1274,2812]


# %%
for col in selected_features:
    print(col)
    if col in numeric_features:
        bins0 = [i for i in list(trans_cm.boundaries_[col]) if i != np.inf]
    elif col in ordinal_features:
        bins0 = [i for i in list(ordinal_encode_dict[col]) if isinstance(i, int) or isinstance(i, float)]
    if col in coarse_bins:
        bins0 = coarse_bins[col]
    print(bins0)
    # continue
    fia.plot_event_dist(result_cm[col],y
                    ,title=f'Feature distribution of {col}'
                    ,x_label=col
                    ,y_label=''
                    ,x_rotation=90
                    ,save=False
                    ,file_name=col)
    ## %%
    if col in coarse_bins:
        new_x = cm.assign_interval_str(X[col].values,bins0) # pass new interval boundaries             (coarse binning)
        woe.woe_vector(new_x, y.values)
        ## %%
        fia.plot_event_dist(new_x,y
                        ,title=f'Feature distribution of {col}'
                        ,x_label=col
                        ,y_label='More valuable than Q90'
                        ,x_rotation=90
                        ,save=False
                        ,file_name=col
                        ,table_vpos=-0.6)
        ## %%
        result_cm[col] = new_x # reasonable explainability and predictability. Select.

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
cmap = sns.light_palette("steelblue", as_cmap=True)
result_woe.corr().style.background_gradient(cmap=cmap)

# %%
corr_matrix = result_woe.corr()
plt.figure(figsize=(3,3))
sns.heatmap(corr_matrix, cmap = 'bwr', center=0)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

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
 'C': np.arange(1,10)/10, 
#  'penalty':['l2','l1'],   
# 'penalty':['l2','none'],   
'penalty':['l2'],   
'class_weight':[weights,None]
}

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

cl = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(cl, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE*5), scoring='roc_auc',verbose=True,n_jobs=DEFAULT_N_JOBS) # n_jobs=-1                                   ##################
grid_search.fit(result_woe, y)

print('Best parameters:',grid_search.best_params_,
      '\n \n Best score',grid_search.best_score_,
      '\n \n Best model:',grid_search.best_estimator_)

# %%
# https://scorecard-bundle.bubu.blue/API/4.model.html   # LogisticRegressionScoreCard function documentation



# model = lrsc.LogisticRegressionScoreCard(trans_woe, PDO=-20, basePoints=100, verbose=True, random_state=DEFAULT_RANDOM_STATE,
#                                         C=0.6, class_weight={0: 0.5555356181589808, 1: 5.0016155088852985},penalty='l2')                                   ##################
model = lrsc.LogisticRegressionScoreCard(trans_woe, PDO=-20, basePoints=100, verbose=True, random_state=DEFAULT_RANDOM_STATE*3, n_jobs=DEFAULT_N_JOBS,
                                        **grid_search.best_params_)     
model.fit(result_woe, y)


## %%
# # Users can use `baseOdds` parameter to set base odds. 
# # Default is None, where base odds will be calculate using the number of positive class divided by the number of negative class in y
# # Assuming Users want base odds to be 1:60 (positive:negative)
# model = lrsc.LogisticRegressionScoreCard(trans_woe, PDO=-20, basePoints=100, baseOdds=1/60,
#                                          verbose=True,C=0.6, class_weight={0: 0.5555356181589808, 1: 5.0016155088852985},penalty='l2')
# model.fit(result_woe, y)

# %%
"""
Access the Scorecard rule table by attribute `woe_df_`. This is the Scorecard model.
"""

# %%
print_ordinal_encodings(ordinal_encode_dict)
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
result.head() # if model object's verbose parameter is set to False, predict will only return Total scores

# ##########################################################################################
print_ordinal_encodings(ordinal_encode_dict)
# toggle_missing_encoder(X_all, numeric_features, numeric_missing_dict)
# toggle_ordinal_encoder(X_all, ordinal_encode_dict, ordinal_features)



# %%
# OR if we load rules from local position.

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
##### Classification performance on differet levels of model scores
"""

# %%
me.pref_table(y,result['TotalScore'].values,thresholds=result['TotalScore'].quantile(np.arange(1,10)/10).values)

# %%
"""
##### Performance plots
"""

# %%
evaluation = me.BinaryTargets(y, result['TotalScore'])
print(evaluation.ks_stat())
# evaluation.plot_ks()
# evaluation.plot_roc()
# evaluation.plot_precision_recall()
evaluation.plot_all()

# %%
"""
#### Validation
"""

# %%
"""
##### Classification performance on differet levels of model scores
"""

# %%
me.pref_table(y_val,result_val['TotalScore'].values,thresholds=result['TotalScore'].quantile(np.arange(1,10)/10).values)

# %%
"""
##### Performance plots
"""

# %%
evaluation = me.BinaryTargets(y_val, result_val['TotalScore'])
print(evaluation.ks_stat())
evaluation.plot_all()

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
me.pref_table(y_val,result_val['TotalScore'].values,thresholds=result['TotalScore'].quantile(np.arange(1,10)/10).values)

# # %%
# """
# Based on the classification performance table, choose 152 as the threshold (precision 20%, recall 79%)
# """

# # %%
# result_val['y_pred'] = result_val['TotalScore'].map(lambda x: 1 if x>152 else 0)                                   ##################
# print(np.mean(result_val['y_pred']))
# result_val


# # %%
# """
# Now we can interprete model scores based on the analysis above. For example, the 4th entry has a total score of 174. 2. The primary driver of house value in this case is housing age ('HouseAge') and position('Longitude')
# """


# %%
"""
### Process, Export Scorecard table
"""
# %%
sc_table = sc_table.replace(NO_INFO_INTERVAL, NO_INFO_STR)
sc_table
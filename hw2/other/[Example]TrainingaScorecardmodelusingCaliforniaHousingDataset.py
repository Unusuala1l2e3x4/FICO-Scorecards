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
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime
import copy

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
from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing() # Download dataset

# with open('dataset_california_housing.pkl','wb') as f:
#     pickle.dump(dataset,f,4)

# %%
# with open('dataset_california_housing.pkl','rb') as f:
#     dataset = pickle.load(f)

# %%
dataset.keys()

# %%
# Features data
housing = pd.DataFrame(dataset['data'],columns=dataset['feature_names'])
print(housing.columns)
print('shape:',housing.shape)
housing.head()

# %%
housing.isna().sum()

# %%
housing.info()

# %%
"""
### Define feature and target
"""

# %%
"""
- Let `median_house_value` be the target variable and all other columns be features

- Set y=1 when medain house value is larger than its q90 and y=0 otherwise
"""

# %%
house_value = pd.Series(dataset['target'],name=dataset['target_names'][0])

# %%
house_value.hist(bins=30)

plt.axvline(x=house_value.quantile(0.9), c='r', alpha=0.5)
plt.annotate(s='Q90',xy=(house_value.quantile(0.9),1200),
            xytext=(house_value.quantile(0.9)+0.5,1300),
            arrowprops={'arrowstyle':'->'})
plt.title('Distribution of median house value')
plt.ylabel('Frequency')
plt.xlabel('Median house value')

# %%
features = list(housing.columns)
q90 = house_value.quantile(0.9)
X_all, y_all = housing[features], house_value.map(lambda x: 1 if x>q90 else 0)

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

# %%
y.value_counts(normalize=True)

# %%
y_val.value_counts(normalize=True)

# %%
"""
## Scorecard Model
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

# %%
# Chi2
res_chi2 = chi2(MinMaxScaler().fit_transform(X),y)

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
print(selected_features)
# print(X[selected_features])

# %%
"""
### Feature Discretization
"""

# %%
# trans_cm = cm.ChiMerge(max_intervals=10, min_intervals=2, decimal=3, output_dataframe=True) # Given the meanings of these features, round them up to 3 decimals                                    ##################
trans_cm = cm.ChiMerge(max_intervals=6, min_intervals=2,
                        initial_intervals=100,
                        decimal=3, output_dataframe=True)
result_cm = trans_cm.fit_transform(X[selected_features], y) 
result_cm

# %%
trans_cm.boundaries_ # show boundaries for all features

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
coarse_bins = dict()
# coarse_bins['AveRooms'] = [5.96,6.426,6.95,7.41]
# coarse_bins['HouseAge'] = [24,36,45]
# coarse_bins['Latitude'] = [34.1,34.47,37.59]
# coarse_bins['Longitude'] = [-121.59,-118.37]
# coarse_bins['Population'] = [420,694,877,1274,2812]


# %%
selected_features

for col in selected_features:
    print(col)
    bins0 = [i for i in list(trans_cm.boundaries_[col]) if i != np.inf]
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
 'penalty':['l2','l1'],   
'class_weight':[weights,None]
}

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

# %%
"""
Based on the classification performance table, choose 152 as the threshold (precision 20%, recall 79%)
"""

# %%
result_val['y_pred'] = result_val['TotalScore'].map(lambda x: 1 if x>152 else 0)                                   ##################
print(np.mean(result_val['y_pred']))
result_val


# %%
"""
Now we can interprete model scores based on the analysis above. For example, the 4th entry has a total score of 174. 2. The primary driver of house value in this case is housing age ('HouseAge') and position('Longitude')
"""
import pandas as pd
import os
import sklearn.preprocessing as skpp
import sklearn.model_selection as skms
import sklearn.impute as skim
import xgboost as xgb
import sklearn.metrics as skmt
import sklearn.feature_selection as skfs
import scipy.stats
import re
import eli5.sklearn
import matplotlib
import shap

# Data I/O

cwd = os.getcwd()

berlin_bnb_path = cwd + '/data/berlin-airbnb-data/listings_summary.csv'
berlin_data = pd.read_csv(berlin_bnb_path)
berlin_data.head().to_csv(path_or_buf='listing_summary_head.csv')


# Selecting informative columns and columns for prediction

selected_cols_X = [28] + list(range(48, 51)) + list(range(52, 58)) + [67] \
                  + list(range(71, 75)) + [76] + list(range(79, 86)) + list(range(89, 93)) + [95]
berlin_data_X = berlin_data.iloc[:, selected_cols_X]

selected_cols_y = [60]
berlin_data_y = berlin_data.iloc[:, selected_cols_y]
berlin_data_y = berlin_data_y.price.str.replace('$', '')
berlin_data_y = berlin_data_y.str.replace(',', '')
berlin_data_y = pd.to_numeric(berlin_data_y)


# Value counts to decide whether one-hot
# Numeric/many values -> numeric
# Numeric/not many values -> categorical
# String/many values -> text
# String/not many values -> categorical

berlin_data_X_vc = [berlin_data_X[col].unique().__len__() for col in berlin_data_X.columns]
berlin_data_X_vc = pd.Series(data=berlin_data_X_vc, index=berlin_data_X.columns)

# Remove vars with uniform value
berlin_data_X = berlin_data_X.drop(columns=berlin_data_X.columns[berlin_data_X_vc == 1])

# Update value counts
berlin_data_X_vc = [berlin_data_X[col].unique().__len__() for col in berlin_data_X.columns]

berlin_data_X_vc = pd.Series(data=berlin_data_X_vc, index=berlin_data_X.columns)

for i in range(berlin_data_X.columns.__len__()):
    if berlin_data_X_vc[i] < 34:
        print(berlin_data_X.columns[i])
        print(berlin_data_X.iloc[:, i].value_counts())
        print('\n')

# Define numerical and categorical columns

cat_vars = berlin_data_X.columns[berlin_data_X_vc < 6]
num_vars = berlin_data_X.columns[berlin_data_X_vc >= 6]


# Bin multi-labels and add the matrix to berlin_data_X; No NAs generated
# Deal with Nan values in 'Super host' columns
# Multi-label bin
# Remove sample points with less than 2000 positive values
# Record column names

berlin_data_X.loc[:, 'host_is_superhost'] = berlin_data_X.loc[:, 'host_is_superhost'].fillna(value='f')

amn_data = berlin_data.iloc[:, 58]
amn_data_word = [list(filter(None, re.split("{|,|\"|}", words))) for words in amn_data]

mlb = skpp.MultiLabelBinarizer()
amn_data_mlb = mlb.fit_transform(amn_data_word)

mlb_data_X = pd.DataFrame(data=amn_data_mlb,
                          index=amn_data.index,
                          columns=mlb.classes_)

mlb_drop = mlb_data_X.apply(sum, axis=0) < 2000
mlb_data_X = mlb_data_X.drop(columns=mlb_data_X.columns[mlb_drop])

mlb_vars = mlb_data_X.columns


# Add mlb data to berlin data
berlin_data_X = pd.concat([berlin_data_X, mlb_data_X], axis=1)


# Dealing with NAs
# To start with, how many NAs are there in each columns we have selected?
numOfNAs = pd.Series([berlin_data_X[col].isnull().sum() for col in berlin_data_X.columns], index=berlin_data_X.columns)

# Inspect manually the reasons of NAs
for col in berlin_data_X.columns:
    if numOfNAs[col] > 0:
        berlin_data_X[berlin_data_X[col].isnull()].head(20).to_csv(path_or_buf=col+'_NA_head20.csv')

# Before imputation, split Train and Validation data
train_X, val_X, train_y, val_y = skms.train_test_split(berlin_data_X, berlin_data_y, random_state=88)

# Imputation: cat var most frequent/ num var mean
# However, we don't have cat var NAs here. So that only num var imputation will be done
# NA imputation for numeric columns, simple mean
imputer_num = skim.SimpleImputer(strategy='mean')

imputed_train_X_num = pd.DataFrame(imputer_num.fit_transform(train_X[num_vars]),
                                   columns=train_X[num_vars].columns,
                                   index=train_X.index)
imputed_val_X_num = pd.DataFrame(imputer_num.transform(val_X[num_vars]),
                                 columns=val_X[num_vars].columns,
                                 index=val_X.index)

# One hot for categorical columns
# For cat vars one hot encoding

OH_encoder = skpp.OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train_X = pd.DataFrame(OH_encoder.fit_transform(train_X[cat_vars]))
OH_cols_val_X = pd.DataFrame(OH_encoder.transform(val_X[cat_vars]))

OH_cols_train_X.index = train_X.index
OH_cols_val_X.index = val_X.index

OH_cols_train_X.columns = OH_encoder.get_feature_names()
OH_cols_val_X.columns = OH_encoder.get_feature_names()

# Delete the first column of binary feature
OH_columns_drop = [False for i in range(0, OH_cols_train_X.columns.__len__())]
OH_value_count = berlin_data_X_vc[berlin_data_X_vc <= 6]
OH_value_firsts = OH_value_count.cumsum()[:-1]
OH_value_firsts = [0] + OH_value_firsts.to_list()

for i in range(0, OH_value_count.__len__()):
    if OH_value_count[i] == 2:
        OH_columns_drop[OH_value_firsts[i]] = True

OH_cols_train_X = OH_cols_train_X.drop(columns=OH_cols_train_X.columns[OH_columns_drop])
OH_cols_val_X = OH_cols_val_X.drop(columns=OH_cols_val_X.columns[OH_columns_drop])

# Concatenate num vars and OH vars

full_train_X = pd.concat([imputed_train_X_num, OH_cols_train_X, train_X[mlb_vars]], axis=1)
full_val_X = pd.concat([imputed_val_X_num, OH_cols_val_X, val_X[mlb_vars]], axis=1)

# Baseline model fitting

# my_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=multiprocessing.cpu_count())

# my_model.fit(full_train_X, train_y,
#              early_stopping_rounds=5,
#              eval_set=[full_val_X, val_y])

my_model = xgb.XGBRegressor()
my_model.fit(full_train_X, train_y)

predictions = my_model.predict(full_val_X)

mae = skmt.mean_absolute_error(y_pred=predictions, y_true=val_y)

print(mae)

# CV to search a grid of tuning parameter

param_grid = [
    {'max_depth': [6],
     'min_child_weight': [1],
     'gamma': [10],
     'colsample_bytree': [0.4],
     'learning_rate': [0.05],
     'n_estimators': [100]}
]

model_tune = skms.GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=param_grid,
                               scoring="neg_mean_absolute_error",
                               cv=5, n_jobs=5)
model_tune.fit(X=full_train_X, y=train_y)

model_tune_results = pd.DataFrame.from_dict(model_tune.cv_results_)
model_tune_results.to_csv(path_or_buf="tuning_result.csv")
print(model_tune_results)


cv_predictions = model_tune.predict(X=full_val_X)
cv_mae = skmt.mean_absolute_error(y_pred=cv_predictions, y_true=val_y)
print(cv_mae)


# Randomized CV search implementation

param_dist = {
        'n_estimators': scipy.stats.randint(100, 500),
        'learning_rate': scipy.stats.uniform(0.01, 0.4),
        'colsample_bytree': scipy.stats.uniform(0.2, 0.4),
        'subsample': scipy.stats.uniform(0.2, 0.4),
        'max_depth': scipy.stats.randint(4, 8),
        'min_child_weight': scipy.stats.randint(1, 5)
    }


model_tune_rand = skms.RandomizedSearchCV(estimator=xgb.XGBRegressor(), param_distributions=param_dist,
                                          n_iter=100,
                                          scoring="neg_mean_absolute_error",
                                          cv=5, n_jobs=12)

model_tune_rand.fit(X=full_train_X, y=train_y)

model_tune_rand_results = pd.DataFrame.from_dict(model_tune_rand.cv_results_)
model_tune_rand_results.to_csv(path_or_buf="tuning_rand_result.csv")

cv_rand_predictions = model_tune_rand.predict(X=full_val_X)
cv_rand_mae = skmt.mean_absolute_error(y_pred=cv_rand_predictions, y_true=val_y)
print(cv_rand_mae)


# Recursive feature elimination with cross-validation
# Set the params of the estimator

my_estimator = xgb.XGBRegressor()

xgb_params = {'max_depth': 6,
              'min_child_weight': 1,
              'gamma': 10,
              'colsample_bytree': 0.4,
              'learning_rate': 0.05,
              'n_estimators': 100}

my_estimator.set_params(**xgb_params)

my_estimator.fit(X=full_train_X, y=train_y)
my_estimator_prediction = my_estimator.predict(full_val_X)

my_estimator_mae = skmt.mean_absolute_error(y_true=val_y, y_pred=my_estimator_prediction)

my_scorer = skmt.make_scorer(score_func=skmt.mean_absolute_error, greater_is_better=False)

# Set the params of the RFE

rfe_cv = skfs.RFECV(estimator=my_estimator,
                    step=1,
                    min_features_to_select=20,
                    cv=5,
                    scoring=my_scorer,
                    n_jobs=5)

rfe_cv.fit(X=full_train_X, y=train_y)
rfe_cv.estimator_.fit(X=full_train_X, y=train_y)

rfe_prediction = rfe_cv.estimator_.predict(full_val_X)
rfe_mae = skmt.mean_absolute_error(y_true=val_y, y_pred=rfe_prediction)

# Update the training set

new_train_X = pd.DataFrame(rfe_cv.transform(full_train_X),
                           columns=full_train_X.columns[rfe_cv.get_support()],
                           index=full_train_X.index)
new_val_X = pd.DataFrame(rfe_cv.transform(full_val_X),
                         columns=full_train_X.columns[rfe_cv.get_support()],
                         index=full_val_X.index)

# Update the model and prediction using new features

model_tune.fit(X=new_train_X, y=train_y)

model_tune_results = pd.DataFrame.from_dict(model_tune.cv_results_)
model_tune_results.to_csv(path_or_buf="tuning_result.csv")
print(model_tune_results)

cv_predictions = model_tune.predict(X=new_val_X)
cv_mae = skmt.mean_absolute_error(y_pred=cv_predictions, y_true=val_y)
print(cv_mae)

# Get feature importance

fea_imp = model_tune.best_estimator_.get_booster().get_score()

fea_imp_plot = xgb.plot_importance(model_tune.best_estimator_.get_booster())
fea_imp_plot.get_figure().savefig('feature_imp.png')

# Permutation importance

perm = eli5.sklearn.PermutationImportance(estimator=model_tune).fit(X=new_val_X, y=val_y)
perm_df = pd.DataFrame(data={'Perm_imp': perm.feature_importances_},
                       index=new_val_X.columns)

# Shap explainer

shap_explainer = shap.TreeExplainer(model_tune.best_estimator_)

shap_values = shap_explainer.shap_values(new_val_X)

shap.initjs()
shap.force_plot(shap_explainer.expected_value[1], shap_values[1], new_val_X)

# show_weights(perm)

# Thinking about the next step to improve the modeling
# 1. try various way of feature selection (DONE)
# 2. Model interpretation: feature importance, single feature plot, SHAP value calculation
# 3. build a pipeline for baseline modelling of tabular data
# 4. Data visualization in python
# 5. At some point, apply natual language processing to description of properties


# Steps (for any estimator)
# 1. Data import train_X, val_X, train_y, val_y
# repeat: 2-4 until no significant improvement of performance
# 2. With all features, broad + narrow cv search for params
# 3. With the chosen params do recursive feature elimination
# 4. With selected features, broad + narrow cv search for params
# 5. Build model using selected feature and params

import os
import pandas as pd
import xgboost
import scipy.stats
import sklearn.model_selection
import sklearn.feature_selection
import sklearn.metrics
import multiprocessing
import joblib


def best_simple_estimator(cv_results, base_estimator_):
    rank_column = cv_results.filter(regex='rank').columns[0]
    cv_results = cv_results.nsmallest(n=5, columns=rank_column)
    cv_results = cv_results.nsmallest(n=1, columns='mean_fit_time')
    simple_param = eval(cv_results.params.values[0])
    base_estimator_.set_params(**simple_param)
    return base_estimator_


def best_metrics(cv_results):
    rank_column = cv_results.filter(regex='rank').columns[0]
    cv_results = cv_results.nsmallest(n=1, columns=rank_column)
    metric_column = cv_results.filter(regex='mean_test').columns[0]
    return cv_results[metric_column].values[0]


# data import
cwd = os.getcwd()
project_name = 'berlin_airbnb'
log_file_path = '/'.join([cwd, project_name, 'log'])

pp_train_X = pd.read_csv(filepath_or_buffer=log_file_path + '/pp_train_X.csv',
                         index_col=0)
pp_val_X = pd.read_csv(filepath_or_buffer=log_file_path + '/pp_val_X.csv',
                       index_col=0)
train_y = pd.read_csv(filepath_or_buffer=log_file_path + '/train_y.csv',
                      index_col=0, header=None, names='y')
val_y = pd.read_csv(filepath_or_buffer=log_file_path + '/val_y.csv',
                    index_col=0, header=None, names='y')


# base estimator and scorer

base_estimator = xgboost.XGBRegressor()
base_scorer = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'explained_variance']

# Random search for params

param_broad = {
        'n_estimators': 100,
        'learning_rate': scipy.stats.uniform(0.01, 0.4),
        'colsample_bytree': scipy.stats.uniform(0.1, 0.9),
        'subsample': scipy.stats.uniform(0.1, 0.9),
        'max_depth': scipy.stats.randint(2, 10),
        'min_child_weight': scipy.stats.randint(1, 5)
    }


try:
    random_cv_broad_results = pd.read_csv(filepath_or_buffer=log_file_path + '/cv_broad_1.csv', index_col=0)
except FileNotFoundError:
    random_cv_broad = sklearn.model_selection.RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_broad,
        n_iter=100,
        scoring=base_scorer,
        refit=base_scorer[0],
        n_jobs=multiprocessing.cpu_count(),
        cv=5
    )
    random_cv_broad.fit(X=pp_train_X, y=train_y)
    joblib.dump(random_cv_broad, log_file_path + 'cv_broad_1.joblib')
    random_cv_broad_results = pd.DataFrame.from_dict(random_cv_broad.cv_results_)
    random_cv_broad_results.to_csv(path_or_buf=log_file_path + '/cv_broad_1.csv')

base_metrics = best_metrics(random_cv_broad_results)

# Get the less complex model and use it for rfe

rfe_estimator = best_simple_estimator(random_cv_broad_results, base_estimator)

rfe_cv = sklearn.feature_selection.RFECV(
        estimator=rfe_estimator,
        step=1,
        min_features_to_select=20,
        cv=5,
        scoring=base_scorer[0],
        n_jobs=multiprocessing.cpu_count()
)

rfe_cv.fit(X=pp_train_X, y=train_y)

rfe_train_X = pd.DataFrame(
    rfe_cv.transform(pp_train_X),
    index=pp_train_X.index,
    columns=pp_train_X.columns[rfe_cv.get_support()]
)

rfe_val_X = pd.DataFrame(
    rfe_cv.transform(pp_val_X),
    index=pp_val_X.index,
    columns=pp_val_X.columns[rfe_cv.get_support()]
)

rfe_metrics = sklearn.metrics.mean_absolute_error(y_pred=rfe_cv.predict(X=pp_train_X),
                                                  y_true=train_y)

rfe_train_X.to_csv(path_or_buf=log_file_path + '/rfe_train_X.csv')
rfe_val_X.to_csv(path_or_buf=log_file_path + '/rfe_val_X.csv')

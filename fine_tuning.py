import os
import pandas as pd
import sklearn.model_selection
import xgboost
import scipy.stats
import multiprocessing
import joblib

# data import
cwd = os.getcwd()
project_name = 'berlin_airbnb'
log_file_path = '/'.join([cwd, project_name, 'log'])

rfe_train_X = pd.read_csv(filepath_or_buffer=log_file_path + '/rfe_train_X.csv',
                          index_col=0)
rfe_val_X = pd.read_csv(filepath_or_buffer=log_file_path + '/rfe_val_X.csv',
                        index_col=0)
train_y = pd.read_csv(filepath_or_buffer=log_file_path + '/train_y.csv',
                      index_col=0, header=None, names='y')
val_y = pd.read_csv(filepath_or_buffer=log_file_path + '/val_y.csv',
                    index_col=0, header=None, names='y')


# base estimator and scorer

base_estimator = xgboost.XGBRegressor()
base_scorer = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'explained_variance']


# Fine tuning

param_broad = {
        'n_estimators': scipy.stats.randint(100, 101),
        'learning_rate': scipy.stats.uniform(0.01, 0.4),
        'colsample_bytree': scipy.stats.uniform(0.1, 0.9),
        'subsample': scipy.stats.uniform(0.1, 0.9),
        'max_depth': scipy.stats.randint(2, 10),
        'min_child_weight': scipy.stats.randint(1, 5)
    }

# Random CV

random_cv_broad = sklearn.model_selection.RandomizedSearchCV(
    estimator=base_estimator,
    param_distributions=param_broad,
    n_iter=100,
    scoring=base_scorer,
    n_jobs=multiprocessing.cpu_count(),
    refit=base_scorer[0],
    cv=5
)

random_cv_broad.fit(X=rfe_train_X, y=train_y)
joblib.dump(random_cv_broad, log_file_path + '/ft_cv_broad_1.joblib')
random_cv_broad_results = pd.DataFrame.from_dict(random_cv_broad.cv_results_)
random_cv_broad_results.to_csv(path_or_buf=log_file_path + '/ft_cv_broad_1.csv')

mae = sklearn.metrics.mean_absolute_error(y_true=val_y,
                                          y_pred=random_cv_broad.predict(rfe_val_X))


full_train = pd.concat([rfe_train_X, rfe_val_X])
full_y = pd.concat([train_y, val_y])

final_estimator = random_cv_broad.best_estimator_

final_estimator.fit(X=full_train, y=full_y)
joblib.dump(final_estimator, log_file_path + '/ft_final_estimator.joblib')


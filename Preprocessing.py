import pandas as pd
import os
import sys
import sklearn.model_selection
import sklearn.impute
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.feature_extraction.text
import sklearn.feature_selection
import sklearn.decomposition


def transform_y(y):
    # function to perform necessary y transformation
    y = y.str.replace('$', '')
    y = y.str.replace(',', '')
    y = pd.to_numeric(y)
    return y


def transform_multi_label(train_raw, val_raw):
    def col_trans(col):
        col = col.str.replace(' ', '_')
        col = col.str.replace('{|,|\"|}', ' ')
        return col
    train_trans = train_raw.apply(col_trans)
    val_trans = val_raw.apply(col_trans)
    return train_trans, val_trans


def transform_categorical(train_raw, val_raw):
    for col_train, col_val in zip(train_raw, val_raw):
        vc_ratio = train_raw[col_train].value_counts() / train_raw[col_train].count()
        train_unique = train_raw[col_train].unique()
        to_replace = vc_ratio.index[vc_ratio < 0.05].to_list()
        train_raw[col_train].replace(to_replace=to_replace, value='AAA_low_freq', inplace=True)
        for i in val_raw[col_val].unique():
            if i not in train_unique:
                to_replace.append(i)
        val_raw[col_val].replace(to_replace=to_replace, value='AAA_low_freq', inplace=True)
    train_trans = train_raw
    val_trans = val_raw
    return train_trans, val_trans


def transform_text(train_raw, val_raw):
    train_trans = train_raw.fillna('')
    val_trans = val_raw.fillna('')
    return train_trans, val_trans


def standard_pp_X(pipeline, train_raw, val_raw):
    pipeline.fit(train_raw)
    try:
        train_pp = pd.DataFrame(pipeline.transform(train_raw).toarray(), index=train_raw.index)
        val_pp = pd.DataFrame(pipeline.transform(val_raw).toarray(), index=val_raw.index)
        ftr_names = pd.Series(pipeline[-2].get_feature_names())[pipeline[-1].get_support()]
        train_pp.columns = ftr_names
        val_pp.columns = ftr_names
    except AttributeError:
        try:
            train_pp = pd.DataFrame(pipeline.transform(train_raw), index=train_raw.index, columns=pipeline[-1].get_feature_names())
            val_pp = pd.DataFrame(pipeline.transform(val_raw), index=val_raw.index, columns=pipeline[-1].get_feature_names())
        except AttributeError:
            try:
                train_pp = pd.DataFrame(pipeline.transform(train_raw), index=train_raw.index, columns=train_raw.columns)
                val_pp = pd.DataFrame(pipeline.transform(val_raw), index=val_raw.index, columns=val_raw.columns)
            except AttributeError:
                train_pp = pd.DataFrame(pipeline.transform(train_raw), index=train_raw.index)
                val_pp = pd.DataFrame(pipeline.transform(val_raw), index=val_raw.index)
    return [train_pp, val_pp]


def columns_pp_X(pipeline, train_raw, val_raw):
    columns_pp = []
    for col_train, col_val in zip(train_raw, val_raw):
        col_pair = standard_pp_X(pipeline, train_raw[col_train], val_raw[col_val])
        col_pair[0] = col_pair[0].add_prefix(col_train + '__')
        col_pair[1] = col_pair[1].add_prefix(col_train + '__')
        columns_pp.append(col_pair)
    train_pp = pd.concat([df[0] for df in columns_pp], axis=1)
    val_pp = pd.concat([df[1] for df in columns_pp], axis=1)
    return [train_pp, val_pp]


# Default values
project_name = 'berlin_airbnb'
data_file_name = 'listings_summary.csv'
minor_fraction = 0.1


# Data input
cwd = os.getcwd()
data_file_path = '/'.join([cwd, project_name, 'data', data_file_name])
log_file_path = '/'.join([cwd, project_name, 'log'])
descriptive_path = '/'.join([log_file_path, 'descriptive_table_manual.csv'])

raw_data = pd.read_csv(data_file_path)
n_obs = raw_data.index.__len__()
descriptive_table_and_label = pd.read_csv(descriptive_path, index_col=0)
label_table = pd.read_csv('/'.join([log_file_path, 'label_table.csv']), index_col=0)


# Check if the every feature is labeled once in label_table
if not all(label_table.apply(sum, axis=1) == 1):
    sys.exit('Some columns have not exactly 1 label!')


# Get target value and do necessary transformation
target = raw_data[raw_data.columns[label_table.is_target]].iloc[:, 0]
target = transform_y(target)


# Get feature df by dropping columns labeled with
# 'is_target', 'is_leaking' or 'is_non_info', or if NA is exceeding a threshold
# or if unique count is 1 or if frequency of most frequent value exceeds threshold
to_drop = label_table.index[
    label_table.is_target |\
    label_table.is_leaking |\
    label_table.is_non_info |\
    (descriptive_table_and_label.Num_NA > minor_fraction * n_obs) |\
    (descriptive_table_and_label.unique == 1) |\
    (descriptive_table_and_label.freq > descriptive_table_and_label['count'] * (1 - minor_fraction))
    ]
features = raw_data.drop(columns=to_drop)
# Get the remaining feature labels too
features_label = label_table.drop(index=to_drop, columns=['is_target', 'is_leaking', 'is_non_info'])
features_descriptive = descriptive_table_and_label.drop(index=to_drop, columns=['is_target', 'is_leaking', 'is_non_info'])


# Train, Val split
raw_train_X, raw_val_X, train_y, val_y = sklearn.model_selection.train_test_split(features, target, random_state=88)


# Init a dict of different types of train, val, pipeline and category orders;
# And list of processed train/val pairs: pairs_pp
train_X_dict = dict([(col_name, raw_train_X[features_label.index[features_label[col_name]]]) for col_name in features_label.columns])
val_X_dict = dict([(col_name, raw_val_X[features_label.index[features_label[col_name]]]) for col_name in features_label.columns])

pipeline_dict = dict()
category_dict = dict()

pairs_pp = []

# Categorical records
category_dict['is_ordinal'] = [cat for cat in features_descriptive.unique_values[features_label['is_ordinal']].str.split('/').get_values()]
category_dict['is_binary'] = [cat for cat in features_descriptive.unique_values[features_label['is_binary']].str.split('/').get_values()]

# Transformation of train or val data; Records of categories
train_X_dict['is_categorical'], val_X_dict['is_categorical'] = transform_categorical(train_raw=train_X_dict['is_categorical'], val_raw=val_X_dict['is_categorical'])
train_X_dict['is_multi_label'], val_X_dict['is_multi_label'] = transform_multi_label(train_raw=train_X_dict['is_multi_label'], val_raw=val_X_dict['is_multi_label'])
train_X_dict['is_text'], val_X_dict['is_text'] = transform_text(train_raw=train_X_dict['is_text'], val_raw=val_X_dict['is_text'])

# Pipelines
pipeline_dict['is_continuous'] = sklearn.pipeline.Pipeline([
    ('imputation', sklearn.impute.SimpleImputer(strategy='mean')),
    ('scaling', sklearn.preprocessing.StandardScaler())
])

pipeline_dict['is_discrete'] = sklearn.pipeline.Pipeline([
    ('imputation', sklearn.impute.SimpleImputer(strategy='most_frequent')),
    ('scaling', sklearn.preprocessing.StandardScaler())
])

pipeline_dict['is_ordinal'] = sklearn.pipeline.Pipeline([
    ('SimpleImputer', sklearn.impute.SimpleImputer(strategy='most_frequent')),
    ('OrdinalEncoder', sklearn.preprocessing.OrdinalEncoder(categories=category_dict['is_ordinal'])),
    ('StandardScaler', sklearn.preprocessing.StandardScaler())
])

pipeline_dict['is_binary'] = sklearn.pipeline.Pipeline([
    ('SimpleImputer', sklearn.impute.SimpleImputer(strategy='most_frequent')),
    ('OrdinalEncoder', sklearn.preprocessing.OrdinalEncoder(categories=category_dict['is_binary']))
])

pipeline_dict['is_categorical'] = sklearn.pipeline.Pipeline([
    ('SimpleImputer', sklearn.impute.SimpleImputer(strategy='most_frequent')),
    ('OneHotEncoder', sklearn.preprocessing.OneHotEncoder(drop='first', sparse=False))
])

pipeline_dict['is_multi_label'] = sklearn.pipeline.Pipeline([
    ('CountVectorizer', sklearn.feature_extraction.text.CountVectorizer(binary=True)),
    ('VarianceThreshold', sklearn.feature_selection.VarianceThreshold(threshold=0.16))
])

pipeline_dict['is_text'] = sklearn.pipeline.Pipeline([
    ('CountVectorizer', sklearn.feature_extraction.text.CountVectorizer()),
    ('TfidfTransformer', sklearn.feature_extraction.text.TfidfTransformer()),
    ('TrancatedSVD', sklearn.decomposition.TruncatedSVD(n_components=20))
])

# Apply the pipelines to each type of data

for i in features_label.columns[:5]:
    pairs_pp.append(
        standard_pp_X(pipeline=pipeline_dict[i],
                      train_raw=train_X_dict[i],
                      val_raw=val_X_dict[i])
    )

for i in features_label.columns[-2:]:
    pairs_pp.append(
        columns_pp_X(pipeline=pipeline_dict[i],
                     train_raw=train_X_dict[i],
                     val_raw=val_X_dict[i])
    )


# Concat
pp_train_X = pd.concat([df[0] for df in pairs_pp], axis=1)
pp_val_X = pd.concat([df[1] for df in pairs_pp], axis=1)


# Save data to log folder
pp_train_X.to_csv(path_or_buf=log_file_path + '/pp_train_X.csv')
pp_val_X.to_csv(path_or_buf=log_file_path + '/pp_val_X.csv')

train_y.to_csv(path_or_buf=log_file_path + '/train_y.csv')
val_y.to_csv(path_or_buf=log_file_path + '/val_y.csv')


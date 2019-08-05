import pandas as pd
import os


def join_unique(column):
    # join the unique values of a df column if not many unique values
    if column.nunique() < 21:
        return '/'.join([str(x) for x in column.dropna().unique()])
    return ''


# Default values
project_name = 'berlin_airbnb'
data_file_name = 'listings_summary.csv'
minor_fraction = 0.1

# Data input
cwd = os.getcwd()
data_file_path = '/'.join([cwd, project_name, 'data', data_file_name])
log_file_path = '/'.join([cwd, project_name, 'log/'])
descriptive_path = '/'.join([log_file_path, 'descriptive_table.csv'])
raw_data = pd.read_csv(data_file_path)


# Log different column descriptive for inspection
os.makedirs(log_file_path, exist_ok=True)

raw_data.head(n=20).to_csv(path_or_buf='/'.join([log_file_path, 'head_of_table.csv']))

# Descriptive
descriptive_table = pd.DataFrame(raw_data.describe(include='all').transpose())
descriptive_table['unique'] = raw_data.nunique()
descriptive_table['freq'].fillna(value=0, inplace=True)
descriptive_table['Num_NA'] = raw_data.isna().apply(sum, axis=0)
descriptive_table['dtypes'] = raw_data.dtypes

# Trimming indicators
descriptive_table['is_target'] = False
descriptive_table['is_leaking'] = False
descriptive_table['is_non_info'] = False

# Feature properties
descriptive_table['is_continuous'] = False
descriptive_table['is_discrete'] = False
descriptive_table['is_ordinal'] = False
descriptive_table['is_binary'] = False
descriptive_table['is_categorical'] = False
descriptive_table['is_multi_label'] = False
descriptive_table['is_text'] = False

# Examples
descriptive_table['unique_values'] = raw_data.apply(join_unique)
descriptive_table = pd.concat([descriptive_table, raw_data.head(5).transpose()], axis=1)

# Remove unnecessary columns
descriptive_table.drop(columns='top', inplace=True)

# Log file to csv
descriptive_table.to_csv(path_or_buf=descriptive_path)

# Log the first rows of columns with NAs to inspect the reason
os.makedirs(log_file_path + 'NA_inspection', exist_ok=True)

for col in raw_data.columns:
    if 0 < descriptive_table.Num_NA[col] < minor_fraction * raw_data.index.__len__():
        raw_data[raw_data[col].isnull()].head(20).to_csv(path_or_buf=log_file_path + 'NA_inspection/' + 'NA_head20_' + col + '.csv')


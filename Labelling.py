import pandas as pd
import os

# Default values
project_name = 'berlin_airbnb'
data_file_name = 'listings_summary.csv'


# Data input
cwd = os.getcwd()
data_file_path = '/'.join([cwd, project_name, 'data', data_file_name])
log_file_path = '/'.join([cwd, project_name, 'log'])
descriptive_path = '/'.join([log_file_path, 'descriptive_table_manual.csv'])

raw_data = pd.read_csv(data_file_path)

# Import the manually processed label
descriptive_table_and_label = pd.read_csv(descriptive_path, index_col=0)

# Extra labeling process

# Save the descriptive table and label table
descriptive_table_and_label.to_csv(descriptive_path)

label_table = descriptive_table_and_label.filter(regex='is_')
label_table.to_csv('/'.join([log_file_path, 'label_table.csv']))

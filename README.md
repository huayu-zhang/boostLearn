**boostLearn Project**  
Aiming to produce a baseline model from any type of tabular data

# Overall road map
1. Description.py  
    * Produce basic descriptive table
    * Every row is a column in data 
    * Add type of column labels, which can be edited in labelling.py or manually
    
2. Labelling.py
    * Label every column with one of the data types (including is_continuous, is_discrete, is_ordinal, is_binary, is_categorical, is_multilabel, is_text, is_noinfo)
    * Labels can be edited manually
    * <project name>/log/descriptive_table_manual.csv will be used in Preprocessing.py

3. Preprocessing.py
    * Train-test split
    * Pipelines to transform and preprocess different types of data
    * Concatenate all data together to produce  
     (pp_train_X, pp_val_X, train_y, val_y)

4. feature_selection.py
    * Crude tune a estimator using all data
    * Recursive feature elimination using the estimator
    * Output rfe-trimmed data  
    (rfe_train_X, rfe_val_X, train_y, val_y)
    
5. fine_tuning.py
    * Fine tune the estimator using rfe-trimmed data
    
6. model_performance
    * Working in progress
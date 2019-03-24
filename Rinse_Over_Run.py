import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Point to data folder
DATA_DIR = '/data/training/'

#Define functions needed for measuring MAPE
def mape(pred, actual):
    denominator = np.abs(actual.values)
    denominator[denominator < 290000] = 290000
    f_pred = ((np.abs(pred - actual.values))/denominator).mean()
    return f_pred

def evaluate(preds, test_labels):
    predictions = preds.values
    errors = abs(predictions - test_labels)
    denominators=np.maximum(test_labels,290000)
    array =  (errors/denominators)
    mape = array.mean()
    accuracy = mape
    print('Model Performance')
    print('Average Error: {:0.4f}  NTU.L.'.format(np.mean(errors)))
    print('Accuracy = {:0.4f}.'.format(accuracy))
    
    return array

#define functions used in clustering 
def create_object_dataset_for_clustering(df_expl, df_target):
    
    df = df_expl.merge(df_target, on='process_id')

    df_kurt = pd.DataFrame(df.groupby('object_id')['final_rinse_total_turbidity_liter'].apply(pd.DataFrame.kurt).reset_index())
    df_kurt.columns = ['object_id', 'kurtosis']
    df_skew = pd.DataFrame(df.groupby('object_id')['final_rinse_total_turbidity_liter'].skew().reset_index())
    df_skew.columns = ['object_id', 'skew']
    df_median = pd.DataFrame(df.groupby('object_id')['final_rinse_total_turbidity_liter'].median().reset_index())
    df_median.columns = ['object_id', 'median']
    df_count = pd.DataFrame(df.groupby('object_id')['process_id'].nunique().reset_index())
    df_count.columns = ['object_id', 'num_processes']

    df_object_summary = df_kurt.merge(df_skew, on='object_id')
    df_object_summary = df_object_summary.merge(df_median, on='object_id')
    df_object_summary = df_object_summary.merge(df_count, on='object_id')

    scaler = StandardScaler()
    df_normalised_object_summary = pd.DataFrame(scaler.fit_transform(df_object_summary[['kurtosis', 'skew', 'median']]))
    df_normalised_object_summary.columns = ['kurtosis_norm', 'skew_norm', 'median_norm']

    df_object_summary = pd.concat([df_object_summary, df_normalised_object_summary], axis=1)
    
    return df_object_summary

def create_object_id_clusters_list(i_num_clusters, df_for_clustering):
    
    y_pred = KMeans(n_clusters=i_num_clusters, random_state=0).fit_predict(df_for_clustering[['kurtosis_norm', 'skew_norm', 'median_norm']])
    df_for_clustering['cluster'] = y_pred

    list_of_list_of_objects = []
    for i in range(0, i_num_clusters):
        l_cur_objects_in_cluster = list(df_for_clustering[df_for_clustering['cluster']==i]['object_id'])
        list_of_list_of_objects.append(l_cur_objects_in_cluster)
        
    return list_of_list_of_objects

# data for training our model
X_raw = pd.read_csv(DATA_DIR+'/train_values.csv', index_col=0, parse_dates=['timestamp'])
recipe_metadata = pd.read_csv(DATA_DIR+'/recipe_metadata.csv',index_col=0)
y_raw = pd.read_csv(DATA_DIR+'/train_labels.csv',index_col=0)
#drop final phase data
X_raw_1 = X_raw[X_raw.phase != 'final_rinse']

#list of object_id's in the training data
l_objid=X_raw1['object_id'].drop_duplicates()

#list of coulums to use in training
ts_cols = [
    'process_id',
    'object_id',
    'phase',
    'supply_flow',
    'supply_pressure',
    'return_temperature',
    'return_conductivity',
    'return_turbidity',
    'return_flow',
    'tank_temperature_pre_rinse',
    'tank_temperature_caustic',
    'tank_temperature_acid',
    'tank_concentration_caustic',
    'tank_concentration_acid',
    'tank_lsh_acid','tank_lsh_pre_rinse']

#list of boolean coulums to use in training
boolean_cols = [
    'process_id',
    'object_id',
    'phase',
    'supply_pump',
    'supply_pre_rinse',
    'supply_caustic',
    'return_caustic',
    'supply_acid',
    'return_acid',
    'supply_clean_water',
    'return_recovery_water',
    'return_drain',
    'object_low_level']    

def prep_data(df_to_prep):
    raw_train_values=X_raw_1
    raw_train_values=raw_train_values[ts_cols]
    raw_train_values['process_phase'] = raw_train_values.process_id.astype(str) + '_' + raw_train_values.phase.astype(str)
    raw_train_values=raw_train_values.drop(columns=['process_id'])
    #create turb feature
    raw_train_values['turb']=raw_train_values.return_turbidity * raw_train_values.return_flow
    raw_features = raw_train_values.groupby(['process_phase','object_id','phase']).agg(['mean','std','min','max','median'])
    raw_features.columns = ['_'.join(col).strip() for col in raw_features.columns.values]
    raw_features=raw_features.fillna(0)
    raw_features.reset_index( inplace=True)
    raw_features['process_id']=raw_features['process_phase'].apply(str).str[0:5]
    raw_features['process_id']=raw_features['process_id'].apply(int)

    # create a boolean features using sum true divided by count
    boolean_train_values=X_raw_1
    boolean_train_values=boolean_train_values[boolean_cols]
    boolean_train_values['process_phase'] = boolean_train_values.process_id.astype(str) + '_' + boolean_train_values.phase.astype(str)
    boolean_train_values=boolean_train_values.drop(columns=['process_id'])
    boolean_features = boolean_train_values.groupby(['process_phase','object_id','phase']).agg(['count','sum'])
    boolean_features.columns = ['_'.join(col).strip() for col in boolean_features.columns.values]
    boolean_features=boolean_features.fillna(0)
    boolean_features.reset_index( inplace=True)
    boolean_features=boolean_features.drop(columns=['object_id','phase'])
    l_drop=['process_phase','supply_pump_count', 'supply_pump_sum',
        'supply_pre_rinse_count', 'supply_pre_rinse_sum',
        'supply_caustic_count', 'supply_caustic_sum', 'return_caustic_count',
        'return_caustic_sum', 'supply_acid_count', 'supply_acid_sum',
        'return_acid_count', 'return_acid_sum', 'supply_clean_water_count',
        'supply_clean_water_sum', 'return_recovery_water_count',
        'return_recovery_water_sum', 'return_drain_count', 'return_drain_sum',
        'object_low_level_count', 'object_low_level_sum',]
    boolean_features['supply_pump_prc']=boolean_features['supply_pump_sum']/boolean_features['supply_pump_count']
    boolean_features['supply_pre_rinse_prc']=boolean_features['supply_pre_rinse_sum']/boolean_features['supply_pre_rinse_count']
    boolean_features['supply_caustic_prc']=boolean_features['supply_caustic_sum']/boolean_features['supply_caustic_count']
    boolean_features['return_caustic_prc']=boolean_features['return_caustic_sum']/boolean_features['return_caustic_count']
    boolean_features['supply_acid_prc']=boolean_features['supply_acid_sum']/boolean_features['supply_acid_count']
    boolean_features['return_acid_prc']=boolean_features['return_acid_sum']/boolean_features['return_acid_count']
    boolean_features['supply_clean_water_prc']=boolean_features['supply_clean_water_sum']/boolean_features['supply_clean_water_count']
    boolean_features['return_recovery_water_prc']=boolean_features['return_recovery_water_sum']/boolean_features['return_recovery_water_count']
    boolean_features['return_drain_prc']=boolean_features['return_drain_sum']/boolean_features['return_drain_count']
    boolean_features['object_low_level_prc']=boolean_features['object_low_level_sum']/boolean_features['object_low_level_count']
    boolean_features=boolean_features.drop(columns=l_drop)
    boolean_features.head()

    #Create phase dummies & merge 
    df_dummies=pd.get_dummies(raw_features['phase'])
    pr_features=pd.concat([raw_features, df_dummies], axis=1)
    p_features=pd.concat([pr_features, boolean_features], axis=1)
    p_features=p_features.drop(columns=['process_phase','phase'])

    #attach result variable to remove outliers later when creating a model, drop extra dummy
    pa_features=p_features.merge(recipe_metadata, on='process_id', how='left')
    Xres_in=pa_features.merge(y_raw, on='process_id', how='left')
    Xres_in=Xres_in.drop(columns=['pre_rinse_x','pre_rinse_y','final_rinse'])
    Xres_in=Xres_in.set_index('process_id')
    df_prepped_data= Xres_in

    return df_prepped_data   

# create the dataset for kmeans
df_object_summary_for_clustering = create_object_dataset_for_clustering(X_raw, y_raw)
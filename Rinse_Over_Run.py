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
DATA_DIR = '/data/training'
DATA_DIR_TEST = '/data/test'

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
# load the test data
test_values = pd.read_csv(DATA_DIR_TEST+'/test_values.csv',index_col=0,parse_dates=['timestamp'])
#drop final phase data
X_raw_1 = X_raw[X_raw.phase != 'final_rinse']

#list of object_id's in the training data
l_objid=X_raw_1['object_id'].drop_duplicates()

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

def prep_train_data(X,y):
    raw_train_values=X
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
    boolean_train_values=X
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

    #Create phase dummies & merge 
    df_dummies=pd.get_dummies(raw_features['phase'])
    pr_features=pd.concat([raw_features, df_dummies], axis=1)
    p_features=pd.concat([pr_features, boolean_features], axis=1)
    p_features=p_features.drop(columns=['process_phase','phase'])

    #attach result variable to remove outliers later when creating a model, drop extra dummy
    pa_features=p_features.merge(recipe_metadata, on='process_id', how='left')
    Xres_in=pa_features.merge(y, on='process_id', how='left')
    Xres_in=Xres_in.drop(columns=['pre_rinse_x','pre_rinse_y','final_rinse'])
    Xres_in=Xres_in.set_index('process_id')
    df_prepped_data= Xres_in
    print('Training Data prep successful')
    return df_prepped_data   

def prep_test_data(X):
    # create a unique phase identifier by joining process_id and phase
    raw_test_values=X
    raw_test_values=raw_test_values[ts_cols]
    raw_test_values['process_phase'] = raw_test_values.process_id.astype(str) + '_' + raw_test_values.phase.astype(str)
    raw_test_values=raw_test_values.drop(columns=['process_id'])
    raw_test_values['turb']=raw_test_values.return_turbidity * raw_test_values.return_flow
    raw_test_features = raw_test_values.groupby(['process_phase','object_id','phase']).agg(['mean','std','min','max','median'])
    raw_test_features.columns = ['_'.join(col).strip() for col in raw_test_features.columns.values]
    raw_test_features=raw_test_features.fillna(0)
    raw_test_features.reset_index( inplace=True)
    raw_test_features['process_id']=raw_test_features['process_phase'].apply(str).str[0:5]
    raw_test_features['process_id']=raw_test_features['process_id'].apply(int)
    # create a unique phase identifier by joining process_id & phase and group by object_id & phase
    boolean_test_values=X
    boolean_test_values=boolean_test_values[boolean_cols]
    boolean_test_values['process_phase'] = boolean_test_values.process_id.astype(str) + '_' + boolean_test_values.phase.astype(str)
    boolean_test_values=boolean_test_values.drop(columns=['process_id'])
    boolean_test_features = boolean_test_values.groupby(['process_phase','object_id','phase']).agg(['count','sum'])
    boolean_test_features.columns = ['_'.join(col).strip() for col in boolean_test_features.columns.values]
    boolean_test_features=boolean_test_features.fillna(0)
    boolean_test_features.reset_index( inplace=True)
    boolean_test_features=boolean_test_features.drop(columns=['object_id','phase'])
    l_drop=['process_phase','supply_pump_count', 'supply_pump_sum',
        'supply_pre_rinse_count', 'supply_pre_rinse_sum',
        'supply_caustic_count', 'supply_caustic_sum', 'return_caustic_count',
        'return_caustic_sum', 'supply_acid_count', 'supply_acid_sum',
        'return_acid_count', 'return_acid_sum', 'supply_clean_water_count',
        'supply_clean_water_sum', 'return_recovery_water_count',
        'return_recovery_water_sum', 'return_drain_count', 'return_drain_sum',
        'object_low_level_count', 'object_low_level_sum',]
    boolean_test_features['supply_pump_prc']=boolean_test_features['supply_pump_sum']/boolean_test_features['supply_pump_count']
    boolean_test_features['supply_pre_rinse_prc']=boolean_test_features['supply_pre_rinse_sum']/boolean_test_features['supply_pre_rinse_count']
    boolean_test_features['supply_caustic_prc']=boolean_test_features['supply_caustic_sum']/boolean_test_features['supply_caustic_count']
    boolean_test_features['return_caustic_prc']=boolean_test_features['return_caustic_sum']/boolean_test_features['return_caustic_count']
    boolean_test_features['supply_acid_prc']=boolean_test_features['supply_acid_sum']/boolean_test_features['supply_acid_count']
    boolean_test_features['return_acid_prc']=boolean_test_features['return_acid_sum']/boolean_test_features['return_acid_count']
    boolean_test_features['supply_clean_water_prc']=boolean_test_features['supply_clean_water_sum']/boolean_test_features['supply_clean_water_count']
    boolean_test_features['return_recovery_water_prc']=boolean_test_features['return_recovery_water_sum']/boolean_test_features['return_recovery_water_count']
    boolean_test_features['return_drain_prc']=boolean_test_features['return_drain_sum']/boolean_test_features['return_drain_count']
    boolean_test_features['object_low_level_prc']=boolean_test_features['object_low_level_sum']/boolean_test_features['object_low_level_count']
    boolean_test_features=boolean_test_features.drop(columns=l_drop)
    df_dummies=pd.get_dummies(raw_test_features['phase'])
    pr_test_features=pd.concat([raw_test_features, df_dummies], axis=1)
    p_test_features=pd.concat([pr_test_features, boolean_test_features], axis=1)
    p_test_features=p_test_features.drop(columns=['process_phase','phase'])
    p_test_features=p_test_features.merge(recipe_metadata, on='process_id', how='left')
    p_test_features=p_test_features.set_index('process_id')
    p_test_features=p_test_features.drop(columns=['pre_rinse_x','pre_rinse_y','final_rinse'])
    print('Test Data prep successful')

def Cluster_Flow2(Xtrain,ytrain,Xtest,aggtype='min'):
    # create the dataset for kmeans
    df_object_summary_for_clustering = create_object_dataset_for_clustering(Xtrain, ytrain)

    i_best_num_clusters = 20
    l_optimum_objects_in_clusters = create_object_id_clusters_list(i_best_num_clusters, df_object_summary_for_clustering)

    #create RF for each object_id
    for i in range(0, i_best_num_clusters):
        X_prep_a=Xtrain[Xtrain.object_id.isin(l_optimum_objects_in_clusters[i])]
        #select data with result within certain distance of the median
        X_prep_b=X_prep_a[np.abs(X_prep_a.final_rinse_total_turbidity_liter-X_prep_a.final_rinse_total_turbidity_liter.median()) <= (2*X_prep_a.final_rinse_total_turbidity_liter.median())]
        train_p_id=pd.DataFrame()
        train_p_id['process_id']=X_prep_b.index
        #drop columns not used in prediction
        y_train=train_p_id.merge(y_raw, on='process_id', how='left')
        y_train=y_train.set_index('process_id')
        #split up training & test
        X_train, X_test, y_train, y_test = train_test_split(X_prep_b, y_train, test_size = 0.20, random_state = 42)
        #drop columns not used in prediction
        X_train=X_train.drop(columns=['object_id','final_rinse_total_turbidity_liter'])
        #create regressor cxg_i for each cluster i
        exec(f"cxg_{i} = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.75, learning_rate = 0.05,max_depth = 9, alpha = 0, n_estimators = 50,subsample = 0.75 ,n_jobs=-1)")
        exec(f'cxg_{i}.fit(X_train, y_train)')
    out_preds=[]
    lout_process=[]
    for i in range(0, i_best_num_clusters):
        X_prep_a=Xtest[Xtest.object_id.isin(l_optimum_objects_in_clusters[i])]
        X_test=X_prep_a.drop(columns=['object_id'])
        process_ids=X_prep_a.index
        exec(f'l_out=cxg_{i}.predict(X_test)')
        out_preds.append(l_out)
        lout_process.append(process_ids)
    tlooksy=pd.DataFrame(np.concatenate( out_preds, axis=0 ))
    tlooksy['process_id']=(np.concatenate( lout_process, axis=0 ))
    df_test_pred=tlooksy
    df_test_pred['pred']=df_test_pred[0]
    df_test_pred=df_test_pred.drop(columns=[0])
    df_pred=df_test_pred.groupby(['process_id']).agg([aggtype])
    df_pred.columns = ['_'.join(col).strip() for col in df_pred.columns.values]
    df_pred.columns=['final_rinse_total_turbidity_liter']
    print('Clustering Flow2 successful')
    return df_pred

def Object_Flow1(Xtrain,ytrain,Xtest,aggtype='min'):
    l_objid=Xtrain['object_id'].drop_duplicates()
    for i in l_objid:
        X_prep_a=Xtrain[Xtrain.object_id == i]
        #select data with result within certain distance of the median
        X_prep_b=X_prep_a[np.abs(X_prep_a.final_rinse_total_turbidity_liter-X_prep_a.final_rinse_total_turbidity_liter.median()) <= (2*X_prep_a.final_rinse_total_turbidity_liter.median())]
        train_p_id=pd.DataFrame()
        train_p_id['process_id']=X_prep_b.index
        #drop columns not used in prediction
        y_train=train_p_id.merge(y_raw, on='process_id', how='left')
        y_train=y_train.set_index('process_id')
        #split up training & test
        X_train, X_test, y_train, y_test = train_test_split(X_prep_b, y_train, test_size = 0.20, random_state = 42)
        #drop columns not used in prediction
        X_train=X_train.drop(columns=['object_id','final_rinse_total_turbidity_liter'])
        #create regressor cxg_i for each cluster i
        exec(f"oxg_{i} = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.75, learning_rate = 0.05,max_depth = 9, alpha = 0, n_estimators = 50,subsample = 0.75 ,n_jobs=-1)")
        exec(f'oxg_{i}.fit(X_train, y_train)')
    out_preds=[]
    lout_process=[]
    test_obj_id=Xtest['object_id'].drop_duplicates()
    test_obj_id=np.array(test_obj_id,dtype=int)  
    #create RF for each object_id
    for i in test_obj_id:
        X_prep_a=Xtest[Xtest.object_id == i]
        X_test=X_prep_a.drop(columns=['object_id'])
        process_ids=X_prep_a.index
        exec(f'l_out=oxg_{i}.predict(X_test)')
        out_preds.append(l_out)
        lout_process.append(process_ids)
    tlooksy=pd.DataFrame(np.concatenate( out_preds, axis=0 ))
    tlooksy['process_id']=(np.concatenate( lout_process, axis=0 ))
    df_test_pred=tlooksy
    df_test_pred['pred']=df_test_pred[0]
    df_test_pred=df_test_pred.drop(columns=[0])
    df_pred=df_test_pred.groupby(['process_id']).agg([aggtype])
    df_pred.columns = ['_'.join(col).strip() for col in df_pred.columns.values]
    df_pred.columns=['final_rinse_total_turbidity_liter']
    print('Object Flow1 successful')
    return df_pred
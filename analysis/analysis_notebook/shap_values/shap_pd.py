# singularity exec --nv --bind /data/dadua2 --workdir /data/dadua2/BaselineClassification docker://anantdadu/baselineclassification_xgb_gpu:1.0 /opt/miniconda3/envs/h2oai_env/bin/python -i shap_pd.py 


import pickle
from pathlib import Path
import xgboost as xgb
import GPUtil
import anndata as ad
from sklearn.preprocessing import MultiLabelBinarizer
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from sklearn import metrics
import pandas as pd
import numpy as np
import copy
from itertools import product

directory_path = Path("ml_generated_data/23_03_30/training/generated_data/example/prediction/apr5/IMAGING_PDPREDICTION_CLASSIFIER")
with open(directory_path / "imaging__pddiagnosisprediction_train100_numpy_array.pkl", "rb") as f:
    data = pickle.load(f)


with open(directory_path / "imaging__pddiagnosisprediction_train100_replication_numpy_array.pkl", "rb") as f:
    replication_data = pickle.load(f)

split_df = pd.read_csv(directory_path / "numerical$imaging__pddiagnosisprediction_train100_split_index.csv", header=None)
split_df.columns = ['ID', 'fold']


rep_split_df = pd.read_csv(directory_path / "imaging__pddiagnosisprediction_train100_replication_id_multi_index.csv", header=None)
rep_split_df.columns = ['sample', 'ID']


ad_imaging = ad.read("ml_generated_data/23_03_30/predictiveModel/pd_imaging.h5ad")
feature_names = ad_imaging.var.index
class_labels = {0: 'Control', 1: 'Dementia'}
replication_ad_imaging = ad.read("ml_generated_data/23_03_30/predictiveModel/replication_pd_imaging.h5ad")
X_rep = np.array(replication_ad_imaging.X)
y_rep = [-1] * len(X_rep)
id_rep = replication_ad_imaging.obs.index
ensemble_prediction = pd.read_csv("ml_generated_data/23_03_30/training/results_data/predictiveAnalysis/classification_detailed_metrics/phenotype_data/replication_ensemble_pd_imaging_probability_scores.csv")

    

def parameters_generator(parameters_dict):            
            output = []
            current_optimized = copy.deepcopy(parameters_dict['default'])
            for key, value in parameters_dict.items():
               if key == 'default':
                   continue
               else:
                   keys1, values1 = zip(*value.items())
                   for bundle in product(*values1):
                       params = copy.deepcopy(current_optimized)
                       d = copy.deepcopy(dict(zip(keys1, bundle)))
                       params.update(d)
                       yield params

def get_shap_data(param, X_train, X_test, X_rep,  y_train, y_test, y_rep, id_train, id_test, id_rep, feature_names, class_labels, shap=False):
        mlb = MultiLabelBinarizer()
        dtrain = xgb.DMatrix(X_train, label=mlb.fit_transform([(str(int(i))) for i in y_train]), feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=mlb.fit_transform([(str(int(i))) for i in y_test]), feature_names=feature_names)
        drep = xgb.DMatrix(X_rep, label=mlb.fit_transform([(str(int(i))) for i in y_rep]), feature_names=feature_names)
        param["tree_method"] = "gpu_hist"
        model  = xgb.train(param, dtrain, verbose_eval=False)
        deviceIDs = GPUtil.getAvailable(order = 'memory', limit = 4, maxLoad = 1, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
        if len(deviceIDs) == 0:
            pass
        else:
            model.set_param({"predictor": "gpu_predictor"})
        shap_values_train = model.predict(dtrain, pred_contribs=True) if shap else {}
        shap_values_test = model.predict(dtest, pred_contribs=True) if shap else {}
        shap_values = {'shap_values_train': shap_values_train, 'shap_values_test': shap_values_test, }
        shap_values_rep = model.predict(drep, pred_contribs=True) if shap else {}
        shap_values['shap_values_rep'] = shap_values_rep
        other_info = {}
        other_info['ID_train'] = id_train
        other_info['ID_test'] = id_test
        other_info['ID_rep'] = id_rep 
        other_info['y_pred_train'] = model.predict(dtrain)
        other_info['y_pred_test'] = model.predict(dtest)
        other_info['y_pred_rep'] = model.predict(drep) 
        if len(deviceIDs) == 0:
            pass
        else:
            model.set_param({"predictor": "cpu_predictor"})
        return model, (shap_values, feature_names, other_info, class_labels)



num_round = 500
params_d = {
            "default":{
                'objective': 'binary:logistic',
                "eta": 0.6,
                "max_depth": 2,
                "tree_method": "gpu_hist",
                "gamma": 1,
                "min_child_weight": 5,
                "max_delta_step": 4,
                "lambda": 0.1,
                "eval_metric": "auc",
                # "num_feature": 10,
                # "booster": "gblinear"
            },
            'block1':{
            'objective': ['binary:logistic'],
            "eta": [0.01, 0.05],
            "max_depth": [1, 2, 3],
            "tree_method": ["gpu_hist"],
            "gamma": [0, 1],
            "min_child_weight": [10],
            "eval_metric": ["auc"],
            "max_delta_step": [1, 20],
            "lambda": [1, 20],
            },
}

params_d = {
            "default":{
                'objective': 'binary:logistic',
                "eta": 0.6,
                "max_depth": 2,
                "tree_method": "gpu_hist",
                "gamma": 1,
                "min_child_weight": 5,
                "max_delta_step": 4,
                "lambda": 0.1,
                "eval_metric": "auc",
                # "num_feature": 10,
                # "booster": "gblinear"
            },
            'block1':{
            'objective': ['binary:logistic'],
            "eta": np.arange(0.01, 1, 0.3),
            "max_depth": range(1, 10, 3),
            "tree_method": ["gpu_hist"],
            "gamma": np.arange(0.01, 1, 0.3),
            "min_child_weight": range(1, 20, 5),
            "eval_metric": ["auc"],
            "max_delta_step": range(1, 20, 5),
            "lambda": range(1, 20, 5),
            },
}
num_round = 500
params_d = {
            "default":{
                'objective': 'binary:logistic',
                "eta": 0.6,
                "max_depth": 2,
                "tree_method": "gpu_hist",
                "gamma": 1,
                "min_child_weight": 5,
                "max_delta_step": 4,
                "lambda": 0.1,
                "eval_metric": "auc",
                # "num_feature": 10,
                # "booster": "gblinear"
            },
            'block1':{
            'objective': ['binary:logistic'],
            "eta": [0.01, 0.05],
            "max_depth": [1, 2, 3],
            "tree_method": ["gpu_hist"],
            "gamma": [0, 1],
            "min_child_weight": [10],
            "eval_metric": ["auc"],
            "max_delta_step": [1, 20],
            "lambda": [1, 20],
            },
}

params_d = {
            "default":{
                'objective': 'binary:logistic',
                "eta": 0.6,
                "max_depth": 2,
                "tree_method": "gpu_hist",
                "gamma": 1,
                "min_child_weight": 5,
                "max_delta_step": 4,
                "lambda": 0.1,
                "eval_metric": "auc",
                # "num_feature": 10,
                # "booster": "gblinear"
            },
            'block1':{
            'objective': ['binary:logistic'],
            "eta": np.arange(0.01, 1, 0.4),
            "max_depth": range(1, 10, 4),
            "tree_method": ["gpu_hist"],
            "gamma": np.arange(0.01, 1, 0.4)[::-1],
            "min_child_weight": range(1, 12, 5)[::-1],
            "eval_metric": ["auc"],
            "max_delta_step": range(1, 10, 5)[::-1],
            "lambda": range(1, 10, 5),
            },
}
params = list(parameters_generator(params_d))
param = params[0]
print (len(params))

def cube(e, show=False):
    # print (e, params[e])
    # model_temp = xgb.train(param, dtrain_train, num_round, feval=calculate_roc_auc_eval_xgb_binary, evals=eval_set, verbose_eval=False, early_stopping_rounds=20)
    all_outputs = []
    y_rep_predictions = []
    for fold in ['fold1', 'fold2', 'fold3']:
        train_index = split_df[~(split_df['fold']==fold)].index
        test_index = split_df[(split_df['fold']==fold)].index
        X_train = data[train_index, 1:]
        y_train = data[train_index, 0]
        id_train = split_df.iloc[train_index]['ID']
        X_test = data[test_index, 1:]
        y_test = data[test_index, 0]
        id_test = split_df.iloc[test_index]['ID']
        output = get_shap_data(copy.deepcopy(params[e]), X_train, X_test, X_rep,  y_train, y_test, y_rep, id_train, id_test, id_rep, feature_names, class_labels, shap=show)
        y_rep_predictions.append(output[1][2]['y_pred_rep'][:, 1])
        all_outputs.append(output)
        if show:
            print (fold, 'Train:', round(metrics.roc_auc_score(y_train, output[1][2]['y_pred_train'][:, 1]), 2) )
            print (fold, 'Test:', round(metrics.roc_auc_score(y_test, output[1][2]['y_pred_test'][:, 1]), 2) )
            
    corr, _ = pearsonr(np.mean(y_rep_predictions, axis=0), ensemble_prediction['rep_y_probas1'])
    print(e, params[e], 'Pearsons correlation: %.3f' % corr)
    if show:
        return all_outputs
    return round(corr, 3)


from tqdm.contrib.concurrent import process_map

if False:
    result_list = process_map(cube, list(range(len(params))), max_workers=32)
    max_index = result_list.index(max(result_list))
    with open('optimized_pd.txt', 'w') as f:
        print (max_index, result_list[max_index], params[max_index], file=f)


max_index = 145
print (max_index, params[max_index])
all_outputs = cube(max_index, show=True)
   
with open("optimized_pd.pkl", "wb") as f:
    pickle.dump(all_outputs, f)

K = [all_outputs[i][1] for i in range(len(all_outputs))]
with open("optimized_pd_shap.pkl", "wb") as f:
    pickle.dump(K, f, protocol=pickle.HIGHEST_PROTOCOL)

    
# all_outputs[0][1][0]['shap_values_rep']

"""
# opt_params = {'objective': 'binary:logistic', 'eta': 0.31, 'max_depth': 4, 'tree_method': 'gpu_hist', 'gamma': 0.9099999999999999, 'min_child_weight': 16, 'max_delta_step': 11, 'lambda': 6, 'eval_metric': 'auc'}

num_round = 500

params_d = {
            "default":{
                'objective': 'reg:squarederror',
                "eta": 0.3,
                "max_depth": 2,
                "tree_method": "gpu_hist",
                "gamma": 1,
                "min_child_weight": 5,
                "max_delta_step": 4,
                "lambda": 0.1,
                "eval_metric": "auc",
                # "num_feature": 10,
                # "booster": "gblinear"
            },
            'block1':{
            'objective': ['reg:squarederror'],
            "eta": np.arange(0.01, 1, 0.3),
            "max_depth": range(1, 10, 3),
            "tree_method": ["gpu_hist"],
            "gamma": np.arange(0.01, 1, 0.3),
            "min_child_weight": range(1, 20, 5),
            "eval_metric": ["auc"],
            "max_delta_step": range(1, 20, 5),
            "lambda": range(1, 20, 5),
            },
}
params = list(parameters_generator(params_d))
param = params[0]
print (len(params))

def get_shap_data_regression(param, X_train, X_test, X_rep,  y_train, y_test, y_rep, id_train, id_test, id_rep, feature_names, class_labels):
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
        drep = xgb.DMatrix(X_rep, label=y_rep, feature_names=feature_names)
        param["tree_method"] = "gpu_hist"
        model  = xgb.train(param, dtrain, verbose_eval=False)
        deviceIDs = GPUtil.getAvailable(order = 'memory', limit = 4, maxLoad = 1, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
        if len(deviceIDs) == 0:
            pass
        else:
            model.set_param({"predictor": "gpu_predictor"})
        shap_values_train = {} # model.predict(dtrain, pred_contribs=True)
        shap_values_test = {} # model.predict(dtest, pred_contribs=True) 
        shap_values = {'shap_values_train': shap_values_train, 'shap_values_test': shap_values_test, }
        shap_values_rep = {} # model.predict(drep, pred_contribs=True)  
        shap_values['shap_values_rep'] = shap_values_rep
        other_info = {}
        other_info['ID_train'] = id_train
        other_info['ID_test'] = id_test
        other_info['ID_rep'] = id_rep 
        other_info['y_pred_train'] = model.predict(dtrain)
        other_info['y_pred_test'] = model.predict(dtest)
        other_info['y_pred_rep'] = model.predict(drep) 
        if len(deviceIDs) == 0:
            pass
        else:
            model.set_param({"predictor": "cpu_predictor"})
        return model, (shap_values, feature_names, other_info, class_labels)
    
    
def new_cube(e):
    # print (e, params[e])
    # model_temp = xgb.train(param, dtrain_train, num_round, feval=calculate_roc_auc_eval_xgb_binary, evals=eval_set, verbose_eval=False, early_stopping_rounds=20)
    # y_rep_predictions = []
    X_train = X_rep
    y_train = ensemble_prediction['rep_y_probas1']
    id_train = id_rep
    X_test = X_rep
    y_test = ensemble_prediction['rep_y_probas1']
    id_test = id_rep
    output = get_shap_data_regression(copy.deepcopy(params[e]), X_train, X_test, X_rep,  y_train, y_test, y_rep, id_train, id_test, id_rep, feature_names, class_labels)
    y_rep_predictions = output[1][2]['y_pred_rep']
    # print ('Train:', round(metrics.roc_auc_score(y_train, output[1][2]['y_pred_train'][:, 1]), 2) )
    # print ('Test:', round(metrics.roc_auc_score(y_test, output[1][2]['y_pred_test'][:, 1]), 2) )
    corr, _ = pearsonr(y_rep_predictions, ensemble_prediction['rep_y_probas1'])
    print(e, params[e], 'Pearsons correlation: %.3f' % corr)
    return round(corr, 3)

# from tqdm.contrib.concurrent import process_map
# result_list = process_map(new_cube, list(range(len(params))), max_workers=8)
# print (result_list)

"""


"""
# corr, _ = pearsonr(output[1][2]['y_pred_rep'][:, 1], ensemble_prediction['rep_y_probas1'])
# print('Pearsons correlation: %.3f' % corr)

# X_rep = replication_data[:, 1:]
# y_rep = replication_data[:, 0]
# id_rep = rep_split_df['ID']







# ad_imaging = ad.read("/data/dadua2/projects_data/project_MLPhenotypesMRIGWAS/ml_generated_data/23_03_30/predictiveModel/ad_imaging.h5ad")
#  replication_ad_imaging = ad.read("/data/dadua2/projects_data/project_MLPhenotypesMRIGWAS/ml_generated_data/23_03_30/predictiveModel/replication_ad_imaging.h5ad")




import pandas as pd
import pickle
import numpy as np
import os
from MachineLearningStreamlitBase.train_model import generateFullData

# Read Data File both pre-processed and raw data
original_data = pd.read_csv("data/scriptToWrangleJessicaDataFreeze5/ALSregistry.AdrianoChio.wrangled.nodates.freeze5.csv")
replication_data = pd.read_csv("data/scriptToWrangleJessicaDataFreeze5/ALSregistry.JessicaMandrioli.wrangled.nodates.freeze5.csv")

# select top columns
# Select Feature and Label Column
with open("feature_list.txt", 'r') as f:
    selected_cols_x_top = f.read().strip().split('\n')

selected_cols_y = 'clinicaltype_at_oneyear'

# mapping class to index
Z_map = {
    "bulbar":0,
    "classical":1,
    "flailArm":2,
    "flailLeg":3,
    "pyramidal":4,
    "respiratory":5
}

obj = generateFullData()

# print unique value counts for each selected feature
for col in original_data[selected_cols_x_top].columns:
    print ('*'*50, col)
    print (len(original_data[col].value_counts()))

original_encoded_data = original_data[['number', selected_cols_y]].copy().rename(columns={'number': 'ID'})
replication_encoded_data = replication_data[['number', selected_cols_y]].copy().rename(columns={'number': 'ID'})
categorical_variable = [
    'smoker', 'elEscorialAtDx', 'anatomicalLevel_at_onset', 'site_of_onset',
    'onset_side', 'ALSFRS1'
]
numerical_variable = [
    "FVCPercentAtDx", "weightAtDx_kg", "rateOfDeclineBMI_per_month", "age_at_onset", "firstALSFRS_daysIntoIllness"
]
col_dict_map = {}
for col in categorical_variable:
    distinct_vals = original_data[col].dropna().unique()
    dict_map = {i:e for e, i in enumerate(list(distinct_vals))}
    col_dict_map[col] = dict_map
    mode = original_data[col].dropna().index[0] 
    for val in distinct_vals:
        # temp = original_data[col].fillna(mode)
        # original_encoded_data['{}_{}'.format(col, val)] = pd.Series(temp == val).astype(int)
        original_encoded_data[col] = original_data[col].map(lambda x: dict_map.get(x, np.nan))
        # temp = replication_data[col].fillna(mode)
        # replication_encoded_data['{}_{}'.format(col, val)] = pd.Series(temp == val).astype(int) 
        replication_encoded_data[col] = replication_data[col].map(lambda x: dict_map.get(x, np.nan))


for col in numerical_variable:
    original_encoded_data[col] = list(original_data[col])
    replication_encoded_data[col] = list(replication_data[col])

original_encoded_data = original_encoded_data[original_encoded_data[selected_cols_y].notna()]
replication_encoded_data = replication_encoded_data[replication_encoded_data[selected_cols_y].notna()]
original_encoded_data[selected_cols_y] = original_encoded_data[selected_cols_y].map(lambda x: Z_map[x])
replication_encoded_data[selected_cols_y] = replication_encoded_data[selected_cols_y].map(lambda x: Z_map[x])
selected_raw_columns = [col for col in original_encoded_data.columns if not col in [selected_cols_y, 'ID'] ] 
# _ = obj.trainXGBModel_multiclass(data=original_encoded_data, feature_names=selected_raw_columns, label_name=selected_cols_y, replication_set=replication_encoded_data)
# _ = obj.trainLightGBMModel_multiclass(data=original_encoded_data, feature_names=selected_raw_columns, label_name=selected_cols_y, replication_set=replication_encoded_data)


os.makedirs('saved_models', exist_ok=True)
for class_name, ind in Z_map.items():
    print ('*'*30, class_name, '*'*30)
    original_encoded_data_temp = original_encoded_data.copy()
    replication_encoded_data_temp = replication_encoded_data.copy()
    original_encoded_data_temp[selected_cols_y] = original_encoded_data_temp[selected_cols_y].map (lambda x: 1 if x==ind else 0)
    replication_encoded_data_temp[selected_cols_y] = replication_encoded_data_temp[selected_cols_y].map (lambda x: 1 if x==ind else 0)
    # data_pol = pd.concat([data, data_rep], axis=0)
    model, train = obj.trainXGBModel_binaryclass(data=original_encoded_data_temp, feature_names=selected_raw_columns, label_name=selected_cols_y, replication_set=replication_encoded_data_temp)
    with open('saved_models/trainXGB_gpu_{}.data'.format(class_name), 'wb') as f:
        pickle.dump(train, f)
    import joblib
    joblib.dump( model, 'saved_models/trainXGB_gpu_{}.model'.format(class_name) )
    # with open('saved_models/trainXGB_gpu_{}.model'.format(class_name), 'wb') as f:
    #     pickle.dump(model, f)

result_aucs = {}
for class_name in Z_map:
    with open('saved_models/trainXGB_gpu_{}.data'.format(class_name), 'rb') as f:
        temp = pickle.load(f)
    result_aucs[class_name] = (temp[3]['AUC_train'], temp[3]['AUC_test'], temp[3]['AUC_rep'] )
    print (class_name, result_aucs[class_name])

with open('saved_models/trainXGB_gpu.aucs', 'wb') as f:
    pickle.dump(result_aucs, f)

with open('saved_models/trainXGB_categorical_map.pkl', 'wb') as f:
    pickle.dump(col_dict_map, f)

with open('saved_models/trainXGB_class_map.pkl', 'wb') as f:
    pickle.dump(Z_map, f)
    
"""

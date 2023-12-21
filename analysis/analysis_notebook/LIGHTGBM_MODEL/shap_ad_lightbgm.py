
import pickle
import joblib
from pathlib import Path
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from sklearn import metrics
import pandas as pd
import numpy as np
import copy
from itertools import product
import anndata as ad
import shap
from lightgbm import LGBMRegressor

directory_path = Path("ml_generated_data/23_03_30/training/generated_data/example/prediction/apr5/IMAGING_ADPREDICTION_CLASSIFIER")
with open(directory_path / "imaging__addiagnosisprediction_train100_numpy_array.pkl", "rb") as f:
    data = pickle.load(f)

with open(directory_path / "imaging__addiagnosisprediction_train100_replication_numpy_array.pkl", "rb") as f:
    replication_data = pickle.load(f)

split_df = pd.read_csv(directory_path / "numerical$imaging__addiagnosisprediction_train100_split_index.csv", header=None)
split_df.columns = ['ID', 'fold']

rep_split_df = pd.read_csv(directory_path / "imaging__addiagnosisprediction_train100_replication_id_multi_index.csv", header=None)
rep_split_df.columns = ['sample', 'ID']

ad_imaging = ad.read("ml_generated_data/23_03_30/predictiveModel/ad_imaging.h5ad")
feature_names = ad_imaging.var.index
class_labels = {0: 'Control', 1: 'Dementia'}
replication_ad_imaging = ad.read("ml_generated_data/23_03_30/predictiveModel/replication_ad_imaging.h5ad")
X_rep = np.array(replication_ad_imaging.X)
y_rep = [-1] * len(X_rep)
id_rep = replication_ad_imaging.obs.index
ensemble_prediction = pd.read_csv("ml_generated_data/23_03_30/training/results_data/predictiveAnalysis/classification_detailed_metrics/phenotype_data/replication_ensemble_ad_imaging_probability_scores.csv")
y = ensemble_prediction['rep_y_probas1'].map(lambda x: np.log2(x / (1 - x))).values
# Create and fit model
model = LGBMRegressor()
model.fit(X_rep, y)
y_pred = model.predict(X_rep)

# pd.Series(y).map(lambda x: (2 ** x) / (1 + 2 ** x))
# 0.900718543752591
from sklearn.metrics import r2_score
print ("R squared_error: ", r2_score(y, y_pred))

# Create Explainer and get shap_values
explainer = shap.Explainer(model, X_rep)
my_shap_values = explainer(X_rep, check_additivity=False)

vals = np.abs(my_shap_values.values).mean(0)
top20_features_index = np.argsort(vals)[::-1][:20]
top20_features = feature_names[top20_features_index]

reduced_model = LGBMRegressor()
reduced_model.fit(X_rep[:, top20_features_index], y)
y_reduced_pred = reduced_model.predict(X_rep[:, top20_features_index])
print ("reduced R squared_error: ", r2_score(y, y_reduced_pred))

# print (feature_names[np.argsort(tops)][:10])
# print (X.columns[np.argsort(np.abs(shap_values).mean(0))])
# print (feature_names[np.argsort(np.abs(my_shap_values).mean(0))])


output = {
    "shap_values": my_shap_values,
    "lightgbm_predictions": y_pred,
    "lightgbm_reduced_predictions": y_reduced_pred,
    "true_probabilities": y,
    "top20_features": top20_features,
}

with open("lightgbm_ad_shap.pkl", "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

joblib.dump(model, 'ad_full_lgb.pkl')
joblib.dump(reduced_model, 'ad_reduced_lgb.pkl')


gbm_pickle1 = joblib.load('ad_full_lgb.pkl')
gbm_pickle2 = joblib.load('ad_reduced_lgb.pkl')

import pickle
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
class_labels = {0: 'Control', 1: 'PD'}
replication_ad_imaging = ad.read("ml_generated_data/23_03_30/predictiveModel/replication_pd_imaging.h5ad")
X_rep = np.array(replication_ad_imaging.X)
y_rep = [-1] * len(X_rep)
id_rep = replication_ad_imaging.obs.index
ensemble_prediction = pd.read_csv("ml_generated_data/23_03_30/training/results_data/predictiveAnalysis/classification_detailed_metrics/phenotype_data/replication_ensemble_pd_imaging_probability_scores.csv")

y = ensemble_prediction['rep_y_probas1'].values
# Create and fit model
model = LGBMRegressor()
model.fit(X_rep, y)
y_pred = model.predict(X_rep)

# 0.7959702538257827
from sklearn.metrics import r2_score
print ("R squared_error: ", r2_score(ensemble_prediction['rep_y_probas1'].values, y_pred))

# Create Explainer and get shap_values
explainer = shap.Explainer(model, X_rep)
my_shap_values = explainer(X_rep, check_additivity=False)

output = {
    "shap_values": my_shap_values,
    "lightgbm_predictions": y_pred,
    "true_probabilities": ensemble_prediction['rep_y_probas1'].values,
}

with open("lightgbm_pd_shap.pkl", "wb") as f:
    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

import joblib
joblib.dump(model, 'pd_lgb.pkl')
gbm_pickle = joblib.load('pd_lgb.pkl')
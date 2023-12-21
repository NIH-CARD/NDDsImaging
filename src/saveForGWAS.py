import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from omegaconf import OmegaConf

config = OmegaConf.load('configs/config.yaml')
organized_generated_dir = Path(config['organized_project_path'])
data = pd.read_parquet(organized_generated_dir / 'combinedData.parquet.gzip')
baseline_date = {}
l = list(data.groupby('eid'))
for i in range(len(l)):
    baseline_date[l[i][0]] = l[i][1]['Date_of_attending_assessment_centre'].min()

data['days_from_first_visit'] = data['Date_of_attending_assessment_centre'] - data['eid_nih'].map(baseline_date)
data['Age_at_image_taken'] = data['Age.when.attended.assessment.centre_nih'].values

required_cols = [
    'eid_nih', 'Age_at_image_taken', 'Sex', 'townsend', 'Age_at_image_taken',
    'days_from_first_visit', 'norm_AD_imaging_score', 'norm_PD_imaging_score',
    'AD_imaging_score', 'PD_imaging_score',
]

os.makedirs("../results_data/files", exist_ok=True)
data[required_cols].to_csv("../results_data/files/ukbb_imaging_scores.tsv", index=False, sep="\t")

# plot histogram
# import pandas as pd
# data = pd.read_csv("ukbb_imaging_scores.tsv", sep="\t")
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.histplot(data=data['norm_AD_imaging_score'])
# plt.show()
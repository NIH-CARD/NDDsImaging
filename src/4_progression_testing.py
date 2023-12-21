import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from omegaconf import OmegaConf

import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(context='notebook', style='ticks', font_scale=1.2, font='sans-serif', rc={"lines.linewidth": 1.2})
sns.set_style("white")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


config = OmegaConf.load('configs/config.yaml')
evaluation_generated_dir = Path(config['evaluation_generated_dir']) / "progression_testing"
organized_generated_dir = Path(config['organized_project_path'])

data_all = pd.read_parquet(organized_generated_dir / 'combinedData.parquet.gzip')
data = data_all[data_all['ukbb_id_nih'].str.contains('20252_2_0')].copy()

# data = pd.read_parquet(organized_generated_dir / 'combinedData.parquet.gzip')

list_variable_of_interest = ["norm_AD_imaging_score", "norm_PD_imaging_score"]
list_outcome = ['norm_AD_imaging_score', 'norm_PD_imaging_score']
list_formula = [
    f"Age_at_image_taken + townsend + Sex + var:AD_prs",
    f"Age_at_image_taken + townsend + Sex + var:PD_prs",
    f"Age_at_image_taken + townsend + Sex + var:AD_prs + var:PD_prs",
]

list_disease = ['PD', 'ALL_DEMENTIA', 'PARKINSONISM', 'AD', "MS", "STROKE", "OtMOVEMENT", "DYSTONIA"]
for disease in list_disease:
    duration_col = f"daysFromDiagnosis_image_taken_{disease}"
    data[duration_col] = data[f'Date_of_{disease}'] - data['Date_of_attending_assessment_centre']
    data['Age_at_image_taken'] = data['Age.when.attended.assessment.centre_nih'].values
    group = []
    for i in range(len(data)):
        x = data.iloc[i][duration_col]
        if pd.isnull(x):
            group.append("Control")
        elif x >= 0:
            group.append("Dx_after_baseline")
        else:
            group.append("Dx_before_baseline")
    data[f'{disease}_GroupColumn'] = group

list_disease_to_plot = ['PD', 'ALL_DEMENTIA', 'AD']
for temp_formula in list_formula:
    for outcome in list_outcome:
        list_variable_of_interest = [var.split(':')[1] for var in temp_formula.split(' + ') if 'var' in var]
        formula = temp_formula.replace('var:', '')
        fname = f"{outcome}&{formula.replace(' + ', '@')}"
        os.makedirs(evaluation_generated_dir / fname, exist_ok=True)
        input_data = data.copy()
        input_data['Age_at_image_taken'] = input_data['Age.when.attended.assessment.centre_nih'].values
        input_data = input_data[formula.split(' + ') + [outcome, ]].dropna().copy()
        model = smf.ols(formula=f'{outcome} ~ {formula}', data=input_data).fit()
        temp = None
        for index in model.params.index:
            if temp is None:
                temp = model.params.loc[index] * input_data[index] if not 'Intercept' in index else model.params.loc[index]
            else:
                temp += model.params.loc[index] * input_data[index] if not 'Intercept' in index else model.params.loc[index]
        input_data[f"adjusted@{outcome}@{formula.replace(' + ', '_')}"] = input_data[outcome] - temp
        for disease in list_disease:
            input_data.loc[:, f'{disease}_GroupColumn'] = data.loc[input_data.index, f'{disease}_GroupColumn']

        input_data.reset_index().to_csv(evaluation_generated_dir / fname / "AdjustedScores.csv", index=False)

        fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(28, 8), gridspec_kw={'hspace': 0.4, 'wspace': 0.1, })
        e = 0
        axs = axs.reshape(-1)
        for disease in list_disease_to_plot:
            for y in [outcome, f"adjusted@{outcome}@{formula.replace(' + ', '_')}"]:
                sns.pointplot(
                    data=input_data, y=y, x=f'{disease}_GroupColumn',
                    order=['Control', 'Dx_before_baseline', 'Dx_after_baseline'],
                    capsize=.4, join=False, color="black", ax=axs[e], errorbar='ci'  # errorbar=("pi", 100),
                )
                axs[e].set_xlabel(disease)
                axs[e].set_ylabel("")
                axs[e].grid()
                axs[e].set_title(outcome if not 'adjusted' in y else f'adjusted@{outcome}')
                e += 1
        # fig.tight_layout()
        fig.savefig(evaluation_generated_dir / fname / "AdjustedScores_DiseaseGroups.png")
        fig.savefig(evaluation_generated_dir / fname / "AdjustedScores_DiseaseGroups.pdf")
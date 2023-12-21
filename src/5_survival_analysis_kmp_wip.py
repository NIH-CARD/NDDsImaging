import os
from pathlib import Path
import pandas as pd
from lifelines import CoxPHFitter
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.datasets import load_waltons
from contextlib import redirect_stdout

from collections import defaultdict

config = OmegaConf.load('configs/config.yaml')
evaluation_generated_dir = Path(config['evaluation_generated_dir']) / "survival_analysis_kmp_wip"
organized_generated_dir = Path(config['organized_project_path'])
os.makedirs(evaluation_generated_dir, exist_ok=True)

data_all = pd.read_parquet(organized_generated_dir / 'combinedData.parquet.gzip')
data = data_all[data_all['ukbb_id_nih'].str.contains('20252_2_0')].copy()

disease = "PD"
formula = f"Age_at_image_taken + townsend + Sex + PD_prs + norm_PD_imaging_score"
variable_of_interest = "norm_PD_imaging_score"
censoring_time = 3650

list_variable_of_interest= ["norm_AD_imaging_score", "norm_PD_imaging_score"]
list_disease = ['PD', 'ALL_DEMENTIA', 'PARKINSONISM', 'AD'] # "MS", "STROKE", "OtMOVEMENT", "DYSTONIA"]
list_formula = [
    f"Age_at_image_taken + townsend + Sex + AD_prs + var:norm_AD_imaging_score",
    f"Age_at_image_taken + townsend + Sex + PD_prs + var:norm_PD_imaging_score",
    f"Age_at_image_taken + townsend + Sex + AD_prs + PD_prs + var:norm_AD_imaging_score + var:norm_PD_imaging_score"
]
for temp_formula in list_formula:
    for disease in list_disease:
        list_variable_of_interest = [var.split(':')[1] for var in temp_formula.split(' + ') if 'var' in var]
        formula = temp_formula.replace('var:', '')
        if True:
            # variable_of_interest = formula.split(' + ')[-1]
            fname = f"{disease}&{formula.replace(' + ', '@')}"
            os.makedirs(evaluation_generated_dir / fname, exist_ok=True)

            duration_col = f"daysFromDiagnosis_image_taken_{disease}"
            data[duration_col] = data[f'Date_of_{disease}'] - data['Date_of_attending_assessment_centre']
            data['Age_at_image_taken'] = data['Age.when.attended.assessment.centre_nih'].values
            input_data = data.copy()
            input_data[duration_col] = input_data[duration_col].map(lambda x: x if not pd.isnull(x) else censoring_time)
            input_data = input_data[input_data[duration_col] >= 0].copy()
            input_data[f"Status_{duration_col}"] = input_data[duration_col].map(lambda x: 1 if not x == censoring_time else 0)
            input_data_columns = formula.split(' + ') + [ duration_col, f"Status_{duration_col}" ]
            input_data_survival = input_data[input_data_columns].copy().dropna(subset=formula.split(' + '))

            input_data_survival_class1 = input_data_survival[input_data_survival[f"Status_{duration_col}"] == 1].copy()
            input_data_survival_class0 = input_data_survival[input_data_survival[f"Status_{duration_col}"] == 0].sample(n=len(input_data_survival_class1)).copy()
            input_data_survival_balanced = pd.concat([input_data_survival_class0, input_data_survival_class1], axis=0)

        for variable_of_interest in list_variable_of_interest:
            v = pd.qcut(input_data_survival[variable_of_interest], 10)

            input_data_survival[f"quantiles_{variable_of_interest}"] = v.cat.codes.map(lambda x: f'Q{x}')

            # fig, ax = plt.subplots()
            # cph.plot(ax=ax)
            # plt.tight_layout()
            # fig.savefig(evaluation_generated_dir / fname / f"EffectSize_{variable_of_interest}.png")
            # fig.savefig(evaluation_generated_dir / fname / f"EffectSize_{variable_of_interest}.pdf")

            # fig, ax = plt.subplots()
            # list_of_values = sorted(v.map(lambda x: x.right).astype(float).unique())
            # cph.plot_partial_effects_on_outcome(covariates=variable_of_interest, values=list_of_values, cmap='coolwarm', ax=ax)
            # plt.tight_layout()
            # fig.savefig(evaluation_generated_dir / fname / f"PartialEffects_Quantiles_{variable_of_interest}.png")
            # fig.savefig(evaluation_generated_dir / fname / f"PartialEffects_Quantiles_{variable_of_interest}.pdf")

            fig, ax = plt.subplots(figsize=(8, 16))
            kmfs = []
            for ijk in sorted(input_data_survival[f"quantiles_{variable_of_interest}"].unique()):
                ix1 = input_data_survival[input_data_survival[f'quantiles_{variable_of_interest}'] == ijk]
                kmf1 = KaplanMeierFitter()
                ax = kmf1.fit(ix1[f"{duration_col}"], ix1[f"Status_{duration_col}"], label=ijk).plot_survival_function(ax=ax, ci_show=False)
                kmfs.append(kmf1)

            add_at_risk_counts(*kmfs, ax=ax)
            plt.tight_layout()
            fig.savefig(evaluation_generated_dir / fname / f"KaplanMeier_Quantiles_{variable_of_interest}.png")
            fig.savefig(evaluation_generated_dir / fname / f"KaplanMeier_Quantiles_{variable_of_interest}.pdf")
            import sys; sys.exit()

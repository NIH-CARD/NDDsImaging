import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from omegaconf import OmegaConf

config = OmegaConf.load('configs/config.yaml')
evaluation_generated_dir = Path(config['evaluation_generated_dir']) / "logistic_regression"
organized_generated_dir = Path(config['organized_project_path'])
os.makedirs(evaluation_generated_dir, exist_ok=True)

data_all = pd.read_parquet(organized_generated_dir / 'combinedData.parquet.gzip')
data = data_all[data_all['ukbb_id_nih'].str.contains('20252_2_0')].copy()

disease = "PD"
formula = f"Age_at_image_taken + townsend + Sex + PD_prs + norm_PD_imaging_score"
variable_of_interest = "norm_PD_imaging_score"
censoring_time = -3650
list_variable_of_interest = ["norm_AD_imaging_score", "norm_PD_imaging_score"]
list_disease = ['ALL_DEMENTIA', 'PD', 'PARKINSONISM', 'AD']
list_formula = [
    f"Age_at_image_taken + townsend + Sex + AD_prs + var:norm_AD_imaging_score",
    f"Age_at_image_taken + townsend + Sex + PD_prs + var:norm_PD_imaging_score",
    f"Age_at_image_taken + townsend + Sex + AD_prs + PD_prs + var:norm_AD_imaging_score + var:norm_PD_imaging_score",
]
for temp_formula in list_formula:
    for disease in list_disease:
        list_variable_of_interest = [var.split(':')[1] for var in temp_formula.split(' + ') if 'var' in var]
        formula = temp_formula.replace('var:', '')
        if True:
            fname = f"{disease}&{formula.replace(' + ', '@')}"
            os.makedirs(evaluation_generated_dir / fname, exist_ok=True)
            duration_col = f"daysFromDiagnosis_image_taken_{disease}"
            data[duration_col] = data[f'Date_of_{disease}'] - data['Date_of_attending_assessment_centre']
            data['Age_at_image_taken'] = data['Age.when.attended.assessment.centre_nih'].values
            input_data = data.copy()
            input_data[duration_col] = input_data[duration_col].map(lambda x: x if not pd.isnull(x) else censoring_time)
            temp_input_data = input_data[(input_data[duration_col] < 0) | (input_data[duration_col] == censoring_time)].copy()
            temp_input_data[f"Status_{duration_col}"] = temp_input_data[duration_col].map(lambda x: 1 if not x == censoring_time else 0)

            input_data_columns = formula.split(' + ') + [duration_col, f"Status_{duration_col}"]
            input_data_survival = temp_input_data[input_data_columns].copy().dropna(subset=formula.split(' + '))

            log_reg = smf.logit(f"Status_{duration_col} ~ {formula}", data=input_data_survival).fit()

            with open(evaluation_generated_dir / fname / 'Print_Summary.out', 'w') as f:
                with redirect_stdout(f):
                    print(log_reg.summary())

            odds_ratios = pd.DataFrame(
                {
                    "OR": log_reg.params,
                    "OR_Lower CI": log_reg.conf_int()[0],
                    "OR_Upper CI": log_reg.conf_int()[1],
                }, index=log_reg.params.index
            )
            odds_ratios = np.exp(odds_ratios)
            odds_ratios['coef_pvalues'] = log_reg.pvalues
            odds_ratios['coef'] = log_reg.params
            odds_ratios['coef_Lower CI'] = log_reg.conf_int()[0]
            odds_ratios['coef_Upper CI'] = log_reg.conf_int()[1]
            odds_ratios['coef_-log10(p)'] = -round(np.log10(log_reg.pvalues), 2)
            odds_ratios.reset_index().to_csv(evaluation_generated_dir / fname / "Summary.csv", index=False)
            imbalance_count = input_data_survival[f"Status_{duration_col}"].value_counts()
            imbalance_count.reset_index().to_csv(evaluation_generated_dir / fname / "Case_Control_Ratio.csv", index=False)
        for variable_of_interest in list_variable_of_interest:
            details_about_model = {}
            # details_about_model['duration_col'] = duration_col
            details_about_model['outcome'] = f"Status_{duration_col}"
            details_about_model['formula'] = formula
            details_about_model['variable_of_interest'] = variable_of_interest
            details_about_model['prsquared'] = log_reg.prsquared
            details_about_model['log_likelihood'] = log_reg.llr
            details_about_model['log_likelihood_p_value'] = log_reg.llr_pvalue
            details_about_model['class0_count'] = imbalance_count.loc[0]
            details_about_model['class1_count'] = imbalance_count.loc[1]
            # imbalance_count.loc[]
            for ind in odds_ratios.loc[variable_of_interest].index:
                details_about_model[ind] = odds_ratios.loc[variable_of_interest][ind]

            for key, val in details_about_model.items():
                details_about_model[key] = [val, ]

            if os.path.exists(evaluation_generated_dir / "results_summary.csv"):
                prev_results = pd.read_csv(evaluation_generated_dir / "results_summary.csv")
                prev_results = pd.concat([prev_results, pd.DataFrame(details_about_model)], axis=0)
                prev_results = prev_results.drop_duplicates(subset=['outcome', 'formula', 'variable_of_interest'])
                prev_results.to_csv(evaluation_generated_dir / "results_summary.csv", index=False)
            else:
                pd.DataFrame(details_about_model).to_csv(evaluation_generated_dir / "results_summary.csv", index=False)

            # print(odds_ratios)
            # import sys; sys.exit()

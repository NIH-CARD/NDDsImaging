import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from omegaconf import OmegaConf

config = OmegaConf.load('configs/config.yaml')
evaluation_generated_dir = Path(config['evaluation_generated_dir']) / "association_testing"
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

for temp_formula in list_formula:
    for outcome in list_outcome:
        list_variable_of_interest = [var.split(':')[1] for var in temp_formula.split(' + ') if 'var' in var]
        formula = temp_formula.replace('var:', '')
        fname = f"{outcome}&{formula.replace(' + ', '@')}"
        os.makedirs(evaluation_generated_dir / fname, exist_ok=True)
        input_data = data.copy()
        input_data['Age_at_image_taken'] = input_data['Age.when.attended.assessment.centre_nih'].values
        input_data = input_data[formula.split(' + ') + [outcome,]].dropna().copy()
        model = smf.ols(formula=f'{outcome} ~ {formula}', data=input_data).fit()
        with open(evaluation_generated_dir / fname / 'Print_Summary.out', 'w') as f:
            with redirect_stdout(f):
                print(model.summary())

        odds_ratios = pd.DataFrame(index=model.params.index )
        odds_ratios['coef_pvalues'] = model.pvalues
        odds_ratios['coef'] = model.params
        odds_ratios['coef_Lower CI'] = model.conf_int()[0]
        odds_ratios['coef_Upper CI'] = model.conf_int()[1]
        odds_ratios['coef_-log10(p)'] = -round(np.log10(model.pvalues + 1e-100), 2)
        odds_ratios.reset_index().to_csv(evaluation_generated_dir / fname / "Summary.csv", index=False)
        for variable_of_interest in list_variable_of_interest:
            details_about_model = {}
            # details_about_model['duration_col'] = duration_col
            details_about_model['outcome'] = outcome
            details_about_model['formula'] = formula
            details_about_model['variable_of_interest'] = variable_of_interest
            details_about_model['rsquared'] = model.rsquared
            # details_about_model['log_likelihood'] = model.llr
            # details_about_model['log_likelihood_p_value'] = model.llr_pvalue
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



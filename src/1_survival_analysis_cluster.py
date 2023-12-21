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
evaluation_generated_dir = Path(config['evaluation_generated_dir']) / "survival_analysis_cluster"
organized_generated_dir = Path(config['organized_project_path'])
os.makedirs(evaluation_generated_dir, exist_ok=True)

data_all = pd.read_parquet(organized_generated_dir / 'combinedData.parquet.gzip')
data_all['id_invicrot1_vol_temporal_rdktlobes'] = 1 - data_all['id_invicrot1_vol_temporal_rdktlobes']
data_all['id_invicrot1_vol_gmtissues'] = 1 - data_all['id_invicrot1_vol_gmtissues']
data_all['id_invicrot1_vol_left_entorhinaldktcortex'] = 1 - data_all['id_invicrot1_vol_left_entorhinaldktcortex']

subject_max = data_all.set_index('eid')[['Age_i0', 'Age_i1', 'Age_i2', 'Age_i3']].agg(max, axis=1) # - data.set_index('eid')[['Age_i0', 'Age_i1', 'Age_i2', 'Age_i3']].agg(min, axis=1)
subject_max = subject_max.reset_index().groupby('eid').agg('max')[0]
study_time = dict(zip(list(subject_max.index), list(subject_max.values)))

data = data_all[data_all['ukbb_id_nih'].str.contains('20252_2_0')].copy().set_index('eid')

disease = "PD"
formula = f"Age_at_image_taken + townsend + Sex + PD_prs + norm_PD_imaging_score"
variable_of_interest = "norm_PD_imaging_score"
censoring_time = 3650

list_disease = ['PD', 'ALL_DEMENTIA', 'PARKINSONISM', 'AD', "MS", "STROKE", "OtMOVEMENT", "DYSTONIA"]
list_disease = ['PD', 'ALL_DEMENTIA', 'PARKINSONISM']
list_formula = [
    f"Age_at_image_taken + townsend + Sex + AD_prs + var:id_invicrot1_vol_temporal_rdktlobes",
    f"Age_at_image_taken + townsend + Sex + AD_prs + var:id_invicrot1_vol_gmtissues",
    f"Age_at_image_taken + townsend + Sex + AD_prs + var:id_invicrot1_vol_left_entorhinaldktcortex",
    # f"Age_at_image_taken + townsend + Sex + AD_prs + PD_prs + var:norm_AD_imaging_score + var:norm_PD_imaging_score"
    f"Age_at_image_taken + townsend + Sex + PD_prs + var:norm_PD_imaging_score",
    f"Age_at_image_taken + townsend + Sex + AD_prs + var:norm_AD_imaging_score",
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


            temp_input_data = data.copy()
            event_subjects = temp_input_data[temp_input_data[duration_col] > 0].index
            event_subjects_data = temp_input_data.loc[event_subjects].copy()
            event_subjects_data[f"Status_{duration_col}"] = 1

            right_censored_subjects = temp_input_data[temp_input_data[duration_col].isnull()].index
            right_censored_subjects_data = temp_input_data.loc[right_censored_subjects].copy()
            right_censored_subjects_data[duration_col] = 365 * (right_censored_subjects_data.index.map(lambda x: study_time[x]) - right_censored_subjects_data['Age_at_image_taken'])
            right_censored_subjects_data = right_censored_subjects_data[right_censored_subjects_data[duration_col] > 0]
            right_censored_subjects_data[f"Status_{duration_col}"] = 0

            input_data = pd.concat([event_subjects_data, right_censored_subjects_data], axis=0)

            input_data_columns = formula.split(' + ') + [ duration_col, f"Status_{duration_col}" ]
            input_data_survival = input_data[input_data_columns].copy().dropna(subset=formula.split(' + '))

            input_data_survival_class1 = input_data_survival[input_data_survival[f"Status_{duration_col}"] == 1].copy()
            input_data_survival_class0 = input_data_survival[input_data_survival[f"Status_{duration_col}"] == 0].sample(n=len(input_data_survival_class1)).copy()
            input_data_survival_balanced = pd.concat([input_data_survival_class0, input_data_survival_class1], axis=0)
            covariate_columns = []
            for variable_of_interest in list_variable_of_interest:
                formula = formula.replace(variable_of_interest, "")
                v = pd.qcut(input_data_survival[variable_of_interest], 4)
                input_data_survival[f"quantiles_{variable_of_interest}"] = v.cat.codes.map(lambda x: f'Q{x}')
                addition_columns = pd.get_dummies(input_data_survival[f"quantiles_{variable_of_interest}"])
                addition_columns.columns = [f"{variable_of_interest}_{j}" for j in addition_columns.columns]
                formula += ' + '.join(list(addition_columns.columns)[1:])
                input_data_survival = pd.concat([input_data_survival, addition_columns], axis=1)
                covariate_columns.extend(list(addition_columns.columns)[1:])

            cph = CoxPHFitter()
            cph.fit(input_data_survival, duration_col=duration_col, event_col=f"Status_{duration_col}", formula=formula)
            fig, ax = plt.subplots()
            v1 = np.zeros((1, len(covariate_columns)))
            v2 = np.eye(len(covariate_columns))
            ax = cph.plot_partial_effects_on_outcome(covariates=covariate_columns, plot_baseline=False, values=np.concatenate([v1, v2]), ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=[f'Q{i}' for i in range(1+len(covariate_columns))])
            fig.savefig(evaluation_generated_dir / fname / "Survival_plot_Quantiles.png")
            fig.savefig(evaluation_generated_dir / fname / "Survival_plot_Quantiles.pdf")

            fig, ax = plt.subplots()
            cph.plot(columns=sorted(covariate_columns)[::-1], hazard_ratios=True, ax=ax)
            labels = [item.get_text().split('_')[-1] for item in ax.get_yticklabels()]
            ax.set_yticklabels(labels)
            fig.savefig(evaluation_generated_dir / fname / "Effect_size_Quantiles.png")
            fig.savefig(evaluation_generated_dir / fname / "Effect_size_Quantiles.pdf")

            with open(evaluation_generated_dir / fname / 'Print_Summary.out', 'w') as f:
                with redirect_stdout(f):
                    cph.print_summary()
            cph.summary.reset_index().to_csv(evaluation_generated_dir / fname / "Summary.csv", index=False)

        for variable_of_interest in list_variable_of_interest:
            details_about_model = {}
            details_about_model['duration_col'] = duration_col
            details_about_model['event_col'] = f"Status_{duration_col}"
            details_about_model['formula'] = formula
            details_about_model['variable_of_interest'] = variable_of_interest
            details_about_model['concordance_index_'] = cph.concordance_index_
            details_about_model['log_likelihood_ratio_test_p_value'] = cph.log_likelihood_ratio_test().p_value
            details_about_model['log_likelihood_ratio_test_test_statistic'] = cph.log_likelihood_ratio_test().test_statistic
            details_about_model['log_likelihood_'] = cph.log_likelihood_

            for vari in cph.summary.index:
                if not variable_of_interest in vari:
                    continue
                for ind in cph.summary.loc[vari].index:
                    if ind == 'p':
                            details_about_model['-lop10(p)'] = -round(np.log10(cph.summary.loc[vari][ind]), 2)
                    details_about_model[f"{vari}_{ind}"] = cph.summary.loc[vari][ind]

            for key, val in details_about_model.items():
                details_about_model[key] = [val,]

            if os.path.exists(evaluation_generated_dir / "results_summary.csv"):
                prev_results = pd.read_csv(evaluation_generated_dir / "results_summary.csv")
                prev_results = pd.concat([prev_results, pd.DataFrame(details_about_model)], axis=0)
                prev_results = prev_results.drop_duplicates(subset=['duration_col', 'event_col', 'formula', 'variable_of_interest'])
                prev_results.to_csv(evaluation_generated_dir / "results_summary.csv", index=False)
            else:
                pd.DataFrame(details_about_model).to_csv(evaluation_generated_dir / "results_summary.csv", index=False)

            fig, ax = plt.subplots()
            cph.plot(hazard_ratios=True, ax=ax)
            plt.tight_layout()
            fig.savefig(evaluation_generated_dir / fname / f"EffectSize_{variable_of_interest}.png")
            fig.savefig(evaluation_generated_dir / fname / f"EffectSize_{variable_of_interest}.pdf")

            fig, ax = plt.subplots()
            list_of_values = sorted(v.map(lambda x: x.right).astype(float).unique())
            cph.plot_partial_effects_on_outcome(covariates=variable_of_interest, values=list_of_values, ax=ax) # , cmap='coolwarm'
            plt.tight_layout()
            fig.savefig(evaluation_generated_dir / fname / f"PartialEffects_Quantiles_{variable_of_interest}.png")
            fig.savefig(evaluation_generated_dir / fname / f"PartialEffects_Quantiles_{variable_of_interest}.pdf")

            fig, ax = plt.subplots(figsize=(8, 8))
            kmfs = []
            for ijk in sorted(input_data_survival[f"quantiles_{variable_of_interest}"].unique()):
                ix1 = input_data_survival[input_data_survival[f'quantiles_{variable_of_interest}'] == ijk]
                kmf1 = KaplanMeierFitter()
                ax = kmf1.fit(ix1[f"{duration_col}"], ix1[f"Status_{duration_col}"], label=ijk).plot_survival_function(ax=ax)
                kmfs.append(kmf1)
            # ix2 = input_data_survival[input_data_survival[f'quantiles_{variable_of_interest}'] == 'Q1']
            # ix3 = input_data_survival[input_data_survival[f'quantiles_{variable_of_interest}'] == 'Q2']
            # ix4 = input_data_survival[input_data_survival[f'quantiles_{variable_of_interest}'] == 'Q3']
            # kmf2 = KaplanMeierFitter()
            # ax = kmf2.fit(ix2[f"{duration_col}"], ix2[f"Status_{duration_col}"], label='Q1').plot_survival_function(ax=ax)
            # kmf3 = KaplanMeierFitter()
            # ax = kmf3.fit(ix3[f"{duration_col}"], ix3[f"Status_{duration_col}"], label='Q2').plot_survival_function(ax=ax)
            # kmf4 = KaplanMeierFitter()
            # ax = kmf4.fit(ix4[f"{duration_col}"], ix4[f"Status_{duration_col}"], label='Q3').plot_survival_function(ax=ax)
            add_at_risk_counts(*kmfs, ax=ax)
            plt.tight_layout()
            fig.savefig(evaluation_generated_dir / fname / f"KaplanMeier_Quantiles_{variable_of_interest}.png")
            fig.savefig(evaluation_generated_dir / fname / f"KaplanMeier_Quantiles_{variable_of_interest}.pdf")


        # import sys; sys.exit()
        # selected_columns = ["Date_of_attending_assessment_centre", "townsend", "Sex",
        #                    'Age.when.attended.assessment.centre_nih', # 'Age.when.attended.assessment.centre_invicro',
        #                    "Date.of.attending.assessment.centre_invicro", 'Date.of.attending.assessment.centre_nih', 'eid',
        #                    'birthMonthYear', 'Age_from_dates']
        # gap of 1 year due to only month available
        # data['Age_from_dates'] = data['Date_of_attending_assessment_centre'] - data['birthMonthYear']
        # data['Age_from_dates'] = data['Age_from_dates'].map(lambda x: x/365)
        # a = data['Age_from_dates'] == data['Age.when.attended.assessment.centre_nih'].map(int)
        # data.loc[a[~a].index][selected_columns]
        # data['Age_from_dates'] = s // 365

from InputModule.build_pref_model import build_pref_model
from PostProcess.sensitivity_analysis import salib_plot_Si_steady_state
from PostProcess.timeseries_plot import convert_key_to_latex_pref
import numpy as np
from SALib.analyze import sobol


def main():

    p_list_pref, d_list_pref, ex_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, internal_prod_list_pref, \
        phi_list_pref, vars_dict_pref, params_dict_pref, params_bounds_dict_pref, init_cond_pref, \
        init_cond_bounds_pref, param_dependencies_dict_pref = build_pref_model()

    pref_latex_dict = convert_key_to_latex_pref(params_dict_pref, init_cond_pref, vars_dict_pref)
    legend_latex_pref = convert_key_to_latex_pref(vars_dict_pref, vars_dict_pref, vars_dict_pref)

    results_pref_forest = '../../Dropbox/Results/PrefModel/SimData1'
    output_pref_forest_Y = '../../Dropbox/Results/PrefModel/SimData/SensitivityAnalysis'

########################################################################################################################
    # Initialize problem

    init_cond_pref = {'Il_0': 50.0, 'Gl_0': 50.0, 'Sl_0': 50.0, 'Rl_0': 50.0, 'Xl_0': 10.0, 'Zl_0': 10.0, 'Gm_0': 50.0,
                      'Sm_0': 50.0, 'Rm_0': 50.0, 'Xm_0': 10.0, 'Zm_0': 10.0}

    params_dict_pref = {'alpha_X': 0.1,
                        'alpha_Z': 0.1,
                        'gam_aX': 0.1,
                        'gam_aZ': 0.1,
                        'gam_dR': 3.9,
                        'gam_dS': 3.9,
                        'gam_sG': 0.4,
                        'gam_sX': 0.4,
                        'gam_sZ': 0.4,
                        'lam_max': 500.0,
                        'lam_prod': 50,
                        'mu_GX': 10.619469027,
                        'mu_GZ': 10.619469027,
                        'mu_IX': 10.619469027,
                        'mu_RS': 1.0,
                        'mu_RX': 2.5,
                        'mu_RZ': 3.0,
                        'mu_SX': 12.627143363,
                        'mu_SZ': 3.0,
                        'omega': 1000,
                        'D': 1.0071942760083039,
                        'K_GX': 0.5,
                        'K_GZ': 0.26539823,
                        'K_IX': 0.26539823,
                        'K_RS': 0.468416,
                        'K_RX': 0.468416,
                        'K_RZ': 0.468416,
                        'K_SX': 0.468416,
                        'K_SZ': 0.468416,
                        'I_inf': 50.0,
                        'V_l': 0.9,
                        'V_m': 0.1,
                        'Y_IX': 1.0,
                        'Y_GX': 1.0,
                        'Y_GZ': 1.0,
                        'Y_RX': 0.34242,
                        'Y_RZ': 0.34242,
                        'Y_SX': 0.34242,
                        'Y_SZ': 0.34242
                        }

    init_cond_bounds_pref = {'Il_0': [0.01, 50.0], 'Gl_0': [0.01, 50.0], 'Sl_0': [0.01, 10.0], 'Rl_0': [0.0, 10.0],
                             'Xl_0': [0.01, 10.0], 'Zl_0': [0.01, 10.0], 'Gm_0': [0.01, 50.0], 'Sm_0': [0.01, 10.0],
                             'Rm_0': [0.01, 10.0], 'Xm_0': [0.01, 10.0], 'Zm_0': [0.01, 10.0]}

    params_bounds_dict_pref = {'alpha_X': [0.01, 0.5],
                               'alpha_Z': [0.01, 0.5],
                               'gam_aX': [0.05, 0.2],
                               'gam_aZ': [0.05, 0.2],
                               'gam_dR': [2.0, 6.0],
                               'gam_dS': [2.0, 6.0],
                               'gam_sG': [0.05, 0.2],
                               'gam_sX': [0.05, 0.2],
                               'gam_sZ': [0.05, 0.2],
                               'lam_max': [100.0, 600.0],
                               'lam_prod': [40.0, 60.0],
                               'mu_GX': [5.0, 15.0],
                               'mu_GZ': [5.0, 15.0],
                               'mu_IX': [5.0, 15.0],
                               'mu_RS': [0.5, 1.5],
                               'mu_RX': [2.0, 3.0],
                               'mu_RZ': [2.5, 3.5],
                               'mu_SX': [6.0, 18.0],
                               'mu_SZ': [1.5, 4.5],
                               'omega': [10, 10000],
                               'D': [0.75, 1.5],
                               'K_GX': [0.25, 0.75],
                               'K_GZ': [0.15, 0.35],
                               'K_IX': [0.1, 0.4],
                               'K_RS': [0.2, 0.6],
                               'K_RX': [0.2, 0.6],
                               'K_RZ': [0.2, 0.6],
                               'K_SX': [0.2, 0.6],
                               'K_SZ': [0.2, 0.6],
                               'I_inf': [0.01, 60.0],
                               'V_l': [0.85, 0.99],
                               'V_m': [0.01, 0.15],
                               'Y_IX': [0.9, 1.1],
                               'Y_GX': [0.9, 1.1],
                               'Y_GZ': [0.9, 1.1],
                               'Y_RX': [0.2, 0.4],
                               'Y_RZ': [0.2, 0.4],
                               'Y_SX': [0.2, 0.4],
                               'Y_SZ': [0.2, 0.4]}

    # Make samples for Random Forest

    Y = np.loadtxt(results_pref_forest + '/steady_state.txt', delimiter=',', usecols=range(11))

    params_init_dict = {**params_dict_pref, **init_cond_pref}

    # X label as latex
    x_label_latex = {key: pref_latex_dict[key] for key in params_init_dict.keys()}

    bounds = np.concatenate((np.array(list(params_bounds_dict_pref.values())), np.array(list(init_cond_bounds_pref.values()))), axis=0)

    num_vars = len(params_init_dict)
    name_params = list(params_init_dict.keys())

    problem = {
        'num_vars': num_vars,
        'names': name_params,
        'bounds': bounds
    }

    # Run Sobol analysis on steady state

    for key, val in vars_dict_pref.items():
        Si_Y = sobol.analyze(problem, Y[:, val], print_to_console=False)
        total_Y, first_Y, second_Y = Si_Y.to_df()
        file_string_total_Y = output_pref_forest_Y + '/' + 'Si_total_Y_' + key + '.csv'
        file_string_first_Y = output_pref_forest_Y + '/' + 'Si_first_Y_' + key + '.csv'
        file_string_second_Y = output_pref_forest_Y + '/' + 'Si_second_Y_' + key + '.csv'
        total_Y.to_csv(file_string_total_Y)
        first_Y.to_csv(file_string_first_Y)
        second_Y.to_csv(file_string_second_Y)

    # Bar plot of first and total order indices (steady states)
    salib_plot_Si_steady_state(vars_dict_pref, 'total', x_label_latex, legend_latex_pref, output_pref_forest_Y)
    salib_plot_Si_steady_state(vars_dict_pref, 'first', x_label_latex, legend_latex_pref, output_pref_forest_Y)


if __name__ == "__main__":
    main()

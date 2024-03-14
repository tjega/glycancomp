from InputModule.build_pref_model import build_pref_model
from InputModule.create_json_input import open_json_input_file, create_input_file
from SimulationModule.run_ode_sim import gen_samples
from PostProcess.timeseries_plot import convert_key_to_latex_pref
from PostProcess.random_forest import run_rf_sims, rf_read_sample_file
from PostProcess.random_forest import run_rf_model
import numpy as np


def main():
    output_folder_pref_forest = '../../Dropbox/Results/PrefModel/SimData'

    p_list_pref, d_list_pref, ex_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, internal_prod_list_pref, \
        phi_list_pref, vars_dict_pref, params_dict_pref, params_bounds_dict_pref, init_cond_pref, \
        init_cond_bounds_pref, param_dependencies_dict_pref = build_pref_model()

    pref_latex_dict = convert_key_to_latex_pref(params_dict_pref, init_cond_pref, vars_dict_pref)

    read_param_from_file = 1
    read_Yss_from_file = 1
    read_classes_from_file = 1
    num_samples_forest = 8
    eps_sim = 1e-8
    eps = 1e-5
    tend = 1000000
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

    # Make samples for Random Forest

    if read_param_from_file == 1:
        param_init_vals = rf_read_sample_file(output_folder_pref_forest + '/sample_params.txt')
        params_vals_forest = rf_read_sample_file(output_folder_pref_forest + '/param_vals.txt')
        init_vals_forest = rf_read_sample_file(output_folder_pref_forest + '/init_vals.txt')
    else:
        sampling_dist = 2  # 0 for Saltelli, 2 for LHS

        params_vals_forest, init_vals_forest, problem_forest = gen_samples(num_samples_forest, params_dict_pref,
                                                                           init_cond_pref, params_bounds_dict_pref,
                                                                           init_cond_bounds_pref, [], 1, sampling_dist)
        param_init_vals = np.hstack((init_vals_forest, params_vals_forest))

    # Run Random Forest Simulations

    if read_Yss_from_file == 1:
        Y = np.loadtxt(output_folder_pref_forest + '/steady_state.txt', delimiter=',', usecols=range(11))
    else:
        Y = run_rf_sims(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref, ex_list_pref,
                          internal_prod_list_pref, d_list_pref, vars_dict_pref, params_dict_pref,
                          param_dependencies_dict_pref, params_vals_forest, init_vals_forest, init_cond_pref, eps_sim, tend,
                          pref_latex_dict, 0, output_folder_pref_forest)

    # Classify steady states

    if read_classes_from_file == 1:
        Yss = open_json_input_file(output_folder_pref_forest + '/classes.json')
    else:
        Yss = []
        for i, j in enumerate(params_vals_forest):
            if Y[i][vars_dict_pref['X_l']] < eps and Y[i][vars_dict_pref['Z_l']] < eps and Y[i][vars_dict_pref['X_m']] < eps and Y[i][vars_dict_pref['Z_m']] < eps:
                Yss.append('trivial')
            elif Y[i][vars_dict_pref['X_l']] < eps and Y[i][vars_dict_pref['Z_l']] >= eps and Y[i][vars_dict_pref['X_m']] < eps and Y[i][vars_dict_pref['Z_m']] >= eps:
                Yss.append('Z')
            elif Y[i][vars_dict_pref['X_l']] >= eps and Y[i][vars_dict_pref['Z_l']] < eps and Y[i][vars_dict_pref['X_m']] >= eps and Y[i][vars_dict_pref['Z_m']] < eps:
                Yss.append('X')
            elif Y[i][vars_dict_pref['X_l']] >= eps and Y[i][vars_dict_pref['Z_l']] >= eps and Y[i][vars_dict_pref['X_m']] >= eps and Y[i][vars_dict_pref['Z_m']] >= eps and Y[i][vars_dict_pref['X_l']] > Y[i][vars_dict_pref['Z_l']] and Y[i][vars_dict_pref['X_m']] > Y[i][vars_dict_pref['Z_m']]:
                Yss.append('coexistenceX')
            elif Y[i][vars_dict_pref['X_l']] >= eps and Y[i][vars_dict_pref['Z_l']] >= eps and Y[i][vars_dict_pref['X_m']] >= eps and Y[i][vars_dict_pref['Z_m']] >= eps and Y[i][vars_dict_pref['Z_l']] > Y[i][vars_dict_pref['X_l']] and Y[i][vars_dict_pref['Z_m']] > Y[i][vars_dict_pref['X_m']]:
                Yss.append('coexistenceZ')
            elif Y[i][vars_dict_pref['X_l']] >= eps and Y[i][vars_dict_pref['Z_l']] >= eps and Y[i][vars_dict_pref['X_m']] >= eps and Y[i][vars_dict_pref['Z_m']] >= eps and Y[i][vars_dict_pref['X_l']] > Y[i][vars_dict_pref['Z_l']] and Y[i][vars_dict_pref['X_m']] < Y[i][vars_dict_pref['Z_m']]:
                Yss.append('coexistenceXlZm')
            elif Y[i][vars_dict_pref['X_l']] >= eps and Y[i][vars_dict_pref['Z_l']] >= eps and Y[i][vars_dict_pref['X_m']] >= eps and Y[i][vars_dict_pref['Z_m']] >= eps and Y[i][vars_dict_pref['X_l']] < Y[i][vars_dict_pref['Z_l']] and Y[i][vars_dict_pref['X_m']] > Y[i][vars_dict_pref['Z_m']]:
                Yss.append('coexistenceXmZl')
            elif (Y[i][vars_dict_pref['X_l']] < eps or Y[i][vars_dict_pref['X_m']] < eps) and Y[i][vars_dict_pref['Z_l']] >= eps and Y[i][vars_dict_pref['Z_m']] >= eps:
                Yss.append('Z')
            elif Y[i][vars_dict_pref['X_l']] >= eps and Y[i][vars_dict_pref['X_m']] >= eps and (Y[i][vars_dict_pref['Z_m']] < eps or Y[i][vars_dict_pref['Z_l']] < eps):
                Yss.append('X')
            else:
                Yss.append('other')
        create_input_file(Yss, output_folder_pref_forest + '/classes.json')

    # Get feature names (parameter names)
    param_names = np.array(list(params_dict_pref.keys()))
    init_val_names = np.array(list(init_cond_pref.keys()))
    feature_names = np.concatenate([init_val_names, param_names])

    # X label as latex
    params_list = list(feature_names)
    x_label_latex = {key: pref_latex_dict[key] for key in params_list}

    # Run Random Forest Model - Figure 12 Feature Importance produced in this function

    run_rf_model(param_init_vals, Yss, feature_names, x_label_latex, output_folder_pref_forest)


if __name__ == "__main__":
    main()

from InputModule.build_pref_model import build_pref_model
from InputModule.create_json_input import create_input_file
from SimulationModule.run_batch_sim import run_sim_print, check_ss_type_pref
from SimulationModule.run_ode_sim import gen_samples
import numpy as np


def main():

    output_folder_pref_sims = '../../Dropbox/Results/PrefModel/SimData3'

    p_list_pref, d_list_pref, ex_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, internal_prod_list_pref, \
        phi_list_pref, vars_dict_pref, params_dict_pref, params_bounds_dict_pref, init_cond_pref, \
        init_cond_bounds_pref, param_dependencies_dict_pref = build_pref_model()

    num_samples = 1024
    eps_ss = 1e-5
    tstart = 0
    tend = 1000000
    tsteps = 1000000
########################################################################################################################

    # Initialize problem

    init_cond_pref = {'Il_0': 0.01, 'Gl_0': 0.01, 'Sl_0': 0.01, 'Rl_0': 0.01, 'Xl_0': 0.01, 'Zl_0': 0.01, 'Gm_0': 0.01,
                      'Sm_0': 0.01, 'Rm_0': 0.01, 'Xm_0': 0.01, 'Zm_0': 0.01}

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

    param_dependencies_dict_pref = {"V_l": [1.0, "-", "V_m"]}

    # Make samples

    sampling_dist = 0  # 0 for Saltelli, 2 for LHS

    params_vals, init_vals, problem = gen_samples(num_samples, params_dict_pref, init_cond_pref,
                                                  params_bounds_dict_pref, init_cond_bounds_pref, [], 0, sampling_dist)

    param_init_vals = np.hstack((init_vals, params_vals))
    # noinspection PyTypeChecker
    np.savetxt(output_folder_pref_sims + '/sample_params.txt', param_init_vals, fmt='%f')
    np.savetxt(output_folder_pref_sims + '/param_vals.txt', params_vals, fmt='%f')
    np.savetxt(output_folder_pref_sims + '/init_vals.txt', init_vals, fmt='%f')

    # Run Simulations and print to file

    Yss, Y_t, Y_sol, t_sol = run_sim_print(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref, ex_list_pref,
                        internal_prod_list_pref, d_list_pref, vars_dict_pref, params_dict_pref,
                        param_dependencies_dict_pref, params_vals, init_vals, init_cond_pref, tstart, tend, tsteps,
                        output_folder_pref_sims)

    # Classify steady states

    Yss_classes = check_ss_type_pref(eps_ss, Yss, vars_dict_pref)

    # Print classes to file

    create_input_file(Yss_classes, output_folder_pref_sims + '/Yss.json')


if __name__ == "__main__":
    main()

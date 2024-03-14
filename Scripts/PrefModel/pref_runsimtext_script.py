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

    tstart = 0
    tend = 1000000
    tsteps = 1000000
########################################################################################################################

    param_dependencies_dict_pref = {"V_l": [1.0, "-", "V_m"]}

    params_vals = np.loadtxt(output_folder_pref_sims + '/param_vals.txt')
    init_vals = np.loadtxt(output_folder_pref_sims + '/init_vals.txt')

    params_vals = np.flipud(params_vals)
    init_vals = np.flipud(init_vals)

    np.savetxt(output_folder_pref_sims + '/param_vals_flip.txt', params_vals, fmt='%f')
    np.savetxt(output_folder_pref_sims + '/init_vals_flip.txt', init_vals, fmt='%f')

    # Run Simulations and print to file

    Yss, Y_t = run_sim_print(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref, ex_list_pref,
                             internal_prod_list_pref, d_list_pref, vars_dict_pref, params_dict_pref,
                             param_dependencies_dict_pref, params_vals, init_vals, init_cond_pref, tstart, tend, tsteps,
                             output_folder_pref_sims)


if __name__ == "__main__":
    main()

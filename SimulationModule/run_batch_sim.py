import numpy as np
from SimulationModule.run_ode_sim import update_param_init_dicts
from SolverModule.build_problem import build_ode_problem_arrays
from SolverModule.solve_ode import chemo_sim


def run_timeseries_sim(p_list, inflow_amount_list, dilution_rate_list, phi_list, ex_list, internal_prod_list, d_list,
                       vars_dict, params_dict, param_dependencies_dict, params_vals, init_vals, init_cond,
                       varied_param_name, tstart, tend, tsteps):
    # Initializing matrix to store output
    Y_ss = np.zeros([len(params_vals), len(init_cond)])
    Y_t = np.zeros([len(params_vals), len(init_cond)])
    Y_sol = []
    t_sol = []
    varied_param_list = []

    # Solve ODE with sample array
    for i, X in enumerate(params_vals):
        sim_params_dict = update_param_init_dicts(params_dict, param_dependencies_dict, params_vals, i)
        problem_dict = build_ode_problem_arrays(p_list, inflow_amount_list, dilution_rate_list, phi_list, ex_list,
                                                internal_prod_list, d_list, sim_params_dict, vars_dict)
        res = chemo_sim(tstart, tend, tsteps, init_vals[i], problem_dict)

        # Steady state
        Y_ss[i][:] = res.y[:, -1]

        # Time to steady state
        Y_t[i][:] = res.t[-1]

        # Full solution
        Y_sol.append(res.y)
        t_sol.append(res.t)

        # Values of varied parameter
        if varied_param_name in sim_params_dict.keys():
            varied_param_list.append(sim_params_dict[varied_param_name])
        elif varied_param_name in init_cond.keys():
            varied_param_list.append(init_vals[i][list(init_cond).index(varied_param_name)]) # won't work in < Python 3.6 (uses insertion order of dictionary)

    return Y_ss, Y_t, Y_sol, t_sol, varied_param_list


def run_sim_print(p_list, inflow_amount_list, dilution_rate_list, phi_list, ex_list, internal_prod_list, d_list,
                  vars_dict, params_dict, param_dependencies_dict, params_vals, init_vals, init_cond, tstart, tend,
                  tsteps, output_folder):

    # Initializing matrix to store output
    Y_ss = np.zeros([len(params_vals), len(init_cond)])
    Y_t = np.zeros([len(params_vals), len(init_cond)])
    Y_sol = []
    t_sol = []

    f = open(output_folder + '/steady_state.txt', 'a+')

    # Solve ODE with sample array
    for i, X in enumerate(params_vals):
        sim_params_dict = update_param_init_dicts(params_dict, param_dependencies_dict, params_vals, i)
        problem_dict = build_ode_problem_arrays(p_list, inflow_amount_list, dilution_rate_list, phi_list, ex_list,
                                                internal_prod_list, d_list, sim_params_dict, vars_dict)
        res = chemo_sim(tstart, tend, tsteps, init_vals[i], problem_dict)

        # Steady state
        Y_ss[i][:] = res.y[:, -1]

        # Write steady state to file
        np.savetxt(f, Y_ss[i], fmt='%f', newline=',')
        f.write("\n")

        # Time to steady state
        Y_t[i][:] = res.t[-1]

        # Full solution
        Y_sol.append(res.y)
        t_sol.append(res.t)

    f.close()

    return Y_ss, Y_t, Y_sol, t_sol


def check_ss_type_pref(eps, sol_array, vars_dict_pref):

    Yss = []

    for sol in sol_array:
        if sol[vars_dict_pref['X_l']] < eps and sol[vars_dict_pref['Z_l']] < eps and sol[vars_dict_pref['X_m']] < eps and sol[vars_dict_pref['Z_m']] < eps:
            Yss.append('trivial')
        elif sol[vars_dict_pref['X_l']] < eps and sol[vars_dict_pref['Z_l']] >= eps and sol[vars_dict_pref['X_m']] < eps and sol[vars_dict_pref['Z_m']] >= eps:
            Yss.append('Z')
        elif sol[vars_dict_pref['X_l']] >= eps and sol[vars_dict_pref['Z_l']] < eps and sol[vars_dict_pref['X_m']] >= eps and sol[vars_dict_pref['Z_m']] < eps:
            Yss.append('X')
        elif sol[vars_dict_pref['X_l']] >= eps and sol[vars_dict_pref['Z_l']] >= eps and sol[vars_dict_pref['X_m']] >= eps and sol[vars_dict_pref['Z_m']] >= eps:
            Yss.append('coexistence')
        else:
            Yss.append('other')

    return Yss

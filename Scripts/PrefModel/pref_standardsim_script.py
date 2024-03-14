from InputModule.build_pref_model import build_pref_model
from PostProcess.timeseries_plot import plot_ss, plot_timeseries_groupby_param, plot_timeseries_groupby_vars, \
    plot_biomass_ss
from SimulationModule.run_batch_sim import run_timeseries_sim
from SimulationModule.run_ode_sim import gen_samples
from PostProcess.timeseries_plot import convert_key_to_latex_pref
from SolverModule.build_problem import build_ode_problem_arrays
from SolverModule.solve_ode import chemo_sim
import matplotlib.pyplot as plt
from itertools import cycle


def main():
    # Load problem

    p_list_pref, d_list_pref, ex_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, internal_prod_list_pref, \
        phi_list_pref, vars_dict_pref, params_dict_pref, params_bounds_dict_pref, init_cond_pref, init_cond_bounds_pref, \
        param_dependencies_dict_pref = build_pref_model()

    pref_latex_dict = convert_key_to_latex_pref(params_dict_pref, init_cond_pref, vars_dict_pref)

    output_folder_pref_IC = '../../Dropbox/Results/PrefModel/Standard/IC'
    output_folder_pref_IC_mucin = '../../Dropbox/Results/PrefModel/Standard/IC_mucin'
    output_folder_pref_standard_outcomes = '../../Dropbox/Results/PrefModel/Standard/Outcomes'

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)

########################################################################################################################

    # # A - IC Bistability, generalist only - Figure 11 (a)
    #
    # init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 10.0, 'Zl_0': 0.0, 'Gm_0': 0.0,
    #                   'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 0.0, 'Zm_0': 0.0}
    #
    # params_dict_pref = {'alpha_X': 0.1,
    #                     'alpha_Z': 0.1,
    #                     'gam_aX': 0.1,
    #                     'gam_aZ': 0.1,
    #                     'gam_dR': 3.9,
    #                     'gam_dS': 3.9,
    #                     'gam_sG': 0.4,
    #                     'gam_sX': 0.4,
    #                     'gam_sZ': 0.4,
    #                     'lam_max': 500.0,
    #                     'lam_prod': 50,
    #                     'mu_GX': 10.619469027,
    #                     'mu_GZ': 10.619469027,
    #                     'mu_IX': 10.619469027,
    #                     'mu_RS': 1.0,
    #                     'mu_RX': 2.5,
    #                     'mu_RZ': 3.0,
    #                     'mu_SX': 12.627143363,
    #                     'mu_SZ': 3.0,
    #                     'omega': 1000,
    #                     'D': 1.0071942760083039,
    #                     'K_GX': 0.5,
    #                     'K_GZ': 0.26539823,
    #                     'K_IX': 0.26539823,
    #                     'K_RS': 0.468416,
    #                     'K_RX': 0.468416,
    #                     'K_RZ': 0.468416,
    #                     'K_SX': 0.468416,
    #                     'K_SZ': 0.468416,
    #                     'I_inf': 50.0,
    #                     'V_l': 0.9,
    #                     'V_m': 0.1,
    #                     'Y_IX': 1.0,
    #                     'Y_GX': 1.0,
    #                     'Y_GZ': 1.0,
    #                     'Y_RX': 0.34242,
    #                     'Y_RZ': 0.34242,
    #                     'Y_SX': 0.34242,
    #                     'Y_SZ': 0.34242
    #                     }
    #
    # exclude_params_Xl_0 = ['alpha_X', 'alpha_Z', 'gam_aX', 'gam_aZ', 'gam_dR', 'gam_dS', 'gam_sG', 'gam_sX', 'gam_sZ',
    #                         'lam_max', 'lam_prod', 'mu_GX', 'mu_GZ', 'mu_IX', 'mu_RS', 'mu_RX', 'mu_RZ', 'mu_SX',
    #                         'mu_SZ', 'omega', 'D', 'K_GX', 'K_GZ', 'K_IX', 'K_RS', 'K_RX', 'K_RZ', 'K_SX', 'K_SZ', 'I_inf', 'V_l',
    #                         'V_m', 'Y_IX', 'Y_GX', 'Y_GZ', 'Y_RX', 'Y_RZ', 'Y_SX', 'Y_SZ', 'Il_0', 'Gl_0',
    #                         'Sl_0', 'Rl_0', 'Zl_0', 'Gm_0', 'Sm_0', 'Rm_0', 'Xm_0', 'Zm_0']
    #
    # init_cond_bounds_pref['Xl_0'] = [0.01, 0.05]
    #
    # params_vals_omega, init_vals_omega, problem_omega = gen_samples(20, params_dict_pref, init_cond_pref,
    #                                                                 params_bounds_dict_pref, init_cond_bounds_pref,
    #                                                                 exclude_params_Xl_0, 0, 1)
    #
    # varied_param_name_omega = 'Xl_0'
    #
    # Y_ss_omega, Y_t_omega, Y_sol_omega, t_sol_omega, varied_param_list_omega = run_timeseries_sim(
    #     p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref, ex_list_pref,
    #     internal_prod_list_pref, d_list_pref, vars_dict_pref, params_dict_pref, param_dependencies_dict_pref,
    #     params_vals_omega, init_vals_omega, init_cond_pref, varied_param_name_omega, 0, 1000,
    #             1000)
    #
    # plot_ss(Y_ss_omega, varied_param_list_omega, vars_dict_pref, pref_latex_dict,
    #         'Effect of Initial Condition of $X_l$ ($X_l^0)$) on '
    #         'Steady State',
    #         'Steady State Value', 'Initial Condition of $X_l$ ($X_l^0$)', output_folder_pref_IC)

########################################################################################################################

    # # B - IC Bistability, specialist only - Figure 11 (b)
    #
    # init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 0.0, 'Zl_0': 10.0, 'Gm_0': 0.0,
    #                   'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 0.0, 'Zm_0': 0.0}
    #
    # params_dict_pref = {'alpha_X': 0.1,
    #                     'alpha_Z': 0.1,
    #                     'gam_aX': 0.1,
    #                     'gam_aZ': 0.1,
    #                     'gam_dR': 3.9,
    #                     'gam_dS': 3.9,
    #                     'gam_sG': 0.4,
    #                     'gam_sX': 0.4,
    #                     'gam_sZ': 0.4,
    #                     'lam_max': 500.0,
    #                     'lam_prod': 50,
    #                     'mu_GX': 10.619469027,
    #                     'mu_GZ': 10.619469027,
    #                     'mu_IX': 10.619469027,
    #                     'mu_RS': 1.0,
    #                     'mu_RX': 2.5,
    #                     'mu_RZ': 3.0,
    #                     'mu_SX': 12.627143363,
    #                     'mu_SZ': 3.0,
    #                     'omega': 1000,
    #                     'D': 1.0071942760083039,
    #                     'K_GX': 0.5,
    #                     'K_GZ': 0.26539823,
    #                     'K_IX': 0.26539823,
    #                     'K_RS': 0.468416,
    #                     'K_RX': 0.468416,
    #                     'K_RZ': 0.468416,
    #                     'K_SX': 0.468416,
    #                     'K_SZ': 0.468416,
    #                     'I_inf': 50.0,
    #                     'V_l': 0.9,
    #                     'V_m': 0.1,
    #                     'Y_IX': 1.0,
    #                     'Y_GX': 1.0,
    #                     'Y_GZ': 1.0,
    #                     'Y_RX': 0.34242,
    #                     'Y_RZ': 0.34242,
    #                     'Y_SX': 0.34242,
    #                     'Y_SZ': 0.34242
    #                     }
    #
    # exclude_params_Zl_0 = ['alpha_X', 'alpha_Z', 'gam_aX', 'gam_aZ', 'gam_dR', 'gam_dS', 'gam_sG', 'gam_sX', 'gam_sZ',
    #                         'lam_max', 'lam_prod', 'mu_GX', 'mu_GZ', 'mu_IX', 'mu_RS', 'mu_RX', 'mu_RZ', 'mu_SX',
    #                         'mu_SZ', 'omega', 'D', 'K_GX', 'K_GZ', 'K_IX', 'K_RS', 'K_RX', 'K_RZ', 'K_SX', 'K_SZ', 'I_inf', 'V_l',
    #                         'V_m', 'Y_IX', 'Y_GX', 'Y_GZ', 'Y_RX', 'Y_RZ', 'Y_SX', 'Y_SZ', 'Il_0', 'Gl_0',
    #                         'Sl_0', 'Rl_0', 'Xl_0', 'Gm_0', 'Sm_0', 'Rm_0', 'Xm_0', 'Zm_0']
    #
    # init_cond_bounds_pref['Zl_0'] = [0.1, 0.2]
    #
    # params_vals_ic_specialist, init_vals_ic_specialist, problem_ic_specialist = gen_samples(20, params_dict_pref, init_cond_pref,
    #                                                                 params_bounds_dict_pref, init_cond_bounds_pref,
    #                                                                 exclude_params_Zl_0, 0, 1)
    #
    # varied_param_name_ic_specialist = 'Zl_0'
    #
    # Y_ss_ic_specialist, Y_t_ic_specialist, Y_sol_ic_specialist, t_sol_ic_specialist, varied_param_list_ic_specialist = run_timeseries_sim(
    #     p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref, ex_list_pref,
    #     internal_prod_list_pref, d_list_pref, vars_dict_pref, params_dict_pref, param_dependencies_dict_pref,
    #     params_vals_ic_specialist, init_vals_ic_specialist, init_cond_pref, varied_param_name_ic_specialist, 0, 1000,
    #             1000)
    #
    # plot_ss(Y_ss_ic_specialist, varied_param_list_ic_specialist, vars_dict_pref, pref_latex_dict, 'Effect of Initial Condition of $Z_l$ ($Z_l^0$) on '
    #                                                                               'Steady State',
    #         'Steady State Value', 'Initial Condition of $Z_l$ ($Z_l^0$)', output_folder_pref_IC_mucin)

########################################################################################################################

    # C - Four Outcomes

    # X survives - Figure 10 (a) and (b)

    init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 1.0, 'Zl_0': 1.0, 'Gm_0': 0.0,
                      'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 0.0, 'Zm_0': 0.0}

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

    problem_dict = build_ode_problem_arrays(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref,
                                            ex_list_pref, internal_prod_list_pref, d_list_pref, params_dict_pref,
                                            vars_dict_pref)

    res = chemo_sim(0, 1000, 10000, list(init_cond_pref.values()), problem_dict)

    latex_vars_dict = []
    for key, value in vars_dict_pref.items():
        latex_vars_dict.append(pref_latex_dict[key])

    # Plot biomass
    plt.figure()
    plt.plot(res.t, res.y[vars_dict_pref['X_l']], label=latex_vars_dict[vars_dict_pref['X_l']], color='C0')
    plt.plot(res.t, res.y[vars_dict_pref['X_m']], label=latex_vars_dict[vars_dict_pref['X_m']], color='C2')
    plt.plot(res.t, res.y[vars_dict_pref['Z_l']], label=latex_vars_dict[vars_dict_pref['Z_l']], color='C1')
    plt.plot(res.t, res.y[vars_dict_pref['Z_m']], label=latex_vars_dict[vars_dict_pref['Z_m']], color='C3')
    plt.legend('', frameon=False)
    plt.ylabel('Concentration of Biomass (g/L)')
    plt.xlabel('Time (days)')
    plt.title('X Survives')
    plt.savefig(output_folder_pref_standard_outcomes + '/X_survives.pdf', bbox_inches='tight')
    plt.close()

    # Z Survives - Figure 1- (c) and (d)

    init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 1.0, 'Zl_0': 1.0, 'Gm_0': 0.0,
                      'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 0.0, 'Zm_0': 0.0}

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
                        'I_inf': 0.0,
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

    problem_dict = build_ode_problem_arrays(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref,
                                            ex_list_pref, internal_prod_list_pref, d_list_pref, params_dict_pref,
                                            vars_dict_pref)

    res = chemo_sim(0, 1000, 10000, list(init_cond_pref.values()), problem_dict)

    latex_vars_dict = []
    for key, value in vars_dict_pref.items():
        latex_vars_dict.append(pref_latex_dict[key])

    # Plot biomass
    plt.figure()
    plt.plot(res.t, res.y[vars_dict_pref['X_l']], label=latex_vars_dict[vars_dict_pref['X_l']], color='C0')
    plt.plot(res.t, res.y[vars_dict_pref['X_m']], label=latex_vars_dict[vars_dict_pref['X_m']], color='C2')
    plt.plot(res.t, res.y[vars_dict_pref['Z_l']], label=latex_vars_dict[vars_dict_pref['Z_l']], color='C1')
    plt.plot(res.t, res.y[vars_dict_pref['Z_m']], label=latex_vars_dict[vars_dict_pref['Z_m']], color='C3')
    plt.legend(loc='best')
    plt.ylabel('Concentration of Biomass (g/L)')
    plt.xlabel('Time (days)')
    plt.title('Z Survives')
    plt.savefig(output_folder_pref_standard_outcomes + '/Z_survives.pdf', bbox_inches='tight')
    plt.close()

    # Coexistence - Figure 9 (c) and (d)

    init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 1.0, 'Zl_0': 1.0, 'Gm_0': 0.0,
                      'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 0.0, 'Zm_0': 0.0}

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
                        'I_inf': 10.0,
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

    problem_dict = build_ode_problem_arrays(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref,
                                            ex_list_pref, internal_prod_list_pref, d_list_pref, params_dict_pref,
                                            vars_dict_pref)

    res = chemo_sim(0, 1000, 10000, list(init_cond_pref.values()), problem_dict)

    latex_vars_dict = []
    for key, value in vars_dict_pref.items():
        latex_vars_dict.append(pref_latex_dict[key])

    # Plot biomass
    plt.figure()
    plt.plot(res.t, res.y[vars_dict_pref['X_l']], label=latex_vars_dict[vars_dict_pref['X_l']], color='C0')
    plt.plot(res.t, res.y[vars_dict_pref['X_m']], label=latex_vars_dict[vars_dict_pref['X_m']], color='C2')
    plt.plot(res.t, res.y[vars_dict_pref['Z_l']], label=latex_vars_dict[vars_dict_pref['Z_l']], color='C1')
    plt.plot(res.t, res.y[vars_dict_pref['Z_m']], label=latex_vars_dict[vars_dict_pref['Z_m']], color='C3')
    plt.legend('', frameon=False)
    plt.ylabel('Concentration of Biomass (g/L)')
    plt.xlabel('Time (days)')
    plt.title('Coexistence')
    plt.savefig(output_folder_pref_standard_outcomes + '/coexistence.pdf', bbox_inches='tight')
    plt.close()

    # Washout - Figure 9 (a) and (b)

    init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 0.01, 'Zl_0': 0.01, 'Gm_0': 0.0,
                      'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 0.0, 'Zm_0': 0.0}

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
                        'I_inf': 10.0,
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

    problem_dict = build_ode_problem_arrays(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref,
                                            ex_list_pref, internal_prod_list_pref, d_list_pref, params_dict_pref,
                                            vars_dict_pref)

    res = chemo_sim(0, 1000, 10000, list(init_cond_pref.values()), problem_dict)

    latex_vars_dict = []
    for key, value in vars_dict_pref.items():
        latex_vars_dict.append(pref_latex_dict[key])

    # Plot biomass
    plt.figure()
    plt.plot(res.t, res.y[vars_dict_pref['X_l']], label=latex_vars_dict[vars_dict_pref['X_l']], color='C0')
    plt.plot(res.t, res.y[vars_dict_pref['X_m']], label=latex_vars_dict[vars_dict_pref['X_m']], color='C2')
    plt.plot(res.t, res.y[vars_dict_pref['Z_l']], label=latex_vars_dict[vars_dict_pref['Z_l']], color='C1')
    plt.plot(res.t, res.y[vars_dict_pref['Z_m']], label=latex_vars_dict[vars_dict_pref['Z_m']], color='C3')
    plt.legend('', frameon=False)
    plt.ylabel('Concentration of Biomass (g/L)')
    plt.xlabel('Time (days)')
    plt.title('Washout')
    plt.savefig(output_folder_pref_standard_outcomes + '/washout.pdf', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()

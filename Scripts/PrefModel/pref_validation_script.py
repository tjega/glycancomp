from InputModule.build_pref_model import build_pref_model
from PostProcess.timeseries_plot import plot_ss, plot_timeseries_groupby_param, plot_timeseries_groupby_vars, \
    plot_biomass_ss
from SimulationModule.run_batch_sim import run_timeseries_sim
from SimulationModule.run_ode_sim import gen_samples
from PostProcess.timeseries_plot import convert_key_to_latex_pref
from itertools import cycle
from InputModule.build_model import build_single_compartment_primary
from InputModule.build_model import build_double_compartment_primary
from PostProcess.timeseries_plot import convert_key_to_latex_double
from PostProcess.timeseries_plot import convert_key_to_latex_single
from SolverModule.build_problem import build_ode_problem_arrays
from SolverModule.solve_ode import chemo_sim
import matplotlib.pyplot as plt


def main():
    # 1 - Testing Simulations

    (p_list_pref, d_list_pref, ex_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, internal_prod_list_pref,
     phi_list_pref, vars_dict_pref, params_dict_pref, params_bounds_dict_pref, init_cond_pref, init_cond_bounds_pref,
     param_dependencies_dict_pref) = build_pref_model()

    pref_latex_dict = convert_key_to_latex_pref(params_dict_pref, init_cond_pref, vars_dict_pref)

    # single compartment comparison

    (p_list_single, d_list_single, ex_list_single, inflow_amount_list_single, dilution_rate_list_single,
     internal_prod_list_single, phi_list_single, vars_dict_single, params_dict_single, params_bounds_dict_single,
     init_cond_single, init_cond_bounds_single, param_dependencies_dict_single) = build_single_compartment_primary()

    single_latex_dict = convert_key_to_latex_single(params_dict_single, init_cond_single, vars_dict_single)

    # dual compartment comparison

    (p_list_dual, d_list_dual, ex_list_dual, inflow_amount_list_dual, dilution_rate_list_dual, internal_prod_list_dual,
     phi_list_dual, vars_dict_dual, params_dict_dual, params_bounds_dict_dual, init_cond_dual, init_cond_bounds_dual,
     param_dependencies_dict_dual) = build_double_compartment_primary()

    dual_latex_dict = convert_key_to_latex_double(params_dict_dual, init_cond_dual, vars_dict_dual)

    output_folder_pref_test = '../../Dropbox/Results/PrefModel/Testing'
    output_folder_pref_omega_genonly_nomucin = output_folder_pref_test + '/OmegaGeneralistOnlyNoMucin'
    output_folder_pref_omega_genonly_withmucin = output_folder_pref_test + '/OmegaGeneralistOnlyWithMucin'
    output_folder_pref_omega_genonly_withmucin_lowfiber = (output_folder_pref_test +
                                                           '/OmegaGeneralistOnlyWithMucinLowFiber')
    output_folder_pref_omega_nofiberflow = output_folder_pref_test + '/OmegaNoInflowFiber'
    output_folder_pref_omega_comp_lowfiber = output_folder_pref_test + '/OmegaTwoSpeciesLowFiber5'
    output_folder_pref_omega_comp_highfiber = output_folder_pref_test + '/OmegaTwoSpeciesHighFiber50'

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)

########################################################################################################################

    # Section 3.1 Code Testing and Verification

########################################################################################################################
    # A - All zero

    init_cond_pref = {"Il_0": 0.0, "Gl_0": 0.0, "Sl_0": 0.0, "Rl_0": 0.0, "Xl_0": 0.0, "Zl_0": 0.0, "Gm_0": 0.0,
                      "Sm_0": 0.0, "Rm_0": 0.0, "Xm_0": 0.0, "Zm_0": 0.0}

    params_dict_pref['lam_prod'] = 0.0
    params_dict_pref['I_inf'] = 0.0

    problem_dict = build_ode_problem_arrays(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, ex_list_pref,
                                            internal_prod_list_pref, d_list_pref, params_dict_pref, vars_dict_pref)

    res = chemo_sim(0, 1000, 10000, list(init_cond_pref.values()), problem_dict)

    latex_vars_dict = []
    for key, value in vars_dict_pref.items():
        latex_vars_dict.append(pref_latex_dict[key])

    plt.figure()
    for i in range(res.y.shape[0]):
        plt.plot(res.t, res.y[i], label=latex_vars_dict[i])
    plt.legend(loc='best')
    plt.ylabel('Concentration')
    plt.xlabel('Time')
    plt.savefig(output_folder_pref_test + '/all_zero.pdf', bbox_inches='tight')
    plt.close()

########################################################################################################################

    # B - Mucin and fiber only, no biomass, no transfer

    init_cond_pref = {"Il_0": 0.0, "Gl_0": 0.0, "Sl_0": 0.0, "Rl_0": 0.0, "Xl_0": 0.0, "Zl_0": 0.0, "Gm_0": 0.0,
                      "Sm_0": 0.0, "Rm_0": 0.0, "Xm_0": 0.0, "Zm_0": 0.0}

    params_dict_pref['lam_prod'] = 50.0
    params_dict_pref['I_inf'] = 10.0
    params_dict_pref['gam_aX'] = 0.0
    params_dict_pref['gam_aZ'] = 0.0
    params_dict_pref['gam_dR'] = 0.0
    params_dict_pref['gam_dS'] = 0.0
    params_dict_pref['gam_sG'] = 0.0
    params_dict_pref['gam_sX'] = 0.0
    params_dict_pref['gam_sZ'] = 0.0

    problem_dict = build_ode_problem_arrays(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref,
                                            phi_list_pref, ex_list_pref, internal_prod_list_pref, d_list_pref,
                                            params_dict_pref, vars_dict_pref)

    res = chemo_sim(0, 1000, 10000, list(init_cond_pref.values()), problem_dict)

    latex_vars_dict = []
    for key, value in vars_dict_pref.items():
        latex_vars_dict.append(pref_latex_dict[key])

    plt.figure()
    for i in range(res.y.shape[0]):
        plt.plot(res.t, res.y[i], label=latex_vars_dict[i], linestyle=next(linecycler))
    plt.legend(loc='best')
    plt.ylabel('Concentration')
    plt.xlabel('Time')
    plt.savefig(output_folder_pref_test + '/nobiomass_notransfer.pdf', bbox_inches='tight')
    plt.close()

########################################################################################################################

    # C - Mucin and fiber only, no biomass, with transfer

    init_cond_pref = {"Il_0": 0.0, "Gl_0": 0.0, "Sl_0": 0.0, "Rl_0": 0.0, "Xl_0": 0.0, "Zl_0": 0.0, "Gm_0": 0.0,
                      "Sm_0": 0.0,
                      "Rm_0": 0.0, "Xm_0": 0.0, "Zm_0": 0.0}

    params_dict_pref['lam_prod'] = 50.0
    params_dict_pref['I_inf'] = 1.0
    params_dict_pref['gam_aX'] = 0.1
    params_dict_pref['gam_aZ'] = 0.1
    params_dict_pref['gam_dR'] = 3.9
    params_dict_pref['gam_dS'] = 3.9
    params_dict_pref['gam_sG'] = 0.1
    params_dict_pref['gam_sX'] = 0.4
    params_dict_pref['gam_sZ'] = 0.4

    problem_dict = build_ode_problem_arrays(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref,
                                            phi_list_pref, ex_list_pref, internal_prod_list_pref, d_list_pref,
                                            params_dict_pref, vars_dict_pref)

    res = chemo_sim(0, 1000, 10000, list(init_cond_pref.values()), problem_dict)

    latex_vars_dict = []
    for key, value in vars_dict_pref.items():
        latex_vars_dict.append(pref_latex_dict[key])

    plt.figure()
    for i in range(res.y.shape[0]):
        plt.plot(res.t, res.y[i], label=latex_vars_dict[i], linestyle=next(linecycler))
    plt.legend(loc='best')
    plt.ylabel('Concentration')
    plt.xlabel('Time')
    plt.savefig(output_folder_pref_test + '/nobiomass.pdf', bbox_inches='tight')
    plt.close()

########################################################################################################################

    # D - Single compartment single species comparison

    problem_dict_single = build_ode_problem_arrays(p_list_single, inflow_amount_list_single, dilution_rate_list_single,
                                                   phi_list_single, ex_list_single, internal_prod_list_single,
                                                   d_list_single, params_dict_single, vars_dict_single)

    res_single = chemo_sim(0, 1000, 10000, list(init_cond_single.values()), problem_dict_single)

    latex_vars_dict = []
    for key, value in vars_dict_pref.items():
        latex_vars_dict.append(pref_latex_dict[key])

    latex_vars_dict_single = []
    for key, value in vars_dict_single.items():
        latex_vars_dict_single.append(single_latex_dict[key])

    plt.figure()
    for i in range(res.y.shape[0]):
        plt.plot(res.t, res.y[i], label=latex_vars_dict[i], linestyle=next(linecycler))
    for i in range(res_single.y.shape[0]):
        plt.plot(res_single.t, res_single.y[i], label=latex_vars_dict_single[i], linestyle=next(linecycler))
    plt.legend(loc='best')
    plt.ylabel('Concentration')
    plt.xlabel('Time')
    plt.savefig(output_folder_pref_test + '/generalist_compare_single.pdf', bbox_inches='tight')
    plt.close()

########################################################################################################################

    # E - Generalist only, omega is zero, dual compartment

    init_cond_pref = {"Il_0": 0.0, "Gl_0": 0.0, "Sl_0": 0.0, "Rl_0": 0.0, "Xl_0": 0.1, "Zl_0": 0.0, "Gm_0": 0.0,
                      "Rm_0": 0.0, "Xm_0": 0.1, "Zm_0": 0.0}

    params_dict_pref['gam_aX'] = 0.1
    params_dict_pref['gam_aZ'] = 0.1
    params_dict_pref['gam_dR'] = 3.9
    params_dict_pref['gam_dS'] = 3.9
    params_dict_pref['gam_sG'] = 0.1
    params_dict_pref['gam_sX'] = 0.4
    params_dict_pref['gam_sZ'] = 0.4
    params_dict_pref['omega'] = 0.0
    params_dict_pref['lam_prod'] = 0.0
    params_dict_pref['lam_max'] = 0.0
    params_dict_pref['I_inf'] = 1.0

    problem_dict = build_ode_problem_arrays(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref,
                                            phi_list_pref, ex_list_pref, internal_prod_list_pref, d_list_pref,
                                            params_dict_pref, vars_dict_pref)

    res = chemo_sim(0, 1000, 10000, list(init_cond_pref.values()), problem_dict)

    # Dual compartment dual species comparison

    init_cond_dual = {'Il_0': 0, 'Sl_0': 0, 'Xl_0': 0.1, 'Im_0': 0, 'Sm_0': 0, 'Xm_0': 0.1}

    params_dict_dual['lam_prod'] = 0.0
    params_dict_dual['lam_max'] = 0.0

    problem_dict_dual = build_ode_problem_arrays(p_list_dual, inflow_amount_list_dual, dilution_rate_list_dual,
                                                 phi_list_dual, ex_list_dual, internal_prod_list_dual, d_list_dual,
                                                 params_dict_dual, vars_dict_dual)

    res_dual = chemo_sim(0, 1000, 10000, list(init_cond_dual.values()), problem_dict_dual)

    latex_vars_dict = []
    for key, value in vars_dict_pref.items():
        latex_vars_dict.append(pref_latex_dict[key])

    latex_vars_dict_dual = []
    for key, value in vars_dict_dual.items():
        latex_vars_dict_dual.append(dual_latex_dict[key])

    plt.figure()
    for i in range(res.y.shape[0]):
        plt.plot(res.t, res.y[i], label=latex_vars_dict[i], linestyle=next(linecycler))
    for i in range(res_dual.y.shape[0]):
        plt.plot(res_dual.t, res_dual.y[i], label=latex_vars_dict_dual[i], linestyle=next(linecycler))
    plt.legend(loc='best')
    plt.ylabel('Concentration')
    plt.xlabel('Time')
    plt.savefig(output_folder_pref_test + '/generalist_compare_dual.pdf', bbox_inches='tight')
    plt.close()

########################################################################################################################

    # F - Varying omega, generalist only, omega should have no effect

    init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 0.1, 'Zl_0': 0.0, 'Gm_0': 0.0,
                      'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 0.1, 'Zm_0': 0.0}

    params_dict_pref['omega'] = 0.0
    params_dict_pref['lam_prod'] = 0.0
    params_dict_pref['lam_max'] = 0.0
    params_dict_pref['I_inf'] = 1.0
    params_dict_pref['D'] = 1.0071942760083039

    exclude_params_omega = ['alpha_X', 'alpha_Z', 'gam_aX', 'gam_aZ', 'gam_dR', 'gam_dS', 'gam_sG', 'gam_sX', 'gam_sZ',
                            'lam_max', 'lam_prod', 'mu_GX', 'mu_GZ', 'mu_IX', 'mu_RS', 'mu_RX', 'mu_RZ', 'mu_SX',
                            'mu_SZ', 'D', 'K_GX', 'K_GZ', 'K_IX', 'K_RS', 'K_RX', 'K_RZ', 'K_SX', 'K_SZ', 'I_inf',
                            'V_l',
                            'V_m', 'Y_IX', 'Y_GX', 'Y_GZ', 'Y_RX', 'Y_RZ', 'Y_SX', 'Y_SZ', 'Il_0', 'Gl_0',
                            'Sl_0', 'Rl_0', 'Xl_0', 'Zl_0', 'Gm_0', 'Sm_0', 'Rm_0', 'Xm_0', 'Zm_0']

    params_bounds_dict_pref['omega'] = [0.0, 5.0]

    params_vals_omega, init_vals_omega, problem_omega = gen_samples(5, params_dict_pref, init_cond_pref,
                                                                    params_bounds_dict_pref, init_cond_bounds_pref,
                                                                    exclude_params_omega, 0, 1)

    varied_param_name_omega = 'omega'

    Y_ss_omega, Y_t_omega, Y_sol_omega, t_sol_omega, varied_param_list_omega = run_timeseries_sim(
        p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref, ex_list_pref,
        internal_prod_list_pref, d_list_pref, vars_dict_pref, params_dict_pref, param_dependencies_dict_pref,
        params_vals_omega, init_vals_omega, init_cond_pref, varied_param_name_omega, 0, 1000, 1000)

    plot_ss(Y_ss_omega, varied_param_list_omega, vars_dict_pref, pref_latex_dict,
            'Effect of Inhibition Constant ($\omega$) on Steady State',
            'Steady State Value', 'Inhibition Constant ($\omega$)', output_folder_pref_omega_genonly_nomucin)

    params_vals_omega_2, init_vals_omega_2, problem_single_omega_2 = gen_samples(10,
                                                                                 params_dict_pref,
                                                                                 init_cond_pref,
                                                                                 params_bounds_dict_pref,
                                                                                 init_cond_bounds_pref,
                                                                                 exclude_params_omega,
                                                                                 0, 1)

    Y_ss_omega_2, Y_t_omega_2, Y_sol_omega_2, t_sol_omega_2, varied_param_list_omega_2 = run_timeseries_sim(p_list_pref,
                                                                                                            inflow_amount_list_pref,
                                                                                                            dilution_rate_list_pref,
                                                                                                            phi_list_pref,
                                                                                                            ex_list_pref,
                                                                                                            internal_prod_list_pref,
                                                                                                            d_list_pref,
                                                                                                            vars_dict_pref,
                                                                                                            params_dict_pref,
                                                                                                            param_dependencies_dict_pref,
                                                                                                            params_vals_omega_2,
                                                                                                            init_vals_omega_2,
                                                                                                            init_cond_pref,
                                                                                                            varied_param_name_omega)

    plot_timeseries_groupby_param(Y_sol_omega_2, t_sol_omega_2, varied_param_list_omega_2,
                                  varied_param_name_omega, vars_dict_pref, pref_latex_dict,
                                  'Concentration vs Time', 'Concentration', '(g/L)', 'Time (Days)',
                                  output_folder_pref_omega_genonly_nomucin)

    plot_timeseries_groupby_vars(Y_sol_omega_2, t_sol_omega_2, varied_param_list_omega_2,
                                 varied_param_name_omega, vars_dict_pref, pref_latex_dict, 'Concentration vs Time for',
                                 'Concentration (g/L)', 'Time (Days)', output_folder_pref_omega_genonly_nomucin)

########################################################################################################################
# Section 3.3
########################################################################################################################

    # G - Varying omega, generalist only, with mucin, high fiber

    init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 0.5, 'Zl_0': 0.0, 'Gm_0': 0.0,
                      'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 0.5, 'Zm_0': 0.0}

    params_dict_pref['omega'] = 0.0
    params_dict_pref['lam_prod'] = 50.0
    params_dict_pref['lam_max'] = 500.0
    params_dict_pref['I_inf'] = 50.0
    params_dict_pref['D'] = 1.0071942760083039

    exclude_params_omega_mucin = ['alpha_X', 'alpha_Z', 'gam_aX', 'gam_aZ', 'gam_dR', 'gam_dS', 'gam_sG', 'gam_sX',
                                  'gam_sZ', 'lam_max', 'lam_prod', 'mu_GX', 'mu_GZ', 'mu_IX', 'mu_RS', 'mu_RX', 'mu_RZ',
                                  'mu_SX', 'mu_SZ', 'D', 'K_GX', 'K_GZ', 'K_IX', 'K_RS', 'K_RX', 'K_RZ', 'K_SX', 'K_SZ',
                                  'I_inf', 'V_l', 'V_m', 'Y_IX', 'Y_GX', 'Y_GZ', 'Y_RX', 'Y_RZ', 'Y_SX', 'Y_SZ', 'Il_0',
                                  'Gl_0', 'Sl_0', 'Rl_0', 'Xl_0', 'Zl_0', 'Gm_0', 'Sm_0', 'Rm_0', 'Xm_0', 'Zm_0']

    params_bounds_dict_pref['omega'] = [0.0, 10000.0]

    params_vals_omega_mucin, init_vals_omega_mucin, problem_omega_mucin = gen_samples(20, params_dict_pref,
                                                                                      init_cond_pref,
                                                                                      params_bounds_dict_pref,
                                                                                      init_cond_bounds_pref,
                                                                                      exclude_params_omega_mucin, 0, 1)

    varied_param_name_omega_mucin = 'omega'

    Y_ss_omega_mucin, Y_t_omega_mucin, Y_sol_omega_mucin, t_sol_omega_mucin, varied_param_list_omega_mucin = \
        run_timeseries_sim(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref, ex_list_pref,
                           internal_prod_list_pref, d_list_pref, vars_dict_pref, params_dict_pref,
                           param_dependencies_dict_pref, params_vals_omega_mucin, init_vals_omega_mucin, init_cond_pref,
                           varied_param_name_omega_mucin, 0, 1000, 10000)

    plot_biomass_ss(Y_ss_omega_mucin, varied_param_list_omega_mucin, vars_dict_pref, pref_latex_dict,
            'High Inflow Fiber Concentration ($I^\infty=50$)', 'Steady State Value (g/L)',
            'Inhibition Constant ($\omega$)', output_folder_pref_omega_genonly_withmucin)

#######################################################################################################################

    # I - Varying omega, generalist only, with mucin, low fiber

    init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 0.5, 'Zl_0': 0.0, 'Gm_0': 0.0,
                      'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 0.5, 'Zm_0': 0.0}

    params_dict_pref['omega'] = 0.0
    params_dict_pref['lam_prod'] = 50.0
    params_dict_pref['lam_max'] = 500.0
    params_dict_pref['I_inf'] = 5.0
    params_dict_pref['D'] = 1.0071942760083039

    exclude_params_omega_mucin_lowfiber = ['alpha_X', 'alpha_Z', 'gam_aX', 'gam_aZ', 'gam_dR', 'gam_dS', 'gam_sG',
                                           'gam_sX',
                                           'gam_sZ', 'lam_max', 'lam_prod', 'mu_GX', 'mu_GZ', 'mu_IX', 'mu_RS', 'mu_RX',
                                           'mu_RZ', 'mu_SX', 'mu_SZ', 'D', 'K_GX', 'K_GZ', 'K_IX', 'K_RS', 'K_RX',
                                           'K_RZ',
                                           'K_SX', 'K_SZ', 'I_inf', 'V_l', 'V_m', 'Y_IX', 'Y_GX', 'Y_GZ', 'Y_RX',
                                           'Y_RZ',
                                           'Y_SX', 'Y_SZ', 'Il_0', 'Gl_0', 'Sl_0', 'Rl_0', 'Xl_0', 'Zl_0', 'Gm_0',
                                           'Sm_0',
                                           'Rm_0', 'Xm_0', 'Zm_0']

    params_bounds_dict_pref['omega'] = [0.0, 10000.0]

    params_vals_omega_mucin_lowfiber, init_vals_omega_mucin_lowfiber, problem_omega_mucin_lowfiber = \
        gen_samples(20, params_dict_pref, init_cond_pref, params_bounds_dict_pref, init_cond_bounds_pref,
                    exclude_params_omega_mucin_lowfiber, 0, 1)

    varied_param_name_omega_mucin_lowfiber = 'omega'

    Y_ss_omega_mucin_lowfiber, Y_t_omega_mucin_lowfiber, Y_sol_omega_mucin_lowfiber, t_sol_omega_mucin_lowfiber, \
        varied_param_list_omega_mucin_lowfiber = run_timeseries_sim(p_list_pref, inflow_amount_list_pref,
                                                                    dilution_rate_list_pref, phi_list_pref,
                                                                    ex_list_pref,
                                                                    internal_prod_list_pref, d_list_pref,
                                                                    vars_dict_pref,
                                                                    params_dict_pref, param_dependencies_dict_pref,
                                                                    params_vals_omega_mucin_lowfiber,
                                                                    init_vals_omega_mucin_lowfiber, init_cond_pref,
                                                                    varied_param_name_omega_mucin_lowfiber, 0, 1000,
                                                                    10000)

    plot_biomass_ss(Y_ss_omega_mucin_lowfiber, varied_param_list_omega_mucin_lowfiber, vars_dict_pref, pref_latex_dict,
            'Low Inflow Fiber Concentration ($I^\infty = 5$)', 'Steady State Value (g/L)',
            'Inhibition Constant ($\omega$)', output_folder_pref_omega_genonly_withmucin_lowfiber)

#######################################################################################################################

    # G - Varying omega, both species, with mucin, high fiber

    init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 0.5, 'Zl_0': 0.5, 'Gm_0': 0.0,
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

    params_dict_pref['omega'] = 0.0
    params_dict_pref['lam_prod'] = 50.0
    params_dict_pref['lam_max'] = 500.0
    params_dict_pref['I_inf'] = 50.0
    params_dict_pref['D'] = 1.0071942760083039

    exclude_params_omega_mucin = ['alpha_X', 'alpha_Z', 'gam_aX', 'gam_aZ', 'gam_dR', 'gam_dS', 'gam_sG', 'gam_sX',
                                  'gam_sZ', 'lam_max', 'lam_prod', 'mu_GX', 'mu_GZ', 'mu_IX', 'mu_RS', 'mu_RX', 'mu_RZ',
                                  'mu_SX', 'mu_SZ', 'D', 'K_GX', 'K_GZ', 'K_IX', 'K_RS', 'K_RX', 'K_RZ', 'K_SX', 'K_SZ',
                                  'I_inf', 'V_l', 'V_m', 'Y_IX', 'Y_GX', 'Y_GZ', 'Y_RX', 'Y_RZ', 'Y_SX', 'Y_SZ', 'Il_0',
                                  'Gl_0', 'Sl_0', 'Rl_0', 'Xl_0', 'Zl_0', 'Gm_0', 'Sm_0', 'Rm_0', 'Xm_0', 'Zm_0']

    params_bounds_dict_pref['omega'] = [0.0, 10000.0]

    params_vals_omega_mucin, init_vals_omega_mucin, problem_omega_mucin = gen_samples(20, params_dict_pref,
                                                                                      init_cond_pref,
                                                                                      params_bounds_dict_pref,
                                                                                      init_cond_bounds_pref,
                                                                                      exclude_params_omega_mucin, 0, 1)

    varied_param_name_omega_mucin = 'omega'

    Y_ss_omega_mucin, Y_t_omega_mucin, Y_sol_omega_mucin, t_sol_omega_mucin, varied_param_list_omega_mucin = \
        run_timeseries_sim(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref, ex_list_pref,
                           internal_prod_list_pref, d_list_pref, vars_dict_pref, params_dict_pref,
                           param_dependencies_dict_pref, params_vals_omega_mucin, init_vals_omega_mucin, init_cond_pref,
                           varied_param_name_omega_mucin, 0, 1000, 10000)

    plot_biomass_ss(Y_ss_omega_mucin, varied_param_list_omega_mucin, vars_dict_pref, pref_latex_dict,
            'High Inflow Fiber Concentration ($I^\infty=50$)', 'Steady State Value (g/L)',
            'Inhibition Constant ($\omega$)', output_folder_pref_omega_comp_highfiber)

#######################################################################################################################

    # Varying omega, both species, with mucin, low fiber

    init_cond_pref = {'Il_0': 0.0, 'Gl_0': 0.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 0.5, 'Zl_0': 0.5, 'Gm_0': 0.0,
                      'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 0.0, 'Zm_0': 0.0}

    params_dict_pref['omega'] = 0.0
    params_dict_pref['lam_prod'] = 50.0
    params_dict_pref['lam_max'] = 500.0
    params_dict_pref['I_inf'] = 5.0
    params_dict_pref['D'] = 1.0071942760083039

    exclude_params_omega_mucin_lowfiber = ['alpha_X', 'alpha_Z', 'gam_aX', 'gam_aZ', 'gam_dR', 'gam_dS', 'gam_sG',
                                           'gam_sX',
                                           'gam_sZ', 'lam_max', 'lam_prod', 'mu_GX', 'mu_GZ', 'mu_IX', 'mu_RS', 'mu_RX',
                                           'mu_RZ', 'mu_SX', 'mu_SZ', 'D', 'K_GX', 'K_GZ', 'K_IX', 'K_RS', 'K_RX',
                                           'K_RZ',
                                           'K_SX', 'K_SZ', 'I_inf', 'V_l', 'V_m', 'Y_IX', 'Y_GX', 'Y_GZ', 'Y_RX',
                                           'Y_RZ',
                                           'Y_SX', 'Y_SZ', 'Il_0', 'Gl_0', 'Sl_0', 'Rl_0', 'Xl_0', 'Zl_0', 'Gm_0',
                                           'Sm_0',
                                           'Rm_0', 'Xm_0', 'Zm_0']

    params_bounds_dict_pref['omega'] = [0.0, 10000.0]

    params_vals_omega_mucin_lowfiber, init_vals_omega_mucin_lowfiber, problem_omega_mucin_lowfiber = \
        gen_samples(20, params_dict_pref, init_cond_pref, params_bounds_dict_pref, init_cond_bounds_pref,
                    exclude_params_omega_mucin_lowfiber, 0, 1)

    varied_param_name_omega_mucin_lowfiber = 'omega'

    Y_ss_omega_mucin_lowfiber, Y_t_omega_mucin_lowfiber, Y_sol_omega_mucin_lowfiber, t_sol_omega_mucin_lowfiber, \
        varied_param_list_omega_mucin_lowfiber = run_timeseries_sim(p_list_pref, inflow_amount_list_pref,
                                                                    dilution_rate_list_pref, phi_list_pref,
                                                                    ex_list_pref,
                                                                    internal_prod_list_pref, d_list_pref,
                                                                    vars_dict_pref,
                                                                    params_dict_pref, param_dependencies_dict_pref,
                                                                    params_vals_omega_mucin_lowfiber,
                                                                    init_vals_omega_mucin_lowfiber, init_cond_pref,
                                                                    varied_param_name_omega_mucin_lowfiber, 0, 1000,
                                                                    10000)

    plot_biomass_ss(Y_ss_omega_mucin_lowfiber, varied_param_list_omega_mucin_lowfiber, vars_dict_pref, pref_latex_dict,
                    'Low Inflow Fiber Concentration ($I^\infty = 5$)', 'Steady State Value (g/L)',
                    'Inhibition Constant ($\omega$)', output_folder_pref_omega_comp_lowfiber)

########################################################################################################################
# Section 3.2
########################################################################################################################

    # I - Varying omega, generalist only, no mucin, no inflow, can see switch

    init_cond_pref = {'Il_0': 50.0, 'Gl_0': 50.0, 'Sl_0': 0.0, 'Rl_0': 0.0, 'Xl_0': 10.0, 'Zl_0': 0.0, 'Gm_0': 0.0,
                      'Sm_0': 0.0, 'Rm_0': 0.0, 'Xm_0': 10.0, 'Zm_0': 0.0}

    params_dict_pref = {'alpha_X': 0.1,
                        'alpha_Z': 0.1,
                        'gam_aX': 0.1,
                        'gam_aZ': 0.1,
                        'gam_dR': 3.9,
                        'gam_dS': 3.9,
                        'gam_sG': 0.4,
                        'gam_sX': 0.4,
                        'gam_sZ': 0.4,
                        'lam_max': 50.0,
                        'lam_prod': 0.0,
                        'mu_GX': 10.619469027,
                        'mu_GZ': 10.619469027,
                        'mu_IX': 10.619469027,
                        'mu_RS': 12.627143363,
                        'mu_RX': 12.627143363,
                        'mu_RZ': 12.627143363,
                        'mu_SX': 12.627143363,
                        'mu_SZ': 12.627143363,
                        'omega': 1000,
                        'D': 0.0,
                        'K_GX': 0.26539823,
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

    exclude_params_omega_mucin_noinflowfiber = ['alpha_X', 'alpha_Z', 'gam_aX', 'gam_aZ', 'gam_dR', 'gam_dS', 'gam_sG',
                                                'gam_sX', 'gam_sZ', 'lam_max', 'lam_prod', 'mu_GX', 'mu_GZ', 'mu_IX',
                                                'mu_RS', 'mu_RX', 'mu_RZ', 'mu_SX', 'mu_SZ', 'D', 'K_GX', 'K_GZ',
                                                'K_IX',
                                                'K_RS', 'K_RX', 'K_RZ', 'K_SX', 'K_SZ', 'I_inf', 'V_l', 'V_m', 'Y_IX',
                                                'Y_GX', 'Y_GZ', 'Y_RX', 'Y_RZ', 'Y_SX', 'Y_SZ', 'Il_0', 'Gl_0', 'Sl_0',
                                                'Rl_0', 'Xl_0', 'Zl_0', 'Gm_0', 'Sm_0', 'Rm_0', 'Xm_0', 'Zm_0']

    params_bounds_dict_pref['omega'] = [0.0, 4.0]

    varied_param_name_omega_mucin_noinflowfiber = 'omega'

    params_vals_omega_mucin_noinflowfiber_2, init_vals_omega_mucin_noinflowfiber_2, \
        problem_single_omega_mucin_noinflowfiber_2 = gen_samples(3, params_dict_pref, init_cond_pref,
                                                                 params_bounds_dict_pref, init_cond_bounds_pref,
                                                                 exclude_params_omega_mucin_noinflowfiber, 0, 3)

    Y_ss_omega_mucin_noinflowfiber_2, Y_t_omega_mucin_noinflowfiber_2, Y_sol_omega_mucin_noinflowfiber_2, \
        t_sol_omega_mucin_noinflowfiber_2, varied_param_list_omega_mucin_noinflowfiber_2 = \
        run_timeseries_sim(p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref, ex_list_pref,
                           internal_prod_list_pref, d_list_pref, vars_dict_pref, params_dict_pref,
                           param_dependencies_dict_pref, params_vals_omega_mucin_noinflowfiber_2,
                           init_vals_omega_mucin_noinflowfiber_2, init_cond_pref,
                           varied_param_name_omega_mucin_noinflowfiber, 0, 2, 100)

    lines = ["-", "--", "-.", ":"]
    plot_title_string = ['Fiber in Lumen',
                         'Mucin in Lumen',
                         'Fiber-Derived Sugar in Lumen',
                         'Mucin-Derived Sugar in Lumen',
                         'Generalist in Lumen',
                         'Specialist in Lumen',
                         'Mucin in Mucus',
                         'Fiber-Derived Sugar in Mucus',
                         'Mucin-Derived Sugar in Mucus',
                         'Generalist in Mucus',
                         'Specialist in Mucus'
                         ]
    for key, value in vars_dict_pref.items():
        linecycler = cycle(lines)
        plt.figure()
        for index, item in enumerate(Y_sol_omega_mucin_noinflowfiber_2):
            legend_val = round(varied_param_list_omega_mucin_noinflowfiber_2[index], 2)
            if key == 'R_l':
                plt.plot(t_sol_omega_mucin_noinflowfiber_2[index], item[value], label=(pref_latex_dict[varied_param_name_omega_mucin_noinflowfiber] + '=' + str(legend_val)), linestyle=next(linecycler))
            else:
                plt.plot(t_sol_omega_mucin_noinflowfiber_2[index], item[value],
                         linestyle=next(linecycler))
        plt.legend(loc='best')
        plt.title(plot_title_string[value] + ' ' + '(' + pref_latex_dict[key] + ')')
        plt.ylabel('Concentration of ' + pref_latex_dict[key] + ' (g/L)')
        plt.xlabel('Time (Days)')
        # plt.xlim(xrange)
        plt.savefig(output_folder_pref_omega_nofiberflow + '/' + 'omega' + ' ' + '(' + key + ')' + '.pdf', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()

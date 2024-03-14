import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from InputModule.build_pref_model import build_pref_model
from InputModule.create_json_input import open_json_input_file
from PostProcess.random_forest import rf_read_sample_file
from PostProcess.timeseries_plot import convert_key_to_latex_pref, plot_ss, plot_biomass_ss, plot_subvar_ss
from SimulationModule.run_batch_sim import run_timeseries_sim
from SimulationModule.run_ode_sim import gen_samples


def main():
    # Load problem

    p_list_pref, d_list_pref, ex_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, internal_prod_list_pref, \
        phi_list_pref, vars_dict_pref, params_dict_pref, params_bounds_dict_pref, init_cond_pref, \
        init_cond_bounds_pref, param_dependencies_dict_pref = build_pref_model()

    pref_latex_dict = convert_key_to_latex_pref(params_dict_pref, init_cond_pref, vars_dict_pref)

    results_folder = '../../Dropbox/Results/PrefModel/SimData'
    outputfolder = '../../Dropbox/Results/PrefModel/ParamImportance/TimeSeries/I_inf_SS/'
    output_folder_pref_param_importance_freq = '../../Dropbox/Results/PrefModel/SimData/feature_importance/40bins'
    outputfolder_omega = outputfolder + 'Omega_Test'
    outputfolder_Y_SX = outputfolder + 'Y_SX'
    outputfolder_Y_RZ = outputfolder + 'Y_RZ'
    outputfolder_mu_IX = outputfolder + 'mu_IX'
    outputfolder_V_m = outputfolder + 'V_m'

########################################################################################################################

    # Initialize problem

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
    params_init_dict = {**init_cond_pref, **params_dict_pref}

########################################################################################################################

    # Read in samples

    param_init_vals = rf_read_sample_file(results_folder + '/sample_params.txt')
    Yss = open_json_input_file(results_folder + '/classes.json')

########################################################################################################################

    # Frequency histogram mu_IX - Figure 15 (b)

    param_pos = list(params_init_dict.keys()).index('mu_IX')
    x_mu_IX_vals = param_init_vals[:, param_pos]
    mu_IX_df = pd.DataFrame(list(zip(Yss, x_mu_IX_vals)), columns=['Class', 'Value'])
    mu_IX_bin = pd.qcut(mu_IX_df['Value'], q=40, precision=2)
    mu_IX_df['mu_IX'] = mu_IX_bin
    mu_IX_df = mu_IX_df.groupby(['Class', 'mu_IX'])
    mu_IX_df = mu_IX_df.size().unstack()
    mu_IX_df = mu_IX_df.transpose()
    interval_list = mu_IX_df.index.values.tolist()
    mid_interval = [(a.left + a.right)/2 for a in interval_list]
    mid_interval = np.around(mid_interval, decimals=1)
    ax = mu_IX_df.plot.bar(stacked=True, width=1.0) #, figsize=(20, 20))
    # ax.set_title('Distribution of Steady States')
    ax.set_xlabel('$\\kappa_{I,X}$')
    ax.set_ylabel('Frequency')
    ax.get_legend().remove()
    # ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=8,  handlelength=1, borderpad=0.8, labelspacing=0.8)
    ax.tick_params(axis='x')#, labelsize=8)
    ax.set_xticklabels(mid_interval, rotation='vertical', fontsize=11)
    ax.set_xticks(ax.get_xticks()[::2])
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.set_xticklabels(rotation='horizontal')
    plt.savefig(output_folder_pref_param_importance_freq + '/mu_IX_freq' + '.pdf', bbox_inches='tight')
    plt.close()

    # Frequency histogram I_inf - Figure 15 (a)

    param_pos = list(params_init_dict.keys()).index('I_inf')
    x_I_inf_vals = param_init_vals[:, param_pos]
    I_inf_df = pd.DataFrame(list(zip(Yss, x_I_inf_vals)), columns=['Class', 'Value'])
    I_inf_bin = pd.qcut(I_inf_df['Value'], q=40, precision=2)
    I_inf_df['I_inf'] = I_inf_bin
    I_inf_df = I_inf_df.groupby(['Class', 'I_inf'])
    I_inf_df = I_inf_df.size().unstack()
    I_inf_df = I_inf_df.transpose()
    interval_list_I_inf = I_inf_df.index.values.tolist()
    mid_interval_I_inf = [(a.left + a.right) / 2 for a in interval_list_I_inf]
    mid_interval_I_inf = np.around(mid_interval_I_inf, decimals=1)
    ax = I_inf_df.plot.bar(stacked=True, width=1.0)
    # ax.set_title('Distribution of Steady States')
    ax.set_xlabel('$I^\infty$')
    ax.set_ylabel('Frequency')
    ax.get_legend().remove()
    # ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=8,  handlelength=1, borderpad=0.8, labelspacing=0.8)
    ax.tick_params(axis='x')  # , labelsize=8)
    ax.set_xticklabels(mid_interval_I_inf, rotation='vertical', fontsize=11)
    ax.set_xticks(ax.get_xticks()[::2])
    plt.savefig(output_folder_pref_param_importance_freq + '/I_inf_freq' + '.pdf', bbox_inches='tight')
    plt.close()

    # Frequency histogram alpha_X - Figure 13 (a)

    param_pos = list(params_init_dict.keys()).index('alpha_X')
    x_alpha_X_vals = param_init_vals[:, param_pos]
    alpha_X_df = pd.DataFrame(list(zip(Yss, x_alpha_X_vals)), columns=['Class', 'Value'])
    alpha_X_bin = pd.qcut(alpha_X_df['Value'], q=40, precision=2)
    alpha_X_df['alpha_X'] = alpha_X_bin
    alpha_X_df = alpha_X_df.groupby(['Class', 'alpha_X'])
    alpha_X_df = alpha_X_df.size().unstack()
    alpha_X_df = alpha_X_df.transpose()
    interval_list_alpha_X = alpha_X_df.index.values.tolist()
    mid_interval_alpha_X = [(a.left + a.right) / 2 for a in interval_list_alpha_X]
    mid_interval_alpha_X = np.around(mid_interval_alpha_X, decimals=2)
    ax = alpha_X_df.plot.bar(stacked=True, width=1.0)
    # ax.set_title('Distribution of Steady States')
    ax.set_xlabel('$\\alpha_X$')
    ax.set_ylabel('Frequency')
    # ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=8,  handlelength=1, borderpad=0.8, labelspacing=0.8)
    ax.get_legend().remove()
    ax.tick_params(axis='x')  # , labelsize=8)
    ax.set_xticklabels(mid_interval_alpha_X, rotation='vertical', fontsize=11)
    ax.set_xticks(ax.get_xticks()[::2])
    plt.savefig(output_folder_pref_param_importance_freq + '/alpha_X_freq' + '.pdf', bbox_inches='tight')
    plt.close()

    # Frequency histogram alpha_Z - Figure 13 (b)

    param_pos = list(params_init_dict.keys()).index('alpha_Z')
    x_alpha_Z_vals = param_init_vals[:, param_pos]
    alpha_Z_df = pd.DataFrame(list(zip(Yss, x_alpha_Z_vals)), columns=['Class', 'Value'])
    alpha_Z_bin = pd.qcut(alpha_Z_df['Value'], q=40, precision=2)
    alpha_Z_df['alpha_Z'] = alpha_Z_bin
    alpha_Z_df = alpha_Z_df.groupby(['Class', 'alpha_Z'])
    alpha_Z_df = alpha_Z_df.size().unstack()
    alpha_Z_df = alpha_Z_df.transpose()
    interval_list_alpha_Z = alpha_Z_df.index.values.tolist()
    mid_interval_alpha_Z = [(a.left + a.right) / 2 for a in interval_list_alpha_Z]
    mid_interval_alpha_Z = np.around(mid_interval_alpha_Z, decimals=2)
    ax = alpha_Z_df.plot.bar(stacked=True, width=1.0)
    # ax.set_title('Distribution of Steady States')
    ax.set_xlabel('$\\alpha_Z$')
    ax.set_ylabel('Frequency')
    # ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=8,  handlelength=1, borderpad=0.8, labelspacing=0.8)
    ax.get_legend().remove()
    ax.tick_params(axis='x')  # , labelsize=8)
    ax.set_xticklabels(mid_interval_alpha_Z, rotation='vertical', fontsize=11)
    ax.set_xticks(ax.get_xticks()[::2])
    plt.savefig(output_folder_pref_param_importance_freq + '/alpha_Z_freq' + '.pdf', bbox_inches='tight')
    plt.close()

    # Frequency histogram Y_SX - Figure 14 (a)

    param_pos = list(params_init_dict.keys()).index('Y_SX')
    x_Y_SX_vals = param_init_vals[:, param_pos]
    Y_SX_df = pd.DataFrame(list(zip(Yss, x_Y_SX_vals)), columns=['Class', 'Value'])
    Y_SX_bin = pd.qcut(Y_SX_df['Value'], q=40, precision=3)
    Y_SX_df['Y_SX'] = Y_SX_bin
    Y_SX_df = Y_SX_df.groupby(['Class', 'Y_SX'])
    Y_SX_df = Y_SX_df.size().unstack()
    Y_SX_df = Y_SX_df.transpose()
    interval_list_Y_SX = Y_SX_df.index.values.tolist()
    mid_interval_Y_SX = [(a.left + a.right) / 2 for a in interval_list_Y_SX]
    mid_interval_Y_SX = np.around(mid_interval_Y_SX, decimals=2)
    ax = Y_SX_df.plot.bar(stacked=True, width=1.0)
    # ax.set_title('Distribution of Steady States')
    ax.set_xlabel('$Y_{S,X}$')
    ax.set_ylabel('Frequency')
    # ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=8,  handlelength=1, borderpad=0.8, labelspacing=0.8)
    ax.get_legend().remove()
    ax.tick_params(axis='x')  # , labelsize=8)
    ax.set_xticklabels(mid_interval_Y_SX, rotation='vertical', fontsize=11)
    ax.set_xticks(ax.get_xticks()[::2])
    plt.savefig(output_folder_pref_param_importance_freq + '/Y_SX_freq' + '.pdf', bbox_inches='tight')
    plt.close()

    # Frequency histogram Y_RZ - Figure 14 (b)

    param_pos = list(params_init_dict.keys()).index('Y_RZ')
    x_Y_RZ_vals = param_init_vals[:, param_pos]
    Y_RZ_df = pd.DataFrame(list(zip(Yss, x_Y_RZ_vals)), columns=['Class', 'Value'])
    Y_RZ_bin = pd.qcut(Y_RZ_df['Value'], q=40, precision=2)
    Y_RZ_df['Y_RZ'] = Y_RZ_bin
    Y_RZ_df = Y_RZ_df.groupby(['Class', 'Y_RZ'])
    Y_RZ_df = Y_RZ_df.size().unstack()
    Y_RZ_df = Y_RZ_df.transpose()
    interval_list_Y_RZ = Y_RZ_df.index.values.tolist()
    mid_interval_Y_RZ = [(a.left + a.right) / 2 for a in interval_list_Y_RZ]
    mid_interval_Y_RZ = np.around(mid_interval_Y_RZ, decimals=3)
    ax = Y_RZ_df.plot.bar(stacked=True, width=1.0)
    # ax.set_title('Distribution of Steady States')
    ax.set_xlabel('$Y_{R,Z}$')
    ax.set_ylabel('Frequency')
    # ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=8,  handlelength=1, borderpad=0.8, labelspacing=0.8)
    ax.get_legend().remove()
    ax.tick_params(axis='x')  # , labelsize=8)
    ax.set_xticklabels(mid_interval_Y_RZ, rotation='vertical', fontsize=11)
    ax.set_xticks(ax.get_xticks()[::2])
    plt.savefig(output_folder_pref_param_importance_freq + '/Y_RZ_freq' + '.pdf', bbox_inches='tight')
    plt.close()

    # Frequency histogram V_m - Figure 16

    param_pos = list(params_init_dict.keys()).index('V_m')
    x_V_m_vals = param_init_vals[:, param_pos]
    V_m_df = pd.DataFrame(list(zip(Yss, x_V_m_vals)), columns=['Class', 'Value'])
    V_m_bin = pd.qcut(V_m_df['Value'], q=40, precision=3)
    V_m_df['V_m'] = V_m_bin
    V_m_df = V_m_df.groupby(['Class', 'V_m'])
    V_m_df = V_m_df.size().unstack()
    V_m_df = V_m_df.transpose()
    interval_list_V_m = V_m_df.index.values.tolist()
    mid_interval_V_m = [(a.left + a.right) / 2 for a in interval_list_V_m]
    mid_interval_V_m = np.around(mid_interval_V_m, decimals=2)
    ax = V_m_df.plot.bar(stacked=True, width=1.0)
    # ax.set_title('Distribution of Steady States')
    ax.set_xlabel('$V_m$')
    ax.set_ylabel('Frequency')
    # ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=11,  handlelength=1, borderpad=0.8, labelspacing=0.8)
    ax.get_legend().remove()
    ax.tick_params(axis='x')  # , labelsize=8)
    ax.set_xticklabels(mid_interval_V_m, rotation='vertical', fontsize=11)
    ax.set_xticks(ax.get_xticks()[::2])
    plt.savefig(output_folder_pref_param_importance_freq + '/V_m_freq' + '.pdf', bbox_inches='tight')
    plt.close()


########################################################################################################################

    def high_low_param_sim_Iinf(init_cond_pref, params_dict_pref, param_name, param_val_array, plot_titles,
                                output_folder):
        exclude_params_I_inf = ['alpha_X', 'alpha_Z', 'gam_aX', 'gam_aZ', 'gam_dR', 'gam_dS', 'gam_sG', 'gam_sX',
                                'gam_sZ', 'lam_max', 'lam_prod', 'mu_GX', 'mu_GZ', 'mu_IX', 'mu_RS', 'mu_RX',
                                'mu_RZ', 'mu_SX', 'mu_SZ', 'omega', 'D', 'K_GX', 'K_GZ', 'K_IX', 'K_RS', 'K_RX',
                                'K_RZ', 'K_SX', 'K_SZ', 'V_l', 'V_m', 'Y_IX', 'Y_GX', 'Y_GZ', 'Y_RX', 'Y_RZ',
                                'Y_SX', 'Y_SZ', 'Il_0', 'Gl_0', 'Sl_0', 'Rl_0', 'Xl_0', 'Zl_0', 'Gm_0', 'Sm_0',
                                'Rm_0', 'Xm_0', 'Zm_0']

        params_bounds_dict_pref['I_inf'] = [0.01, 60.0]

        varied_param_name_I_inf = 'I_inf'

        for val, title in zip(param_val_array, plot_titles):
            params_dict_pref[param_name] = val
            param_dependencies_dict_pref = {"V_l": [1.0, "-", "V_m"]}

            params_vals_I_inf, init_vals_I_inf, problem_I_inf = gen_samples(50, params_dict_pref, init_cond_pref,
                                                                            params_bounds_dict_pref,
                                                                            init_cond_bounds_pref,
                                                                            exclude_params_I_inf, 0, 1)

            Y_ss_I_inf, Y_t_I_inf, Y_sol_I_inf, t_sol_I_inf, varied_param_list_I_inf = run_timeseries_sim(
                p_list_pref, inflow_amount_list_pref, dilution_rate_list_pref, phi_list_pref, ex_list_pref,
                internal_prod_list_pref, d_list_pref, vars_dict_pref, params_dict_pref, param_dependencies_dict_pref,
                params_vals_I_inf, init_vals_I_inf, init_cond_pref, varied_param_name_I_inf, 0, 10000,
                1000)

            # plot_subvar_ss(Y_ss_I_inf, varied_param_list_I_inf, vars_dict_pref, pref_latex_dict,
            #                 'Effect of $I^\infty$ on Steady State',
            #                 'Steady State Value (g/L)',
            #                 'Inflow Dietary Fiber Concentration $I^\infty$ (g/L)', output_folder)

            plot_biomass_ss(Y_ss_I_inf, varied_param_list_I_inf, vars_dict_pref, pref_latex_dict,
                            title,
                            'Steady State Value (g/L)',
                            'Inflow Dietary Fiber Concentration $I^\infty$ (g/L)', output_folder)

            plot_ss(Y_ss_I_inf, varied_param_list_I_inf, vars_dict_pref, pref_latex_dict,
                    title,
                    'Steady State Value (g/L)',
                    'Inflow Dietary Fiber Concentration $I^\infty$ (g/L)', output_folder)

    # Figure 5 - High and low omega steady states as a function of I_inf
    high_low_param_sim_Iinf(init_cond_pref, params_dict_pref, 'omega',
                            [100, 1000], ['Low Inhibition Constant ($\omega=100$)',
                                          'High Inhibition Constant ($\omega=1000$)'],
                            outputfolder_omega)

    # Figure 20 - High and low Y_RZ steady states as a function of I_inf
    high_low_param_sim_Iinf(init_cond_pref, params_dict_pref, 'Y_SX',
                            [0.2, 0.4], ['Low Yield Coefficient ($Y_{S,X}=0.2)',
                                         'High Yield Coefficient ($Y_{S,X}=0.4$)'],
                            outputfolder_Y_SX)

    # Figure 21 - High and low Y_RZ steady states as a function of I_inf
    high_low_param_sim_Iinf(init_cond_pref, params_dict_pref, 'Y_RZ',
                            [0.2, 0.4], ['Low Yield Coefficient ($Y_{R,Z}=0.2)',
                                         'High Yield Coefficient ($Y_{R,Z}=0.4$)'],
                            outputfolder_Y_RZ)

    # Figure 22 - High and low mu_IX steady states as a function of I_inf
    high_low_param_sim_Iinf(init_cond_pref, params_dict_pref, 'mu_IX',
                            [5.0, 15.0], ['Low Max Conversion Rate of Fiber ($I$) to Sugar ($S$) ($\kappa_{I,X}=5.0)',
                                         'High Max Conversion Rate of Riber ($I$) to Sugar ($S$) ($\kappa_{I,X}=15.0$)'],
                            outputfolder_mu_IX)

    # Figure 23 - High and low V_m steady states as a function of I_inf
    high_low_param_sim_Iinf(init_cond_pref, params_dict_pref, 'V_m',
                            [0.05, 0.15], ['Low Mucus Volume ($V_m=0.05$)',
                                         'High Mucus Volume ($V_m=0.15$)'],
                            outputfolder_V_m)


########################################################################################################################


if __name__ == "__main__":
    main()

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
import numpy as np
plt.rcParams.update({'font.size':14})
# plt.rc('legend', fontsize=20)
import os


def plot_ss(Y_ss, varied_param_list, vars_dict, latex_dict, plot_title_string, plot_ylabel_string, plot_xlabel_string,
            results_folder):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    # # legend_colors = matplotlib.colors.LinearSegmentedColormap.from_list("", ['C4', 'C5', 'C6', 'C7', 'C0', 'C1', 'C8', 'C9', 'C10', 'C2', 'C3'])
    # legend_colors = matplotlib.colors.LinearSegmentedColormap.from_list("", ['C4', 'C5', 'C6', 'C7', 'C0', 'C8', 'aqua', 'navy', 'C2'])
    legend_colors = matplotlib.colors.LinearSegmentedColormap.from_list("", ['C4', 'C5', 'C6', 'C7', 'C1', 'C8', 'aqua', 'navy', 'C3'])
    Y_ss_df = pd.DataFrame(Y_ss, index=varied_param_list, columns=[*var_latex.values()])
    Y_ss_df = Y_ss_df[Y_ss_df.columns.drop(list(Y_ss_df.filter(regex='X')))]
    ax = Y_ss_df.plot(title=plot_title_string, legend='best', colormap=legend_colors)
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '.pdf', bbox_inches='tight')
    plt.close()


def plot_biomass_ss(Y_ss, varied_param_list, vars_dict, latex_dict, plot_title_string, plot_ylabel_string,
                    plot_xlabel_string, results_folder):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    Y_ss_df = pd.DataFrame(Y_ss, index=varied_param_list, columns=[*var_latex.values()])
    # Y_ss_biomass_df = Y_ss_df.filter(regex='X|Z')
    Y_ss_biomass_df = Y_ss_df.filter(regex='X')
    legend_colors = matplotlib.colors.LinearSegmentedColormap.from_list("", ['C0', 'C2'])
    ax = Y_ss_biomass_df.plot(title=plot_title_string, legend='best', colormap=legend_colors)
    # ax.get_legend().remove()
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '_biomass.pdf', bbox_inches='tight')
    plt.close()


def plot_subvar_ss(Y_ss, varied_param_list, vars_dict, latex_dict, plot_title_string, plot_ylabel_string,
                    plot_xlabel_string, results_folder):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    Y_ss_df = pd.DataFrame(Y_ss, index=varied_param_list, columns=[*var_latex.values()])
    # Y_ss_biomass_df = Y_ss_df.filter(regex='X|Z')
    Y_ss_biomass_df = Y_ss_df.filter(regex='X|Z|G')
    legend_colors = matplotlib.colors.LinearSegmentedColormap.from_list("", ['C5', 'C0', 'C1', 'C8', 'C2', 'C3'])
    ax = Y_ss_biomass_df.plot(title=plot_title_string, legend='best', colormap=legend_colors)
    # ax.get_legend().remove()
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + 'test' + '_biomass.pdf', bbox_inches='tight')
    plt.close()


def plot_t2ss(Y_t, varied_param_list, plot_title_string, plot_ylabel_string, plot_xlabel_string, results_folder):
    # Y_t is a list of times to steady state for each param

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    Y_t_df = pd.DataFrame(Y_t, index=varied_param_list)
    ax = Y_t_df.plot(title=plot_title_string, legend=False, linestyle=next(linecycler))
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '.pdf', bbox_inches='tight')
    plt.close()


def plot_timeseries_groupby_param(Y_sol, t_sol, varied_param_list, varied_param_name, vars_dict, latex_dict, plot_title_string, plot_ylabel_string, plot_ylabel_units_string, plot_xlabel_string, results_folder):
    # Each index of Y_sol is the solution of one simulation, t_sol contains the corresponding time steps

    lines = ["-", "--", "-.", ":"]
    for key, value in vars_dict.items():
        linecycler = cycle(lines)
        plt.figure()
        for index, item in enumerate(Y_sol):
            legend_val = round(varied_param_list[index], 2)
            plt.plot(t_sol[index], item[value], label=(latex_dict[varied_param_name] + '=' + str(legend_val)), linestyle=next(linecycler))
        plt.legend(loc='best')
        plt.ylabel(plot_ylabel_string + 'of ' + latex_dict[key] + plot_ylabel_units_string)
        plt.xlabel(plot_xlabel_string)
        plt.title(plot_title_string + ' ' + '(' + latex_dict[key] + ')')
        # plt.xlim(xrange)
        plt.savefig(results_folder + '/' + plot_title_string + ' ' + '(' + key + ')' + '.pdf', bbox_inches='tight')
        plt.close()


def plot_timeseries_groupby_vars(Y_sol, t_sol, varied_param_list, varied_param_name, vars_dict, latex_dict, plot_title_string, plot_ylabel_string, plot_xlabel_string, results_folder):

    lines = ["-", "--", "-.", ":"]
    for index, item in enumerate(Y_sol):
        linecycler = cycle(lines)
        plt.figure()
        for key, value in vars_dict.items():
            plt.plot(t_sol[index], item[value], label=latex_dict[key], linestyle=next(linecycler))
        plt.legend(loc='best')
        plt.ylabel(plot_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.title(plot_title_string + ' ' + latex_dict[varied_param_name] + '=' + str(round(varied_param_list[index], 1)))
        plt.savefig(results_folder + '/' + plot_title_string + '(' + varied_param_name + '=' + str(varied_param_list[index]) + ')' + '.pdf', bbox_inches='tight')
        plt.close()


def convert_key_to_latex_single(params_dict, init_cond_dict, vars_dict):
    latex_dict = {}
    for key, value in params_dict.items():
        if key == 'mu_S':
            latex_dict[key] = '$\kappa_S$'
        if key == 'mu_I':
            latex_dict[key] = '$\kappa_I$'
        if key == 'K_S':
            latex_dict[key] = '$K_S$'
        if key == 'K_I':
            latex_dict[key] = '$K_I$'
        if key == 'D':
            latex_dict[key] = '$D$'
        if key == 'Y_S':
            latex_dict[key] = '$Y_S$'
        if key == 'Y_I':
            latex_dict[key] = '$Y_I$'
        if key == 'I_inf':
            latex_dict[key] = '$I^\infty$'
        if key == 'alpha':
            latex_dict[key] = '$\\alpha$'
    for key, value in init_cond_dict.items():
        if key == 'I_0':
            latex_dict[key] = '$I^0$'
        if key == 'S_0':
            latex_dict[key] = '$S^0$'
        if key == 'X_0':
            latex_dict[key] = '$X^0$'
    for key, value in vars_dict.items():
        if key == 'I':
            latex_dict[key] = '$I$'
        if key == 'S':
            latex_dict[key] = '$S$'
        if key == 'X':
            latex_dict[key] = '$X$'
    return latex_dict


def convert_key_to_latex_double(params_dict, init_cond_dict, vars_dict):
    latex_dict = {}
    for key, value in params_dict.items():
        if key == 'mu_S':
            latex_dict[key] = '$\kappa_S$'
        if key == 'mu_I':
            latex_dict[key] = '$\kappa_I$'
        if key == 'K_S':
            latex_dict[key] = '$K_S$'
        if key == 'K_I':
            latex_dict[key] = '$K_I$'
        if key == 'D':
            latex_dict[key] = '$D$'
        if key == 'Y_S':
            latex_dict[key] = '$Y_S$'
        if key == 'Y_I':
            latex_dict[key] = '$Y_I$'
        if key == 'I_inf':
            latex_dict[key] = '$I^\infty$'
        if key == 'alpha':
            latex_dict[key] = '$\\alpha$'
        if key == 'V_l':
            latex_dict[key] = '$V_l$'
        if key == 'V_m':
            latex_dict[key] = '$V_m$'
        if key == 'gam_a_X':
            latex_dict[key] = '$\gamma_{a,X}$'
        if key == 'gam_d':
            latex_dict[key] = '$\gamma_d$'
        if key == 'gam_s_I':
            latex_dict[key] = '$\gamma_{s,I}$'
        if key == 'gam_s_X':
            latex_dict[key] = '$\gamma_{s,X}$'
        if key == 'lam_max':
            latex_dict[key] = '$\Gamma_{max}$'
        if key == 'lam_prod':
            latex_dict[key] = '$\Gamma_{prod}$'
    for key, value in init_cond_dict.items():
        if key == 'Il_0':
            latex_dict[key] = '$I^0_l$'
        if key == 'Sl_0':
            latex_dict[key] = '$S^0_l$'
        if key == 'Xl_0':
            latex_dict[key] = '$X^0_l$'
        if key == 'Im_0':
            latex_dict[key] = '$I^0_m$'
        if key == 'Sm_0':
            latex_dict[key] = '$S^0_m$'
        if key == 'Xm_0':
            latex_dict[key] = '$X^0_m$'
    for key, value in vars_dict.items():
        if key == 'I_l':
            latex_dict[key] = '$I_l$'
        if key == 'S_l':
            latex_dict[key] = '$S_l$'
        if key == 'X_l':
            latex_dict[key] = '$X_l$'
        if key == 'I_m':
            latex_dict[key] = '$I_m$'
        if key == 'S_m':
            latex_dict[key] = '$S_m$'
        if key == 'X_m':
            latex_dict[key] = '$X_m$'
    return latex_dict


def convert_key_to_latex_pref(params_dict, init_cond_dict, vars_dict):
    latex_dict = {}
    for key, value in params_dict.items():
        if key == 'alpha_X':
            latex_dict[key] = '$\\alpha_X$'
        if key == 'alpha_Z':
            latex_dict[key] = '$\\alpha_Z$'
        if key == 'gam_aX':
            latex_dict[key] = '$\gamma_{a,X}$'
        if key == 'gam_aZ':
            latex_dict[key] = '$\gamma_{a,Z}$'
        if key == 'gam_dR':
            latex_dict[key] = '$\gamma_{d,R}$'
        if key == 'gam_dS':
            latex_dict[key] = '$\gamma_{d,S}$'
        if key == 'gam_sG':
            latex_dict[key] = '$\gamma_{s,G}$'
        if key == 'gam_sX':
            latex_dict[key] = '$\gamma_{s,X}$'
        if key == 'gam_sZ':
            latex_dict[key] = '$\gamma_{s,Z}$'
        if key == 'lam_max':
            latex_dict[key] = '$\Gamma_{max}$'
        if key == 'lam_prod':
            latex_dict[key] = '$\Gamma_{prod}$'
        if key == 'mu_GX':
            latex_dict[key] = '$\kappa_{G,X}$'
        if key == 'mu_GZ':
            latex_dict[key] = '$\kappa_{G,Z}$'
        if key == 'mu_IX':
            latex_dict[key] = '$\kappa_{I,X}$'
        if key == 'mu_RS':
            latex_dict[key] = '$\kappa_{R,S}$'
        if key == 'mu_RX':
            latex_dict[key] = '$\kappa_{R,X}$'
        if key == 'mu_RZ':
            latex_dict[key] = '$\kappa_{R,Z}$'
        if key == 'mu_SX':
            latex_dict[key] = '$\kappa_{S,X}$'
        if key == 'mu_SZ':
            latex_dict[key] = '$\kappa_{S,Z}$'
        if key == 'omega':
            latex_dict[key] = '$\omega$'
        if key == 'D':
            latex_dict[key] = '$D$'
        if key == 'K_GX':
            latex_dict[key] = '$K_{G,X}$'
        if key == 'K_GZ':
            latex_dict[key] = '$K_{G,Z}$'
        if key == 'K_IX':
            latex_dict[key] = '$K_{I,X}$'
        if key == 'K_RS':
            latex_dict[key] = '$K_{R,S}$'
        if key == 'K_RX':
            latex_dict[key] = '$K_{R,X}$'
        if key == 'K_RZ':
            latex_dict[key] = '$K_{R,Z}$'
        if key == 'K_SX':
            latex_dict[key] = '$K_{S,X}$'
        if key == 'K_SZ':
            latex_dict[key] = '$K_{S,Z}$'
        if key == 'I_inf':
            latex_dict[key] = '$I_{\infty}$'
        if key == 'V_l':
            latex_dict[key] = '$V_l$'
        if key == 'V_m':
            latex_dict[key] = '$V_m$'
        if key == 'Y_IX':
            latex_dict[key] = '$Y_{I,X}$'
        if key == 'Y_GX':
            latex_dict[key] = '$Y_{G,X}$'
        if key == 'Y_GZ':
            latex_dict[key] = '$Y_{G,Z}$'
        if key == 'Y_RX':
            latex_dict[key] = '$Y_{R,X}$'
        if key == 'Y_RZ':
            latex_dict[key] = '$Y_{R,Z}$'
        if key == 'Y_SX':
            latex_dict[key] = '$Y_{S,X}$'
        if key == 'Y_SZ':
            latex_dict[key] = '$Y_{S,Z}$'
    for key, value in init_cond_dict.items():
        if key == 'Il_0':
            latex_dict[key] = '$I^0_l$'
        if key == 'Gl_0':
            latex_dict[key] = '$G^0_l$'
        if key == 'Sl_0':
            latex_dict[key] = '$S^0_l$'
        if key == 'Rl_0':
            latex_dict[key] = '$R^0_l$'
        if key == 'Xl_0':
            latex_dict[key] = '$X^0_l$'
        if key == 'Zl_0':
            latex_dict[key] = '$Z^0_l$'
        if key == 'Gm_0':
            latex_dict[key] = '$G^0_m$'
        if key == 'Sm_0':
            latex_dict[key] = '$S^0_m$'
        if key == 'Rm_0':
            latex_dict[key] = '$R^0_m$'
        if key == 'Xm_0':
            latex_dict[key] = '$X^0_m$'
        if key == 'Zm_0':
            latex_dict[key] = '$Z^0_m$'
    for key, value in vars_dict.items():
        if key == 'I_l':
            latex_dict[key] = '$I_l$'
        if key == 'G_l':
            latex_dict[key] = '$G_l$'
        if key == 'S_l':
            latex_dict[key] = '$S_l$'
        if key == 'R_l':
            latex_dict[key] = '$R_l$'
        if key == 'X_l':
            latex_dict[key] = '$X_l$'
        if key == 'Z_l':
            latex_dict[key] = '$Z_l$'
        if key == 'G_m':
            latex_dict[key] = '$G_m$'
        if key == 'S_m':
            latex_dict[key] = '$S_m$'
        if key == 'R_m':
            latex_dict[key] = '$R_m$'
        if key == 'X_m':
            latex_dict[key] = '$X_m$'
        if key == 'Z_m':
            latex_dict[key] = '$Z_m$'

    return latex_dict


def convert_key_to_latex_ant(params_dict, init_cond_dict, vars_dict):
    latex_dict = {}
    for key, value in params_dict.items():
        if key == 'alpha_X':
            latex_dict[key] = '$\\alpha_X$'
        if key == 'alpha_Z':
            latex_dict[key] = '$\\alpha_Z$'
        if key == 'gam_aA':
            latex_dict[key] = '$\gamma_{a,A}$'
        if key == 'gam_aX':
            latex_dict[key] = '$\gamma_{a,X}$'
        if key == 'gam_aZ':
            latex_dict[key] = '$\gamma_{a,Z}$'
        if key == 'gam_dA':
            latex_dict[key] = '$\gamma_{d,A}$'
        if key == 'gam_dR':
            latex_dict[key] = '$\gamma_{d,R}$'
        if key == 'gam_dS':
            latex_dict[key] = '$\gamma_{d,S}$'
        if key == 'gam_sA':
            latex_dict[key] = '$\gamma_{s,A}$'
        if key == 'gam_sG':
            latex_dict[key] = '$\gamma_{s,G}$'
        if key == 'gam_sX':
            latex_dict[key] = '$\gamma_{s,X}$'
        if key == 'gam_sZ':
            latex_dict[key] = '$\gamma_{s,Z}$'
        if key == 'lam_max':
            latex_dict[key] = '$\Gamma_{max}$'
        if key == 'lam_prod':
            latex_dict[key] = '$\Gamma_{prod}$'
        if key == 'mu_GX':
            latex_dict[key] = '$\kappa_{G,X}$'
        if key == 'mu_GZ':
            latex_dict[key] = '$\kappa_{G,Z}$'
        if key == 'mu_IX':
            latex_dict[key] = '$\kappa_{I,X}$'
        if key == 'mu_RS':
            latex_dict[key] = '$\kappa_{R,S}$'
        if key == 'mu_RX':
            latex_dict[key] = '$\kappa_{R,X}$'
        if key == 'mu_RZ':
            latex_dict[key] = '$\kappa_{R,Z}$'
        if key == 'mu_SX':
            latex_dict[key] = '$\kappa_{S,X}$'
        if key == 'mu_SZ':
            latex_dict[key] = '$\kappa_{S,Z}$'
        if key == 'omega':
            latex_dict[key] = '$\omega$'
        if key == 'A_in':
            latex_dict[key] = '$A_{in}$'
        if key == 'D':
            latex_dict[key] = '$D$'
        if key == 'K_AX':
            latex_dict[key] = '$K_{A,X}$'
        if key == 'K_AZ':
            latex_dict[key] = '$K_{A,Z}$'
        if key == 'K_GX':
            latex_dict[key] = '$K_{G,X}$'
        if key == 'K_GZ':
            latex_dict[key] = '$K_{G,Z}$'
        if key == 'K_IX':
            latex_dict[key] = '$K_{I,X}$'
        if key == 'K_RS':
            latex_dict[key] = '$K_{R,S}$'
        if key == 'K_RX':
            latex_dict[key] = '$K_{R,X}$'
        if key == 'K_RZ':
            latex_dict[key] = '$K_{R,Z}$'
        if key == 'K_SX':
            latex_dict[key] = '$K_{S,X}$'
        if key == 'K_SZ':
            latex_dict[key] = '$K_{S,Z}$'
        if key == 'I_inf':
            latex_dict[key] = '$I_{\infty}$'
        if key == 'V_l':
            latex_dict[key] = '$V_l$'
        if key == 'V_m':
            latex_dict[key] = '$V_m$'
        if key == 'Y_IX':
            latex_dict[key] = '$Y_{I,X}$'
        if key == 'Y_GX':
            latex_dict[key] = '$Y_{G,X}$'
        if key == 'Y_GZ':
            latex_dict[key] = '$Y_{G,Z}$'
        if key == 'Y_RX':
            latex_dict[key] = '$Y_{R,X}$'
        if key == 'Y_RZ':
            latex_dict[key] = '$Y_{R,Z}$'
        if key == 'Y_SX':
            latex_dict[key] = '$Y_{S,X}$'
        if key == 'Y_SZ':
            latex_dict[key] = '$Y_{S,Z}$'
    for key, value in init_cond_dict.items():
        if key == 'Il_0':
            latex_dict[key] = '$I^0_l$'
        if key == 'Gl_0':
            latex_dict[key] = '$G^0_l$'
        if key == 'Sl_0':
            latex_dict[key] = '$S^0_l$'
        if key == 'Rl_0':
            latex_dict[key] = '$R^0_l$'
        if key == 'Xl_0':
            latex_dict[key] = '$X^0_l$'
        if key == 'Zl_0':
            latex_dict[key] = '$Z^0_l$'
        if key == 'Al_0':
            latex_dict[key] = '$A^0_l$'
        if key == 'Gm_0':
            latex_dict[key] = '$G^0_m$'
        if key == 'Sm_0':
            latex_dict[key] = '$S^0_m$'
        if key == 'Rm_0':
            latex_dict[key] = '$R^0_m$'
        if key == 'Xm_0':
            latex_dict[key] = '$X^0_m$'
        if key == 'Zm_0':
            latex_dict[key] = '$Z^0_m$'
        if key == 'Am_0':
            latex_dict[key] = '$A^0_m$'
    for key, value in vars_dict.items():
        if key == 'I_l':
            latex_dict[key] = '$I_l$'
        if key == 'G_l':
            latex_dict[key] = '$G_l$'
        if key == 'S_l':
            latex_dict[key] = '$S_l$'
        if key == 'R_l':
            latex_dict[key] = '$R_l$'
        if key == 'X_l':
            latex_dict[key] = '$X_l$'
        if key == 'Z_l':
            latex_dict[key] = '$Z_l$'
        if key == 'A_l':
            latex_dict[key] = '$A_l$'
        if key == 'G_m':
            latex_dict[key] = '$G_m$'
        if key == 'S_m':
            latex_dict[key] = '$S_m$'
        if key == 'R_m':
            latex_dict[key] = '$R_m$'
        if key == 'X_m':
            latex_dict[key] = '$X_m$'
        if key == 'Z_m':
            latex_dict[key] = '$Z_m$'
        if key == 'A_m':
            latex_dict[key] = '$A_m$'
    return latex_dict

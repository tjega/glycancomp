import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

plt.rcParams.update({'font.size': 14})
from matplotlib.offsetbox import AnchoredText


def plot_t2_recovery(Y_ss, varied_param_list, vars_dict, latex_dict, plot_title_string, plot_ylabel_string,
                     plot_xlabel_string, results_folder):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    Y_ss_df = pd.DataFrame(Y_ss, index=varied_param_list, columns=[*var_latex.values()])
    ax = Y_ss_df.plot(title=plot_title_string, legend='best', colormap='tab20')
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '.pdf', bbox_inches='tight')
    plt.close()


def plot_min_val(Y_ss, varied_param_list, vars_dict, latex_dict, plot_title_string, plot_ylabel_string,
                 plot_xlabel_string,
                 results_folder):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    Y_ss_df = pd.DataFrame(Y_ss, index=varied_param_list, columns=[*var_latex.values()])
    ax = Y_ss_df.plot(title=plot_title_string, legend='best', colormap='tab20')
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '.pdf', bbox_inches='tight')
    plt.close()


def critical_KA(Y_ss, varied_param_list, vars_dict, latex_dict, varied_param_name, crit_vars_name, plot_title_string,
                plot_ylabel_string,
                plot_xlabel_string, results_folder):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    Y_ss_df = pd.DataFrame(Y_ss, index=varied_param_list, columns=[*var_latex.values()])
    Y_ss_df = Y_ss_df.filter(regex=crit_vars_name)
    critical_KA_lumen = varied_param_list[(Y_ss[:, vars_dict[crit_vars_name + '_l']] > 0.0001).argmax(axis=0)]
    critical_KA_mucus = varied_param_list[(Y_ss[:, vars_dict[crit_vars_name + '_m']] > 0.0001).argmax(axis=0)]
    # anchored_text = AnchoredText('$K_{A,X}$ = 0.0000001', loc=1)
    anchored_text = AnchoredText(latex_dict[varied_param_name] + '=' + str(round(critical_KA_lumen, 3)), loc=1)
    ax = Y_ss_df.plot(title=plot_title_string, legend='best')  # , colormap='Set1')
    ax.add_artist(anchored_text)
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '.pdf', bbox_inches='tight')
    plt.close()


def critical_KA_XZ(Y_ss, varied_param_list, vars_dict, latex_dict, crit_vars_name,
                   plot_title_string, plot_ylabel_string, plot_xlabel_string, results_folder):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    Y_ss_df = pd.DataFrame(Y_ss, index=varied_param_list, columns=[*var_latex.values()])
    Y_ss_X_df = Y_ss_df.filter(regex='X')
    Y_ss_Z_df = Y_ss_df.filter(regex='Z')
    critical_KA_Xl = varied_param_list[(Y_ss[:, vars_dict['X_l']] > 0.0001).argmax(axis=0)]
    critical_KA_Xm = varied_param_list[(Y_ss[:, vars_dict['X_m']] > 0.0001).argmax(axis=0)]
    critical_KA_Zl = varied_param_list[(Y_ss[:, vars_dict['Z_l']] > 0.0001).argmax(axis=0)]
    critical_KA_Zm = varied_param_list[(Y_ss[:, vars_dict['Z_m']] > 0.0001).argmax(axis=0)]
    if crit_vars_name == 'X':
        # anchored_text = AnchoredText('$K_{A,X}$ = 0.0000001', loc=1)
        anchored_text = AnchoredText('$K_{A,X}$ = ' + str(round(critical_KA_Xl, 3)), loc=1)
    elif crit_vars_name == 'Z':
        # anchored_text = AnchoredText('$K_{A,X}$ = 0.0000001', loc=1)
        anchored_text = AnchoredText('$K_{A,Z}$ = ' + str(round(critical_KA_Zl, 3)), loc=1)
    elif crit_vars_name == 'both':
        crit_KA = min(critical_KA_Xl, critical_KA_Zl)
        # anchored_text = AnchoredText('$K_{A,X}$ = 0.0000001', loc=1)
        anchored_text = AnchoredText('$K_{A}$ = ' + str(round(crit_KA, 3)), loc=1)
    df_plot = pd.concat([Y_ss_X_df, Y_ss_Z_df], axis=1)
    ax = df_plot.plot(title=plot_title_string, legend='best')
    ax.add_artist(anchored_text)
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '.pdf', bbox_inches='tight')
    plt.close()


def ant_recovery_XY(Y_sol, t_sol, ant_start, varied_param_list, vars_dict, latex_dict, varied_param_name,
                    recovery_vars_name,
                    plot_title_string_rec, plot_rec_ylabel_string, plot_title_string_min, plot_min_ylabel_string,
                    plot_title_string_tmin, plot_tmin_ylabel_string, plot_title_string_perdiff,
                    plot_perdiff_ylabel_string, plot_xlabel_string, results_folder, plot_vals=None, pre_start=None):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    t_df = pd.DataFrame(t_sol[0])
    ant_index = t_df[t_df >= ant_start].first_valid_index()
    pre_start_index = t_df[t_df >= pre_start].first_valid_index()
    t2_rec_Xl_list = []
    t2_rec_Xm_list = []
    t2_min_Xl_list = []
    t2_min_Xm_list = []
    min_Xl_list = []
    min_Xm_list = []
    min_perdiff_Xl_list = []
    min_perdiff_Xm_list = []
    t2_rec_Zl_list = []
    t2_rec_Zm_list = []
    t2_min_Zl_list = []
    t2_min_Zm_list = []
    min_Zl_list = []
    min_Zm_list = []
    min_perdiff_Zl_list = []
    min_perdiff_Zm_list = []
    for sol in Y_sol:
        sol = sol.transpose()
        sol_df = pd.DataFrame(sol, columns=[*var_latex.values()])
        sol_X_df = sol_df.filter(regex='X')
        sol_Z_df = sol_df.filter(regex='Z')
        sol_X_df = pd.DataFrame(sol_X_df)
        sol_Z_df = pd.DataFrame(sol_Z_df)
        ss_Xl = sol_X_df['$X_l$'].iloc[-1]
        ss_Xm = sol_X_df['$X_m$'].iloc[-1]
        ss_Zl = sol_Z_df['$Z_l$'].iloc[-1]
        ss_Zm = sol_Z_df['$Z_m$'].iloc[-1]
        # sol_X_df['t'] = t_sol[0]
        # sol_Z_df['t'] = t_sol[0]
        if sol_df['$X_l$'].iloc[-1] < 0.000001 and sol_df['$X_m$'].iloc[-1] < 0.000001:
            t2_recovery_Xl = 0.0
            t2_recovery_Xm = 0.0
            min_val_Xl = 0.0
            min_val_Xm = 0.0
            min_perdiff_Xl = 0.0
            min_perdiff_Xm = 0.0
            t2_min_Xl_list.append(0.0)
            t2_min_Xm_list.append(0.0)
        else:
            min_index_Xl = sol_X_df['$X_l$'].idxmin()
            min_index_Xm = sol_X_df['$X_m$'].idxmin()
            min_val_Xl = sol_X_df['$X_l$'].min()
            min_val_Xm = sol_X_df['$X_m$'].min()
            min_perdiff_Xl = ((ss_Xl - min_val_Xl) / ss_Xl) * 100
            min_perdiff_Xm = ((ss_Xm - min_val_Xm) / ss_Xm) * 100
            sol_X_df = ((sol_X_df - sol_X_df.iloc[ant_index]).abs()) / ((sol_X_df + sol_X_df.iloc[ant_index]) / 2)
            # ss_X = sol_X_df.iloc[ant_index]
            sol_X_df['t'] = t_sol[0]
            t2_min_Xl_list.append(sol_X_df['t'][min_index_Xl])
            t2_min_Xm_list.append(sol_X_df['t'][min_index_Xm])
            sol_X_df = sol_X_df[(sol_X_df['t'] > min_index_Xm).idxmax():]
            recovery_index_Xl = sol_X_df[sol_X_df['$X_l$'] < 0.01].first_valid_index()
            recovery_index_Xm = sol_X_df[sol_X_df['$X_m$'] < 0.01].first_valid_index()
            # recovery_index_Xl = sol_X_df[sol_X_df['$X_l$'] > (0.99*ss_X['$X_l$'])].first_valid_index()
            # recovery_index_Xm = sol_X_df[sol_X_df['$X_m$'] > (0.99*ss_X['$X_m$'])].first_valid_index()
            if recovery_index_Xl is None:
                t2_recovery_Xl = 0.0
            else:
                t2_recovery_Xl = sol_X_df['t'][recovery_index_Xl] - (ant_start)
            if recovery_index_Xm is None:
                t2_recovery_Xm = 0.0
            else:
                t2_recovery_Xm = sol_X_df['t'][recovery_index_Xm] - (ant_start)
        if sol_df['$Z_l$'].iloc[-1] < 0.000001 and sol_df['$Z_m$'].iloc[-1] < 0.000001:
            t2_recovery_Zl = 0.0
            t2_recovery_Zm = 0.0
            min_val_Zl = 0.0
            min_val_Zm = 0.0
            min_perdiff_Zl = 0.0
            min_perdiff_Zm = 0.0
            t2_min_Zl_list.append(0.0)
            t2_min_Zm_list.append(0.0)
        else:
            min_index_Zl = sol_Z_df['$Z_l$'].idxmin()
            min_index_Zm = sol_Z_df['$Z_m$'].idxmin()
            min_val_Zl = sol_Z_df['$Z_l$'].min()
            min_val_Zm = sol_Z_df['$Z_m$'].min()
            min_perdiff_Zl = ((ss_Zl - min_val_Zl) / ss_Zl) * 100
            min_perdiff_Zm = ((ss_Zm - min_val_Zm) / ss_Zm) * 100
            # ss_Z = sol_Z_df.iloc[ant_index]
            sol_Z_df = ((sol_Z_df - sol_Z_df.iloc[ant_index]).abs()) / ((sol_Z_df + sol_Z_df.iloc[ant_index]) / 2)
            sol_Z_df['t'] = t_sol[0]
            t2_min_Zl_list.append(sol_Z_df['t'][min_index_Zl])
            t2_min_Zm_list.append(sol_Z_df['t'][min_index_Zm])
            sol_Z_df = sol_Z_df[(sol_Z_df['t'] > min_index_Zm).idxmax():]
            recovery_index_Zl = sol_Z_df[sol_Z_df['$Z_l$'] < 0.01].first_valid_index()
            recovery_index_Zm = sol_Z_df[sol_Z_df['$Z_m$'] < 0.01].first_valid_index()
            # recovery_index_Zl = sol_Z_df[sol_Z_df['$Z_l$'] > (0.99*ss_Z['$Z_l$'])].first_valid_index()
            # recovery_index_Zm = sol_Z_df[sol_Z_df['$Z_m$'] > (0.99*ss_Z['$Z_m$'])].first_valid_index()
            if recovery_index_Zl is None:
                t2_recovery_Zl = 0.0
            else:
                t2_recovery_Zl = sol_Z_df['t'][recovery_index_Zl] - (ant_start)
            if recovery_index_Zm is None:
                t2_recovery_Zm = 0.0
            else:
                t2_recovery_Zm = sol_Z_df['t'][recovery_index_Zm] - (ant_start)
        min_Xl_list.append(min_val_Xl)
        min_Xm_list.append(min_val_Xm)
        t2_rec_Xl_list.append(t2_recovery_Xl)
        t2_rec_Xm_list.append(t2_recovery_Xm)
        min_perdiff_Xl_list.append(min_perdiff_Xl)
        min_perdiff_Xm_list.append(min_perdiff_Xm)
        min_Zl_list.append(min_val_Zl)
        min_Zm_list.append(min_val_Zm)
        t2_rec_Zl_list.append(t2_recovery_Zl)
        t2_rec_Zm_list.append(t2_recovery_Zm)
        min_perdiff_Zl_list.append(min_perdiff_Zl)
        min_perdiff_Zm_list.append(min_perdiff_Zm)

    # Plot time to recovery
    if recovery_vars_name == 'both':
        x_name = '$K_A$'
    else:
        x_name = latex_dict.get(varied_param_name, varied_param_name)

    if plot_vals is None:
        plot_list_rec = np.column_stack([varied_param_list, t2_rec_Xl_list, t2_rec_Xm_list, t2_rec_Zl_list, t2_rec_Zm_list])
        plot_df_rec = pd.DataFrame(plot_list_rec,
                                   columns=[x_name, '$X_l$', '$X_m$', '$Z_l$', '$Z_m$'])
        plot_df_rec = plot_df_rec.set_index(x_name)
        ax = plot_df_rec.plot(title=plot_title_string_rec, legend='best', colormap='tab20')
        plt.ylabel(plot_rec_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_rec + '.pdf', bbox_inches='tight')
        plt.close()

        # Plot time to minimum
        plot_list = np.column_stack([varied_param_list, t2_min_Xl_list, t2_min_Xm_list, t2_min_Zl_list, t2_min_Zm_list])
        plot_df = pd.DataFrame(plot_list, columns=[x_name, '$X_l$', '$X_m$', '$Z_l$', '$Z_m$'])
        plot_df = plot_df.set_index(x_name)
        ax = plot_df.plot(title=plot_title_string_tmin, legend='best', colormap='tab20')
        plt.ylabel(plot_tmin_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_tmin + '.pdf', bbox_inches='tight')
        plt.close()

        # Plot minimum value
        plot_list = np.column_stack([varied_param_list, min_Xl_list, min_Xm_list, min_Zl_list, min_Zm_list])
        plot_df = pd.DataFrame(plot_list, columns=[x_name, '$X_l$', '$X_m$', '$Z_l$', '$Z_m$'])
        plot_df = plot_df.set_index(x_name)
        ax = plot_df.plot(title=plot_title_string_min, legend='best', colormap='tab20')
        plt.ylabel(plot_min_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_min + '.pdf', bbox_inches='tight')
        plt.close()

        # Plot percent decrease
        plot_list = np.column_stack(
            [varied_param_list, min_perdiff_Xl_list, min_perdiff_Xm_list, min_perdiff_Zl_list, min_perdiff_Zm_list])
        plot_df = pd.DataFrame(plot_list, columns=[x_name, '$X_l$', '$X_m$', '$Z_l$', '$Z_m$'])
        plot_df = plot_df.set_index(x_name)
        ax = plot_df.plot(title=plot_title_string_perdiff, legend='best', colormap='tab20')
        plt.ylabel(plot_perdiff_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_perdiff + '.pdf', bbox_inches='tight')
        plt.close()

    return min_perdiff_Xl_list, min_perdiff_Xm_list, min_perdiff_Zl_list, min_perdiff_Zm_list


def ant_recovery(Y_sol, t_sol, ant_start, varied_param_list, vars_dict, latex_dict, varied_param_name,
                 recovery_vars_name, plot_title_string_rec, plot_rec_ylabel_string, plot_title_string_min,
                 plot_min_ylabel_string, plot_title_string_tmin, plot_tmin_ylabel_string, plot_title_string_perdiff,
                 plot_perdiff_ylabel_string, plot_xlabel_string, results_folder, plot_vals=None, pre_start=None):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    t_df = pd.DataFrame(t_sol[0])
    ant_index = t_df[t_df > ant_start].first_valid_index()
    pre_start_index = t_df[t_df > pre_start].first_valid_index()
    t2_rec_lumen_list = []
    t2_rec_mucus_list = []
    t2_min_lumen_list = []
    t2_min_mucus_list = []
    min_lumen_list = []
    min_mucus_list = []
    min_perdiff_lumen_list = []
    min_perdiff_mucus_list = []
    for sol in Y_sol:
        sol = sol.transpose()
        sol_df = pd.DataFrame(sol, columns=[*var_latex.values()])
        sol_df = sol_df.filter(regex=recovery_vars_name)
        ss_l = sol_df['$' + recovery_vars_name + '_l$'].iloc[-1]
        ss_m = sol_df['$' + recovery_vars_name + '_m$'].iloc [-1]

        if sol_df['$' + recovery_vars_name + '_l$'].iloc[-1] < 0.000001 and \
                sol_df['$' + recovery_vars_name + '_m$'].iloc[-1] < 0.000001:
            t2_recovery_lumen = 0.0
            t2_recovery_mucus = 0.0
            min_val_lumen = 0.0
            min_val_mucus = 0.0
            min_perdiff_lumen = 0.0
            min_perdiff_mucus = 0.0
            t2_min_lumen_list.append(0.0)
            t2_min_mucus_list.append(0.0)
        else:
            min_index_lumen = sol_df['$' + recovery_vars_name + '_l$'].idxmin()
            min_index_mucus = sol_df['$' + recovery_vars_name + '_m$'].idxmin()
            min_val_lumen = sol_df['$' + recovery_vars_name + '_l$'].min()
            min_val_mucus = sol_df['$' + recovery_vars_name + '_m$'].min()
            min_perdiff_lumen = ((ss_l - min_val_lumen) / ss_l) * 100
            min_perdiff_mucus = ((ss_m - min_val_mucus) / ss_m) * 100
            sol_df = ((sol_df - sol_df.iloc[ant_index]).abs()) / ((sol_df + sol_df.iloc[ant_index]) / 2)
            sol_df['t'] = t_sol[0]
            t2_min_lumen_list.append(sol_df['t'][min_index_lumen])
            t2_min_mucus_list.append(sol_df['t'][min_index_mucus])
            sol_df = sol_df[(sol_df['t'] > min_index_mucus).idxmax():]
            recovery_index_lumen = sol_df[sol_df['$' + recovery_vars_name + '_l$'] < 0.01].first_valid_index()
            recovery_index_mucus = sol_df[sol_df['$' + recovery_vars_name + '_m$'] < 0.01].first_valid_index()
            if recovery_index_lumen is None:
                t2_recovery_lumen = 0.0
            else:
                t2_recovery_lumen = sol_df['t'][recovery_index_lumen] - (ant_start)
            if recovery_index_mucus is None:
                t2_recovery_mucus = 0.0
            else:
                t2_recovery_mucus = sol_df['t'][recovery_index_mucus] - (ant_start)
        min_lumen_list.append(min_val_lumen)
        min_mucus_list.append(min_val_mucus)
        t2_rec_lumen_list.append(t2_recovery_lumen)
        t2_rec_mucus_list.append(t2_recovery_mucus)
        min_perdiff_lumen_list.append(min_perdiff_lumen)
        min_perdiff_mucus_list.append(min_perdiff_mucus)

    if plot_vals is None:
        # Plot time to recovery
        plot_list_rec = np.column_stack([varied_param_list, t2_rec_lumen_list, t2_rec_mucus_list])
        plot_df_rec = pd.DataFrame(plot_list_rec,
                                   columns=[latex_dict[varied_param_name], '$' + recovery_vars_name + '_l$', '$' +
                                            recovery_vars_name + '_m$'])
        plot_df_rec = plot_df_rec.set_index(latex_dict[varied_param_name])
        ax = plot_df_rec.plot(title=plot_title_string_rec, legend='best', colormap='tab20')
        plt.ylabel(plot_rec_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_rec + '.pdf', bbox_inches='tight')
        plt.close()

        # Plot time to minimum
        plot_list = np.column_stack([varied_param_list, t2_min_lumen_list, t2_min_mucus_list])
        plot_df = pd.DataFrame(plot_list, columns=[latex_dict[varied_param_name], '$' + recovery_vars_name + '_l$', '$' +
                                                   recovery_vars_name + '_m$'])
        plot_df = plot_df.set_index(latex_dict[varied_param_name])
        ax = plot_df.plot(title=plot_title_string_tmin, legend='best', colormap='tab20')
        plt.ylabel(plot_tmin_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_tmin + '.pdf', bbox_inches='tight')
        plt.close()

        # Plot minimum value
        plot_list = np.column_stack([varied_param_list, min_lumen_list, min_mucus_list])
        plot_df = pd.DataFrame(plot_list, columns=[latex_dict[varied_param_name], '$' + recovery_vars_name + '_l$', '$' +
                                                   recovery_vars_name + '_m$'])
        plot_df = plot_df.set_index(latex_dict[varied_param_name])
        ax = plot_df.plot(title=plot_title_string_min, legend='best', colormap='tab20')
        plt.ylabel(plot_min_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_min + '.pdf', bbox_inches='tight')
        plt.close()

        # Plot percent decrease
        plot_list = np.column_stack([varied_param_list, min_perdiff_lumen_list, min_perdiff_mucus_list])
        plot_df = pd.DataFrame(plot_list, columns=[latex_dict[varied_param_name], '$' + recovery_vars_name + '_l$', '$' +
                                                   recovery_vars_name + '_m$'])
        plot_df = plot_df.set_index(latex_dict[varied_param_name])
        ax = plot_df.plot(title=plot_title_string_perdiff, legend='best', colormap='tab20')
        plt.ylabel(plot_perdiff_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_perdiff + '.pdf', bbox_inches='tight')
        plt.close()

    return min_perdiff_lumen_list, min_perdiff_mucus_list


def plot_all(varied_param_list, vars_dict, latex_dict, recovery_vars_name, varied_param_name,
             min_perdiff_Xl_list_Xsingle, min_perdiff_Xm_list_Xsingle,
             min_perdiff_Zl_list_Zsingle, min_perdiff_Zm_list_Zsingle,
             min_perdiff_Xl_list_compXant, min_perdiff_Xm_list_compXant, min_perdiff_Zl_list_compXant,
             min_perdiff_Zm_list_compXant,
             min_perdiff_Xl_list_compZant, min_perdiff_Xm_list_compZant, min_perdiff_Zl_list_compZant,
             min_perdiff_Zm_list_compZant,
             min_perdiff_Xl_list_compantboth, min_perdiff_Xm_list_compantboth, min_perdiff_Zl_list_compantboth,
             min_perdiff_Zm_list_compantboth,
             plot_title_string, plot_ylabel_string, plot_xlabel_string, results_folder):
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    if recovery_vars_name == 'both':
        x_name = '$K_A$'
    else:
        x_name = latex_dict[varied_param_name]

    # Plot percent decrease
    plot_list = np.column_stack(
        [varied_param_list, min_perdiff_Xl_list_Xsingle, min_perdiff_Xm_list_Xsingle,
         min_perdiff_Zl_list_Zsingle, min_perdiff_Zm_list_Zsingle,
         min_perdiff_Xl_list_compXant, min_perdiff_Xm_list_compXant, min_perdiff_Zl_list_compXant,
         min_perdiff_Zm_list_compXant,
         min_perdiff_Xl_list_compZant, min_perdiff_Xm_list_compZant, min_perdiff_Zl_list_compZant,
         min_perdiff_Zm_list_compZant,
         min_perdiff_Xl_list_compantboth, min_perdiff_Xm_list_compantboth, min_perdiff_Zl_list_compantboth,
         min_perdiff_Zm_list_compantboth])

    plot_list_single = np.column_stack(
        [varied_param_list, min_perdiff_Xl_list_Xsingle, min_perdiff_Xm_list_Xsingle,
         min_perdiff_Zl_list_Zsingle, min_perdiff_Zm_list_Zsingle])

    plot_list_comp = np.column_stack(
        [varied_param_list, min_perdiff_Xl_list_compXant, min_perdiff_Xm_list_compXant, min_perdiff_Zl_list_compXant,
         min_perdiff_Zm_list_compXant,
         min_perdiff_Xl_list_compZant, min_perdiff_Xm_list_compZant, min_perdiff_Zl_list_compZant,
         min_perdiff_Zm_list_compZant,
         min_perdiff_Xl_list_compantboth, min_perdiff_Xm_list_compantboth, min_perdiff_Zl_list_compantboth,
         min_perdiff_Zm_list_compantboth])
    plot_df = pd.DataFrame(plot_list, columns=[x_name, '$X_l$ (I)', '$X_m$ (I)',
                                               '$Z_l$ (II)', '$Z_m$ (II)',
                                               '$X_l$ (III)', '$X_m$ (III)', '$Z_l$ (III)', '$Z_m$ (III)',
                                               '$X_l$ (IV)', '$X_m$ (IV)', '$Z_l$ (IV)', '$Z_m$ (IV)',
                                               '$X_l$ (V)', '$X_m$ (V)', '$Z_l$ (V)', '$Z_m$ (V)'])

    plot_df_single = pd.DataFrame(plot_list_single, columns=[x_name, '$X_l$ (I)', '$X_m$ (I)', '$Z_l$ (II)', '$Z_m$ (II)'])

    plot_df_comp = pd.DataFrame(plot_list_comp, columns=[x_name, '$X_l$ (III)', '$X_m$ (III)', '$Z_l$ (III)', '$Z_m$ (III)',
                                                    '$X_l$ (IV)', '$X_m$ (IV)', '$Z_l$ (IV)', '$Z_m$ (IV)', '$X_l$ (V)',
                                                    '$X_m$ (V)', '$Z_l$ (V)', '$Z_m$ (V)'])

    plot_df = plot_df.set_index(x_name)
    ax = plot_df.plot(title=plot_title_string, legend='best', colormap='tab20')
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.legend(loc='center left',  bbox_to_anchor=(1, 0.5))
    plt.savefig(results_folder + '/' + plot_title_string + '.pdf', bbox_inches='tight')
    plt.close()

    plot_df_single = plot_df_single.set_index(x_name)
    ax = plot_df_single.plot(title=plot_title_string, legend='best', colormap='tab20')
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.legend(loc='center left',  bbox_to_anchor=(1, 0.5))
    plt.savefig(results_folder + '/' + plot_title_string + ' (Single Species).pdf', bbox_inches='tight')
    plt.close()

    plot_df_comp = plot_df_comp.set_index(x_name)
    ax = plot_df_comp.plot(title=plot_title_string, legend='best', colormap='tab20')
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.legend(loc='center left',  bbox_to_anchor=(1, 0.5))
    plt.savefig(results_folder + '/' + plot_title_string + ' (with Competition).pdf', bbox_inches='tight')
    plt.close()


def ant_relative_abundance(Y_sol, t_sol, ant_start, ant_end, vars_names, varied_param_list, vars_dict, latex_dict,
                           varied_param_name, plot_title_string, plot_ylabel_string, plot_xlabel_string,
                           results_folder):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    t_df = pd.DataFrame(t_sol[0])
    ant_start_index = t_df[t_df >= ant_start].first_valid_index()
    ant_end_index = t_df[t_df >= ant_end].first_valid_index()
    rel_abd_before_Xl_list = []
    rel_abd_before_Xm_list = []
    rel_abd_before_Zl_list = []
    rel_abd_before_Zm_list = []
    rel_abd_during_Xl_list = []
    rel_abd_during_Xm_list = []
    rel_abd_during_Zl_list = []
    rel_abd_during_Zm_list = []
    rel_abd_after_Xl_list = []
    rel_abd_after_Xm_list = []
    rel_abd_after_Zl_list = []
    rel_abd_after_Zm_list = []
    for sol in Y_sol:
        sol = sol.transpose()
        sol_df = pd.DataFrame(sol, columns=[*var_latex.values()])
        sol_df = sol_df[sol_df.keys()[sol_df.keys().isin(vars_names)]]
        sol_df['total_l'] = sol_df.filter(regex='l').sum(axis=1)
        sol_df['total_m'] = sol_df.filter(regex='m').sum(axis=1)
        rel_abd_before_Xl_list.append(
            (sol_df.loc[ant_start_index, '$X_l$'] / sol_df.loc[ant_start_index, 'total_l']) * 100)
        rel_abd_before_Xm_list.append(
            (sol_df.loc[ant_start_index, '$X_m$'] / sol_df.loc[ant_start_index, 'total_m']) * 100)
        rel_abd_before_Zl_list.append(
            (sol_df.loc[ant_start_index, '$Z_l$'] / sol_df.loc[ant_start_index, 'total_l']) * 100)
        rel_abd_before_Zm_list.append(
            (sol_df.loc[ant_start_index, '$Z_m$'] / sol_df.loc[ant_start_index, 'total_m']) * 100)
        rel_abd_during_Xl_list.append(
            (sol_df.loc[(ant_start_index / 2), '$X_l$'] / sol_df.loc[(ant_start_index / 2), 'total_l']) * 100)
        rel_abd_during_Xm_list.append(
            (sol_df.loc[(ant_start_index / 2), '$X_m$'] / sol_df.loc[(ant_start_index / 2), 'total_m']) * 100)
        rel_abd_during_Zl_list.append(
            (sol_df.loc[(ant_start_index / 2), '$Z_l$'] / sol_df.loc[(ant_start_index / 2), 'total_l']) * 100)
        rel_abd_during_Zm_list.append(
            (sol_df.loc[(ant_start_index / 2), '$Z_m$'] / sol_df.loc[(ant_start_index / 2), 'total_m']) * 100)
        rel_abd_after_Xl_list.append(
            (sol_df.loc[ant_end_index, '$X_l$'] / sol_df.loc[ant_start_index, 'total_l']) * 100)
        rel_abd_after_Xm_list.append(
            (sol_df.loc[ant_end_index, '$X_m$'] / sol_df.loc[ant_start_index, 'total_m']) * 100)
        rel_abd_after_Zl_list.append(
            (sol_df.loc[ant_end_index, '$Z_l$'] / sol_df.loc[ant_start_index, 'total_l']) * 100)
        rel_abd_after_Zm_list.append(
            (sol_df.loc[ant_end_index, '$Z_m$'] / sol_df.loc[ant_start_index, 'total_m']) * 100)
    rel_abd_b_df = pd.DataFrame(
        zip(varied_param_list, rel_abd_before_Xl_list, rel_abd_before_Xm_list, rel_abd_before_Zl_list,
            rel_abd_before_Zm_list), columns=[latex_dict[varied_param_name], '$X_l$', '$X_m$', '$Z_l$', '$Z_m$'])
    rel_abd_d_df = pd.DataFrame(
        zip(varied_param_list, rel_abd_during_Xl_list, rel_abd_during_Xm_list, rel_abd_during_Zl_list,
            rel_abd_during_Zm_list), columns=[latex_dict[varied_param_name], '$X_l$', '$X_m$', '$Z_l$', '$Z_m$'])
    rel_abd_a_df = pd.DataFrame(
        zip(varied_param_list, rel_abd_after_Xl_list, rel_abd_after_Xm_list, rel_abd_after_Zl_list,
            rel_abd_after_Zm_list), columns=[latex_dict[varied_param_name], '$X_l$', '$X_m$', '$Z_l$', '$Z_m$'])

    rel_abd_b_df = rel_abd_b_df.set_index(latex_dict[varied_param_name])
    rel_abd_d_df = rel_abd_d_df.set_index(latex_dict[varied_param_name])
    rel_abd_a_df = rel_abd_a_df.set_index(latex_dict[varied_param_name])
    rel_abd_change = rel_abd_b_df - rel_abd_a_df

    # Plot relative abundance before treatment
    ax = rel_abd_b_df.plot(title=plot_title_string + ' Before Treatment', legend='best', colormap='tab20')
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '_before' + '.pdf', bbox_inches='tight')
    plt.close()

    # Plot relative abundance during treatment
    ax = rel_abd_d_df.plot(title=plot_title_string + ' During Treatment', legend='best', colormap='tab20')
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '_during' + '.pdf', bbox_inches='tight')
    plt.close()

    # Plot relative abundance at the end of treatment
    ax = rel_abd_a_df.plot(title=plot_title_string + ' At the End of Treatment', legend='best', colormap='tab20')
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '_after' + '.pdf', bbox_inches='tight')
    plt.close()

    # Plot change in relative abundance at the end of treatment
    ax = rel_abd_change.plot(title='Change in ' + plot_title_string, legend='best', colormap='tab20')
    plt.ylabel(plot_ylabel_string)
    plt.xlabel(plot_xlabel_string)
    plt.savefig(results_folder + '/' + plot_title_string + '_change' + '.pdf', bbox_inches='tight')
    plt.close()


def ant_t2_recovery_XY(Y_sol, t_sol, ant_start, pre_start, per_rec, varied_param_list, vars_dict, latex_dict,
                       varied_param_name, recovery_vars_name, plot_title_string_rec, plot_rec_ylabel_string,
                       plot_xlabel_string, results_folder, plot_vals=None):
    # Y_ss is a list of steady state values (list of lists)
    var_latex = {}
    for key, value in vars_dict.items():
        var_latex[key] = latex_dict[key]
    t2_rec_Xl_list = []
    t2_rec_Xm_list = []
    t2_rec_Zl_list = []
    t2_rec_Zm_list = []
    for sol in Y_sol:
        sol = sol.transpose()
        sol_df = pd.DataFrame(sol, columns=[*var_latex.values()])
        sol_X_df = sol_df.filter(regex='X')
        sol_Z_df = sol_df.filter(regex='Z')
        sol_A_df = sol_df.filter(regex='A')
        sol_X_df = pd.DataFrame(sol_X_df)
        sol_X_df['t'] = t_sol[0]
        sol_Z_df = pd.DataFrame(sol_Z_df)
        sol_Z_df['t'] = t_sol[0]

        min_index_Xl = sol_X_df['$X_l$'].idxmin()
        min_index_Xm = sol_X_df['$X_m$'].idxmin()
        min_index_Zl = sol_Z_df['$Z_l$'].idxmin()
        min_index_Zm = sol_Z_df['$Z_m$'].idxmin()
        ant_start_index = sol_X_df[sol_X_df['t'] > ant_start].first_valid_index()
        pre_start_index = sol_X_df[sol_X_df['t'] > pre_start].first_valid_index()
        ss_Xl = sol_X_df['$X_l$'].iloc[-1]
        ss_Xm = sol_X_df['$X_m$'].iloc[-1]
        ss_Zl = sol_Z_df['$Z_l$'].iloc[-1]
        ss_Zm = sol_Z_df['$Z_m$'].iloc[-1]
        sol_A_df = sol_A_df.truncate(before=ant_start_index)
        ant_end_index_l = sol_A_df[sol_A_df['$A_l$'] < 0.000001].first_valid_index()
        ant_end_index_m = sol_A_df[sol_A_df['$A_m$'] < 0.000001].first_valid_index()

        # Recovery time for X
        if sol_df['$X_l$'].iloc[-1] < 0.000001 and sol_df['$X_m$'].iloc[-1] < 0.000001:
            t2_recovery_Xl = 0.0
            t2_recovery_Xm = 0.0
        elif (min_index_Xl <= ant_start_index or min_index_Xl >= ant_end_index_l) and (
                min_index_Xm <= ant_start_index or min_index_Xm >= ant_end_index_m):
            t2_recovery_Xl = 0.0
            t2_recovery_Xm = 0.0
        else:
            # ss_Xl = sol_X_df['$X_l$'].iloc[0]
            # ss_Xm = sol_X_df['$X_m$'].iloc[0]
            if min_index_Xl <= min_index_Xm and ant_start_index < min_index_Xl < ant_end_index_l:
                min_index = min_index_Xl
            if min_index_Xm <= min_index_Xl and ant_start_index < min_index_Xm < ant_end_index_m:
                min_index = min_index_Xm
            elif ant_start_index < min_index_Xm < ant_end_index_m:
                min_index = min_index_Xm
            sol_X_df = sol_X_df.truncate(before=min_index)
            recovery_index_Xl = sol_X_df[sol_X_df['$X_l$'] > (per_rec * ss_Xl)].first_valid_index()
            recovery_index_Xm = sol_X_df[sol_X_df['$X_m$'] > (per_rec * ss_Xm)].first_valid_index()
            if recovery_index_Xl is None:
                t2_recovery_Xl = 0.0
            else:
                t2_recovery_Xl = sol_X_df['t'][recovery_index_Xl] - (ant_start)
            if recovery_index_Xm is None:
                t2_recovery_Xm = 0.0
            else:
                t2_recovery_Xm = sol_X_df['t'][recovery_index_Xm] - (ant_start)
        # Recovery time for Z
        if sol_df['$Z_l$'].iloc[-1] < 0.000001 and sol_df['$Z_m$'].iloc[-1] < 0.000001:
            t2_recovery_Zl = 0.0
            t2_recovery_Zm = 0.0
        elif (min_index_Zl <= ant_start_index or min_index_Zl >= ant_end_index_l) and (
                min_index_Zm <= ant_start_index or min_index_Zm >= ant_end_index_m):
            t2_recovery_Zl = 0.0
            t2_recovery_Zm = 0.0
        else:
            # ss_Zl = sol_Z_df['$Z_l$'].iloc[0]
            # ss_Zm = sol_Z_df['$Z_m$'].iloc[0]
            if min_index_Zl <= min_index_Zm and ant_start_index < min_index_Zl < ant_end_index_l:
                min_index = min_index_Zl
            elif min_index_Zm <= min_index_Zl and ant_start_index < min_index_Zm < ant_end_index_m:
                min_index = min_index_Zm
            else:
                min_index = 0
            sol_Z_df = sol_Z_df.truncate(before=min_index)
            recovery_index_Zl = sol_Z_df[sol_Z_df['$Z_l$'] > (per_rec * ss_Zl)].first_valid_index()
            recovery_index_Zm = sol_Z_df[sol_Z_df['$Z_m$'] > (per_rec * ss_Zm)].first_valid_index()
            if recovery_index_Zl is None:
                t2_recovery_Zl = 0.0
            else:
                t2_recovery_Zl = sol_Z_df['t'][recovery_index_Zl] - (ant_start)
            if recovery_index_Zm is None:
                t2_recovery_Zm = 0.0
            else:
                t2_recovery_Zm = sol_Z_df['t'][recovery_index_Zm] - (ant_start)
        t2_rec_Xl_list.append(t2_recovery_Xl)
        t2_rec_Xm_list.append(t2_recovery_Xm)
        t2_rec_Zl_list.append(t2_recovery_Zl)
        t2_rec_Zm_list.append(t2_recovery_Zm)

    # Plot time to recovery
    if recovery_vars_name == 'both':
        x_name = '$K_A$'
    else:
        x_name = latex_dict.get(varied_param_name, varied_param_name)

    if plot_vals is None:
        plot_list_rec = np.column_stack([varied_param_list, t2_rec_Xl_list, t2_rec_Xm_list, t2_rec_Zl_list, t2_rec_Zm_list])
        plot_df_rec = pd.DataFrame(plot_list_rec,
                                   columns=[x_name, '$X_l$', '$X_m$', '$Z_l$', '$Z_m$'])
        plot_df_rec = plot_df_rec.set_index(x_name)
        ax = plot_df_rec.plot(title=plot_title_string_rec, legend='best', colormap='tab20')
        plt.ylabel(plot_rec_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_rec + '.pdf', bbox_inches='tight')
        plt.close()

        plot_list_X = np.column_stack([varied_param_list, t2_rec_Xl_list, t2_rec_Xm_list])
        plot_df_X = pd.DataFrame(plot_list_X, columns=[x_name, '$X_l$', '$X_m$'])
        plot_df_X = plot_df_X.set_index(x_name)
        ax = plot_df_X.plot(title=plot_title_string_rec, legend='best', colormap='tab20')
        plt.ylabel(plot_rec_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_rec + '_X.pdf', bbox_inches='tight')
        plt.close()

        plot_list_Z = np.column_stack([varied_param_list, t2_rec_Zl_list, t2_rec_Zm_list])
        plot_df_Z = pd.DataFrame(plot_list_Z, columns=[x_name, '$Z_l$', '$Z_m$'])
        plot_df_Z = plot_df_Z.set_index(x_name)
        ax = plot_df_Z.plot(title=plot_title_string_rec, legend='best', colormap='tab20')
        plt.ylabel(plot_rec_ylabel_string)
        plt.xlabel(plot_xlabel_string)
        plt.savefig(results_folder + '/' + plot_title_string_rec + '_Z.pdf', bbox_inches='tight')
        plt.close()

    return t2_rec_Xl_list, t2_rec_Xm_list, t2_rec_Zl_list, t2_rec_Zm_list

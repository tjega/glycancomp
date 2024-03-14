import pandas as pd
from SALib.analyze import sobol
import numpy as np
import re
from matplotlib import pyplot as plt
from SimulationModule.run_ode_sim import update_param_init_dicts
from SolverModule.build_problem import build_ode_problem_arrays
from SolverModule.solve_ode import chemo_sim


def salib_analysis(p_list, inflow_amount_list, dilution_rate_list, phi_list, ex_list, internal_prod_list, d_list,
                   vars_dict, params_dict, param_dependencies_dict, params_vals, init_vals, init_cond, problem, results_folder):
    # Initializing matrix to store output
    Y = np.zeros([len(params_vals), len(init_cond)])
    Yt = np.zeros([len(params_vals), len(init_cond)])

    # Solve ODE with sample array
    for i, X in enumerate(params_vals):
        sim_params_dict = update_param_init_dicts(params_dict, param_dependencies_dict, params_vals, i)
        problem_dict = build_ode_problem_arrays(p_list, inflow_amount_list, dilution_rate_list, phi_list, ex_list,
                                                internal_prod_list, d_list, sim_params_dict, vars_dict)
        res = chemo_sim(0, 1000, 1000, init_vals[i], problem_dict)

        # Steady state
        Y[i][:] = res.y[:, -1]

        # Time to steady state
        Yt[i][:] = res.t[-1]

    for key, val in vars_dict.items():
        Si_Y = sobol.analyze(problem, Y[:, val], print_to_console=False)
        total_Y, first_Y, second_Y = Si_Y.to_df()
        file_string_total_Y = results_folder + '/' + 'Si_total_Y_' + key + '.csv'
        file_string_first_Y = results_folder + '/' + 'Si_first_Y_' + key + '.csv'
        file_string_second_Y = results_folder + '/' + 'Si_second_Y_' + key + '.csv'
        total_Y.to_csv(file_string_total_Y)
        first_Y.to_csv(file_string_first_Y)
        second_Y.to_csv(file_string_second_Y)
    for key, val in vars_dict.items():
        Si_Yt = sobol.analyze(problem, Yt[:, val], print_to_console=False)
        total_Yt, first_Yt, second_Yt = Si_Yt.to_df()
        file_string_total_Yt = results_folder + '/' + 'Si_total_Yt_' + key + '.csv'
        file_string_first_Yt = results_folder + '/' + 'Si_first_Yt_' + key + '.csv'
        file_string_second_Yt = results_folder + '/' + 'Si_second_Yt_' + key + '.csv'
        total_Yt.to_csv(file_string_total_Yt)
        first_Yt.to_csv(file_string_first_Yt)
        second_Yt.to_csv(file_string_second_Yt)


def salib_plot_Si_steady_state(vars_dict, sobol_index, x_label_latex, legend_latex, results_folder):
    # sobol index must be total, first or second
    if sobol_index == 'total':
        conf_col = 'ST_conf'
    else:
        conf_col = 'S1_conf'
    Si_df_list = []
    Si_df = None
    for i, (key, val) in enumerate(vars_dict.items()):
        file_string_total = results_folder + '/' + 'Si_' + sobol_index + '_Y_' + key + '.csv'
        Si_df_list.append(pd.read_csv(file_string_total, index_col=0).drop(columns=[conf_col]))
        Si_df_list[i]['Key'] = key
        if i == 0:
            Si_df = Si_df_list[i]
        else:
            Si_df = pd.concat([Si_df, Si_df_list[i]])
    Si_df = Si_df.groupby([Si_df.index, 'Key'])
    Si_df = Si_df.sum().unstack('Key')
    Si_df[Si_df < 0] = 0
    Si_df = Si_df.loc[~(Si_df < 1e-1).all(axis=1)]

    # Remove extra latex xticks
    params_list = Si_df.index.tolist()
    x_label_latex_subset = {key: x_label_latex[key] for key in params_list}

    # Dataframes with biomass only
    Si_biomass_df = Si_df.filter(regex='X|Z')
    biomass_legend = {k: legend_latex[k] for k in legend_latex if re.match('X|Z', k)}

    # Plot
    ax = Si_df.plot(kind='bar')
    locs, labels = plt.xticks()
    plt.xticks(locs, x_label_latex_subset.values(), rotation='vertical', fontsize=11)
    # plt.xticks(locs, x_label_latex_subset.values(), rotation='vertical', fontsize=6)
    alphabetize_legend_latex = sorted(legend_latex.values())
    ax.legend(alphabetize_legend_latex,  prop={'size': 11}, loc='best')
    # ax.legend(alphabetize_legend_latex,  prop={'size': 6}, loc='best')
    plt.ylabel('Sobol Index')
    plt.xlabel('Parameter')
    plt.title(sobol_index.capitalize() + '-order Sobol Index for Steady State')
    plt.tight_layout()
    plt.savefig(results_folder + '/' + 'Y_Si_' + sobol_index + '.pdf')
    plt.close()

    # Plot biomass only
    ax = Si_biomass_df.plot(kind='bar')
    locs, labels = plt.xticks()
    plt.xticks(locs, x_label_latex_subset.values(), rotation='vertical', fontsize=11)
    # plt.xticks(locs, x_label_latex_subset.values(), rotation='vertical', fontsize=6)
    alphabetize_legend_latex = sorted(biomass_legend.values())
    ax.legend(alphabetize_legend_latex,  prop={'size': 11}, loc='best')
    # ax.legend(alphabetize_legend_latex,  prop={'size': 6}, loc='best')
    plt.ylabel('Sobol Index')
    plt.xlabel('Parameter')
    plt.title(sobol_index.capitalize() + '-order Sobol Index for Steady State')
    plt.tight_layout()
    plt.savefig(results_folder + '/' + 'Y_Si_biomass_' + sobol_index + '.pdf')
    plt.close()


def salib_plot_Si_time_to_ss(vars_dict, sobol_index, x_label_latex, results_folder):
    # sobol index must be total, first or second
    # sobol indices will be the same for each variable (time to steady
    # state is the same for all variables)-->only need to plot one
    if sobol_index == 'total':
        conf_col = 'ST_conf'
        s_col = 'ST'
    else:
        conf_col = 'S1_conf'
        s_col = 'S1'
    key = str(next(iter(vars_dict)))
    file_string_total = results_folder + '/Si_' + sobol_index + '_Yt_' + key + '.csv'
    Si_df = pd.read_csv(file_string_total, index_col=0)
    Si_df = Si_df.drop(columns=conf_col)
    Si_df[Si_df[s_col] < 0] = 0
    Si_df = Si_df.loc[~(Si_df < 1e-2).all(axis=1)]


    # Remove extra latex xticks
    params_list = Si_df.index.tolist()
    x_label_latex_subset = {key: x_label_latex[key] for key in params_list}

    ax = Si_df.plot(kind='bar', legend=False)
    locs, labels = plt.xticks()
    plt.xticks(locs, x_label_latex_subset.values(), rotation='vertical')
    plt.ylabel('Sobol Index')
    plt.xlabel('Parameter')
    plt.title(sobol_index.capitalize() + '-order Sobol Index for Time to Steady State')
    plt.tight_layout()
    plt.savefig(results_folder + '/' + 'Yt_Si_' + sobol_index + '.pdf')
    plt.close()


def salib_plot_Si_classes(sobol_index, x_label_latex, results_folder):
    # sobol index must be total, first or second
    # sobol indices will be the same for each variable (time to steady
    # state is the same for all variables)-->only need to plot one
    if sobol_index == 'total':
        conf_col = 'ST_conf'
        s_col = 'ST'
    else:
        conf_col = 'S1_conf'
        s_col = 'S1'
    file_string_total = results_folder + '/Si_' + sobol_index + '_Y_ss.csv'
    Si_df = pd.read_csv(file_string_total, index_col=0)
    Si_df = Si_df.drop(columns=conf_col)
    Si_df[Si_df[s_col] < 0] = 0
    Si_df = Si_df.loc[~(Si_df < 1e-2).all(axis=1)]


    # Remove extra latex xticks
    params_list = Si_df.index.tolist()
    x_label_latex_subset = {key: x_label_latex[key] for key in params_list}

    ax = Si_df.plot(kind='bar', legend=False)
    locs, labels = plt.xticks()
    plt.xticks(locs, x_label_latex_subset.values(), rotation='vertical')
    plt.ylabel('Sobol Index')
    plt.xlabel('Parameter')
    plt.title(sobol_index.capitalize() + '-order Sobol Index for Classes')
    plt.tight_layout()
    plt.savefig(results_folder + '/' + 'Yss_Si_' + sobol_index + '.pdf')
    plt.close()
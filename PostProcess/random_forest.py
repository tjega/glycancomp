import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
from InputModule.create_json_input import create_input_file
from SimulationModule.run_ode_sim import update_param_init_dicts
from SolverModule.build_problem import build_ode_problem_arrays
from SolverModule.solve_ode import chemo_sim
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from itertools import cycle, islice
import pandas as pd
from sklearn.tree import export_text
from sklearn.tree import _tree
import matplotlib.patches as mpatches


def run_rf_sims(p_list, inflow_amount_list, dilution_rate_list, phi_list, ex_list, internal_prod_list, d_list, vars_dict,
              params_dict, param_dependencies_dict, params_vals, init_vals, init_cond, eps, tend,
              latex_dict, testing, results_folder):

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)

    # Initializing matrix to store output
    Y = np.zeros([len(params_vals), len(init_cond)])  # steady state
    Yt = np.zeros([len(params_vals), len(init_cond)])  # time to steady state
    Yss = []  # classify steady state

    param_init_vals = np.hstack((init_vals, params_vals))
    # noinspection PyTypeChecker
    np.savetxt(results_folder + '/sample_params.txt', param_init_vals, fmt='%f')
    np.savetxt(results_folder + '/param_vals.txt', params_vals, fmt='%f')
    np.savetxt(results_folder + '/init_vals.txt', init_vals, fmt='%f')

    # Solve ODE with sample array
    for i, j in enumerate(params_vals):
        sim_params_dict = update_param_init_dicts(params_dict, param_dependencies_dict, params_vals, i)
        problem_dict = build_ode_problem_arrays(p_list, inflow_amount_list, dilution_rate_list, phi_list, ex_list,
                                                internal_prod_list, d_list, sim_params_dict, vars_dict)
        res = chemo_sim(0, tend, 1000, init_vals[i], problem_dict)

        # Steady state
        Y[i][:] = res.y[:, -1]

        # Time to steady state
        Yt[i][:] = res.t[-1]
        if Yt[i, 0] == tend:
            if testing == 1:
                Yss.append('noss')
                latex_vars_dict = []
                for key, value in vars_dict.items():
                    latex_vars_dict.append(latex_dict[key])
                # Plot lumen biomass
                plt.figure()
                plt.plot(res.t, res.y[vars_dict['X_l']], label=latex_vars_dict[vars_dict['X_l']], linestyle=next(linecycler))
                plt.plot(res.t, res.y[vars_dict['Z_l']], label=latex_vars_dict[vars_dict['Z_l']], linestyle=next(linecycler))
                plt.legend(loc='best')
                plt.ylabel('Concentration')
                plt.xlabel('Time')
                plt.savefig(results_folder + '/noss_lumen_' + str(i) + '.pdf', bbox_inches='tight')
                plt.close()
                # Plot mucus biomass
                plt.figure()
                plt.plot(res.t, res.y[vars_dict['X_m']], label=latex_vars_dict[vars_dict['X_m']], linestyle=next(linecycler))
                plt.plot(res.t, res.y[vars_dict['Z_m']], label=latex_vars_dict[vars_dict['Z_m']], linestyle=next(linecycler))
                plt.legend(loc='best')
                plt.ylabel('Concentration')
                plt.xlabel('Time')
                plt.savefig(results_folder + '/noss_mucus_' + str(i) + '.pdf', bbox_inches='tight')
                plt.close()
                # Plot all
                plt.figure()
                for k in range(res.y.shape[0]):
                    plt.plot(res.t, res.y[k], label=latex_vars_dict[k])
                plt.legend(loc='best')
                plt.ylabel('Concentration')
                plt.xlabel('Time')
                plt.savefig(results_folder + '/noss_all_' + str(i) + '.pdf', bbox_inches='tight')
                plt.close()
            else:
                if Y[i][vars_dict['X_l']] < eps and Y[i][vars_dict['Z_l']] < eps and Y[i][vars_dict['X_m']] < eps and \
                        Y[i][vars_dict['Z_m']] < eps:
                    Yss.append('trivial')
                elif Y[i][vars_dict['X_l']] < eps and Y[i][vars_dict['Z_l']] >= eps and Y[i][vars_dict['X_m']] < eps and \
                        Y[i][vars_dict['Z_m']] >= eps:
                    Yss.append('Z')
                elif Y[i][vars_dict['X_l']] >= eps and Y[i][vars_dict['Z_l']] < eps and Y[i][
                    vars_dict['X_m']] >= eps and Y[i][vars_dict['Z_m']] < eps:
                    Yss.append('X')
                elif Y[i][vars_dict['X_l']] >= eps and Y[i][vars_dict['Z_l']] >= eps and Y[i][
                        vars_dict['X_m']] >= eps and Y[i][vars_dict['Z_m']] >= eps and Y[i][vars_dict['X_l']] > Y[i][
                             vars_dict['Z_l']] and Y[i][vars_dict['X_m']] > Y[i][vars_dict['Z_m']]:
                    Yss.append('coexistenceX')
                elif Y[i][vars_dict['X_l']] >= eps and Y[i][vars_dict['Z_l']] >= eps and Y[i][
                    vars_dict['X_m']] >= eps and Y[i][vars_dict['Z_m']] >= eps and Y[i][vars_dict['Z_l']] > Y[i][
                    vars_dict['X_l']] and Y[i][vars_dict['Z_m']] > Y[i][vars_dict['X_m']]:
                    Yss.append('coexistenceZ')
                elif Y[i][vars_dict['X_l']] >= eps and Y[i][vars_dict['Z_l']] >= eps and Y[i][
                    vars_dict['X_m']] >= eps and Y[i][vars_dict['Z_m']] >= eps:
                    Yss.append('coexistence')
                else:
                    Yss.append('other')
        else:
            if Y[i][vars_dict['X_l']] < eps and Y[i][vars_dict['Z_l']] < eps and Y[i][vars_dict['X_m']] < eps and Y[i][vars_dict['Z_m']] < eps:
                Yss.append('trivial')
            elif Y[i][vars_dict['X_l']] < eps and Y[i][vars_dict['Z_l']] >= eps and Y[i][vars_dict['X_m']] < eps and Y[i][vars_dict['Z_m']] >= eps:
                Yss.append('Z')
            elif Y[i][vars_dict['X_l']] >= eps and Y[i][vars_dict['Z_l']] < eps and Y[i][vars_dict['X_m']] >= eps and Y[i][vars_dict['Z_m']] < eps:
                Yss.append('X')
            elif Y[i][vars_dict['X_l']] >= eps and Y[i][vars_dict['Z_l']] >= eps and Y[i][vars_dict['X_m']] >= eps and Y[i][vars_dict['Z_m']] >= eps and Y[i][vars_dict['X_l']] > Y[i][vars_dict['Z_l']] and Y[i][vars_dict['X_m']] > Y[i][vars_dict['Z_m']]:
                Yss.append('coexistenceX')
            elif Y[i][vars_dict['X_l']] >= eps and Y[i][vars_dict['Z_l']] >= eps and Y[i][vars_dict['X_m']] >= eps and Y[i][vars_dict['Z_m']] >= eps and Y[i][vars_dict['Z_l']] > Y[i][vars_dict['X_l']] and Y[i][vars_dict['Z_m']] > Y[i][vars_dict['X_m']]:
                Yss.append('coexistenceZ')
            elif Y[i][vars_dict['X_l']] >= eps and Y[i][vars_dict['Z_l']] >= eps and Y[i][vars_dict['X_m']] >= eps and Y[i][vars_dict['Z_m']] >= eps:
                Yss.append('coexistence')
            else:
                Yss.append('other')

    np.savetxt(results_folder + '/steady_state.txt', Y, fmt='%f')
    create_input_file(Yss, results_folder + '/Yss.json')
    return Yss


def run_rf_model(param_init_vals, Yss, feature_names, x_label_latex, results_folder):

    X_train, X_test, y_train, y_test = train_test_split(param_init_vals, Yss, stratify=Yss, shuffle=True, random_state=42, test_size=0.3)
    forest_model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    forest_model.fit(X_train, y_train)
    y_pred = forest_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    rf_file = open(results_folder + '/rf_output.txt', 'w')
    rf_file.write('accuracy=' + repr(accuracy))
    rf_file.close()
    conf_mat = confusion_matrix(y_test, y_pred)
    np.savetxt(results_folder + '/conf_mat.txt', conf_mat, fmt='%d')
    np.savetxt(results_folder + '/class_names.txt', forest_model.classes_, fmt='%s')

    # Print decision tree rules

    # for i in range(0, 5):
    #     tree_rules = export_text(forest_model.estimators_[i], spacing=3, decimals=3, feature_names=list(feature_names))
    #     tree_file = open(results_folder + '/tree_' + str(i) + '.txt', 'w')
    #     tree_file.write(tree_rules)
    #     tree_file.close()

    # Plot feature importance
    importances = forest_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest_model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    # forest_importances.sort_values(ascending=False, inplace=True)
    result = permutation_importance(forest_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=10)

    importance_df = pd.DataFrame(list(zip(feature_names, importances, x_label_latex.values(), result.importances_std)), columns=['Params', 'Importances', 'Latex', 'STD'])
    importance_df = importance_df.sort_values(by=['Importances'], ascending=False)

    # # Feature importance based on mean decrease impurity
    # ax = forest_importances.plot(kind='bar')
    # locs, labels = plt.xticks()
    # plt.xticks(locs, x_label_latex.values(), rotation='horizontal')
    # plt.title("Feature importances using MDI")
    # plt.ylabel("Mean decrease in impurity")
    # plt.xlabel('Parameter')
    # plt.tight_layout()
    # plt.savefig(results_folder + '/MDI_Feature_Importance.pdf')
    # plt.close()
    #
    # # Feature importance based on feature permutation
    # ax = forest_importances.plot(kind='bar')
    # plt.title("Feature importances using permutation on full model")
    # plt.ylabel("Mean accuracy decrease")
    # plt.xlabel('Parameter')
    # plt.tight_layout()
    # plt.savefig(results_folder + '/Permutation_Feature_Importance.pdf')
    # plt.close()

    # # Feature importance based on mean decrease impurity
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # locs, labels = plt.xticks()
    # plt.xticks(locs, x_label_latex.values(), rotation='vertical', fontsize=6)
    # plt.savefig(results_folder + '/MDI_Feature_Importance.pdf', bbox_inches='tight')
    # plt.close()

    importance_df.loc[importance_df['Params'].str.contains("alpha"), 'colours'] = 'blue'
    importance_df.loc[importance_df['Params'].str.contains("mu"), 'colours'] = 'orange'
    importance_df.loc[importance_df['Params'].str.contains("gam"), 'colours'] = 'green'
    importance_df.loc[importance_df['Params'].str.contains("K_"), 'colours'] = 'red'
    importance_df.loc[importance_df['Params'].str.contains("omega"), 'colours'] = 'magenta'
    importance_df.loc[importance_df['Params'].str.contains("I_inf"), 'colours'] = 'purple'
    importance_df.loc[importance_df['Params'].str.contains("lam"), 'colours'] = 'purple'
    importance_df.loc[importance_df['Params'].str.contains("_0"), 'colours'] = 'aqua'
    importance_df.loc[importance_df['Params'].str.contains("V"), 'colours'] = 'pink'
    importance_df.loc[importance_df['Params'].str.contains("D"), 'colours'] = 'pink'
    importance_df.loc[importance_df['Params'].str.contains("Y"), 'colours'] = 'gray'

    # Feature importance based on feature permutation
    # bar_colours = list(islice(cycle(['blue', 'blue', 'purple', 'orange', 'gray', 'gray', 'pink', 'red', 'red', 'green',
    #                                  'pink', 'orange', 'gray', 'orange', 'green', 'purple', 'orange', 'green', 'orange',
    #                                  'gray', 'orange', 'red', 'aqua', 'aqua', 'red', 'aqua', 'green', 'green',
    #                                  'red', 'aqua', 'green', 'orange', 'red', 'pink', 'gray', 'orange', 'purple',
    #                                  'red', 'orange', 'orange', 'orange', 'red', 'orange', 'aqua', 'aqua', 'orange',
    #                                  'aqua', 'aqua', 'pink', 'orange']), None, len(importance_df)))
    ax = importance_df.plot.bar(x='Latex', y='Importances', yerr='STD', color=list(importance_df['colours']))
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    ax.set_xlabel("Parameter")
    plt.xticks(rotation='vertical', fontsize=5)
    death = mpatches.Patch(color='blue', label='Death Rates')
    growth = mpatches.Patch(color='orange', label='Growth Rates')
    halfsat = mpatches.Patch(color='red', label='Half-Saturation Constants')
    omeg = mpatches.Patch(color='magenta', label='Inhibition Constant')
    glycan = mpatches.Patch(color='purple', label='Glycan')
    ic = mpatches.Patch(color='aqua', label='Initial Conditions')
    reactor = mpatches.Patch(color='pink', label='Reactor')
    ycoeff = mpatches.Patch(color='gray', label='Yield Coefficients')
    plt.legend(handles=[death, growth, halfsat, omeg, glycan, ic, reactor, ycoeff], title='Parameter Type', fontsize=8)

    # ax.get_legend().remove()
    plt.savefig(results_folder + '/Permutation_Feature_Importance_Ordered.pdf', bbox_inches='tight')
    plt.close()

    # # Feature importance based on feature permutation
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # locs, labels = plt.xticks()
    # plt.xticks(locs, x_label_latex.values(), rotation='vertical', fontsize=6)
    # plt.savefig(results_folder + '/Permutation_Feature_Importance.pdf', bbox_inches='tight')
    # plt.close()


def rf_read_sample_file(sample_filename):
    param_init_vals = np.loadtxt(sample_filename)
    return param_init_vals


def param_tuning_estimators(param_init_vals, Yss, hyperparam_name, hyperparam_range, output_folder):

    train_scores, test_scores = validation_curve(RandomForestClassifier(), X=param_init_vals, y=Yss,
                                                 param_name=hyperparam_name, param_range=hyperparam_range, cv=2,
                                                 n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    lw = 2

    plt.title('Validation Curve')
    plt.xlabel(hyperparam_name)
    plt.ylabel('Score')
    plt.plot(hyperparam_range, test_scores_mean, label='Cross-validation score', color='navy', lw=lw)
    plt.fill_between(
        hyperparam_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.plot(hyperparam_range, train_scores_mean, label='Training score', color='darkorange')
    plt.fill_between(
        hyperparam_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.legend(loc='best')

    plt.savefig(output_folder + '/validationcurve_' + hyperparam_name + '.pdf', bbox_inches='tight')
    plt.close()


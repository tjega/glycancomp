def build_single_compartment_primary():
    p_list = [[-1.0, 0.0, 0.0], ['Y_I', -1.0, 0], [0, 'Y_S', -1.0]]

    d_list = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]

    ex_list = [[0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]]

    inflow_amount_list = ['I_inf', 0.0, 0.0]

    dilution_rate_list = ['D', 'D', 'D']

    internal_prod_list = [0.0, 0.0, 0.0]

    phi_list = [{'contois': [['mu_I', 'K_I'], ['I', 'X']]}, {'monod': [['mu_S', 'K_S'], ['S', 'X']]},
                {'decay': [['alpha'], ['X']]}]

    vars_dict = {"I": 0, "S": 1, "X": 2}

    params_dict = {'mu_S': 12.627143363,
                   'mu_I': 10.619469027,
                   'K_S': 0.468416,
                   'K_I': 0.26539823,
                   'D': 1.0071942760083039,
                   'Y_S': 0.34242,
                   'Y_I': 1.0,
                   'I_inf': 1.0,
                   'alpha': 0.1}

    params_bounds_dict = {'mu_S': [0.8, 1.2],
                          'mu_I': [0.8, 1.2],
                          'K_S': [0.8, 1.2],
                          'K_I': [0.8, 1.2],
                          'D': [0.8, 1.2],
                          'Y_S': [0.8, 1.2],
                          'Y_I': [0.8, 1.2],
                          'I_inf': [0.8, 1.2],
                          'alpha': [0.8, 1.2]}

    init_cond = {'I_0': 0.0, 'S_0': 0.0, 'X_0': 0.1}

    init_cond_bounds = {'I_0': [0.8, 1.2], 'S_0': [0.8, 1.2], 'X_0': [0.8, 1.2]}

    param_dependencies = {}

    return p_list, d_list, ex_list, inflow_amount_list, dilution_rate_list, internal_prod_list, phi_list, vars_dict, params_dict, params_bounds_dict, init_cond, init_cond_bounds, param_dependencies


def build_double_compartment_primary():
    p_list = [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], ['Y_I', -1.0, 0, 0, 0, 0], [0, 'Y_S', -1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 'Y_I', -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 'Y_S', -1.0]]

    d_list = [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    ex_list = [[0.0, 0.0, 0.0, {'exin': [['gam_s_I', 'V_l', 'V_m'], None]}, 0.0, 0.0],
               [0.0, {'diffusion': [['-1', 'gam_d', 'V_l'], None]}, 0.0, 0.0,
                {'diffusion': [['1', 'gam_d', 'V_l'], None]},
                0.0], [0.0, 0.0, {'exout': [['gam_a_X'], None]}, 0.0, 0.0, {'exin': [['gam_s_X', 'V_l', 'V_m'], None]}],
               [0.0, 0.0, 0.0, {'exout': [['gam_s_I'], None]}, 0.0, 0.0],
               [0.0, {'diffusion': [['1', 'gam_d', 'V_m'], None]}, 0.0, 0.0,
                {'diffusion': [['-1', 'gam_d', 'V_m'], None]},
                0.0], [0.0, 0.0, {'exin': [['gam_a_X', 'V_m', 'V_l'], None]}, 0.0, 0.0, {'exout': [['gam_s_X'], None]}]]

    inflow_amount_list = ['I_inf', 0.0, 0.0, 0.0, 0.0, 0.0]

    dilution_rate_list = ['D', 'D', 'D', 'D', 'D', 'D']

    internal_prod_list = [0.0, 0.0, 0.0, {'mucprod': [['lam_max', 'lam_prod'], ['I_m']]}, 0.0, 0.0]

    phi_list = [{'contois': [['mu_I', 'K_I'], ['I_l', 'X_l']]}, {'monod': [['mu_S', 'K_S'], ['S_l', 'X_l']]},
                {'decay': [['alpha'], ['X_l']]}, {'contois': [['mu_I', 'K_I'], ['I_m', 'X_m']]},
                {'monod': [['mu_S', 'K_S'], ['S_m', 'X_m']]}, {'decay': [['alpha'], ['X_m']]}]

    vars_dict = {"I_l": 0, "S_l": 1, "X_l": 2, "I_m": 3, "S_m": 4, "X_m": 5}

    params_dict = {'D': 1.0071942760083039,
                   'I_inf': 1.0,
                   'K_I': 0.26539823,
                   'K_S': 0.468416,
                   'V_l': 0.9,
                   'V_m': 0.1,
                   'Y_I': 1.0,
                   'Y_S': 0.34242,
                   'alpha': 0.1,
                   'gam_a_X': 0.1,
                   'gam_d': 3.9,
                   'gam_s_I': 0.1,
                   'gam_s_X': 0.4,
                   'lam_max': 500.0,
                   'lam_prod': 50,
                   'mu_I': 10.619469027,
                   'mu_S': 12.627143363}

    params_bounds_dict = {'D': [0.8, 1.2],
                          'I_inf': [0.8, 1.2],
                          'K_I': [0.8, 1.2],
                          'K_S': [0.8, 1.2],
                          'V_l': [0.8, 1.2],
                          'V_m': [0.8, 1.2],
                          'Y_I': [0.8, 1.2],
                          'Y_S': [0.8, 1.2],
                          'alpha': [0.8, 1.2],
                          'gam_a_X': [0.8, 1.2],
                          'gam_d': [0.8, 1.2],
                          'gam_s_I': [0.8, 1.2],
                          'gam_s_X': [0.8, 1.2],
                          'lam_max': [0.8, 1.2],
                          'lam_prod': [0.8, 1.2],
                          'mu_I': [0.8, 1.2],
                          'mu_S': [0.8, 1.2],
                          }

    init_cond = {'Il_0': 0, 'Sl_0': 0, 'Xl_0': 0.1, 'Im_0': 0, 'Sm_0': 0, 'Xm_0': 0.1}

    init_cond_bounds = {'Il_0': [0.8, 1.2], 'Sl_0': [0.8, 1.2], 'Xl_0': [0.8, 1.2], 'Im_0': [0.8, 1.2],
                        'Sm_0': [0.8, 1.2],
                        'Xm_0': [0.8, 1.2]}

    param_dependencies = {}

    return p_list, d_list, ex_list, inflow_amount_list, dilution_rate_list, internal_prod_list, phi_list, vars_dict, params_dict, params_bounds_dict, init_cond, init_cond_bounds, param_dependencies

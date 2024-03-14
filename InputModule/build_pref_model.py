def build_pref_model():
    p_list = [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # I_l
              [0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # G_l
              ['Y_IX', 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # S_l
              [0.0, 'Y_GX', 0.0, -1.0, 0.0, 'Y_GZ', 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # R_l
              [0.0, 0.0, 'Y_SX', 'Y_RX', -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # X_l
              # [0.0, 0.0, 'Y_SX', 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # X_l
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Y_RZ', 'Y_SZ', -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Z_l
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # G_m
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # S_m
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Y_GX', 0.0, -1.0, 0.0, 'Y_GZ', 0.0, -1.0, -1.0, 0.0, 0.0],  # R_m
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Y_SX', 'Y_RX', -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # X_m
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Y_RZ', 'Y_SZ', -1.0]]  # Z_m

    d_list = [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 1
              [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2
              [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 3
              [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 4
              [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 5
              [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 6
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 7
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 8
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 9
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 10
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # 11

    ex_list = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # I_l
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {'exin': [['gam_sG', 'V_l', 'V_m'], None]}, 0.0, 0.0, 0.0, 0.0],  # G_l
               [0.0, 0.0, {'diffusion': [['-1', 'gam_dS', 'V_l'], None]}, 0.0, 0.0, 0.0, 0.0, {'diffusion': [['1', 'gam_dS', 'V_l'], None]}, 0.0, 0.0, 0.0],  # S_l
               [0.0, 0.0, 0.0, {'diffusion': [['-1', 'gam_dR', 'V_l'], None]}, 0.0, 0.0, 0.0, 0.0, {'diffusion': [['1', 'gam_dR', 'V_l'], None]}, 0.0, 0.0],  # R_l
               [0.0, 0.0, 0.0, 0.0, {'exout': [['gam_aX'], None]}, 0.0, 0.0, 0.0, 0.0, {'exin': [['gam_sX', 'V_l', 'V_m'], None]}, 0.0],  # X_l
               [0.0, 0.0, 0.0, 0.0, 0.0, {'exout': [['gam_aZ'], None]}, 0.0, 0.0, 0.0, 0.0, {'exin': [['gam_sZ', 'V_l', 'V_m'], None]}],  # Z_l
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {'exout': [['gam_sG'], None]}, 0.0, 0.0, 0.0, 0.0],  # G_m
               [0.0, 0.0, {'diffusion': [['1', 'gam_dS', 'V_m'], None]}, 0.0, 0.0, 0.0, 0.0, {'diffusion': [['-1', 'gam_dS', 'V_m'], None]}, 0.0, 0.0, 0.0],  # S_m
               [0.0, 0.0, 0.0, {'diffusion': [['1', 'gam_dR', 'V_m'], None]}, 0.0, 0.0, 0.0, 0.0, {'diffusion': [['-1', 'gam_dR', 'V_m'], None]}, 0.0, 0.0],  # R_m
               [0.0, 0.0, 0.0, 0.0, {'exin': [['gam_aX', 'V_m', 'V_l'], None]}, 0.0, 0.0, 0.0, 0.0, {'exout': [['gam_sX'], None]}, 0.0],  # X_m
               [0.0, 0.0, 0.0, 0.0, 0.0, {'exin': [['gam_aZ', 'V_m', 'V_l'], None]}, 0.0, 0.0, 0.0, 0.0, {'exout': [['gam_sZ'], None]}]]  # Z_m

    inflow_amount_list = ['I_inf', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    dilution_rate_list = ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']

    internal_prod_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {'mucprod': [['lam_prod', 'lam_max'], ['G_m']]}, 0.0, 0.0, 0.0, 0.0]

    phi_list = [{'contois': [['mu_IX', 'K_IX'], ['I_l', 'X_l']]},  # 1
                {'prefcontois': [['mu_GX', 'K_GX', 'omega'], ['G_l', 'I_l', 'X_l']]},  # 2
                {'monod': [['mu_SX', 'K_SX'], ['S_l', 'X_l']]},  # 3
                {'monod': [['mu_RX', 'K_RX'], ['R_l', 'X_l']]},  # 4
                {'decay': [['alpha_X'], ['X_l']]},  # 5
                {'contois': [['mu_GZ', 'K_GZ'], ['G_l', 'Z_l']]},  # 6
                {'monod': [['mu_SZ', 'K_SZ'], ['S_l', 'Z_l']]},  # 7
                {'monod': [['mu_RS', 'K_RS'], ['R_l', 'Z_l']]},  # 8
                {'monod': [['mu_RZ', 'K_RZ'], ['R_l', 'Z_l']]},  # 9
                {'doublemonod': [['mu_SZ', 'K_SZ', 'mu_RS', 'K_RS'], ['S_l', 'R_l', 'Z_l']]},  # 10
                {'decay': [['alpha_Z'], ['Z_l']]},  # 11
                {'prefcontois': [['mu_GX', 'K_GX', 'omega'], ['G_m', 'I_l', 'X_m']]},  # 12,
                {'monod': [['mu_SX', 'K_SX'], ['S_m', 'X_m']]},  # 13
                {'monod': [['mu_RX', 'K_RX'], ['R_m', 'X_m']]},  # 14
                {'decay': [['alpha_X'], ['X_m']]},  # 15
                {'contois': [['mu_GZ', 'K_GZ'], ['G_m', 'Z_m']]},  # 16
                {'monod': [['mu_SZ', 'K_SZ'], ['S_m', 'Z_m']]},  # 17
                {'monod': [['mu_RS', 'K_RS'], ['R_m', 'Z_m']]},  # 18
                {'monod': [['mu_RZ', 'K_RZ'], ['R_m', 'Z_m']]},  # 19
                {'doublemonod': [['mu_SZ', 'K_SZ', 'mu_RS', 'K_RS'], ['S_m', 'R_m', 'Z_m']]},  # 20
                {'decay': [['alpha_Z'], ['Z_m']]}]  # 21

    vars_dict = {"I_l": 0, "G_l": 1, "S_l": 2, "R_l": 3, "X_l": 4, "Z_l": 5, "G_m": 6, "S_m": 7, "R_m": 8, "X_m": 9, "Z_m": 10}

    params_dict = {'alpha_X': 0.1,
                   'alpha_Z': 0.1,
                   'gam_aX': 0.1,
                   'gam_aZ': 0.1,
                   'gam_dR': 3.9,
                   'gam_dS': 3.9,
                   'gam_sG': 0.1,
                   'gam_sX': 0.4,
                   'gam_sZ': 0.4,
                   'lam_max': 500.0,
                   'lam_prod': 50,
                   'mu_GX': 10.619469027,
                   'mu_GZ': 10.619469027,
                   'mu_IX': 10.619469027,
                   'mu_RS': 10.619469027,
                   'mu_RX': 12.627143363,
                   'mu_RZ': 12.627143363,
                   'mu_SX': 12.627143363,
                   'mu_SZ': 12.627143363,
                   'omega': 0.1,
                   'D': 1.0071942760083039,
                   'K_GX': 0.26539823,
                   'K_GZ': 0.26539823,
                   'K_IX': 0.26539823,
                   'K_RS': 0.468416,
                   'K_RX': 0.468416,
                   'K_RZ': 0.468416,
                   'K_SX': 0.468416,
                   'K_SZ': 0.468416,
                   'I_inf': 1.0,
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

    params_bounds_dict = {'alpha_X': [0.8, 1.2],
                   'alpha_Z': [0.8, 1.2],
                   'gam_aX': [0.8, 1.2],
                   'gam_aZ': [0.8, 1.2],
                   'gam_dR': [0.8, 1.2],
                   'gam_dS': [0.8, 1.2],
                   'gam_sG': [0.8, 1.2],
                   'gam_sX': [0.8, 1.2],
                   'gam_sZ': [0.8, 1.2],
                   'lam_max': [0.8, 1.2],
                   'lam_prod': [0.8, 1.2],
                   'mu_GX': [0.8, 1.2],
                   'mu_GZ': [0.8, 1.2],
                   'mu_IX': [0.8, 1.2],
                   'mu_RS': [0.8, 1.2],
                   'mu_RX': [0.8, 1.2],
                   'mu_RZ': [0.8, 1.2],
                   'mu_SX': [0.8, 1.2],
                   'mu_SZ': [0.8, 1.2],
                   'omega': [0.8, 1.2],
                   'D': [0.8, 1.2],
                   'K_GX': [0.8, 1.2],
                   'K_GZ': [0.8, 1.2],
                   'K_IX': [0.8, 1.2],
                   'K_RS': [0.8, 1.2],
                   'K_RX': [0.8, 1.2],
                   'K_RZ': [0.8, 1.2],
                   'K_SX':[0.8, 1.2],
                   'K_SZ': [0.8, 1.2],
                   'I_inf': [0.8, 1.2],
                   'V_l': [0.8, 1.2],
                   'V_m': [0.8, 1.2],
                   'Y_IX': [0.8, 1.2],
                   'Y_GX': [0.8, 1.2],
                   'Y_GZ': [0.8, 1.2],
                   'Y_RX': [0.8, 1.2],
                   'Y_RZ': [0.8, 1.2],
                   'Y_SX': [0.8, 1.2],
                   'Y_SZ': [0.8, 1.2]}

    init_cond = {"Il_0": 0.0, "Gl_0": 0.0, "Sl_0": 0.0, "Rl_0": 0.0, "Xl_0": 0.1, "Zl_0": 0.1, "Gm_0": 0.0, "Sm_0": 0.0, "Rm_0": 0.0, "Xm_0": 0.1, "Zm_0": 0.1}

    init_cond_bounds = {'Il_0': [0.8, 1.2], 'Gl_0': [0.8, 1.2], 'Sl_0': [0.8, 1.2], 'Rl_0': [0.8, 1.2],
                        'Xl_0': [0.8, 1.2], 'Zl_0': [0.8, 1.2], 'Gm_0': [0.8, 1.2], 'Sm_0': [0.8, 1.2],
                        'Rm_0': [0.8, 1.2], 'Xm_0': [0.8, 1.2], 'Zm_0': [0.8, 1.2]}

    param_dependencies = {}

    return p_list, d_list, ex_list, inflow_amount_list, dilution_rate_list, internal_prod_list, phi_list, vars_dict, \
        params_dict, params_bounds_dict, init_cond, init_cond_bounds, param_dependencies
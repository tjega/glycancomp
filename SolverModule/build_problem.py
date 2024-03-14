import numpy as np
import copy
from ProblemModule.reaction_class import Monod
from ProblemModule.reaction_class import Antmonod
from ProblemModule.reaction_class import Contois
from ProblemModule.reaction_class import Decay
from ProblemModule.reaction_class import Mucprod
from ProblemModule.reaction_class import Mucprodspec
from ProblemModule.reaction_class import Prefcontois
from ProblemModule.reaction_class import Doublemonod
from ProblemModule.reaction_class import Antdoublemonod
from ProblemModule.exchange_class import Exin
from ProblemModule.exchange_class import Diffusion
from ProblemModule.exchange_class import Exout


def build_petersen_and_friends(petersen_list, params_dict):
    petersen_array = copy.deepcopy(petersen_list)
    if isinstance(petersen_list[0], list):
        for row in petersen_array:
            for index, item in enumerate(row):
                if item in params_dict.keys():
                    row[index] = params_dict[item]
        petersen_array = np.array(petersen_array)
    else:
        petersen_array = [params_dict.get(item, item) for item in petersen_array]
        petersen_array = np.array(petersen_array)
    return petersen_array


def build_inflow_vector(list1, list2, params_dict):
    vector1 = build_petersen_and_friends(list1, params_dict)
    vector2 = build_petersen_and_friends(list2, params_dict)
    finalvector = vector1 * vector2
    return finalvector


def build_input_object(input_dict, params_dict):
    class_name = list(input_dict.keys())[0].capitalize()
    arg_list = build_petersen_and_friends(list(input_dict.values())[0][0], params_dict)
    input_object = globals()[class_name](*arg_list)
    return input_object


def get_vars_array(input_dict, z, vars_dict):
    if list(input_dict.values())[0][1] is not None and vars_dict is not None:
        vars_index = []
        for k, v in vars_dict.items():
            if k in list(input_dict.values())[0][1]:
                vars_index.append(v)
        z_vars = [z[i] for i in vars_index]
    else:
        z_vars = None
    return z_vars


def build_object_method(input_object, input_dict, z_vars=None):
    if z_vars is not None:
        methodout = getattr(input_object, list(input_dict.keys())[0])(*z_vars)
    else:
        methodout = getattr(input_object, list(input_dict.keys())[0])()
    return methodout


def build_object_list(input_list, params_dict):
    object_list = copy.deepcopy(input_list)
    if isinstance(object_list[0], list):
        for row in object_list:
            for index, item in enumerate(row):
                if isinstance(item, dict):
                    row[index] = build_input_object(item, params_dict)
    else:
        for index, item in enumerate(object_list):
            if isinstance(item, dict):
                object_list[index] = build_input_object(item, params_dict)
    return object_list


def eval_object_array(input_array, input_object_list, input_list, vars_dict=None, z=None):
    if isinstance(input_object_list[0], list):
        for i, row in enumerate(input_object_list):
            for j, item in enumerate(row):
                if hasattr(item, '__dict__') or hasattr(item, '__slots__'):
                    z_vars = get_vars_array(input_list[i][j], z, vars_dict)
                    input_array[i][j] = build_object_method(item, input_list[i][j], z_vars)
    else:
        for index, item in enumerate(input_object_list):
            if hasattr(item, '__dict__') or hasattr(item, '__slots__'):
                z_vars = get_vars_array(input_list[index], z, vars_dict)
                input_array[index] = build_object_method(item, input_list[index], z_vars)
    return input_array


def build_ode_problem_arrays(p_list, inflow_amount_list, dilution_rate_list, phi_list, ex_list, internal_prod_list,
                             d_list, params_dict, vars_dict):
    # Build np arrays (p_list, inflow_amount_list, dilution_rate_list)

    p_array = build_petersen_and_friends(p_list, params_dict)
    inflow_array = build_petersen_and_friends(inflow_amount_list, params_dict)
    dilution_rate_array = build_petersen_and_friends(dilution_rate_list, params_dict)

    # Build object arrays (phi_list, ex_list, internal_prod_list)
    phi_object_list = build_object_list(phi_list, params_dict)
    ex_object_list = build_object_list(ex_list, params_dict)
    internal_prod_object_list = build_object_list(internal_prod_list, params_dict)

    # Evaluate methods in object arrays with no z args (ex_object_list, internal_prod_object_list)
    ex_array = copy.deepcopy(ex_object_list)
    ex_array = eval_object_array(ex_array, ex_object_list, ex_list)

    # Make arrays for methods in object arrays with z args
    phi_array = copy.deepcopy(phi_object_list)
    internal_prod_array = copy.deepcopy(internal_prod_object_list)

    # Simplify problem arrays
    transport_array = np.multiply(params_dict['D'], d_list) + ex_array
    inflow_input_array = dilution_rate_array * inflow_array

    # Make dict to build dzdt
    problem_dict = {'p_array': p_array,
                    'phi_object_list': phi_object_list,
                    'phi_list': phi_list,
                    'phi_array': phi_array,
                    'transport_array': transport_array,
                    'inflow_input_array': inflow_input_array,
                    'internal_prod_object_list': internal_prod_object_list,
                    'internal_prod_list': internal_prod_list,
                    'internal_prod_array': internal_prod_array,
                    'params_dict': params_dict,
                    'vars_dict': vars_dict}
    return problem_dict


def build_problem_matrix(t, z, problem_dict):
    # Evaluate methods object arrays with z args

    problem_dict['phi_array'] = eval_object_array(problem_dict['phi_array'], problem_dict['phi_object_list'],
                                                  problem_dict['phi_list'], problem_dict['vars_dict'], z)
    problem_dict['internal_prod_array'] = eval_object_array(problem_dict['internal_prod_array'],
                                                            problem_dict['internal_prod_object_list'],
                                                            problem_dict['internal_prod_list'],
                                                            problem_dict['vars_dict'], z)

    # Multiply together to form dzdt
    dzdt = problem_dict['p_array'] @ problem_dict['phi_array'] + problem_dict['transport_array'] @ z + problem_dict[
            'inflow_input_array'] + problem_dict['internal_prod_array']

    return dzdt


def test_ode_dual(y, z, problem_dict):
    params = problem_dict['params_dict']
    mu_S = params['mu_S']
    mu_I = params['mu_I']
    K_S = params['K_S']
    K_I = params['K_I']
    D = params['D']
    Y_S = params['Y_S']
    Y_I = params['Y_I']
    I_inf = params['I_inf']
    alpha = params['alpha']
    V_l = params['V_l']
    V_m = params['V_m']
    gam_d = params['gam_d']
    gam_s_I = params['gam_s_I']
    gam_s_X = params['gam_s_X']
    gam_a_X = params['gam_a_X']
    lam_max = params['lam_max']
    lam_prod = params['lam_prod']

    I_l = z[0]
    S_l = z[1]
    X_l = z[2]
    I_m = z[3]
    S_m = z[4]
    X_m = z[5]

    def contois(I, X, mu_I, K_I):
        con = (mu_I * I) / ((K_I * X) + I)
        return con

    def monod(S, mu_S, K_S):
        mon = (mu_S * S) / (K_S + S)
        return mon

    def mucus_prod(I_m, lam_max, lam_prod):
        if (I_m / lam_max) > 1:
            muc = 0
        else:
            muc = (1 - (I_m / lam_max)) * lam_prod
        return muc

    dI_l = D * (I_inf - I_l) - contois(I_l, X_l, mu_I, K_I) * X_l + (V_m / V_l) * gam_s_I * I_m
    dS_l = -D * S_l - monod(S_l, mu_S, K_S) * X_l + Y_I * contois(I_l, X_l, mu_I, K_I) * X_l - (gam_d / V_l) * (
                S_l - S_m)
    dX_l = -D * X_l + Y_S * monod(S_l, mu_S, K_S) * X_l - alpha * X_l + (V_m / V_l) * gam_s_X * X_m - gam_a_X * X_l
    dI_m = mucus_prod(I_m, lam_max, lam_prod) - contois(I_m, X_m, mu_I, K_I) * X_m - gam_s_I * I_m
    # dI_m = - contois(I_m, X_m, mu_I, K_I) * X_m - gam_s_I * I_m
    dS_m = -monod(S_m, mu_S, K_S) * X_m + Y_I * contois(I_m, X_m, mu_I, K_I) * X_m + (gam_d / V_m) * (S_l - S_m)
    dX_m = Y_S * monod(S_m, mu_S, K_S) * X_m - alpha * X_m + (V_l / V_m) * gam_a_X * X_l - gam_s_X * X_m

    dzdt = [dI_l, dS_l, dX_l, dI_m, dS_m, dX_m]

    return dzdt

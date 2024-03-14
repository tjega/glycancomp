import numpy as np
from SALib.sample import saltelli, latin
from InputModule.param_dependencies import param_dependencies


def gen_samples(num_samples, params_dict, init_cond_dict, params_bounds, init_cond_bounds,
                exclude_params_list, bounds_per, even):

    params_init_dict = {**params_dict, **init_cond_dict}
    params_key_pos, sim_dict = exclude_params(params_init_dict, exclude_params_list)

    bounds = create_bounds(sim_dict, params_bounds, init_cond_bounds, exclude_params_list, bounds_per)
    bounds_lower = bounds[:, 0]
    bounds_upper = bounds[:, 1]

    num_vars = len(sim_dict)
    num_params = len(params_dict)
    name_params = list(sim_dict.keys())

    problem = {
        'num_vars': num_vars,
        'names': name_params,
        'bounds': bounds
    }
    if even == 0:
        vals = saltelli.sample(problem, num_samples, calc_second_order=True)
    elif even == 2:
        vals = latin.sample(problem, num_samples)
    elif even == 1:
        vals = np.linspace(bounds_lower, bounds_upper, num_samples, axis=0)
    else:
        vals = np.linspace(bounds_lower, bounds_upper, num_samples, axis=0)
        vals = 10 ** vals
        vals[0] = 0

    vals = update_sample_array(params_init_dict, params_key_pos, vals, exclude_params_list)

    params_vals = vals[:, :num_params]
    init_vals = vals[:, num_params:]

    return params_vals, init_vals, problem


def exclude_params(params_init_dict, exclude_params_list):
    sim_dict = dict(params_init_dict)
    # Get position of removed keys
    params_key_pos = []
    for index, (key, value) in enumerate(params_init_dict.items()):
        if key in exclude_params_list:
            params_key_pos.append(index)
    # Remove excluded parameters in sims_params_dict
    for k in exclude_params_list:
        sim_dict.pop(k, None)
    return params_key_pos, sim_dict


def create_bounds(sim_dict, params_bounds_dict, init_bounds_dict, exclude_params_list, bounds_per):
    for k in exclude_params_list:
        params_bounds_dict.pop(k, None)
        init_bounds_dict.pop(k, None)
    if bounds_per == 1:
        bounds = np.concatenate(
            (np.array(list(params_bounds_dict.values())), np.array(list(init_bounds_dict.values()))), axis=0)
        bounds = np.column_stack((np.multiply(np.array(list(sim_dict.values())), bounds[:, 0]), np.multiply(np.array(list(sim_dict.values())), bounds[:, 1])))
    else:
        if len(params_bounds_dict) == 0:
            bounds = np.array(list(init_bounds_dict.values()))
        elif len(init_bounds_dict) == 0:
            bounds = np.array(list(params_bounds_dict.values()))
        else:
            bounds = np.concatenate((np.array(list(params_bounds_dict.values())), np.array(list(init_bounds_dict.values()))), axis=0)
    return bounds


def update_sample_array(params_init_dict, params_key_pos, vals, exclude_params_list):
    sample_len = len(vals)
    exclude_params_dict = {key: val for key, val in zip(exclude_params_list, params_key_pos)}
    for key, val in exclude_params_dict.items():
        remove_params_array = np.full((1, sample_len), params_init_dict[key])
        vals = np.hstack((vals[:, :val], remove_params_array.reshape(-1, 1), vals[:, val:]))

    return vals


def update_param_init_dicts(params_dict, param_dependencies_dict, params_vals, sim_num):
    sim_params_dict = {key: val for key, val in zip(params_dict.keys(), params_vals[sim_num, :])}
    param_dependencies(sim_params_dict, param_dependencies_dict)
    return sim_params_dict

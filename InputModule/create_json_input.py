import json
from pathlib import Path


def create_input_file(input_list, input_filename):
    json_string = json.dumps(input_list, indent=4, sort_keys=True)
    json_file = open(input_filename, 'w')
    json_file.write(json_string)
    json_file.close()


def open_json_input_file(input_filename):
    file_object = open(input_filename, 'r')
    json_content = file_object.read()
    input_list = json.loads(json_content)
    return input_list


def create_problem_files(p_list, d_list, ex_list, inflow_amount_list, dilution_rate_list, internal_prod_list, phi_list, vars_dict, params_dict, init_cond, params_bounds_dict, init_cond_bounds, param_dependencies, folderpath):
    create_input_file(p_list, folderpath+'/p_list.json')
    create_input_file(d_list, folderpath+'/d_list.json')
    create_input_file(ex_list, folderpath+'/ex_list.json')
    create_input_file(inflow_amount_list, folderpath+'/inflow_amount_list.json')
    create_input_file(dilution_rate_list, folderpath+'/dilution_rate_list.json')
    create_input_file(internal_prod_list, folderpath+'/internal_prod_list.json')
    create_input_file(phi_list, folderpath+'/phi_list.json')
    create_input_file(vars_dict, folderpath+'/vars_dict.json')
    create_input_file(params_dict, folderpath+'/params_dict.json')
    create_input_file(init_cond, folderpath+'/init_cond.json')
    create_input_file(params_bounds_dict, folderpath+'/params_bounds_dict.json')
    create_input_file(init_cond_bounds, folderpath+'/init_cond_bounds.json')
    create_input_file(param_dependencies, folderpath+'/param_dependencies.json')


def open_problem_files(folderpath):
    p_list = open_json_input_file(folderpath+'/p_list.json')
    d_list = open_json_input_file(folderpath+'/d_list.json')
    ex_list = open_json_input_file(folderpath+'/ex_list.json')
    inflow_amount_list = open_json_input_file(folderpath+'/inflow_amount_list.json')
    dilution_rate_list = open_json_input_file(folderpath+'/dilution_rate_list.json')
    internal_prod_list = open_json_input_file(folderpath+'/internal_prod_list.json')
    phi_list = open_json_input_file(folderpath+'/phi_list.json')
    vars_dict = open_json_input_file(folderpath+'/vars_dict.json')
    params_dict = open_json_input_file(folderpath+'/params_dict.json')
    init_cond = open_json_input_file(folderpath+'/init_cond.json')
    params_bounds_dict = open_json_input_file(folderpath+'/params_bounds_dict.json')
    init_cond_bounds = open_json_input_file(folderpath+'/init_cond_bounds.json')
    param_dependencies = open_json_input_file(folderpath+'/param_dependencies.json')

    return p_list, d_list, ex_list, inflow_amount_list, dilution_rate_list, internal_prod_list, phi_list, vars_dict, params_dict, init_cond, params_bounds_dict, init_cond_bounds, param_dependencies


def update_sampled_json(params_dict, init_cond_dict, params_vals, init_vals, sim_folder):
    rows = params_vals.shape[0]
    for i in range(rows):
        simpath = sim_folder + '/Sim' + str(i)
        Path(simpath).mkdir(parents=True, exist_ok=True)
        sim_params_dict = {A: B for A, B in zip(params_dict.keys(), params_vals[i, :])}
        create_input_file(sim_params_dict, simpath + '/params_dict.json')
        sim_init_cond_dict = {X: Y for X, Y in zip(init_cond_dict.keys(), init_vals[i, :])}
        create_input_file(sim_init_cond_dict, simpath + '/init_cond.json')

# def multiple_rep_p_list(p_list, num_reps_dict):

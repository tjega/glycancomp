def param_dependencies(params_dict, param_dependencies_dict):
    arithop = None
    s = 0

    for key in param_dependencies_dict:
        for item in param_dependencies_dict[key]:
            if isinstance(item, float) is True or isinstance(item, int) is True:
                if arithop == '-':
                    s -= item
                elif arithop == '*':
                    s *= item
                elif arithop == '/':
                    s /= item
                elif arithop is None or '+':
                    s += item
                arithop = None
            if isinstance(item, str):
                if item in ['+', '-', '*', '/']:
                    arithop = item
                elif item in params_dict:
                    if arithop == '-':
                        s -= params_dict[item]
                    elif arithop == '*':
                        s *= params_dict[item]
                    elif arithop == '/':
                        s /= params_dict[item]
                    elif arithop is None or '+':
                        s += params_dict[item]
                    arithop = None
        params_dict[key] = s
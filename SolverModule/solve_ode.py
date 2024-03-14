import numpy as np
from scipy.integrate import solve_ivp
from SolverModule.build_problem import build_problem_matrix
from numpy import linalg as LA


def sim_steady_state(t, z, params):
    dzdt = build_problem_matrix(t, z, params)
    test = LA.norm(dzdt) - 1e-8
    return test


sim_steady_state.terminal = True
sim_steady_state.direction = -1
global flag
flag = 1


def chemo_sim(tstart, tend, tsteps, init_cond, problem_dict):
    t = np.linspace(tstart, tend, tsteps)

    res = solve_ivp(fun=build_problem_matrix, t_span=[tstart, tend], y0=init_cond, t_eval=t,
                    args=[problem_dict], events=sim_steady_state, method='Radau')

    return res


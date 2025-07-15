import math
from copy import deepcopy

import numpy as np


def get_benchmark_functions(names="branin,Currin"):
    """
    Returns a list of benchmark functions.
    """
    names = names.split(",")
    available_functions = {
        "branin": mfbranin,
        "Currin": mfCurrin,
        # Add other functions here as needed
    }
    functions = []
    for name in names:
        if name in available_functions:
            functions.append(available_functions[name])
        else:
            raise ValueError(f"Function '{name}' is not available.")
    return functions


def get_cost(names="branin,Currin"):
    """
    Returns a list of cost functions.
    """
    names = names.split(",")
    available_cost_functions = {
        "branin": [1, 10],
        "Currin": [1, 10],
        # Add other cost functions here as needed
    }
    cost_functions = []
    for name in names:
        if name in available_cost_functions:
            cost_functions.append(available_cost_functions[name])
        else:
            raise ValueError(f"Cost function '{name}' is not available.")
    return cost_functions


########################################################
def Currin(x1, d):
    x = deepcopy(x1)
    if x[1] == 0:
        x[1] = 1e-100
    return -1 * float(
        (
            (1 - math.exp(-0.5 * (1 / x[1])))
            * (
                (2300 * pow(x[0], 3) + 1900 * x[0] * x[0] + 2092 * x[0] + 60)
                / (100 * pow(x[0], 3) + 500 * x[0] * x[0] + 4 * x[0] + 20)
            )
        )
    )


def lowFideliltyCurrin(x, d):
    a = [x[0] + 0.05, x[1] + 0.05]
    b = [x[0] + 0.05, max(0, x[1] - 0.05)]
    c = [x[0] - 0.05, x[1] + 0.05]
    f = [x[0] - 0.05, max(0, x[1] - 0.05)]
    ### we do not multply by -1 because it will come from Currin
    return 0.25 * (Currin(a, d) + Currin(b, d) + Currin(c, d) + Currin(f, d))


def mfCurrin(x, d, f):
    if f == 1:
        return Currin(x, d)
    elif f == 0:
        return lowFideliltyCurrin(x, d)


##################################################################3
def branin(x1, d):
    x = deepcopy(x1)
    x[0] = 15 * x[0] - 5
    x[1] = 15 * x[1]
    return -1 * float(
        np.square(
            x[1]
            - (5.1 / (4 * np.square(math.pi))) * np.square(x[0])
            + (5 / math.pi) * x[0]
            - 6
        )
        + 10 * (1 - (1.0 / (8 * math.pi))) * np.cos(x[0])
        + 10
    )


def lowFideliltybranin(x1, d):
    x = deepcopy(x1)
    x[0] = 15 * x[0] - 5
    x[1] = 15 * x[1]
    first_term = (
        x[1]
        - (5.1 / (4 * pow(np.pi, 2)) - 0.01) * x[0] ** 2
        + (5 / np.pi - 0.1) * x[0]
        - 6
    ) ** 2
    second_term = 10 * (1 - 1.0 / (8 * np.pi) - 0.05) * np.cos(x[0]) + 10
    return -(first_term + second_term)  # + (x[0]+5)/3


def mfbranin(x1, d, f):
    if f == 1:
        return branin(x1, d)
    elif f == 0:
        return lowFideliltybranin(x1, d)


##############################################################

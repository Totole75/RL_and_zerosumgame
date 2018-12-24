# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 10:16:49 2018

@author: anatole parre
@adam5ny/linear-programming-in-python-cvxopt-and-game-theory-8626a143d428
"""

import numpy as np
from cvxopt import matrix, solvers
solvers.options['glpk'] = {'tm_lim': 1000} # max timeout for glpk


def print_sol(sol):
    print("Reward array for this game : ")
    print(reward_array)
    print("")
    print("Value of this game : ")
    print(opt_valeur)
    print("An optimal strategy : ")
    print(opt_strategy)
    print("")

def solve_2players(A, solver="glpk", verbose=True):
    """
    Solve a zero sum game and return its value and an optimal strategy
    """
    num_vars = len(A)
    # minimize matrix c
    c = [-1] + [0 for i in range(num_vars)]
    c = np.array(c, dtype="float")
    c = matrix(c)
    # constraints G*x <= h
    G = np.matrix(A, dtype="float").T # reformat each variable is in a row
    G *= -1 # minimization constraint
    G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
    new_col = [1 for i in range(num_vars)] + [0 for i in range(num_vars)]
    G = np.insert(G, 0, new_col, axis=1) # insert utility column
    G = matrix(G)
    h = ([0 for i in range(num_vars)] + 
         [0 for i in range(num_vars)])
    h = np.array(h, dtype="float")
    h = matrix(h)
    # contraints Ax = b
    A = [0] + [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver, verbose=False)
    
    if verbose:
        print_sol(sol)
    
    return(sol['primal objective'], sol['x'][1:])
    
    
###############################################################
    
def build_ce_constraints(A):
    num_vars = int(len(A) ** (1/2))
    G = []
    # row player
    for i in range(num_vars): # action row i
        for j in range(num_vars): # action row j
            if i != j:
                constraints = [0 for i in A]
                base_idx = i * num_vars
                comp_idx = j * num_vars
                for k in range(num_vars):
                    constraints[base_idx+k] = (- A[base_idx+k][0]
                                               + A[comp_idx+k][0])
                G += [constraints]
    # col player
    for i in range(num_vars): # action column i
        for j in range(num_vars): # action column j
            if i != j:
                constraints = [0 for i in A]
                for k in range(num_vars):
                    constraints[i + (k * num_vars)] = (
                        - A[i + (k * num_vars)][1]
                        + A[j + (k * num_vars)][1])
                G += [constraints]
    return(np.matrix(G, dtype="float"))


def ce(A, solver=None):
    num_vars = len(A)
    # maximize matrix c
    c = [sum(i) for i in A] # sum of payoffs for both players
    c = np.array(c, dtype="float")
    c = matrix(c)
    c *= -1 # cvxopt minimizes so *-1 to maximize
    # constraints G*x <= h
    G = build_ce_constraints(A=A)
    G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
    h_size = len(G)
    G = matrix(G)
    h = [0 for i in range(h_size)]
    h = np.array(h, dtype="float")
    h = matrix(h)
    # contraints Ax = b
    A = [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver, verbose=False)
    return(sol['primal objective'], sol['x'])
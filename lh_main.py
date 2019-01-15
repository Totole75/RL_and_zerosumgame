# -*- coding: utf-8 -*-
#!/usr/bin/env python
# vim:fileencoding=utf8
#
# Project: Implementation of the Lemke-Howson algorithm for finding MNE
# Author:  Petr Zemek <s3rvac@gmail.com>, 2009
#

"""Runs a program which computes MNE in the given 2-player game
using the Lemke-Howson algorithm.
"""


import sys
import numpy as np

def solve(loss_array):
    # These imports must be here because of possible
    # SyntaxError exceptions in different versions of python
    # (this program needs python 2.5)
    import io_lh
    import lh

    with open("game.txt", 'w') as file:
        for elem in [" ".join(item) for item in (-loss_array).astype(str)]:
            file.write(elem)
            file.write("\n")
        
        file.write("\n")
        
        for elem in [" ".join(item) for item in loss_array.astype(str)]:
            file.write(elem)
            file.write("\n")
        
    file_object = open('game.txt')
    txt = "".join(file_object.readlines())
    #print(txt)
    m1, m2 = io_lh.parseInputMatrices(txt)

    # Obtain input matrices from the standard input
    #m1, m2 = io_lh.parseInputMatrices(sys.stdin.read())

    # Compute the equilibirum
    eq = lh.lemkeHowson(m1, m2)
    player_1 = [rational_p1.nom()/rational_p1.denom() for rational_p1 in eq[0]]
    
    player_2 = [rational_p2.nom()/rational_p2.denom() for rational_p2 in eq[1]]
    
    
    opt_valeur = np.array(player_1).dot(-loss_array.dot(np.array(player_2)))
    
    player_1 = np.array(player_1).reshape(-1,1)
    player_2 = np.array(player_2).reshape(-1,1)
    opt_strategies = [player_1, player_2]
    
    return(opt_valeur, opt_strategies)
    # Print both matrices and the result
    #io_lh.printGameInfo(m1, m2, eq, sys.stdout)

    #return 0


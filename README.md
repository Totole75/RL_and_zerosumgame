# RL and zero sum games : finding equilibria

Work mainly based on the book "Prediction, Learning, and Games" by CESA-BIANCHI and LUGOSI

For more details, see section 

## Strategies that have been implemented

Fictitious play (not Hannan consistent) and Perturbed fictitious play (Hannan consistent)

Exponentially weighted average (Hannan consistent)

Regret matching (Hannan consistent)

Deterministic exploration exploitation

## What to do with the code ?

Each strategy is implemented as a class, and all of them inherit from a common class. The objective is to use a unified framework when confronting them. The corresponding code can be found in `strategy.py`.

Simulations are done thanks to `simulation.py`.

`convergence_check.py` proposes to run more efficiently different strategies against random games.

`speed.py` is here to get more graphical insights on how fast strategies converge to the optimum.

`repeated_simulations.py` helps get experimental validation on theoretical bounds.
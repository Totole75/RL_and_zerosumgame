# RL_and_zerosumgame
Find equilibrium in games with RL

Work based on the book "Prediction, Learning, and Games" by CESA-BIANCHI and LUGOSI

## Points bloquants

Difference entre equilibre de Nash et stratégie optimale associée à la valeur

Remarque 7.4, convergence vers l'équilibre de Nash, mais pas vers l'équilibre de Nash

Explication de l'algorithme d'exploration exploitation 

## Code

Implémentation en classe des méthodes

## Méthodes à implémenter

Résolution numérique du minmax
Descente/Montée de gradient

Fictitious play (not Hannan consistent)
Possibilité de rendre Hannan consistent en perturbant légèrement

Exponentially weighted average (Hannan consistent)
Affichage de la borne (Corollaire 4.2)

Regret matching Vianney Perchet

section 7.11
Deterministic exploration exploitation

Multi Armed Bandit UCB

## Bonus

Regret interne et jeu à m joueurs (pour atteindre des équilibres corrélés)

Uncoupled strategy to find PURE/MIX equilibria (p206)
Bad complexity in the case of m players, quadratic in our case ?
Version stationnaire en exercice 7.25

Time varying games (6.7 and 7.5), résolu avec regret interne

Unknown game : Experimental Regret Testing
Th 7.8 pour calibrer les parametres
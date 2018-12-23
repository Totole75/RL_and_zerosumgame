# RL_and_zerosumgame

Find equilibrium in games with RL

Work based on the book "Prediction, Learning, and Games" by CESA-BIANCHI and LUGOSI

Any two player zero sum game played with Hannan consistent strategies leads to the convergence of the frequency distributions
to a Nash Equilibrium (Corollary 7.1 p.189)

## Concerning the UCB

Doesn't work well against an oblivious opponent equipped with an optimal strategy, or himself.

The algorithm learns well when there's a noticeable difference between the different arms to pull.

Here the gap is too narrow so it just focuses on one arm, the one that worked well in the beginning,
and it keeps choosing it over the other. And the upper bound term grows too slowly to make a difference on
10000 steps or so.

For the situation opposing two bandits, they both play in the beginning their arms in a predetermined order,
and the results do have a huge influence on the game afterwards.

I don't know if it converges to a correlated equilibirum or else though.

## Implementation of the fictitious play (aka follow the leader)

Details for the implementation (value for the perturbation interval, etc ...) are provided in section 4 p.76 of the book.

## Exponentially weighted average

Details for the implementation (for instance the value for the coefficient \eta) are also provided in section 4 of the book.
See corollary 4.3 p.73

## Points bloquants

Difference entre equilibre de Nash et stratégie optimale associée à la valeur

Remarque 7.4, convergence vers l'équilibre de Nash, mais pas vers l'équilibre de Nash

Explication de l'algorithme d'exploration exploitation 

## Méthodes à implémenter

Résolution numérique du minmax
Descente/Montée de gradient

Fictitious play (not Hannan consistent)
Possibilité de rendre Hannan consistent en perturbant légèrement
Borne à tracer (corollaire 4.4 p76, non oblivious opponent case)

Exponentially weighted average (Hannan consistent)
Affichage de la borne (Corollaire 4.2)

Regret matching Perchet

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
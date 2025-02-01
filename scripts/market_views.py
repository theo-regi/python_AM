import numpy as np
"""=================================================================================
 PYTHON FILE FOR MARKET VIEWS, FOR MORE CLARITY VIEWS ARE EXPLAINED AND DEFINED HERE
===================================================================================="""
"""
Pour rappel, l'ordre des actifs est:
['Cash', 'Eq DM', 'Eq EM', 'Gov DM', 'Corp DM IG', 'Corp HY', 'Bonds EM AGG']
"""

#=============================================================================
#                       MARKET VIEWS from PARTIE 6
#=============================================================================
#Le 02/08/2002 (un an après eclatement de la bulle internet)
#Par conséquent, il nous faudrait des actifs plus défensif, 
#et ainsi sous ponderer les actions. Puis, ponderer plus fortement 
#les obligations notamment obligations HY et investment grades.
W1= [0, 0.0, 0.0, 0, 0.2, 0.8, 0.0]

#Le 30/06/2007 (avant la grande crise financière), on cherche donc a reduire les 
#actifs risqués en connaissance de cause afin de limiter notre chute 
#face a la crise des subprimes -> 100 % oblig d'etats.
W2 = [0, 0.00, 0.00, 1, 0.00, 0.0, 0.00]

#Le 09/03/2009 (après la grande crise financière):
#on cherche ici a profiter de l'importante baisse au niveau des actions
#suite a la crise. On pondere plus fortement les actions.
W3 = [0, 0.75, 0.25, 0.0, 0.00, 0.0, 0]

#Avant le Covid, le 31/12/2019: on repart sur une stratégie plus defensive
#en augmentant la part obligations, on garde encore tres legerement des actions
#qui peuvent continuer de performer un peu a cette date.
W4 = [0, 0.05, 0.05, 0.70, 0.20, 0.0, 0]

#Pendant le Covid, le 15/03/2020: on ralentit notre stratégie plus defensive
#car aides importantes des Etats pour relancer l'economie, par conséquent on 
#favorise les actions qui sont principalement touchées par ces plans de relance,
#c'est à dire les actions des marchés développés.
W5 = [0, 0.6, 0.20, 0.0, 0.05, 0.15, 0.0]

#Le 31/12/2021 (après le Covid): periode tres volatile pour les actions,
#on se met sur les obligations d'état notamment.
W6 = [0.0, 0.2, 0.1, 0.7, 0.0, 0.0, 0.0]

#Le 31/12/2022 (après le pic d’inflation): on remet sur les actions pour 
#prendre le plus de performance possible sur le marché qui reprend de l'élan 
#notamment sur les marchés développés.
W7= [0, 0.95, 0.05, 0.00, 0.0, 0.0, 0]

#=============================================================================
#                     MARKET VIEWS FOR BLACK-LITTERMAN BACKTEST
#=============================================================================
# Vues sur les actifs exprimées par plusieurs relations
views = np.array([
    [0, -0.5, 0.5, 0, 0, 0, 0], # Vue 1 : Actions des marchés émergents > Actions des marchés développés
    [0, 0, 0, 1, -0.5, 0, 0],   # Vue 2 : Obligations d'État > Obligations Investment Grade
    [0, 1, -1, 0, 0, 0, 0],     # Vue 3 : Actions des marchés développés > Actions des marchés émergents
    [0, 0, 0, -1, 0, 1, 0],     # Vue 4 : Obligations à haut rendement > Obligations d'État
    [0, 0, 1, -1, 0, 0, 0],     # Vue 5 : Actions des marchés émergents > Obligations d'État
    [0, 0, 0, 0, -1, 1, 0],     # Vue 6 : Obligations Investment Grade < Obligations à haut rendement
    [0, 1, 0, 0,-0.4,-0.6, 0],  # Vue 7 : Actions des marchés émergents > Obligations Investment Grade et High Yield
    [0, 0, 0, 1, 0, 0, -1],     # Vue 8 : Obligations d'État > Obligations Emergentes Agrégées
    [0, 0, 0, 1, 0, -1, 0],     # Vue 9 : Obligations d'État > Obligations à haut rendement
    [0, 0, -1, 1, 0, 0, 0],     # Vue 10 : Obligations d'État > Actions des marchés émergents
    [0, 0, 0, 0, 1, -1, 0],     # Vue 11 : Obligations Investment Grade > Obligations à haut rendement
    [0, 0, 1, 0, 0, 0, -1],     # Vue 12 : Actions des marchés émergents > Obligations Emergentes Agrégées
    [0,0.5,-0.5,0.2,-0.2,0,0],  # Vue 13 : Actions des marchés développés > Actions des marchés émergents, avec un focus sur obligations d'État
    [0, 0, 0, -1, 0.5, 0.5, 0], # Vue 14 : Obligations d'État < Obligations Investment Grade et Obligations à haut rendement
    [0, 0.8, 0, -0.8, 0, 0, 0], # Vue 15 : Actions des marchés développés > Obligations d'État
    [0, 0.8, 0.8, 0, 0, -1, 0], # Vue 16 : Actions > Obligations à haut rendement
    [0, -1, 0.3, 0.7, 0, 0, 0], # Vue 17 : Obligations d'État > Actions des marchés développés partiellement
    [0, 0, 0.4, -0.6,0,0.2, 0], # Vue 18 : Actions des marchés émergents avec un léger avantage par rapport aux obligations d'État
    [0, -0.5, 0.5, 0, 0, 0, 1]  # Vue 19 : Obligations Emergentes Agrégées > Combinaison d'actions développées et émergentes
])

# Rendements espérés pour chaque vue
exp_returns = np.array([
    0.06,  # Vue 1 : Actions des marchés émergents > Actions des marchés développés
    0.03,  # Vue 2 : Obligations d'État > Obligations Investment Grade
    0.09,  # Vue 3 : Actions des marchés développés > Actions des marchés émergents
    0.07,  # Vue 4 : Obligations à haut rendement > Obligations d'État
    0.06,  # Vue 5 : Actions des marchés émergents > Obligations d'État
    0.025, # Vue 6 : Obligations Investment Grade < Obligations à haut rendement
    0.045, # Vue 7 : Actions des marchés émergents > Obligations Investment Grade et High Yield
    0.03,  # Vue 8 : Obligations d'État > Obligations Emergentes Agrégées
    0.035, # Vue 9 : Obligations d'État > Obligations à haut rendement
    0.05,  # Vue 10 : Obligations d'État > Actions des marchés émergents
    0.025, # Vue 11 : Obligations Investment Grade > Obligations à haut rendement
    0.05,  # Vue 12 : Actions des marchés émergents > Obligations Emergentes Agrégées
    0.035, # Vue 13 : Actions des marchés développés > Actions des marchés émergents (focus obligations d'État)
    0.03,  # Vue 14 : Obligations d'État < Obligations Investment Grade et Obligations à haut rendement
    0.04,  # Vue 15 : Actions des marchés développés > Obligations d'État
    0.045, # Vue 16 : Actions > Obligations à haut rendement
    0.02,  # Vue 17 : Obligations d'État > Actions des marchés développés partiellement
    0.05,  # Vue 18 : Actions des marchés émergents avec léger avantage par rapport aux obligations d'État
    0.04   # Vue 19 : Obligations Emergentes Agrégées > Combinaison d'actions développées et émergentes
])

def get_viewsBL():
    return views

def get_returnsBL():
    return exp_returns

def get_market_views():
    return W1, W2, W3, W4, W5, W6, W7
# HorseProno Quinté V4.6 — Dual Engine

V4.6 sépare volontairement les deux sources d'information :

1. **Moteur fondamental (0 % cotes)** : forme historique, musique, cheval, jockey, entraîneur, poids, corde, distance, fraîcheur, discipline et contextes antérieurs.
2. **Moteur marché V4.5** : sélection V4.2/V4.3 + ordre Winner V4.4 + dynamique des cotes projetée vers T-2.

## Statut de production

Le Top7 officiel reste celui de V4.5. La fusion automatique n'est pas activée car les règles de substitution testées sur le développement n'ont pas amélioré le bloc temporel suivant. Le moteur fondamental est donc exploité en **shadow mode** comme contre-signal indépendant.

L'interface affiche :
- **Noyau consensus** : chevaux présents dans les deux Top7 ;
- **Marché / V4.5 seulement** : chevaux retenus uniquement par le moteur actuel ;
- **Réserve fondamentale** : chevaux du Top7 sans cotes absents du Top7 officiel ;
- le **Top7 fondamental 0 % cotes** complet.

## Backtest walk-forward — 58 Quintés

- V4.4/V4.5 statique : 3,379 chevaux Top5 retrouvés / 5 en moyenne ; 9 Quintés complets dans les 7.
- Fondamental seul : 2,759 / 5 ; 1 Quinté complet.
- Union des deux Top7 (plafond d'information, environ 9 chevaux et non une sélection jouable de 7) : 3,776 / 5 ; 15 Quintés complets ; gagnant présent dans 87,9 % des courses.
- Consensus moyen : 5,02 chevaux communs sur 7.
- Première réserve fondamentale : cheval de l'arrivée Top5 dans 15,5 % des courses.

L'union est uniquement un **plafond de complémentarité**, pas un pronostic de 7 chevaux. Elle montre que le moteur fondamental apporte parfois une information indépendante du marché, mais qu'on ne sait pas encore de manière robuste quand l'utiliser pour remplacer un cheval V4.5.

## Déploiement

Main file Streamlit : `streamlit_app.py`

La version reste portable : `lightgbm` n'est pas requis à l'exécution. `portable_lgbm.py` lit directement les deux fichiers texte LambdaMART.

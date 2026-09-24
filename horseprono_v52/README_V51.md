# HorseProno Quinté V5.1 — Meta-Consensus

V5.1 ajoute une couche de méta-consensus au-dessus de V4.6. Elle **ne modifie pas** le Top7 officiel tant que la validation forward n'a pas démontré un gain robuste.

## Ce que V5.1 calcule

- score Meta Top5 pour chaque candidat de l'union Marché + Fondamental ;
- score Meta Winner ;
- noyau de consensus et indice de divergence ;
- confiance de consensus 0–100 % ;
- lecture de course : consensus fort, marché plus convaincant, fondamental à surveiller, divergence élevée ;
- Top7 et Winner Top3 alternatifs en **shadow mode**.

## Protocole historique

Le méta-modèle a été réglé sur les 30 premières courses walk-forward puis observé sur 28 courses suivantes. Le modèle libre n'a pas battu la baseline :

- V4.4/V4.5 baseline holdout : 3,32/5 ; 10 Trios ; 8 Quartés ; 5 Quintés ; gagnant Top3 14/28.
- Meta libre holdout : 3,29/5 ; 10 Trios ; 8 Quartés ; 4 Quintés ; gagnant Top3 12/28.

En revanche, l'union Marché + Fondamental sur ces mêmes 28 courses contient 3,75/5 chevaux de l'arrivée en moyenne et 9 Quintés complets. L'information complémentaire existe donc, mais l'arbitrage n'est pas encore assez fiable.

## Validation forward

Chaque lecture Quinté peut être enregistrée dans `public.v51_meta_shadow_predictions` si la clé serveur Supabase est configurée. Le Top7 officiel, le Top7/Top3 shadow, consensus, divergence et scores runners sont figés avant course pour une vraie évaluation forward.

## Déploiement

Main file : `streamlit_app.py`

La production n'a besoin ni de scikit-learn ni de LightGBM : les coefficients du méta-modèle sont exportés en JSON et le ranker reste portable Python.

## e-Trio — 10 combinaisons, 2 bases Top5

L'application génère désormais automatiquement 10 tickets e-Trio :
- les 5 premiers du classement officiel fournissent les bases ;
- les 10 paires possibles parmi ces 5 chevaux sont utilisées exactement une fois ;
- le 3e cheval est toujours hors Top5 ;
- jusqu'à 5 associés sont priorisés par le consensus V5.1/V4.6, le Top7 officiel, le Top7 shadow et la réserve fondamentale ;
- les associés sont répartis de manière équilibrée afin de diversifier les 10 tickets.

Cette couche de construction de tickets ne modifie ni le Top7 officiel ni les scores des modèles.

## Générateur Quinté 70 intelligent

La version actuelle ajoute un portefeuille de 70 combinaisons parmi les 252 combinaisons possibles du Top10. Ce portefeuille n'est pas aléatoire : il mélange quatre structures de risque (5+0, 4+1, 3+2, 2+3), tient compte du consensus V5.1/V4.6 et limite la redondance des paires et triples.

L'`Indice modèle` affiché pour chaque ticket est un indicateur comparatif interne, pas une probabilité de gain.

## Quinté 70 rétro-calibré

Le générateur de 70 combinaisons n'utilise plus des quotas fixes 5+0 / 4+1 / 3+2 / 2+3.
Il conserve les 7 premiers du classement officiel et complète le pool à 10 avec les 3 meilleures réserves fondamentales. Parmi les 252 combinaisons possibles, chaque ticket reçoit :

- 60 % de signal courant V5.1/V4.6 ;
- 40 % de prior historique appris sur les 30 premières courses walk-forward.

Le prior historique tient compte de la fréquence des rangs dans le Top5 réel et de la structure du ticket (nombre de chevaux issus des 5 premiers rangs).
Sur le bloc temporel de contrôle de 28 courses, le pool de 10 contenait le Quinté complet 13 fois. L'ancien générateur 70 couvrait 6 de ces 13 cas ; la version rétro-calibrée en couvre 8. Ce résultat historique reste exploratoire et doit être confirmé en forward.

Fichiers de contrôle :
- `v51_quinte70_history_artifact.json`
- `quinte70_retro_backtest_58.csv`
- `quinte70_retro_summary.json`

## V5.2 — Adaptive Tickets

Le portefeuille Quinté n'est plus fixé à 70 tickets. Les 70 candidats rétro-calibrés sont toujours calculés, puis une règle pré-course gelée retient 30, 50 ou 70 tickets :

- **Core 30** : consensus V5.1 >= 6/7, divergence < 30 %, confiance >= 58 %.
- **Extension 50** : cas intermédiaire.
- **Couverture 70** : consensus <= 2/7, divergence >= 50 % ou confiance <= 35 %.

Sur la reconstruction temporelle HOLDOUT28, cette politique utilise 45 tickets/course en moyenne contre 70 systématiques, soit environ 36 % de mise théorique en moins à 2 €/ticket, tout en conservant les 8 Quintés exacts couverts par les 70 tickets dans cette reconstruction. Ce résultat est historique et doit être confirmé en forward.

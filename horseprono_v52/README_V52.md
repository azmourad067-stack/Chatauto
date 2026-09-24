# HorseProno Quinté V5.2 — Adaptive Tickets

V5.2 conserve intégralement le moteur V5.1 Meta-Consensus et le générateur Quinté 70 rétro-calibré, mais ajoute une couche de gestion du nombre de tickets.

## Règle gelée 30 / 50 / 70

- CORE 30 : consensus >= 6/7, divergence < 30 %, confiance >= 58 %.
- EXTENSION 50 : profil intermédiaire.
- COUVERTURE 70 : consensus <= 2/7, divergence >= 50 % ou confiance <= 35 %.

La décision utilise uniquement des informations disponibles avant la course.

## Reconstruction temporelle

Sur le bloc HOLDOUT28 de la reconstruction V5.2 :
- 7 courses en CORE 30 ;
- 21 courses en EXTENSION 50 ;
- 0 course en COUVERTURE 70 ;
- 45 tickets par course en moyenne ;
- 2 520 € de mise théorique à 2 €/ticket contre 3 920 € avec 70 tickets systématiques ;
- 8 Quintés exacts couverts dans la reconstruction, identique à la couverture obtenue avec les 70 tickets complets sur ce bloc.

Le bloc DEV30 donne 11 / 17 / 2 courses dans les trois profils, soit 44 tickets/course en moyenne. Une des cinq combinaisons exactes couvertes par les 70 tickets se situait au rang 61 et n'est pas retenue par la politique adaptative : V5.2 privilégie volontairement l'efficience du capital plutôt que la couverture maximale.

Ces chiffres sont historiques et exploratoires. Ils doivent être confirmés en forward. Ils ne constituent pas une garantie de rentabilité.

# HorseProno Quinté V2

## Ce qui change par rapport à V1

V2 est centré sur le problème réel du Quinté : **faire entrer le plus possible de chevaux de l'arrivée officielle Top 5 dans une shortlist Top 7**.

La recette de production a été choisie avec un protocole à deux niveaux :

1. walk-forward strict sur les supports Quinté officiels ;
2. choix de la recette sur les 30 premières courses de la zone walk-forward ;
3. contrôle final sur les 28 dernières courses laissées de côté.

La recette retenue est volontairement simple :

- **6 chevaux noyau** = les six premiers du marché PMU normalisé dans la course ;
- **1 challenger modèle** = le meilleur cheval hors noyau selon la probabilité Top-5 logistique ;
- la probabilité Top-5 est recalibrée par Platt sur des prédictions OOF walk-forward ;
- LambdaMART est conservé dans le package comme **challenger offline**, car il n'a pas apporté un gain assez robuste pour remplacer la recette de production.

## Backtest strict

Population : 99 supports Quinté+ officiels entre le 01/06/2026 et le 08/09/2026, dont **98 évaluables**. Le 30/08/2026 R1C3 est exclu à cause d'un ex aequo. Cela représente **1 525 partants**.

Le walk-forward commence après 40 supports d'apprentissage et évalue **58 courses**.

### 58 courses walk-forward

| Métrique | Marché seul | V2 |
|---|---:|---:|
| Chevaux du Top5 réel présents dans Top7 (moyenne) | 3,155 | **3,293** |
| 4/5 ou mieux dans Top7 | 37,9 % | **41,4 %** |
| 5/5 dans Top7 | 6,9 % | **8,6 %** |
| Gagnant dans Top7 | **81,0 %** | 79,3 % |
| NDCG@7 | 0,5529 | **0,5539** |

### Holdout final de 28 courses

| Métrique | Marché seul | V2 |
|---|---:|---:|
| Chevaux du Top5 réel présents dans Top7 (moyenne) | 3,107 | **3,250** |
| 4/5 ou mieux dans Top7 | 35,7 % | **42,9 %** |
| 5/5 dans Top7 | 7,1 % | **10,7 %** |
| Gagnant dans Top7 | **75,0 %** | 71,4 % |

Le gain sur `4/5` et `5/5` est encourageant, mais l'échantillon reste limité. Les intervalles bootstrap sont enregistrés dans `quinte_v2_backtest_summary.json` et ne permettent pas encore d'affirmer une amélioration statistiquement certaine à 95 %.

## Fichiers

- `streamlit_app.py` : application Streamlit, programme PMU complet + sélection de course.
- `quinte_candidate.py` : préparation temporelle anti-fuite héritée de V1.
- `quinte_v2.py` : recette de production V2.
- `quinte_v2_artifact.json` : coefficients du modèle Top5 calibré + métadonnées.
- `quinte_v2_ranker.txt` : LambdaMART offline challenger.
- `train_quinte_v2.py` : entraînement + walk-forward reproductible.
- `quinte_v2_walkforward.csv` : résultats course par course.
- `quinte_v2_backtest_summary.json` : synthèse des métriques et bootstrap.
- `quinte_supports_99.csv` : supports Quinté+ officiels utilisés.
- `validated_history.csv` : historique validé toutes courses pour les variables antérieures.
- `capture_market_snapshot.py` : capture horodatée des cotes PMU.
- `supabase_quinte_v2.sql` : schéma reproductible pour les nouvelles tables Supabase.

## Vérification locale rapide

Avant le déploiement, tu peux vérifier que l'artefact, l'historique et le moteur sont compatibles :

```bash
python smoke_test_v2.py
```

Le test ne contacte ni PMU ni Supabase et doit terminer par `OK - HorseProno Quinté V2`.

## Déploiement Streamlit Community Cloud

Main file path :

```text
streamlit_app.py
```

Dans les Secrets Streamlit :

```toml
SUPABASE_URL = "https://VOTRE-PROJET.supabase.co"
SUPABASE_KEY = "VOTRE_CLE_DE_LECTURE"
```

Pour enregistrer automatiquement un snapshot au moment où tu analyses un Quinté dans l'app, ajoute aussi côté **serveur Streamlit** :

```toml
SUPABASE_SERVICE_KEY = "VOTRE_SERVICE_ROLE_KEY"
```

La service key ne doit jamais être exposée au navigateur ou commitée dans GitHub.

## Snapshots de cotes

La table `quinte_market_snapshots` stocke : date, R/C, numéro du cheval, heure de capture, cote et probabilité de marché normalisée.

L'app capture un snapshot lorsqu'un support Quinté est analysé si la service key est disponible. Pour un protocole T-60 / T-30 / T-15 / T-5 / T-2, utilise aussi :

```bash
python capture_market_snapshot.py --date 2026-09-17
```

à chaque horaire souhaité via un scheduler externe. Sans snapshots horodatés, tout ROI historique doit rester présenté comme un **proxy**, pas comme un ROI réellement exécutable.

## Réentraîner V2

Installer les dépendances de training :

```bash
pip install -r requirements-train.txt
```

Puis :

```bash
python train_quinte_v2.py \
  --history validated_history.csv \
  --supports quinte_supports_99.csv \
  --output-dir .
```

Le script réécrit l'artefact, le modèle LambdaMART et le rapport walk-forward.

# 🐎 HorseProba — Pronostics hippiques probabilistes, auto-appris (Streamlit)

Application Streamlit qui classe les partants d'une course hippique par **probabilité de
victoire et de placé**, en combinant un **modèle logit conditionnel** (Bradley-Terry /
Plackett-Luce généralisé) et un **modèle de gradient boosting**, sur cotes, mouvement de
marché, forme récente, jockey, entraîneur, corde, poids, et **aptitude terrain/distance/piste**
de chaque cheval. Le modèle se **ré-entraîne et se journalise automatiquement chaque jour**
(boucle d'auto-apprentissage avec garde-fou de promotion), et met en avant les **outsiders à
potentiel** que le marché semble sous-évaluer — pas seulement les favoris. Chaque classement
est accompagné du raisonnement qui le justifie, et l'interface rappelle explicitement les
limites du modèle.

> ⚠️ Les paris hippiques sont incertains par nature. Ce projet est un outil d'analyse
> statistique et pédagogique ; il ne garantit aucun gain, et la détection d'« outsiders » n'est
> pas une recommandation de pari. Jouez de façon responsable.

---

## Fonctionnalités

| Fonction | État |
|---|---|
| Sélection d'une course du programme du jour (source web publique PMU, sans clé) | ✅ |
| Import CSV d'une course ou d'un historique (séparateur `,`/`;`, alias FR/EN) | ✅ |
| Saisie manuelle d'une course dans un tableau éditable | ✅ |
| **Modèle d'ensemble** : logit conditionnel régularisé + gradient boosting (interactions non-linéaires) | ✅ |
| **Détection d'outsiders à potentiel** (cote élevée + écart significatif au marché) | ✅ |
| **Variables enrichies** : mouvement de cote (probable→directe), aptitude terrain/distance/hippodrome par cheval, duo jockey-cheval | ✅ |
| Probabilité de placé (top-k) par Plackett-Luce (exact ou Monte-Carlo) | ✅ |
| Comparaison modèle vs marché, cote « juste », détection d'écart | ✅ |
| Explications par cheval (contributions de chaque variable, y compris la composante boosting) | ✅ |
| Backtest walk-forward : log-loss, Brier, calibration, hit rate, ROI illustratif | ✅ |
| **Boucle d'auto-apprentissage** : collecte quotidienne → journal de pronostics → ré-entraînement avec garde-fou de promotion → suivi de performance réelle | ✅ |
| **GitHub Action planifiée** (cron quotidien) orchestrant toute la boucle | ✅ |
| **Registre de modèles versionné** (`models/`) + journal d'entraînement (`models/training_log.csv`) | ✅ |
| Script de constitution d'un historique réel (`scripts/build_history.py`, résultats PMU, incrémental) | ✅ |
| Données synthétiques réalistes pour la démo / les tests | ✅ |
| Export CSV du pronostic | ✅ |
| Gestion des erreurs (course introuvable, réseau, CSV invalide, modèle indisponible) sans crash | ✅ |
| Tests unitaires (51) + CI GitHub Actions (pytest + ruff) | ✅ |

### Non implémenté (pistes)
- Intégration d'APIs sous licence (The Racing API, Equidia Pro) via `st.secrets`.
- Variables supplémentaires : œillères, ferrage (trot), valeur handicap officielle, forme de
  l'entraîneur sur 30 jours glissants (actuellement : taux lissé sur tout l'historique).
- Un vrai ranker listwise (LightGBM/XGBoost `rank:*`) à la place de l'approximation
  « classification binaire + normalisation intra-course » du GBM actuel (voir
  `horseproba/model_gbm.py` pour la justification de ce choix).
- Optimisation de mises (Kelly fractionnaire) — volontairement absente pour ne pas
  encourager le jeu.
- Alerte automatique (email/Slack) si le garde-fou de promotion refuse plusieurs fois de
  suite (signe possible de dérive des données collectées).

---

## Structure du dépôt

```
.
├── app.py                          # Point d'entrée Streamlit
├── requirements.txt                # Dépendances (Streamlit Cloud)
├── requirements-dev.txt            # + pytest, ruff
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml                 # Thème / serveur
│   └── secrets.toml.example        # Modèle de secrets (aucun secret obligatoire)
├── .github/workflows/
│   ├── ci.yml                      # Tests + lint automatiques (push/PR)
│   └── learning_loop.yml           # 🧠 Boucle d'auto-apprentissage quotidienne (cron)
├── data/
│   ├── sample_races.csv            # Jeu d'exemple (5 courses, résultats inclus)
│   ├── history_pmu.csv             # Historique réel collecté (créé par build_history.py)
│   └── predictions_log.csv         # Journal des pronostics publiés (créé par log_predictions.py)
├── models/                         # 🧠 Registre de modèles (créé par train_model.py)
│   ├── production/
│   │   ├── logit.json              # Coefficients du logit conditionnel
│   │   ├── gbm.joblib               # Modèle de boosting entraîné (si historique suffisant)
│   │   └── meta.json                # Métadonnées + métriques de validation
│   └── training_log.csv            # Historique de tous les ré-entraînements (promus ou non)
├── horseproba/
│   ├── __init__.py
│   ├── schema.py                   # Schéma de colonnes canonique
│   ├── features.py                 # Ingénierie des variables (musique, cotes, aptitudes, taux lissés…)
│   ├── model.py                    # Logit conditionnel + Plackett-Luce + explications + flag_outsiders
│   ├── model_gbm.py                 # Modèle gradient boosting (interactions non-linéaires)
│   ├── ensemble.py                  # Combinaison logit + GBM (opinion pooling)
│   ├── registry.py                  # Persistance du modèle de production + journal d'entraînement
│   ├── evaluate.py                  # Métriques probabilistes, backtest walk-forward, suivi live
│   ├── data/
│   │   ├── __init__.py             # 📌 Documentation centralisée des sources de données
│   │   ├── loader.py                # Lecture/validation CSV
│   │   ├── synthetic.py             # Générateur d'historique synthétique
│   │   └── pmu.py                   # Source web publique PMU (non officielle)
│   └── ui/
│       ├── __init__.py
│       └── components.py            # Tableaux, graphiques Plotly, outsiders, courbes d'apprentissage
├── scripts/
│   ├── build_history.py             # 1️⃣ Collecte des résultats PMU → data/history_pmu.csv
│   ├── resolve_predictions.py       # 2️⃣ Rapproche les pronostics journalisés des résultats réels
│   ├── train_model.py               # 3️⃣ Ré-entraîne + promeut (avec garde-fou) → models/
│   └── log_predictions.py           # 4️⃣ Journalise les pronostics du jour → data/predictions_log.csv
└── tests/
    ├── test_pipeline.py
    ├── test_build_history.py
    ├── test_features_context.py     # aptitudes terrain/distance, synergie, mouvement de cote, outsiders
    ├── test_model_gbm_ensemble.py
    ├── test_registry.py
    ├── test_train_model_script.py   # garde-fou de promotion
    └── test_prediction_logging.py   # journal de pronostics + résolution
```

---

## Approche statistique (résumé)

**Composante 1 — logit conditionnel (Bradley-Terry / Plackett-Luce généralisé).** Une course
est un problème de *choix discret* : un seul gagnant parmi *n*. Le modèle attribue à chaque
cheval une force latente `s_i = β·x_i` et pose `P(i gagne) = exp(s_i) / Σ_j exp(s_j)`. Les
probabilités somment à 1 par construction, les features sont **centrées par course** (seules
les différences entre partants comptent), et les coefficients sont directement
interprétables. Estimation par maximum de vraisemblance conditionnelle **pénalisé** (ridge
centré sur des coefficients a priori), robuste avec peu ou pas d'historique. Références :
McFadden (1974), Bolton & Chapman (1986), Benter (1994). Détail dans `horseproba/model.py`.

**Composante 2 — gradient boosting (`HistGradientBoostingClassifier`, scikit-learn).**
Le logit est additif : il ne capture pas les interactions du type « tel jockey performe
mieux seulement sur terrain lourd et grande distance ». Le GBM apprend ces interactions
directement depuis les données — c'est le principal levier pour repérer des **outsiders à
potentiel**. Approche : classification binaire gagnant/non-gagnant, score reconverti en force
latente puis renormalisé par le même softmax intra-course que le logit. Ne s'active qu'à
partir d'un historique suffisant (~40 courses gagnantes exploitables) ; en-deçà, l'ensemble
repose entièrement sur le logit. Justification complète et limites assumées documentées en
tête de `horseproba/model_gbm.py`.

**Combinaison.** Moyenne géométrique pondérée des deux probabilités (opinion pooling
log-linéaire, poids GBM 35 % par défaut), renormalisée par course — voir `horseproba/ensemble.py`.

**Détection d'outsiders (`horseproba/model.py::flag_outsiders`).** Un partant est signalé
« 🎯 Outsider » si sa cote est élevée (≥ 8 par défaut), que le modèle l'estime sensiblement
au-dessus du marché (écart ≥ 3 points de probabilité), et qu'il lui donne une probabilité de
victoire non négligeable (≥ 3 %). C'est un signal d'attention à examiner via les explications
par variable, **pas une recommandation de pari** : le modèle ignore la marge du PMU et toute
notion de gestion de bankroll.

**Variables** (voir `FEATURE_LABELS` dans `horseproba/features.py` pour la liste complète et
l'onglet *Méthode & données* de l'app) : probabilité implicite du marché, **mouvement de
cote** (probable → directe), score de forme pondéré (musique), taux de victoire/place récents
et en carrière (lissage bayésien), gains par course, fraîcheur, réussite jockey/entraîneur,
**duo jockey-cheval**, **aptitude terrain/distance/hippodrome par cheval** (toutes avec
lissage bayésien renforcé pour éviter le sur-ajustement sur des échantillons fins), corde
relative, poids relatif, âge relatif.

**Évaluation** : log-loss et Brier vs marché et vs uniforme, calibration par tranches,
backtest walk-forward sans fuite d'information (`horseproba/evaluate.py`), **et** suivi de la
performance réelle des pronostics publiés dans le temps (voir plus bas).

---

## 🧠 Boucle d'auto-apprentissage

C'est le cœur de cette évolution du projet. Avant, le modèle était ré-entraîné à chaque
session Streamlit, sans mémoire d'une session à l'autre. Désormais :

```
        chaque jour (GitHub Action, cron 04:30 UTC)
        ┌──────────────────────────────────────────────────────────────────┐
        │                                                                  │
        ▼                                                                  │
1. build_history.py        2. resolve_predictions.py     3. train_model.py │
   collecte les course-       rapproche les pronostics       ré-entraîne un   
   s terminées de la           journalisés hier des           modèle CANDIDAT  
   veille (résultats,          résultats maintenant           et ne le PROMEUT
   terrain…)                   connus (is_winner)              qu'avec un
        │                            │                          garde-fou    │
        │                            │                               │      │
        │                            ▼                               ▼      │
        │                   data/predictions_log.csv          models/       │
        │                   (résolu, is_winner rempli)        production/   │
        │                                                            │      │
        │                                                            ▼      │
        └──────────────────────────────────────────────► 4. log_predictions.py
                                                              journalise les pronostics
                                                              DU JOUR avec le modèle
                                                              (nouvellement promu ou non)
                                                                     │
                                                                     └──► (résolu demain, étape 2)
```

### Le garde-fou de promotion (« challenger vs champion »)

Ré-entraîner régulièrement ne suffit pas : rien ne garantit qu'un nouveau modèle soit
meilleur (variance d'échantillonnage, mauvaise passe temporaire...). `scripts/train_model.py`
applique donc le protocole standard des pipelines de ML en production :

1. Le **candidat** est évalué par un **backtest walk-forward** complet (plusieurs blocs
   hors-échantillon sur tout l'historique disponible aujourd'hui) → `candidate_logloss`.
2. Il est comparé au log-loss backtest **enregistré au moment de la promotion** du modèle de
   production actuel (pas ré-évalué sur les nouvelles données, pour éviter toute fuite).
3. **Promotion** si le candidat bat le repère uniforme, et n'est pas significativement pire
   (tolérance ±5 % par défaut) que la production actuelle — ou si aucune production n'existe
   encore.
4. Sinon, la production est **conservée**, et le refus est journalisé (ce n'est pas une
   erreur : c'est le garde-fou qui fonctionne).
5. Si l'historique n'a pas grandi depuis le dernier passage, le script ne fait rien
   (idempotent — utile un jour sans nouvelle course terminée).

Chaque tentative (promue ou non) est ajoutée à `models/training_log.csv` : c'est ce qui
permet de visualiser la **progression réelle du modèle dans le temps** dans l'onglet
*🧠 Auto-apprentissage* de l'application (courbe de log-loss backtest au fil des
ré-entraînements, comparée au marché et au repère uniforme).

### Suivi de performance réelle (pas seulement un backtest)

Le backtest rejoue le passé ; il ne prouve pas qu'un pronostic **publié à l'avance** était
bon. D'où la boucle `log_predictions.py` → `resolve_predictions.py` : chaque pronostic est
journalisé avant la course, puis rapproché du résultat réel une fois connu. L'onglet
*🧠 Auto-apprentissage* affiche la courbe de log-loss quotidien et le taux de réussite du
favori du modèle sur les pronostics **effectivement publiés**, pas rejoués.

### Lancer la boucle manuellement

```bash
python scripts/build_history.py --days 3
python scripts/resolve_predictions.py --verbose
python scripts/train_model.py --verbose                 # ajoute --force pour ignorer le garde-fou
python scripts/log_predictions.py --verbose
```

La GitHub Action `.github/workflows/learning_loop.yml` exécute ces 4 étapes chaque jour à
04:30 UTC (après les dernières réunions du soir en France) et commite les fichiers modifiés
(`data/history_pmu.csv`, `data/predictions_log.csv`, `models/`) — Streamlit Community Cloud
redéploie automatiquement à chaque push, donc l'app en ligne reste à jour sans intervention.
Déclenchement manuel toujours possible depuis l'onglet *Actions* de GitHub (`workflow_dispatch`).

> Pourquoi pas un script orchestrateur unique ? Chaque étape est déjà tolérante aux pannes
> individuellement (une source PMU indisponible n'interrompt pas le reste), et 4 étapes
> GitHub Actions séparées (`continue-on-error`) donnent des logs plus lisibles qu'un seul
> script monolithique — standard pour ce type de pipeline.

---

## Sources de données

Documentées dans `horseproba/data/__init__.py` et dans l'onglet *Méthode & données*.

| Source | Clé requise | Rafraîchissement |
|---|---|---|
| **CSV utilisateur** (partants ou historique avec `finish_position`) | Non | Re-téléverser |
| **`data/sample_races.csv`** (exemple embarqué) | Non | Éditer dans le dépôt |
| **Synthétique** (`generate_history`) | Non | Graine / nombre de courses |
| **PMU web public** — programme, partants, **cotes probable et directe séparément** (`online.turfinfo.api.pmu.fr`, endpoints JSON consommés par pmu.fr — *non officiel*) | Non | Bouton « Actualiser », cache 2–10 min |
| APIs tierces sous licence | Oui (`st.secrets`) | Non intégré |

Usage responsable de la source web : User-Agent identifiable, délai entre requêtes, cache,
aucun contournement. La structure peut changer sans préavis : l'application affiche alors un
message et bascule sur l'import CSV.

### Constituer un historique réel (indispensable avant un vrai déploiement)

```bash
# 30 derniers jours, toutes disciplines (≈ 2–4 min par journée collectée)
python scripts/build_history.py --days 30 --user-agent "HorseProba (contact: vous@example.com)"

# Période précise, plat uniquement
python scripts/build_history.py --start 2024-09-01 --end 2024-09-30 --discipline plat

# Mise à jour incrémentale (les courses déjà présentes ne sont pas re-téléchargées)
python scripts/build_history.py --days 7
```

Le script parcourt le programme de chaque journée, récupère partants + ordre d'arrivée
officiel + état du terrain + **cotes probable et directe**, et écrit `data/history_pmu.csv`
au schéma de l'application. Il est **tolérant aux pannes**, **incrémental**, **reprend**
après interruption, respecte un débit limité (≈ 1 requête/s) et recalcule
`days_since_last_run`.

Une fois l'historique constitué, lancez `python scripts/train_model.py` pour produire le
premier modèle de production (sinon l'app utilise des coefficients a priori). **Commitez**
`data/history_pmu.csv` et `models/` pour que l'application déployée en dispose (Streamlit
Cloud n'a pas de stockage persistant) — ou laissez la GitHub Action planifiée s'en charger.

Ordre de grandeur : 30 jours ≈ 1 500–2 500 courses (largement suffisant pour un pseudo-R²
stable et pour activer le GBM) ; 6 mois+ recommandés pour des taux jockey/entraîneur et des
aptitudes terrain/distance/piste vraiment fiables.

### Format CSV

Obligatoire : `race_id`, `horse`. Recommandé : `odds` (cote décimale), `musique`
(`1p 3p Da 2p`), `jockey`, `trainer`, `draw`, `weight_kg`, `age`, `days_since_last_run`,
`career_starts`, `career_wins`, `career_places`, `earnings`. Contexte : `race_date`, `track`,
`discipline` (`plat|trot|obstacle`), `distance_m`, `going`. Optionnel : `odds_probable`,
`odds_direct` (mouvement de cote). Cible (historique seulement) : `finish_position`. Des
alias français (`cheval`, `cote`, `entraineur`, `arrivee`…) sont reconnus.

---

## Installation locale

```bash
git clone https://github.com/<votre-compte>/horseproba.git
cd horseproba
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                              # tests (51)
streamlit run app.py
```

Secrets optionnels : copier `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml`.

---

## Déploiement sur Streamlit Community Cloud

1. **Pousser le dépôt sur GitHub** (public ou privé) avec `app.py` et `requirements.txt` à la racine.
   ```bash
   git init && git add . && git commit -m "HorseProba v2.0 — auto-apprentissage"
   git branch -M main
   git remote add origin https://github.com/<votre-compte>/horseproba.git
   git push -u origin main
   ```
2. Aller sur **https://share.streamlit.io**, se connecter avec GitHub, cliquer **« Create app »**
   → **« Deploy a public app from GitHub »**.
3. Renseigner :
   - *Repository* : `<votre-compte>/horseproba`
   - *Branch* : `main`
   - *Main file path* : `app.py`
   - *Advanced settings* → Python version **3.11** (recommandé).
4. **Secrets (optionnels)** : *Advanced settings → Secrets*, coller le contenu de
   `.streamlit/secrets.toml.example` adapté (`HTTP_USER_AGENT` avec votre contact,
   `ENABLE_WEB_FETCH = false` pour désactiver la source web).
5. Cliquer **Deploy**. L'application est en ligne en 1–3 minutes.
6. **Activer la boucle d'auto-apprentissage** : dans les *Settings* du dépôt GitHub →
   *Actions*, vérifier que les workflows sont activés. La GitHub Action `learning_loop.yml`
   tourne alors chaque nuit et pousse les mises à jour ; Streamlit Cloud redéploie
   automatiquement à chaque `git push` sur `main`.

Dépannage : si le build échoue, consulter les logs (*Manage app*) ; vérifier que
`requirements.txt` est à la racine et que la version Python est 3.10+. Si la GitHub Action
échoue silencieusement côté commit, vérifier que la permission `contents: write` est bien
accordée aux Actions dans les *Settings → Actions → General* du dépôt.

---

## Données & modèles (référence technique)

- **Structures** : `pandas.DataFrame` au schéma `horseproba/schema.py`. Le logit conditionnel
  se sérialise en JSON (`ConditionalLogitModel.to_json/from_json`) ; le GBM se sérialise en
  binaire via `joblib` ; l'ensemble des deux est géré par `horseproba/registry.py`.
- **Stockage** : pas de base de données. `data/history_pmu.csv`, `data/predictions_log.csv`
  et `models/` sont des fichiers **committés dans le dépôt** (voir `.gitignore` pour la
  justification) — c'est ce qui permet à la fois la persistance sur Streamlit Cloud (sans
  stockage serveur) et la traçabilité de l'auto-apprentissage (`git log` sur ces fichiers =
  historique des évolutions du modèle). Les caches Streamlit (`st.cache_data`,
  `st.cache_resource`) restent en mémoire et volatils.
- **Entrées fonctionnelles** : `app.py` (page unique, 4 onglets : Pronostic, Modèle &
  backtest, 🧠 Auto-apprentissage, Méthode & données) ; barre latérale pour le choix du
  modèle (production / entraîné à la volée / a priori) et l'historique d'entraînement.

---

## Limites (rappelées dans l'interface)

- Un cheval estimé à 30 % perd 7 fois sur 10 : ce sont des probabilités, pas des prédictions.
- Sans historique réel, les coefficients a priori ne sont pas calibrés sur vos courses, et le
  GBM ne s'active pas.
- Les cotes de référence évoluent jusqu'au départ ; un « écart de valeur » ou un outsider
  détecté peut disparaître d'ici le départ réel.
- Le marché est un concurrent redoutable : battre systématiquement les cotes est rare, et le
  garde-fou de promotion n'élimine pas le risque de sur-ajustement à long terme — surveillez
  l'onglet *Auto-apprentissage*.
- La détection d'outsiders est un signal statistique, pas une recommandation de pari : elle
  ignore la marge du PMU et toute gestion de bankroll.
- Jeu responsable : *Joueurs Info Service* 09 74 75 13 13. Interdit aux mineurs.

## Licence

MIT. Les données PMU restent la propriété de leurs éditeurs ; respectez leurs conditions d'utilisation.

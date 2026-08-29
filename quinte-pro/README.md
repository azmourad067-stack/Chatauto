# 🐎 Quinté+ Predictor — pronostic hippique assisté (Streamlit)

Application Streamlit qui combine **deux mécanismes distincts** :

1. **Recherche web en temps réel** (module `scraper/`) : collecte de données
   actuelles sur la course saisie (driver, cheval, hippodrome, actualités).
2. **Modèle de scoring pré-entraîné** (module `model/`) : transforme les
   données collectées en classement probabiliste des 5 premiers.

> ⚠️ **Avertissement** : outil d'aide à l'analyse, **aucune garantie de
> résultat**. Les paris comportent un risque — jouez avec modération.
> Jeux d'argent interdits aux mineurs. Aide joueurs : ANJ 09 74 75 13 13.

---

## Structure du projet

```
quinte-pro/
├── app.py                  # Application Streamlit (UI + orchestration)
├── requirements.txt
├── README.md
├── .streamlit/config.toml  # Thème
├── scraper/
│   ├── __init__.py
│   ├── base.py             # HTTP poli : timeout, retries, cache TTL
│   ├── sources.py          # Adaptateurs par source (ouverts/désactivés)
│   └── engine.py           # Fusion multi-sources, données incomplètes
├── model/
│   ├── __init__.py
│   ├── features.py         # 21 features par partant (schéma unique)
│   ├── dataset.py          # Générateur de données synthétiques documenté
│   ├── train.py            # Pipeline d'entraînement (séparé de l'app)
│   ├── predict.py          # Inférence : classement top 5
│   └── test_inference.py   # Smoke test
└── artifacts/
    ├── model.joblib        # Modèle entraîné + médianes + métriques
    └── metrics.json
```

## Sources de données en temps réel

| Source | Statut | Détail |
|---|---|---|
| [`open-pmu-api`](https://github.com/nanaelie/open-pmu-api) | ✅ **actif par défaut** (instance publique `https://open-pmu-api.vercel.app`, MIT) | Endpoint `GET /api/arrivees?date=MM/DD/YYYY&hippo=...`. **Arrivées historiques uniquement** (22/01/2004 → 18/08/2026) : sert à enrichir un cheval sur son passé (musique, gains, cote, entraîneur, distance préférée), pas à construire la grille des partants d'une course à venir. Override via `OPEN_PMU_API_URL` (auto-hébergement : `npm install` + déploiement Vercel, base Postgres requise). |
| Recherche web DuckDuckGo HTML / SerpAPI | 🟡 active par défaut, fragile | Snippets d'actualité par cheval+driver → sentiment. Anti-bot possible ; SerpAPI (clé) plus fiable. |
| geny.com, paris-turf.com, letrot.com, france-galop.com | ⛔ désactivées | Pas d'API publique ; anti-bot + conditions d'utilisation restrictives. Adaptateurs prêts mais OFF par défaut (activer à vos risques). |

Aucune API officielle publique (PMU, France Galop, LeTrot) ne distribue de
données partants. Les échecs de source ne bloquent jamais l'application :
le pronostic est toujours rendu (qualité des données affichée).

## Modèle de scoring

- **Features (21)** : taux de réussite cheval/driver, dernier rang, rang moyen
  sur 6, gains (log), âge, affinité discipline/hippodrome, ratio distance,
  jours de repos, musique (top-5 ratio), proxy de cote, nb de partants,
  sentiment web, qualité des données.
- **Méthode** : `HistGradientBoostingRegressor` sur la place normalisée
  (donne l'**ordre**) + `HistGradientBoostingClassifier` top-5 (donne la
  **probabilité**). Proba affichée = sortie du modèle, normalisée.
- **Données d'entraînement** : générateur synthétique réaliste
  (niveaux latents + bruit, cf. docstring de `dataset.py`), car aucune base
  historique n'est fournie. **Remplacement par un vrai historique** :
  `python model/train.py --real historique.csv` (schéma = colonnes FEATURES).

## Installation locale

```bash
git clone https://github.com/<vous>/quinte-pro.git
cd quinte-pro
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python model/train.py          # génère artifacts/model.joblib
streamlit run app.py
```

## Déploiement Streamlit Community Cloud (via GitHub)

1. **Créer le dépôt GitHub** : `github.com` → *New repository* (public ou privé
   — pour le Cloud gratuit, public recommandé) → `quinte-pro`.
2. **Pousser le projet** :
   ```bash
   git init && git add . && git commit -m "Quinté+ Predictor"
   git branch -M main
   git remote add origin https://github.com/<vous>/quinte-pro.git
   git push -u origin main
   ```
3. **Se connecter au Cloud** : `share.streamlit.io` → *Sign in* avec votre
   compte GitHub (autorisez Streamlit à accéder à vos dépôts).
4. **Nouvelle app** : *Create app* → sélectionner le dépôt `quinte-pro`,
   branche `main`, fichier principal `app.py` → *Deploy*.
5. **Attendre le build** : le Cloud installe `requirements.txt` puis lance
   `streamlit run app.py`. (Le modèle `.joblib` est déjà dans le dépôt ;
   pour régénérer : page de l'app → « Ré-entraîner ».)
6. **Secrets** (optionnel) : *Advanced settings* → *Secrets* →
   ```toml
   OPEN_PMU_API_URL = "https://votre-instance-open-pmu"   # si auto-hébergée
   QUINTE_SEARCH_ENABLED = "1"
   # QUINTE_SERPAPI_KEY = "votre-cle"                     # alternative plus fiable
   ```
7. **Ressources** : plan gratuit ≈ 1 CPU / 1 Go RAM. La recherche web est
   limitée en débit (délai 1,2 s entre requêtes) pour rester dans les clous.
8. **Mises à jour** : tout `git push` sur `main` redéploie automatiquement.

## Limites techniques et légales

- **Fiabilité du scraping** : sites protégés par anti-bot ; aucun accès API
  garanti. Les données partielles dégradent la qualité du pronostic
  (indicateur de qualité affiché).
- **Modèle** : entraîné sur données **synthétiques** par défaut — les
  performances affichées (Spearman, AUC) reflètent la simulation, **pas** le
  réel. Rien ne garantit un gain.
- **Légal** : respectez `robots.txt`, les conditions d'utilisation et la
  réglementation des données (hébergeur = votre responsabilité). Le scraping
  à grande échelle de sites commerciaux peut violer leurs CGU.
- **Jeu responsable** : outil éducatif ; les paris sont risqués. 18 ans
  minimum en France. Aide : ANJ 09 74 75 13 13 / `anj.fr`.

# WHY — application Streamlit

Version web (Streamlit) du script Tkinter « WHY » : un agent conversationnel minimaliste
qui maintient un état (problème, but, représentation, découvertes, questions) et ne pose
des questions que lorsque l'état le justifie.

## Structure du dépôt

```
├── app.py              # application principale
├── requirements.txt    # dépendances
└── README.md
```

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déployer sur Streamlit Community Cloud via GitHub

1. Crée un nouveau dépôt sur GitHub (public ou privé).
2. Pousse ces fichiers :

   ```bash
   git init
   git add app.py requirements.txt README.md
   git commit -m "WHY - app streamlit"
   git branch -M main
   git remote add origin https://github.com/<ton-user>/<ton-repo>.git
   git push -u origin main
   ```

3. Va sur https://share.streamlit.io → **New app** → choisis ton dépôt,
   branche `main`, fichier `app.py` → **Deploy**.

## Notes

- L'état est sauvegardé dans `why_state.json` (ajouté au `.gitignore` idéalement).
  Sur Streamlit Cloud, le système de fichiers est éphémère : l'état vit surtout
  dans la session (`st.session_state`) ; la sauvegarde disque est tentée mais non
  garantie entre redémarrages.
- Commandes disponibles dans le chat : `/exec`, `/etat`, `/reset`.

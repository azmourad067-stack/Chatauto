import streamlit as st
import easyocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import time

# ------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------
st.set_page_config(page_title="Pronostics Hippiques Adaptatif", layout="wide")

# ------------------------------------------------------------
# Initialisation de l'OCR (mise en cache)
# ------------------------------------------------------------
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['fr'], gpu=False)

# ------------------------------------------------------------
# Fonctions génériques de nettoyage et parsing
# ------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return text
    # Remplacer les caractères mal interprétés
    text = text.replace('|', '1').replace('O', '0').replace('l', '1')
    text = text.strip()
    return text

def parse_number(text):
    """Extrait un nombre depuis une chaîne (entier ou décimal)."""
    if not isinstance(text, str):
        return None
    # Remplacer la virgule par un point
    text = text.replace(',', '.')
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
    return None

def parse_gains(text):
    """Parse les gains (ex: '14 165' -> 14165)."""
    if isinstance(text, str):
        text = text.replace(' ', '').replace('\u202f', '')
        match = re.search(r'(\d+)', text)
        if match:
            return int(match.group(1))
    return None

def parse_record(text):
    """Parse un record (ex: '1\'15"4' -> 75.4 secondes)."""
    if isinstance(text, str):
        match = re.search(r"(\d+)'(\d+)\"?(\d?)", text)
        if match:
            minutes = int(match.group(1))
            secondes = int(match.group(2))
            dixiemes = int(match.group(3)) if match.group(3) else 0
            return minutes * 60 + secondes + dixiemes / 10
    return None

def parse_musique_trot(text):
    """Extrait les performances pour le trot (ex: '5a4a2a' -> ['5a','4a','2a'])."""
    if not isinstance(text, str):
        return []
    text = re.sub(r'\(\d+\)', '', text)
    pattern = r'(\d*[aAmM]?[aA]?|Da|Dm|0a)'
    parts = re.findall(pattern, text)
    return [p for p in parts if p]

def parse_musique_plat(text):
    """Extrait les performances pour le plat (ex: '[[8p][1p] (42,5)' -> ['8p','1p'])."""
    if not isinstance(text, str):
        return []
    text = re.sub(r'\([^)]*\)', '', text)
    text = text.replace('[', '').replace(']', '')
    pattern = r'(\d+p)'
    parts = re.findall(pattern, text)
    return parts

def score_musique(musique_list, type_course):
    """
    Calcule un score à partir de la musique selon le type.
    Pour le trot : a = 10 pour 1a, 8 pour 2a, etc.
    Pour le plat : p = 10 pour 1p, etc.
    """
    if not musique_list:
        return 0
    # Définir les poids selon le type
    if type_course == 'trot':
        weights = {
            '1a':10,'1m':10,
            '2a':8,'2m':8,
            '3a':6,'3m':6,
            '4a':5,'4m':5,
            '5a':4,'5m':4,
            '6a':3,'6m':3,
            '7a':2,'7m':2,
            '8a':1,'8m':1,
            '9a':0,'9m':0,
            'Da':0,'Dm':0,'0a':0
        }
        suffix = 'a'
    else:  # plat
        weights = {
            '1p':10, '2p':8, '3p':6, '4p':5, '5p':4,
            '6p':3, '7p':2, '8p':1, '9p':0
        }
        suffix = 'p'
    
    total = 0
    count = 0
    for perf in musique_list[:5]:  # 5 dernières
        perf_lower = perf.lower()
        if perf_lower in weights:
            total += weights[perf_lower]
            count += 1
        else:
            # Essayer d'extraire le chiffre
            match = re.match(r'(\d+)' + suffix, perf_lower)
            if match:
                place = int(match.group(1))
                if place <= 9:
                    total += max(0, 10 - place)
                    count += 1
    return total / count if count > 0 else 0

# ------------------------------------------------------------
# Détection de la structure du tableau (OCR → DataFrame)
# ------------------------------------------------------------
def extract_table_from_image(image):
    """Extrait un DataFrame à partir d'une image en utilisant EasyOCR."""
    reader = load_ocr_reader()
    results = reader.readtext(np.array(image))
    
    text_boxes = []
    for (bbox, text, conf) in results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        text_boxes.append({
            'text': text.strip(),
            'x': (min(x_coords) + max(x_coords)) / 2,
            'y': (min(y_coords) + max(y_coords)) / 2,
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords)
        })
    
    if not text_boxes:
        return pd.DataFrame()
    
    # Trier par position verticale
    text_boxes.sort(key=lambda x: x['y'])
    
    # Regrouper en lignes
    heights = [box['y_max'] - box['y_min'] for box in text_boxes]
    avg_height = np.mean(heights) if heights else 20
    line_threshold = avg_height * 0.6
    
    lines = []
    current_line = []
    current_y = None
    for box in text_boxes:
        if current_y is None or abs(box['y'] - current_y) < line_threshold:
            current_line.append(box)
            current_y = box['y']
        else:
            lines.append(sorted(current_line, key=lambda x: x['x']))
            current_line = [box]
            current_y = box['y']
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x['x']))
    
    if not lines:
        return pd.DataFrame()
    
    # Identifier la ligne d'en-tête : celle qui contient le plus de mots-clés
    header_keywords = ['n°', 'cheval', 'driver', 'jockey', 'entraîneur', 'poids', 'musique', 'gains', 'record']
    best_score = -1
    best_idx = 0
    for i, line in enumerate(lines):
        texts = ' '.join([item['text'].lower() for item in line])
        score = sum(1 for kw in header_keywords if kw in texts)
        if score > best_score:
            best_score = score
            best_idx = i
    
    header_line = lines[best_idx]
    columns = [item['text'] for item in header_line]
    
    # Lignes de données
    data_lines = lines[best_idx+1:]
    data_rows = []
    for line in data_lines:
        row = {}
        for item in line:
            # Trouver la colonne la plus proche
            distances = [abs(item['x'] - h['x']) for h in header_line]
            best_col_idx = np.argmin(distances)
            best_col = columns[best_col_idx]
            row[best_col] = item['text']
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    # Garder seulement les colonnes de l'en-tête
    cols_present = [c for c in columns if c in df.columns]
    if cols_present:
        df = df[cols_present]
    return df

# ------------------------------------------------------------
# Détection du type de course et nettoyage adapté
# ------------------------------------------------------------
def detecter_type_course(df):
    """Analyse les colonnes et le contenu pour déterminer trot ou plat."""
    # Convertir les noms de colonnes en minuscules
    cols_lower = [str(c).lower() for c in df.columns]
    all_text = ' '.join(cols_lower) + ' ' + ' '.join(df.astype(str).values.flatten()).lower()
    
    if 'driver' in all_text or 'entraîneur' in all_text and 'jockey' not in all_text:
        return 'trot'
    elif 'jockey' in all_text or 'poids' in all_text:
        return 'plat'
    else:
        # Par défaut, on regarde la présence de 'a' dans la musique
        # Chercher une colonne qui pourrait être la musique
        musique_col = None
        for col in df.columns:
            if 'musique' in col.lower():
                musique_col = col
                break
        if musique_col:
            # Examiner les premières valeurs
            sample = df[musique_col].iloc[0] if not df.empty else ''
            if 'a' in str(sample):
                return 'trot'
            elif 'p' in str(sample):
                return 'plat'
        return 'inconnu'

def nettoyer_df_selon_type(df, type_course):
    """Nettoie le DataFrame en fonction du type détecté et ajoute les colonnes calculées."""
    df = df.copy()
    # Nettoyer toutes les cellules
    for col in df.columns:
        df[col] = df[col].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    
    # Normaliser les noms de colonnes (minuscules, sans accents)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'n°' in col_lower or 'no' in col_lower or 'num' in col_lower:
            col_mapping[col] = 'N°'
        elif 'cheval' in col_lower:
            col_mapping[col] = 'Cheval'
        elif 'driver' in col_lower:
            col_mapping[col] = 'Driver'
        elif 'jockey' in col_lower:
            col_mapping[col] = 'Jockey'
        elif 'entraîneur' in col_lower or 'entraineur' in col_lower:
            col_mapping[col] = 'Entraîneur'
        elif 'poids' in col_lower:
            col_mapping[col] = 'Poids'
        elif 'musique' in col_lower:
            col_mapping[col] = 'Musique'
        elif 'gains' in col_lower:
            col_mapping[col] = 'Gains'
        elif 'record' in col_lower:
            col_mapping[col] = 'Record'
        elif 'valeur' in col_lower or 'val' in col_lower:
            col_mapping[col] = 'Valeur'
    df.rename(columns=col_mapping, inplace=True)
    
    # Convertir les colonnes numériques
    if 'N°' in df.columns:
        df['N°'] = pd.to_numeric(df['N°'], errors='coerce')
    if 'Gains' in df.columns:
        df['Gains'] = df['Gains'].apply(parse_gains)
    if 'Record' in df.columns:
        df['Record_secondes'] = df['Record'].apply(parse_record)
    if 'Valeur' in df.columns:
        df['Valeur'] = df['Valeur'].apply(parse_number)
    if 'Poids' in df.columns:
        df['Poids'] = df['Poids'].apply(parse_number)
    
    # Traitement de la musique
    if 'Musique' in df.columns:
        if type_course == 'trot':
            df['Musique_list'] = df['Musique'].apply(parse_musique_trot)
        else:
            df['Musique_list'] = df['Musique'].apply(parse_musique_plat)
        df['Score_musique'] = df['Musique_list'].apply(lambda x: score_musique(x, type_course))
    
    return df

# ------------------------------------------------------------
# Calcul du score selon le type
# ------------------------------------------------------------
def calculer_score(row, type_course):
    """Calcule un score normalisé pour un cheval selon le type de course."""
    score = 0
    poids = {}
    
    if type_course == 'trot':
        # Poids pour le trot
        poids = {
            'record': 0.25,
            'gains': 0.20,
            'musique': 0.30,
            'reussite_driver': 0.15,
            'reussite_entraineur': 0.10
        }
        # Note: ici on n'a pas les réusssites car elles sont dans d'autres tableaux
        # On utilisera seulement les données disponibles
        if pd.notna(row.get('Record_secondes')):
            score += poids['record'] * (1 / row['Record_secondes'])  # provisoire
        if pd.notna(row.get('Gains')):
            # Normalisation approximative
            score += poids['gains'] * (row['Gains'] / 100000)
        if pd.notna(row.get('Score_musique')):
            score += poids['musique'] * (row['Score_musique'] / 10)
    else:  # plat
        poids = {
            'valeur': 0.50,
            'musique': 0.40,
            'poids': 0.10
        }
        if pd.notna(row.get('Valeur')):
            # Normalisation : on suppose une fourchette de 30 à 45
            val_norm = (row['Valeur'] - 30) / 15
            val_norm = max(0, min(1, val_norm))
            score += poids['valeur'] * val_norm
        if pd.notna(row.get('Score_musique')):
            score += poids['musique'] * (row['Score_musique'] / 10)
        if pd.notna(row.get('Poids')):
            # Plus le poids est élevé, mieux c'est ? À ajuster
            poids_norm = (row['Poids'] - 54) / 8  # exemple
            poids_norm = max(0, min(1, poids_norm))
            score += poids['poids'] * poids_norm
    
    return score

def classer_chevaux(df, type_course):
    """Ajoute le score et classe les chevaux."""
    df['score_brut'] = df.apply(lambda row: calculer_score(row, type_course), axis=1)
    # Normalisation
    min_score = df['score_brut'].min()
    max_score = df['score_brut'].max()
    if max_score - min_score > 0:
        df['score_normalise'] = (df['score_brut'] - min_score) / (max_score - min_score)
    else:
        df['score_normalise'] = 0.5
    df = df.sort_values('score_normalise', ascending=False).reset_index(drop=True)
    df['rang'] = df.index + 1
    return df

# ------------------------------------------------------------
# Fonctions de pronostics (inchangées)
# ------------------------------------------------------------
def generer_top_3(df):
    return df.head(3)[['N°', 'Cheval', 'score_normalise']].to_dict('records')

def generer_bases(df, n=2):
    return df.head(n)[['N°', 'Cheval']].to_dict('records')

def generer_outsiders(df, n=5, seuil_score=0.3):
    outsiders = df.iloc[3:][df.iloc[3:]['score_normalise'] > seuil_score].head(n)
    return outsiders[['N°', 'Cheval', 'score_normalise']].to_dict('records')

def generer_combinaisons_trio(df, n_combinaisons=10):
    chevaux = df[['N°', 'Cheval', 'score_normalise']].to_dict('records')
    scores = [c['score_normalise'] for c in chevaux]
    total_score = sum(scores)
    if total_score == 0:
        probabilities = [1/len(chevaux)]*len(chevaux)
    else:
        probabilities = [s/total_score for s in scores]
    combinaisons = set()
    while len(combinaisons) < n_combinaisons:
        indices = np.random.choice(len(chevaux), size=3, replace=False, p=probabilities)
        comb = tuple(sorted([chevaux[i]['N°'] for i in indices]))
        combinaisons.add(comb)
    result = []
    for comb in combinaisons:
        noms = [df[df['N°'] == n]['Cheval'].values[0] for n in comb]
        result.append({'combinaison': comb, 'chevaux': noms})
    return result

def generer_combinaisons_quinte(df, n_combinaisons=10):
    chevaux = df[['N°', 'Cheval', 'score_normalise']].to_dict('records')
    scores = [c['score_normalise'] for c in chevaux]
    total_score = sum(scores)
    if total_score == 0:
        probabilities = [1/len(chevaux)]*len(chevaux)
    else:
        probabilities = [s/total_score for s in scores]
    combinaisons = set()
    while len(combinaisons) < n_combinaisons:
        indices = np.random.choice(len(chevaux), size=5, replace=False, p=probabilities)
        comb = tuple(sorted([chevaux[i]['N°'] for i in indices]))
        combinaisons.add(comb)
    result = []
    for comb in combinaisons:
        noms = [df[df['N°'] == n]['Cheval'].values[0] for n in comb]
        result.append({'combinaison': comb, 'chevaux': noms})
    return result

# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------
st.title("🐎 Application de Pronostics Hippiques Adaptative")
st.markdown("S'adapte automatiquement aux courses de trot et de plat.")

uploaded_files = st.file_uploader(
    "📤 Télécharger les photos (PNG, JPG, JPEG)",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_final = None
    st.session_state.type_course = None

if uploaded_files:
    st.subheader("Aperçu des images téléchargées")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, file in enumerate(uploaded_files):
        with cols[i % 4]:
            image = Image.open(file)
            st.image(image, caption=file.name, use_container_width=True)

if st.button("🔍 Analyser la course", type="primary") and uploaded_files:
    with st.spinner("Analyse en cours..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # On prend la première image comme table principale (partants)
        # Si plusieurs images, on pourrait les fusionner, mais simplifions
        image = Image.open(uploaded_files[0])
        status_text.text("Extraction du tableau...")
        df_raw = extract_table_from_image(image)
        progress_bar.progress(50)
        
        if df_raw.empty:
            st.error("Impossible d'extraire un tableau. Vérifiez la qualité de l'image.")
        else:
            status_text.text("Détection du type de course...")
            type_course = detecter_type_course(df_raw)
            st.info(f"Type de course détecté : **{type_course}**")
            
            status_text.text("Nettoyage des données...")
            df_clean = nettoyer_df_selon_type(df_raw, type_course)
            
            # S'assurer que la colonne N° existe
            if 'N°' not in df_clean.columns:
                df_clean['N°'] = range(1, len(df_clean)+1)
            
            status_text.text("Calcul des scores...")
            df_final = classer_chevaux(df_clean, type_course)
            
            st.session_state.df_final = df_final
            st.session_state.type_course = type_course
            st.session_state.data_loaded = True
            
            progress_bar.progress(100)
            status_text.text("Analyse terminée!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

if st.session_state.data_loaded and st.session_state.df_final is not None:
    df = st.session_state.df_final
    type_course = st.session_state.type_course
    
    st.subheader(f"📊 Données extraites et scores (course de {type_course})")
    # Afficher les colonnes pertinentes
    base_cols = ['N°', 'Cheval']
    if type_course == 'trot':
        extra_cols = ['Driver', 'Entraîneur', 'Gains', 'Record_secondes', 'Score_musique', 'score_normalise', 'rang']
    else:
        extra_cols = ['Jockey', 'Entraîneur', 'Valeur', 'Poids', 'Score_musique', 'score_normalise', 'rang']
    cols_afficher = [c for c in base_cols + extra_cols if c in df.columns]
    st.dataframe(df[cols_afficher])
    
    # Graphique
    st.subheader("📈 Scores des chevaux")
    fig, ax = plt.subplots(figsize=(10, 6))
    chevaux = df['Cheval'].astype(str) + " (N°" + df['N°'].astype(str) + ")"
    ax.barh(chevaux, df['score_normalise'])
    ax.set_xlabel("Score normalisé")
    ax.set_title(f"Classement - {type_course}")
    st.pyplot(fig)
    
    # Pronostics
    st.subheader("🏆 Pronostics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 3 probable")
        top3 = generer_top_3(df)
        for i, c in enumerate(top3):
            st.write(f"{i+1}. N°{c['N°']} - {c['Cheval']} (score: {c['score_normalise']:.3f})")
    with col2:
        st.markdown("#### Bases solides")
        bases = generer_bases(df, 2)
        for c in bases:
            st.write(f"🔹 N°{c['N°']} - {c['Cheval']}")
    
    st.markdown("#### Outsiders intéressants")
    outsiders = generer_outsiders(df, 5)
    for c in outsiders:
        st.write(f"👀 N°{c['N°']} - {c['Cheval']} (score: {c['score_normalise']:.3f})")
    
    st.markdown("#### 🎲 10 combinaisons pour le Trio")
    trio = generer_combinaisons_trio(df, 10)
    for i, comb in enumerate(trio):
        st.write(f"{i+1}. {', '.join(comb['chevaux'])} (N°{', '.join(map(str, comb['combinaison']))})")
    
    st.markdown("#### 🎲 10 combinaisons pour le Quinté+")
    quinte = generer_combinaisons_quinte(df, 10)
    for i, comb in enumerate(quinte):
        st.write(f"{i+1}. {', '.join(comb['chevaux'])} (N°{', '.join(map(str, comb['combinaison']))})")
    
    # Export CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger les données (CSV)", csv, "pronostics.csv")
    
    # Bouton pour réinitialiser (permettre une nouvelle analyse)
    if st.button("🔄 Nouvelle analyse"):
        st.session_state.data_loaded = False
        st.session_state.df_final = None
        st.session_state.type_course = None
        st.rerun()
else:
    if uploaded_files:
        st.info("Cliquez sur 'Analyser la course' pour lancer l'extraction.")
    else:
        st.info("Veuillez télécharger une image pour commencer.")

import streamlit as st
import easyocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import time

# Tentative d'import des modèles Transformers (optionnel)
try:
    from transformers import TableTransformerForObjectDetection, DetrImageProcessor
    import timm
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------
st.set_page_config(page_title="Pronostics Hippiques IA", layout="wide")

# ------------------------------------------------------------
# Initialisation des modèles
# ------------------------------------------------------------
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['fr'], gpu=False)

@st.cache_resource
def load_table_detector():
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        return processor, model
    except Exception as e:
        st.warning(f"Modèle de détection de tableaux non disponible : {e}. Utilisation de l'OCR standard.")
        return None, None

# ------------------------------------------------------------
# Fonctions de parsing spécifiques aux courses de plat
# ------------------------------------------------------------
def parse_musique_plat(musique_str):
    """
    Extrait les performances récentes à partir d'une chaîne de musique de plat.
    Exemples : "[[8p][1p] (42,5)" -> ["8p", "1p"]
                "3p(25)4p" -> ["3p", "4p"]
                "[4p](25)1p" -> ["4p", "1p"]
    """
    if not isinstance(musique_str, str):
        return []
    # Supprimer les parenthèses avec années et les crochets
    musique_str = re.sub(r'\([^)]*\)', '', musique_str)  # enlever (25)
    musique_str = musique_str.replace('[', '').replace(']', '')
    # Trouver toutes les occurrences de chiffres suivis de p (parfois avec des crochets)
    pattern = r'(\d+p)'
    parts = re.findall(pattern, musique_str)
    return parts

def score_musique_plat(musique_list, max_items=5):
    """
    Calcule un score à partir des dernières performances en plat.
    Pondération: 1p=10, 2p=8, 3p=6, 4p=5, 5p=4, 6p=3, 7p=2, 8p=1, 9p=0, etc.
    """
    weights = {
        '1p': 10, '2p': 8, '3p': 6, '4p': 5, '5p': 4,
        '6p': 3, '7p': 2, '8p': 1, '9p': 0, '10p': 0,
        '11p': 0, '12p': 0, '13p': 0, '14p': 0
    }
    recent = musique_list[:max_items]
    if not recent:
        return 0
    total = 0
    for perf in recent:
        perf_lower = perf.lower()
        if perf_lower in weights:
            total += weights[perf_lower]
        else:
            # Si c'est comme '4p' mais pas dans le dictionnaire, on essaie d'extraire le chiffre
            match = re.match(r'(\d+)p', perf_lower)
            if match:
                place = int(match.group(1))
                if place <= 9:
                    total += max(0, 10 - place)
                else:
                    total += 0
            else:
                total += 0
    return total / len(recent)

def parse_valeur(valeur_str):
    """Extrait la valeur (handicap) d'une chaîne comme '42' ou '38,5'."""
    if isinstance(valeur_str, str):
        # Remplacer la virgule par un point pour conversion
        valeur_str = valeur_str.replace(',', '.')
        match = re.search(r'(\d+(?:\.\d+)?)', valeur_str)
        if match:
            return float(match.group(1))
    return None

# ------------------------------------------------------------
# Fonctions de nettoyage génériques
# ------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('|', '1').replace('O', '0').replace('l', '1')
    return text.strip()

# ------------------------------------------------------------
# Détection de la structure du tableau (améliorée)
# ------------------------------------------------------------
def detect_table_structure_fallback(text_boxes):
    """
    Version améliorée pour détecter les lignes et colonnes.
    Retourne un DataFrame avec les données brutes.
    """
    if not text_boxes:
        return pd.DataFrame()
    
    # Trier par y
    text_boxes.sort(key=lambda x: x['y'])
    
    # Regrouper par lignes (avec seuil adaptatif)
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
    
    # Identifier la ligne d'en-tête : on prend la ligne qui contient le plus de mots-clés possibles
    header_keywords = ['n°', 'cheval', 'jockey', 'entraîneur', 'poids', 'musique', 'valeur']
    best_header_idx = 0
    best_score = -1
    for i, line in enumerate(lines):
        texts = ' '.join([item['text'].lower() for item in line])
        score = sum(1 for kw in header_keywords if kw in texts)
        if score > best_score:
            best_score = score
            best_header_idx = i
    
    header_line = lines[best_header_idx]
    columns = [item['text'] for item in header_line]
    
    # Lignes de données : après l'en-tête
    data_lines = lines[best_header_idx+1:]
    
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
    # Ne garder que les colonnes présentes dans l'en-tête
    cols_present = [c for c in columns if c in df.columns]
    if cols_present:
        df = df[cols_present]
    return df

# ------------------------------------------------------------
# Nettoyage spécifique selon le type de tableau détecté
# ------------------------------------------------------------
def detect_table_type(df):
    """Détermine le type de tableau en fonction des colonnes présentes."""
    text = ' '.join(df.columns).lower()
    if 'jockey' in text or 'jockey' in str(df.iloc[0] if not df.empty else ''):
        return 'plat'
    elif 'driver' in text:
        return 'trot'
    elif 'record' in text:
        return 'records'
    else:
        return 'unknown'

def clean_dataframe_plat(df):
    """Nettoie un DataFrame de type plat (partants)."""
    df = df.copy()
    df = df.dropna(how='all')
    
    # Renommer les colonnes si nécessaire (par exemple si OCR a mal interprété)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'n°' in col_lower or 'no' in col_lower or 'num' in col_lower:
            col_mapping[col] = 'N°'
        elif 'cheval' in col_lower:
            col_mapping[col] = 'Cheval'
        elif 'jockey' in col_lower:
            col_mapping[col] = 'Jockey'
        elif 'entraîneur' in col_lower or 'entraineur' in col_lower:
            col_mapping[col] = 'Entraîneur'
        elif 'poids' in col_lower:
            col_mapping[col] = 'Poids'
        elif 'musique' in col_lower:
            col_mapping[col] = 'Musique'
        elif 'valeur' in col_lower or 'val' in col_lower:
            col_mapping[col] = 'Valeur'
    df.rename(columns=col_mapping, inplace=True)
    
    # Nettoyer chaque cellule
    for col in df.columns:
        df[col] = df[col].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    
    # Convertir les types
    if 'N°' in df.columns:
        df['N°'] = pd.to_numeric(df['N°'], errors='coerce')
    if 'Valeur' in df.columns:
        df['Valeur'] = df['Valeur'].apply(parse_valeur)
    if 'Musique' in df.columns:
        df['Musique_list'] = df['Musique'].apply(parse_musique_plat)
        df['Score_musique'] = df['Musique_list'].apply(lambda x: score_musique_plat(x) if isinstance(x, list) else 0)
    
    return df

# ------------------------------------------------------------
# Scoring adapté au plat
# ------------------------------------------------------------
def calculer_score_plat(row):
    """
    Score basé sur la valeur (handicap) et la musique.
    Plus la valeur est élevée, meilleur est le cheval (handicap).
    """
    score = 0
    weights = {
        'valeur': 0.6,
        'musique': 0.4
    }
    if pd.notna(row.get('Valeur')):
        # Normaliser la valeur par rapport à la fourchette (ex: entre 30 et 45)
        # Ici on suppose que les valeurs sont entre 30 et 45, on normalise entre 0 et 1
        val = row['Valeur']
        # À ajuster selon les données réelles
        val_norm = (val - 30) / 15  # donne entre 0 et 1 si val entre 30 et 45
        val_norm = max(0, min(1, val_norm))
        score += weights['valeur'] * val_norm
    
    if pd.notna(row.get('Score_musique')):
        # Score musique déjà entre 0 et 10
        score += weights['musique'] * (row['Score_musique'] / 10)
    
    return score

def classer_chevaux_plat(df):
    df['score_brut'] = df.apply(calculer_score_plat, axis=1)
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
# Extraction intelligente avec OCR
# ------------------------------------------------------------
class IntelligentTableDetector:
    def __init__(self):
        self.processor, self.model = load_table_detector()
        self.use_ai = self.processor is not None and self.model is not None
        
    def detect_table_regions(self, image):
        if not self.use_ai:
            return []
        max_size = 800
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
        regions = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.7:
                box = [round(i, 2) for i in box.tolist()]
                regions.append({'bbox': box, 'confidence': score.item()})
        return regions
    
    def crop_table(self, image, bbox):
        return image.crop(bbox)

def intelligent_table_extraction(image):
    detector = IntelligentTableDetector()
    reader = load_ocr_reader()
    if detector.use_ai:
        regions = detector.detect_table_regions(image)
        if regions:
            best = max(regions, key=lambda x: x['confidence'])
            table_image = detector.crop_table(image, best['bbox'])
            results = reader.readtext(np.array(table_image))
            text_boxes = []
            for (bbox, text, conf) in results:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                text_boxes.append({
                    'text': text,
                    'x': (min(x_coords)+max(x_coords))/2,
                    'y': (min(y_coords)+max(y_coords))/2,
                    'x_min': min(x_coords),
                    'x_max': max(x_coords),
                    'y_min': min(y_coords),
                    'y_max': max(y_coords)
                })
            df = detect_table_structure_fallback(text_boxes)
            return [{'type': 'unknown', 'dataframe': df, 'confidence': best['confidence']}]
    # Fallback : OCR sur toute l'image
    results = reader.readtext(np.array(image))
    text_boxes = []
    for (bbox, text, conf) in results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        text_boxes.append({
            'text': text,
            'x': (min(x_coords)+max(x_coords))/2,
            'y': (min(y_coords)+max(y_coords))/2,
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords)
        })
    df = detect_table_structure_fallback(text_boxes)
    return [{'type': 'unknown', 'dataframe': df, 'confidence': 0.5}]

# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------
st.title("🐎 Application de Pronostics Hippiques IA")
st.markdown("### Version optimisée pour les courses de plat")

st.sidebar.title("⚙️ Configuration")
use_ai_detection = st.sidebar.checkbox("Utiliser la détection IA (si disponible)", value=True)

uploaded_files = st.file_uploader(
    "📤 Télécharger les photos (PNG, JPG, JPEG)",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_final = None

if uploaded_files:
    st.subheader("Aperçu des images téléchargées")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, file in enumerate(uploaded_files):
        with cols[i % 4]:
            image = Image.open(file)
            st.image(image, caption=file.name, use_container_width=True)

if st.button("🔍 Analyser avec IA", type="primary") and uploaded_files:
    with st.spinner("Analyse en cours..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_tables = []
        for idx, file in enumerate(uploaded_files):
            image = Image.open(file)
            if use_ai_detection:
                tables = intelligent_table_extraction(image)
            else:
                reader = load_ocr_reader()
                results = reader.readtext(np.array(image))
                text_boxes = []
                for (bbox, text, conf) in results:
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    text_boxes.append({
                        'text': text,
                        'x': (min(x_coords)+max(x_coords))/2,
                        'y': (min(y_coords)+max(y_coords))/2,
                        'x_min': min(x_coords),
                        'x_max': max(x_coords),
                        'y_min': min(y_coords),
                        'y_max': max(y_coords)
                    })
                df = detect_table_structure_fallback(text_boxes)
                tables = [{'type': 'unknown', 'dataframe': df, 'confidence': 0.5}]
            all_tables.extend(tables)
            progress_bar.progress((idx+1)/len(uploaded_files))
        
        # Prendre le premier tableau non vide
        df_final = None
        for table in all_tables:
            if table['dataframe'] is not None and not table['dataframe'].empty:
                df_final = table['dataframe']
                break
        
        if df_final is None or df_final.empty:
            st.error("Aucune donnée valide n'a pu être extraite. Vérifiez vos images.")
        else:
            # Déterminer le type de tableau et nettoyer
            table_type = detect_table_type(df_final)
            if table_type == 'plat':
                df_final = clean_dataframe_plat(df_final)
                # S'assurer que la colonne N° existe
                if 'N°' not in df_final.columns:
                    df_final['N°'] = range(1, len(df_final)+1)
                # Classer
                df_final = classer_chevaux_plat(df_final)
            else:
                st.warning("Type de tableau non reconnu, utilisation du scoring par défaut (tous les scores à 0.5).")
                df_final['N°'] = range(1, len(df_final)+1)
                df_final['score_normalise'] = 0.5
                df_final['rang'] = range(1, len(df_final)+1)
            
            st.session_state.df_final = df_final
            st.session_state.data_loaded = True
            
            progress_bar.progress(100)
            status_text.text("Analyse terminée!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

if st.session_state.data_loaded and st.session_state.df_final is not None:
    df = st.session_state.df_final
    
    st.subheader("📊 Données extraites et scores")
    # Afficher les colonnes disponibles
    cols_afficher = ['N°', 'Cheval', 'Jockey', 'Entraîneur', 'Valeur', 'Score_musique', 'score_normalise', 'rang']
    cols_afficher = [c for c in cols_afficher if c in df.columns]
    st.dataframe(df[cols_afficher])
    
    # Graphique
    st.subheader("📈 Scores des chevaux")
    fig, ax = plt.subplots(figsize=(10, 6))
    chevaux = df['Cheval'].astype(str) + " (N°" + df['N°'].astype(str) + ")"
    ax.barh(chevaux, df['score_normalise'])
    ax.set_xlabel("Score normalisé")
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
else:
    if uploaded_files:
        st.info("Cliquez sur 'Analyser avec IA' pour lancer l'extraction.")
    else:
        st.info("Veuillez télécharger des images pour commencer.")

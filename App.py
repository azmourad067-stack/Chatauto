import streamlit as st
import easyocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import time

# Tentative d'import des modèles Transformers (peut échouer)
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
# Initialisation des modèles (mise en cache)
# ------------------------------------------------------------
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['fr'], gpu=False)

@st.cache_resource
def load_table_detector():
    """Charge le modèle Table Transformer si possible"""
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
# Fonctions de nettoyage et parsing (reprises de la version précédente)
# ------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('|', '1').replace('O', '0').replace('l', '1')
    return text.strip()

def parse_percentage(pct_str):
    if isinstance(pct_str, str):
        match = re.search(r'(\d+(?:[.,]\d+)?)\s*%', pct_str)
        if match:
            return float(match.group(1).replace(',', '.')) / 100
    return None

def parse_gains(gains_str):
    if isinstance(gains_str, str):
        gains_str = gains_str.replace(' ', '').replace('\u202f', '')
        match = re.search(r'(\d+)', gains_str)
        if match:
            return int(match.group(1))
    return None

def parse_record(record_str):
    if isinstance(record_str, str):
        match = re.search(r"(\d+)'(\d+)\"?(\d?)", record_str)
        if match:
            minutes = int(match.group(1))
            secondes = int(match.group(2))
            dixiemes = int(match.group(3)) if match.group(3) else 0
            return minutes * 60 + secondes + dixiemes / 10
    return None

def parse_musique(musique_str):
    if not isinstance(musique_str, str):
        return []
    musique_str = re.sub(r'\(\d+\)', '', musique_str)
    pattern = r'(\d*[aAmM]?[aA]?|Da|Dm|0a)'
    parts = re.findall(pattern, musique_str)
    return [p for p in parts if p]

def score_musique(musique_list, max_items=5):
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
    recent = musique_list[:max_items]
    if not recent:
        return 0
    total = 0
    for perf in recent:
        perf_lower = perf.lower()
        if perf_lower in weights:
            total += weights[perf_lower]
        else:
            match = re.match(r'(\d+)a', perf_lower)
            if match:
                place = int(match.group(1))
                if 1 <= place <= 9:
                    total += max(0, 10 - place)
    return total / len(recent)

def clean_dataframe(df, table_type):
    df = df.copy()
    df = df.dropna(how='all')
    for col in df.columns:
        df[col] = df[col].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    
    if table_type == 'partants':
        if 'N°' in df.columns:
            df['N°'] = pd.to_numeric(df['N°'], errors='coerce')
        if 'Gains' in df.columns:
            df['Gains'] = df['Gains'].apply(parse_gains)
        if 'Musique' in df.columns:
            df['Musique_cheval'] = df['Musique'].apply(parse_musique)
    elif table_type == 'drivers':
        if 'Réussite' in df.columns:
            df['Reussite_driver'] = df['Réussite'].apply(parse_percentage)
        if 'Courses' in df.columns:
            df['Courses_driver'] = pd.to_numeric(df['Courses'], errors='coerce')
        if 'Victoires' in df.columns:
            df['Victoires_driver'] = pd.to_numeric(df['Victoires'], errors='coerce')
        if 'Ecart' in df.columns:
            df['Ecart_driver'] = pd.to_numeric(df['Ecart'], errors='coerce')
        if 'Musique Driver' in df.columns:
            df['Musique_driver'] = df['Musique Driver'].apply(parse_musique)
    elif table_type == 'entraineurs':
        if 'Réussite' in df.columns:
            df['Reussite_entraineur'] = df['Réussite'].apply(parse_percentage)
        if 'Courses' in df.columns:
            df['Courses_entraineur'] = pd.to_numeric(df['Courses'], errors='coerce')
        if 'Victoires' in df.columns:
            df['Victoires_entraineur'] = pd.to_numeric(df['Victoires'], errors='coerce')
        if 'Ecart' in df.columns:
            df['Ecart_entraineur'] = pd.to_numeric(df['Ecart'], errors='coerce')
        if 'Musique Entraîneur' in df.columns:
            df['Musique_entraineur'] = df['Musique Entraîneur'].apply(parse_musique)
    elif table_type == 'records':
        if 'Record' in df.columns:
            df['Record_secondes'] = df['Record'].apply(parse_record)
    return df

def fusionner_donnees(partants_df, drivers_df, entraineurs_df, records_df):
    if partants_df is None or partants_df.empty:
        return pd.DataFrame()
    
    def normalize_name(name):
        if isinstance(name, str):
            name = name.strip().lower()
            name = name.replace('é','e').replace('è','e').replace('ê','e').replace('à','a').replace('ç','c')
            return name
        return ''
    
    partants_df['cheval_norm'] = partants_df['Cheval'].apply(normalize_name)
    
    def merge_table(base_df, other_df, suffix):
        if other_df is None or other_df.empty:
            return base_df
        other_df['cheval_norm'] = other_df['Cheval'].apply(normalize_name)
        merged = pd.merge(base_df, other_df, on=['N°', 'cheval_norm'], how='left', suffixes=('', suffix))
        return merged
    
    if drivers_df is not None:
        partants_df = merge_table(partants_df, drivers_df, '_driver')
    if entraineurs_df is not None:
        partants_df = merge_table(partants_df, entraineurs_df, '_entraineur')
    if records_df is not None:
        partants_df = merge_table(partants_df, records_df, '_record')
    
    partants_df['score_musique_cheval'] = partants_df['Musique_cheval'].apply(lambda x: score_musique(x) if isinstance(x, list) else 0)
    if 'Musique_driver' in partants_df.columns:
        partants_df['score_musique_driver'] = partants_df['Musique_driver'].apply(lambda x: score_musique(x) if isinstance(x, list) else 0)
    if 'Musique_entraineur' in partants_df.columns:
        partants_df['score_musique_entraineur'] = partants_df['Musique_entraineur'].apply(lambda x: score_musique(x) if isinstance(x, list) else 0)
    
    return partants_df

# ------------------------------------------------------------
# Fonctions de scoring
# ------------------------------------------------------------
def calculer_score_cheval(row, weights=None):
    if weights is None:
        weights = {
            'record': 0.20,
            'reussite_driver': 0.15,
            'reussite_entraineur': 0.15,
            'musique_cheval': 0.20,
            'musique_driver': 0.10,
            'musique_entraineur': 0.10,
            'gains': 0.05,
            'ecart_driver': 0.03,
            'ecart_entraineur': 0.02
        }
    score = 0
    if pd.notna(row.get('Record_secondes')):
        score += weights['record'] * (1 / row['Record_secondes'])
    if pd.notna(row.get('Reussite_driver')):
        score += weights['reussite_driver'] * row['Reussite_driver']
    if pd.notna(row.get('Reussite_entraineur')):
        score += weights['reussite_entraineur'] * row['Reussite_entraineur']
    if pd.notna(row.get('score_musique_cheval')):
        score += weights['musique_cheval'] * (row['score_musique_cheval'] / 10)
    if pd.notna(row.get('score_musique_driver')):
        score += weights['musique_driver'] * (row['score_musique_driver'] / 10)
    if pd.notna(row.get('score_musique_entraineur')):
        score += weights['musique_entraineur'] * (row['score_musique_entraineur'] / 10)
    if pd.notna(row.get('Gains')):
        score += weights['gains'] * (row['Gains'] / 100000)
    if pd.notna(row.get('Ecart_driver')):
        score += weights['ecart_driver'] * (1 / (1 + row['Ecart_driver']))
    if pd.notna(row.get('Ecart_entraineur')):
        score += weights['ecart_entraineur'] * (1 / (1 + row['Ecart_entraineur']))
    return score

def normaliser_scores(df, score_col='score_brut'):
    min_score = df[score_col].min()
    max_score = df[score_col].max()
    if max_score - min_score > 0:
        df['score_normalise'] = (df[score_col] - min_score) / (max_score - min_score)
    else:
        df['score_normalise'] = 0.5
    return df

def classer_chevaux(df):
    df['score_brut'] = df.apply(calculer_score_cheval, axis=1)
    df = normaliser_scores(df)
    df = df.sort_values('score_normalise', ascending=False).reset_index(drop=True)
    df['rang'] = df.index + 1
    return df

# ------------------------------------------------------------
# Fonctions de pronostics
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
# Classes pour l'extraction intelligente (avec fallback)
# ------------------------------------------------------------
class IntelligentTableDetector:
    def __init__(self):
        self.processor, self.model = load_table_detector()
        self.use_ai = self.processor is not None and self.model is not None
        
    def detect_table_regions(self, image):
        if not self.use_ai:
            return []
        # Redimensionner si nécessaire
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

def detect_table_structure_fallback(text_boxes):
    """Version simple de détection de structure (sans classification)"""
    if not text_boxes:
        return pd.DataFrame()
    text_boxes.sort(key=lambda x: x['y'])
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
    header_line = lines[0]
    columns = [item['text'] for item in header_line]
    data_lines = lines[1:]
    data_rows = []
    for line in data_lines:
        row = {}
        for item in line:
            best_col_idx = min(range(len(columns)), key=lambda i: abs(item['x'] - header_line[i]['x']))
            row[columns[best_col_idx]] = item['text']
        data_rows.append(row)
    df = pd.DataFrame(data_rows)
    cols_present = [c for c in columns if c in df.columns]
    if cols_present:
        df = df[cols_present]
    return df

def intelligent_table_extraction(image):
    """Extraction avec détection IA si possible, sinon OCR classique"""
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
st.markdown("### Version avec intelligence artificielle pour une extraction précise des données")

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
                # Méthode simple (OCR direct)
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
        
        # Prendre le premier tableau non vide (on suppose que c'est la table des partants)
        df_final = None
        for table in all_tables:
            if table['dataframe'] is not None and not table['dataframe'].empty:
                df_final = table['dataframe']
                break
        
        if df_final is None or df_final.empty:
            st.error("Aucune donnée valide n'a pu être extraite. Vérifiez vos images.")
        else:
            # Nettoyage basique (en supposant que c'est une table partants)
            df_final = clean_dataframe(df_final, 'partants')
            if 'N°' not in df_final.columns:
                st.warning("La colonne N° n'a pas été détectée. Utilisation de l'index comme numéro.")
                df_final['N°'] = range(1, len(df_final)+1)
            
            df_final = classer_chevaux(df_final)
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
    cols_afficher = ['N°', 'Cheval', 'Driver', 'Entraîneur', 'Gains', 'score_normalise', 'rang']
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

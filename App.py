import streamlit as st
import easyocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import time
import torch

# Tentative d'import des modèles Transformers (peut échouer)
try:
    from transformers import TableTransformerForObjectDetection, DetrImageProcessor
    import timm
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------
st.set_page_config(page_title="Pronostics Hippiques IA", layout="wide")

# ------------------------------------------------------------
# Initialisation des modèles IA (avec cache et fallback)
# ------------------------------------------------------------
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

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['fr'], gpu=False)

# ------------------------------------------------------------
# Classe pour la détection intelligente des tableaux (avec fallback)
# ------------------------------------------------------------
class IntelligentTableDetector:
    def __init__(self):
        self.processor, self.model = load_table_detector()
        self.use_ai = self.processor is not None and self.model is not None
        
    def detect_table_regions(self, image):
        """Détecte les régions de tableaux dans l'image (retourne [] si IA non dispo)"""
        if not self.use_ai:
            return []
        
        # Redimensionner si l'image est trop grande (pour éviter les erreurs mémoire)
        max_size = 800
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
        
        table_regions = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.7:
                box = [round(i, 2) for i in box.tolist()]
                table_regions.append({
                    'bbox': box,
                    'confidence': score.item()
                })
        return table_regions
    
    def crop_table(self, image, bbox):
        return image.crop(bbox)

# ------------------------------------------------------------
# Fonction d'extraction intelligente avec fallback
# ------------------------------------------------------------
def intelligent_table_extraction(image):
    """Extraction avec détection IA si possible, sinon OCR classique"""
    detector = IntelligentTableDetector()
    ocr_reader = load_ocr_reader()
    
    if detector.use_ai:
        regions = detector.detect_table_regions(image)
        if regions:
            # Utiliser la région avec la meilleure confiance
            best_region = max(regions, key=lambda x: x['confidence'])
            table_image = detector.crop_table(image, best_region['bbox'])
            # OCR sur cette région
            results = ocr_reader.readtext(np.array(table_image))
            # Construire les text_boxes
            text_boxes = []
            for (bbox, text, conf) in results:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                text_boxes.append({
                    'text': text,
                    'x': (min(x_coords) + max(x_coords)) / 2,
                    'y': (min(y_coords) + max(y_coords)) / 2,
                    'x_min': min(x_coords),
                    'x_max': max(x_coords),
                    'y_min': min(y_coords),
                    'y_max': max(y_coords)
                })
            # Détection de la structure
            df = detect_table_structure_fallback(text_boxes)
            return [{'type': 'unknown', 'dataframe': df, 'confidence': best_region['confidence']}]
    
    # Fallback : OCR sur toute l'image
    results = ocr_reader.readtext(np.array(image))
    text_boxes = []
    for (bbox, text, conf) in results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        text_boxes.append({
            'text': text,
            'x': (min(x_coords) + max(x_coords)) / 2,
            'y': (min(y_coords) + max(y_coords)) / 2,
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords)
        })
    df = detect_table_structure_fallback(text_boxes)
    return [{'type': 'unknown', 'dataframe': df, 'confidence': 0.5}]

def detect_table_structure_fallback(text_boxes):
    """Version simplifiée de détection de structure (sans classification)"""
    if not text_boxes:
        return pd.DataFrame()
    
    text_boxes.sort(key=lambda x: x['y'])
    
    # Regrouper par lignes
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
    
    # Première ligne comme en-tête
    header_line = lines[0]
    columns = [item['text'] for item in header_line]
    
    data_lines = lines[1:]
    data_rows = []
    for line in data_lines:
        row = {}
        for item in line:
            # Trouver la colonne la plus proche
            best_col_idx = min(range(len(columns)), key=lambda i: abs(item['x'] - header_line[i]['x']))
            row[columns[best_col_idx]] = item['text']
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    # Ne garder que les colonnes présentes
    cols_present = [c for c in columns if c in df.columns]
    if cols_present:
        df = df[cols_present]
    return df

# ------------------------------------------------------------
# [Insérez ici toutes les fonctions de scoring et pronostics]
# (parse_percentage, parse_gains, parse_record, parse_musique, 
#  score_musique, clean_dataframe, fusionner_donnees, 
#  calculer_score_cheval, classer_chevaux, generer_top_3, etc.)
# ------------------------------------------------------------
# ... (copiez les fonctions des versions précédentes) ...

# ------------------------------------------------------------
# Interface Streamlit (adaptée)
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
                        'x': (min(x_coords) + max(x_coords)) / 2,
                        'y': (min(y_coords) + max(y_coords)) / 2,
                        'x_min': min(x_coords),
                        'x_max': max(x_coords),
                        'y_min': min(y_coords),
                        'y_max': max(y_coords)
                    })
                df = detect_table_structure_fallback(text_boxes)
                tables = [{'type': 'unknown', 'dataframe': df, 'confidence': 0.5}]
            
            all_tables.extend(tables)
            progress_bar.progress((idx+1)/len(uploaded_files))
        
        # Fusionner les DataFrames (simplifié : on prend le premier non vide)
        df_final = None
        for table in all_tables:
            if table['dataframe'] is not None and not table['dataframe'].empty:
                df_final = table['dataframe']
                break
        
        if df_final is None or df_final.empty:
            st.error("Aucune donnée extraite.")
        else:
            # Nettoyage basique
            df_final = clean_dataframe(df_final, 'partants')  # suppose que c'est une table partants
            if 'N°' not in df_final.columns:
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
    st.dataframe(df[['N°', 'Cheval', 'Driver', 'Entraîneur', 'Gains', 'score_normalise', 'rang']])
    
    # Graphique
    fig, ax = plt.subplots()
    chevaux = df['Cheval'] + " (N°" + df['N°'].astype(str) + ")"
    ax.barh(chevaux, df['score_normalise'])
    ax.set_xlabel("Score")
    st.pyplot(fig)
    
    # Pronostics (simplifiés)
    st.subheader("🏆 Top 3")
    top3 = df.head(3)
    for i, row in top3.iterrows():
        st.write(f"{i+1}. N°{row['N°']} - {row['Cheval']} ({row['score_normalise']:.3f})")
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger CSV", csv, "pronostics.csv")

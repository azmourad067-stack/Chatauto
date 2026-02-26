import streamlit as st
import easyocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import time
import torch
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
import torchvision.transforms as T
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# ------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------
st.set_page_config(page_title="Pronostics Hippiques IA", layout="wide")

# ------------------------------------------------------------
# Initialisation des modèles IA (avec cache)
# ------------------------------------------------------------
@st.cache_resource
def load_table_detector():
    """Charge le modèle Table Transformer pour détecter les tableaux dans les images"""
    processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    return processor, model

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['fr'], gpu=False)

@st.cache_resource
def load_table_classifier():
    """Charge un classifieur pour identifier le type de tableau"""
    # Définir les mots-clés pour chaque type
    type_keywords = {
        'partants': ['n°', 'cheval', 'driver', 'entraîneur', 'musique', 'gains'],
        'drivers': ['driver', 'courses', 'victoires', 'réussite', 'écart'],
        'entraineurs': ['entraîneur', 'courses', 'victoires', 'réussite', 'écart'],
        'records': ['record', 'date', 'temps']
    }
    return type_keywords

# ------------------------------------------------------------
# Classe pour la détection intelligente des tableaux
# ------------------------------------------------------------
class IntelligentTableDetector:
    def __init__(self):
        self.processor, self.model = load_table_detector()
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def detect_table_regions(self, image):
        """Détecte les régions de tableaux dans l'image"""
        # Préparer l'image pour le modèle
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convertir les sorties en boîtes
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
        
        table_regions = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.7:  # Seuil de confiance
                box = [round(i, 2) for i in box.tolist()]
                table_regions.append({
                    'bbox': box,
                    'confidence': score.item()
                })
        
        return table_regions
    
    def crop_table(self, image, bbox):
        """Extrait une région de tableau de l'image"""
        left, top, right, bottom = bbox
        return image.crop((left, top, right, bottom))

# ------------------------------------------------------------
# Classe pour l'OCR contextuel avec correction IA
# ------------------------------------------------------------
class ContextualOCR:
    def __init__(self):
        self.reader = load_ocr_reader()
        self.type_keywords = load_table_classifier()
        
        # Dictionnaire de correction pour les erreurs courantes
        self.common_corrections = {
            'N°': ['No', 'N0', 'N*', 'N°'],
            'Cheval': ['ChevaI', 'Cheva1', 'Chevai'],
            'Driver': ['Drlver', 'Dr1ver', 'Oriver'],
            'Entraîneur': ['Entralneur', 'EntraIneur', 'Entraineur'],
        }
    
    def extract_text_with_context(self, image):
        """Extrait le texte avec analyse contextuelle"""
        results = self.reader.readtext(np.array(image))
        
        text_boxes = []
        for (bbox, text, conf) in results:
            # Correction contextuelle
            corrected_text = self.correct_text(text)
            
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            text_boxes.append({
                'text': corrected_text,
                'original_text': text,
                'confidence': conf,
                'x': (min(x_coords) + max(x_coords)) / 2,
                'y': (min(y_coords) + max(y_coords)) / 2,
                'x_min': min(x_coords),
                'x_max': max(x_coords),
                'y_min': min(y_coords),
                'y_max': max(y_coords)
            })
        
        return text_boxes
    
    def correct_text(self, text):
        """Corrige le texte basé sur les erreurs courantes"""
        corrected = text
        for correct, variations in self.common_corrections.items():
            for var in variations:
                if var in corrected:
                    corrected = corrected.replace(var, correct)
        return corrected
    
    def classify_table_type(self, text_boxes):
        """Classifie automatiquement le type de tableau basé sur le contenu"""
        # Extraire tout le texte
        all_text = ' '.join([box['text'] for box in text_boxes]).lower()
        
        # Calculer un score pour chaque type
        scores = {}
        for table_type, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            scores[table_type] = score
        
        # Retourner le type avec le score le plus élevé
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'unknown'

# ------------------------------------------------------------
# Fonctions améliorées d'extraction de tableaux
# ------------------------------------------------------------
def intelligent_table_extraction(image):
    """Extraction intelligente des tableaux avec détection de région et classification"""
    
    # Initialiser les détecteurs
    table_detector = IntelligentTableDetector()
    contextual_ocr = ContextualOCR()
    
    # Étape 1: Détecter les régions de tableaux
    table_regions = table_detector.detect_table_regions(image)
    
    all_tables = []
    
    if table_regions:
        # Si des régions sont détectées, les extraire
        for region in table_regions:
            table_image = table_detector.crop_table(image, region['bbox'])
            text_boxes = contextual_ocr.extract_text_with_context(table_image)
            table_type = contextual_ocr.classify_table_type(text_boxes)
            
            # Détecter la structure du tableau
            df = detect_table_structure_improved(text_boxes, table_type)
            
            if not df.empty:
                all_tables.append({
                    'type': table_type,
                    'dataframe': df,
                    'confidence': region['confidence']
                })
    else:
        # Fallback: OCR sur toute l'image
        text_boxes = contextual_ocr.extract_text_with_context(image)
        table_type = contextual_ocr.classify_table_type(text_boxes)
        df = detect_table_structure_improved(text_boxes, table_type)
        
        if not df.empty:
            all_tables.append({
                'type': table_type,
                'dataframe': df,
                'confidence': 0.5
            })
    
    return all_tables

def detect_table_structure_improved(text_boxes, table_type):
    """Version améliorée de la détection de structure avec connaissances du contexte"""
    
    if not text_boxes:
        return pd.DataFrame()
    
    # Trier par position verticale
    text_boxes.sort(key=lambda x: x['y'])
    
    # Regrouper par lignes avec seuil adaptatif
    heights = [box['y_max'] - box['y_min'] for box in text_boxes]
    avg_height = np.mean(heights) if heights else 20
    line_threshold = avg_height * 0.6  # Plus petit seuil pour meilleure détection
    
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
    
    # Identifier la ligne d'en-tête basée sur le type de tableau
    header_keywords = {
        'partants': ['n°', 'cheval', 'driver'],
        'drivers': ['driver', 'courses', 'victoires'],
        'entraineurs': ['entraîneur', 'courses', 'victoires'],
        'records': ['record', 'date']
    }
    
    keywords = header_keywords.get(table_type, [])
    
    header_line = None
    header_index = 0
    
    for i, line in enumerate(lines):
        texts = [item['text'].lower() for item in line]
        if any(any(kw in t for kw in keywords) for t in texts):
            header_line = line
            header_index = i
            break
    
    if header_line is None:
        # Si pas d'en-tête trouvé, utiliser la première ligne
        header_line = lines[0]
        header_index = 0
    
    # Extraire les colonnes
    columns = [item['text'] for item in header_line]
    
    # Traiter les lignes de données
    data_lines = lines[header_index+1:]
    data_rows = []
    
    for line in data_lines:
        row = {}
        for item in line:
            # Trouver la colonne la plus proche
            col_distances = []
            for col_idx, col_item in enumerate(header_line):
                dist = abs(item['x'] - col_item['x'])
                col_distances.append((col_idx, dist))
            
            best_col_idx = min(col_distances, key=lambda x: x[1])[0]
            best_col = columns[best_col_idx]
            
            # Nettoyer le texte selon le contexte de la colonne
            cleaned_text = clean_text_contextual(item['text'], best_col)
            row[best_col] = cleaned_text
        
        data_rows.append(row)
    
    # Créer DataFrame
    df = pd.DataFrame(data_rows)
    
    # Réordonner les colonnes
    if not df.empty:
        # Garder seulement les colonnes qui existent
        cols_present = [c for c in columns if c in df.columns]
        if cols_present:
            df = df[cols_present]
    
    return df

def clean_text_contextual(text, column_name):
    """Nettoie le texte en fonction du contexte de la colonne"""
    if not isinstance(text, str):
        return text
    
    # Nettoyage de base
    text = text.strip()
    
    # Nettoyage spécifique selon le type de colonne
    col_lower = column_name.lower()
    
    if 'n°' in col_lower or 'num' in col_lower:
        # Extraire uniquement les chiffres
        numbers = re.findall(r'\d+', text)
        return numbers[0] if numbers else text
    
    elif 'gains' in col_lower or 'record' in col_lower:
        # Garder les chiffres et certains caractères spéciaux
        text = re.sub(r'[^\d\s\'\"]', '', text)
    
    elif '%' in col_lower or 'réussite' in col_lower:
        # S'assurer que le pourcentage est bien formaté
        if '%' not in text:
            text = text + '%'
    
    return text

# ------------------------------------------------------------
# Reste du code (identique à avant avec les fonctions de scoring et pronostics)
# ------------------------------------------------------------
# [Insérer ici toutes les fonctions de scoring et pronostics de la version précédente]
# (parse_percentage, parse_gains, parse_record, parse_musique, score_musique, 
#  clean_dataframe, fusionner_donnees, calculer_score_cheval, classer_chevaux,
#  generer_top_3, generer_bases, generer_outsiders, generer_combinaisons_trio,
#  generer_combinaisons_quinte)

# ------------------------------------------------------------
# Interface Streamlit améliorée
# ------------------------------------------------------------
st.title("🐎 Application de Pronostics Hippiques IA")
st.markdown("### Version avec intelligence artificielle pour une extraction précise des données")

st.sidebar.title("⚙️ Configuration IA")
use_advanced_detection = st.sidebar.checkbox("Utiliser la détection avancée des tableaux", value=True)
confidence_threshold = st.sidebar.slider("Seuil de confiance", 0.0, 1.0, 0.7)

st.markdown("Téléchargez les captures d'écran des statistiques (partants, drivers, entraîneurs, records) pour obtenir une analyse complète.")

uploaded_files = st.file_uploader(
    "📤 Télécharger les photos (PNG, JPG, JPEG)",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

# Initialisation de l'état de session
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_final = None

# Aperçu des images
if uploaded_files:
    st.subheader("Aperçu des images téléchargées")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, file in enumerate(uploaded_files):
        with cols[i % 4]:
            image = Image.open(file)
            st.image(image, caption=file.name, use_container_width=True)

# Bouton d'analyse avec IA
if st.button("🔍 Analyser avec IA", type="primary") and uploaded_files:
    with st.spinner("Analyse IA en cours... Veuillez patienter."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Étape 1: Extraction intelligente
        status_text.text("Détection des tableaux par IA...")
        all_extracted_tables = []
        
        for idx, file in enumerate(uploaded_files):
            image = Image.open(file)
            
            if use_advanced_detection:
                # Utiliser la détection avancée
                tables = intelligent_table_extraction(image)
                all_extracted_tables.extend(tables)
            else:
                # Fallback sur la méthode simple
                from ocr_utils import extract_table_from_image
                headers_keywords = {
                    'partants': ["N°", "Cheval", "Driver", "Entraîneur"],
                    'drivers': ["N°", "Cheval", "Driver", "Courses", "Victoires", "Réussite"],
                    'entraineurs': ["N°", "Cheval", "Entraîneur", "Courses", "Victoires", "Réussite"],
                    'records': ["N°", "Cheval", "Record", "Date"]
                }
                for table_type, keywords in headers_keywords.items():
                    df = extract_table_from_image(image, keywords)
                    if not df.empty and len(df) > 1:
                        all_extracted_tables.append({
                            'type': table_type,
                            'dataframe': df,
                            'confidence': 0.8
                        })
                        break
            
            progress_bar.progress((idx+1)/len(uploaded_files))
        
        # Organiser les tables par type (prendre la meilleure confiance pour chaque type)
        status_text.text("Organisation et fusion des données...")
        
        tables_by_type = {}
        for table in all_extracted_tables:
            table_type = table['type']
            confidence = table['confidence']
            
            if table_type not in tables_by_type or confidence > tables_by_type[table_type]['confidence']:
                tables_by_type[table_type] = {
                    'dataframe': table['dataframe'],
                    'confidence': confidence
                }
        
        # Nettoyer les DataFrames
        cleaned_tables = {}
        for table_type, table_info in tables_by_type.items():
            cleaned_tables[table_type] = clean_dataframe(table_info['dataframe'], table_type)
        
        # Fusion
        df_final = fusionner_donnees(
            cleaned_tables.get('partants'),
            cleaned_tables.get('drivers'),
            cleaned_tables.get('entraineurs'),
            cleaned_tables.get('records')
        )
        
        if df_final.empty:
            st.error("Aucune donnée valide n'a pu être extraite. Vérifiez vos images.")
        else:
            # Vérifier la présence de la colonne N°
            if 'N°' not in df_final.columns:
                st.warning("Le numéro des chevaux n'a pas été détecté. Utilisation de l'index.")
                df_final['N°'] = range(1, len(df_final) + 1)
            
            # Scoring
            status_text.text("Calcul des scores et génération des pronostics...")
            df_final = classer_chevaux(df_final)
            
            st.session_state.df_final = df_final
            st.session_state.data_loaded = True
            
            # Afficher un résumé de la détection
            st.sidebar.success("✅ Analyse IA terminée")
            st.sidebar.write("Tables détectées :")
            for table_type, table_info in tables_by_type.items():
                st.sidebar.write(f"- {table_type}: confiance {table_info['confidence']:.2f}")
            
            progress_bar.progress(100)
            status_text.text("Analyse terminée!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

# Affichage des résultats (identique à la version précédente)
if st.session_state.data_loaded and st.session_state.df_final is not None:
    df = st.session_state.df_final
    
    st.subheader("📊 Données extraites et scores")
    display_cols = ['N°', 'Cheval', 'Driver', 'Entraîneur', 'Gains', 'Record_secondes',
                    'Reussite_driver', 'Reussite_entraineur', 'score_normalise', 'rang']
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols])
    
    # Graphique des scores
    st.subheader("📈 Scores des chevaux")
    fig, ax = plt.subplots(figsize=(10, 6))
    chevaux = df['Cheval'].astype(str) + " (N°" + df['N°'].astype(str) + ")"
    ax.barh(chevaux, df['score_normalise'])
    ax.set_xlabel("Score normalisé")
    ax.set_title("Classement par score")
    st.pyplot(fig)
    
    # Pronostics
    st.subheader("🏆 Pronostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 3 probable")
        top3 = generer_top_3(df)
        for i, cheval in enumerate(top3):
            st.write(f"{i+1}. N°{cheval['N°']} - {cheval['Cheval']} (score: {cheval['score_normalise']:.3f})")
    
    with col2:
        st.markdown("#### Bases solides")
        bases = generer_bases(df, 2)
        for cheval in bases:
            st.write(f"🔹 N°{cheval['N°']} - {cheval['Cheval']}")
    
    st.markdown("#### Outsiders intéressants")
    outsiders = generer_outsiders(df, 5)
    for cheval in outsiders:
        st.write(f"👀 N°{cheval['N°']} - {cheval['Cheval']} (score: {cheval['score_normalise']:.3f})")
    
    # Combinaisons Trio
    st.markdown("#### 🎲 10 combinaisons pour le Trio")
    trio_comb = generer_combinaisons_trio(df, 10)
    for i, comb in enumerate(trio_comb):
        st.write(f"{i+1}. {', '.join(comb['chevaux'])} (N°{', '.join(map(str, comb['combinaison']))})")
    
    # Combinaisons Quinté
    st.markdown("#### 🎲 10 combinaisons pour le Quinté+")
    quinte_comb = generer_combinaisons_quinte(df, 10)
    for i, comb in enumerate(quinte_comb):
        st.write(f"{i+1}. {', '.join(comb['chevaux'])} (N°{', '.join(map(str, comb['combinaison']))})")
    
    # Téléchargement CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données en CSV",
        data=csv,
        file_name='pronostics_hippiques_ia.csv',
        mime='text/csv'
    )

else:
    if uploaded_files:
        st.info("Cliquez sur 'Analyser avec IA' pour lancer l'extraction intelligente.")
    else:
        st.info("Veuillez télécharger des images pour commencer.")

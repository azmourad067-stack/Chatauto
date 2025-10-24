# -*- coding: utf-8 -*-
"""
Streamlit app — Analyseur Hippique IA (Auto Deep Learning + ML hybride)
Fichier prêt à pousser sur GitHub et déployer sur Streamlit.
- Auto-entraînement incrémental : lorsqu'une nouvelle course est chargée (CSV ou URL),
  les données sont fusionnées dans data/historique.csv et le modèle DL est ré-entraîné automatiquement.
- Fusion ML (RandomForest/GBM) + DL (Keras) pour un score final hybride.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Deep learning
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# SKLearn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(
    page_title="🏇 Analyseur Hippique IA (Auto-DL)",
    page_icon="🏇",
    layout="wide"
)

# --------------------- Directories ---------------------
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

MODEL_PATH = os.path.join('models', 'dl_model.keras')
SCALER_PATH = os.path.join('models', 'scaler.joblib')
HIST_PATH = os.path.join('data', 'historique.csv')
LOG_PATH = os.path.join('logs', 'training_log.csv')

# --------------------- Styling ---------------------
st.markdown("""
<style>
    .main-header { font-size: 2.6rem; color: #1e3a8a; text-align: center; margin-bottom: 1rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0.7rem; border-radius: 10px; color: white; text-align: center; margin: 0.4rem 0; }
    .prediction-box { border-left: 4px solid #f59e0b; padding-left: 1rem; background-color: #fffbeb; margin: 0.6rem 0; }
</style>
""", unsafe_allow_html=True)

CONFIGS = {
    "PLAT": {"description": "🏃 Course de galop - Handicap poids + avantage corde intérieure", "optimal_draws": [1,2,3,4]},
    "ATTELE_AUTOSTART": {"description": "🚗 Trot attelé autostart - Numéros 4-6 optimaux", "optimal_draws": [4,5,6]},
    "ATTELE_VOLTE": {"description": "🔄 Trot attelé volté - Numéro sans importance", "optimal_draws": []}
}

# --------------------- Utilities ---------------------
def safe_convert(value, convert_func, default=0):
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return convert_func(cleaned)
    except:
        return default

def extract_weight(poids_str):
    if pd.isna(poids_str):
        return 60.0
    match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    return float(match.group(1).replace(',', '.')) if match else 60.0

# --------------------- Scraper ---------------------
@st.cache_data(ttl=300)
def scrape_race_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        if not table:
            return None, 'Aucun tableau trouvé'
        rows = table.find_all('tr')[1:]
        horses_data = []
        for row in rows:
            cols = row.find_all(['td','th'])
            if len(cols) >= 4:
                horses_data.append({
                    'Numéro de corde': cols[0].get_text(strip=True),
                    'Nom': cols[1].get_text(strip=True),
                    'Musique': cols[2].get_text(strip=True) if len(cols) > 2 else '',
                    'Âge/Sexe': cols[3].get_text(strip=True) if len(cols) > 3 else '',
                    'Poids': cols[-2].get_text(strip=True) if len(cols) > 3 else '60',
                    'Cote': cols[-1].get_text(strip=True)
                })
        if not horses_data:
            return None, 'Aucune donnée extraite'
        return pd.DataFrame(horses_data), 'Succès'
    except Exception as e:
        return None, f'Erreur: {e}'

# --------------------- Feature engineering ---------------------
def music_to_features(music_str):
    s = str(music_str)
    digits = [int(ch) for ch in re.findall(r'\d+', s)]
    if not digits:
        return 0, 0, 0.0
    recent_wins = sum(1 for d in digits if d == 1)
    recent_top3 = sum(1 for d in digits if d <= 3)
    weights = np.linspace(1, 0.3, num=len(digits))
    weighted_score = sum((4 - d) * w for d,w in zip(digits, weights)) / (len(digits) + 1e-6)
    return recent_wins, recent_top3, weighted_score


def prepare_data(df):
    df = df.copy()
    df['Cote'] = df['Cote'].astype(str)
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 999))
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    df['weight_kg'] = df['Poids'].apply(extract_weight)

    ages = []
    is_female = []
    r_wins=[]; r_top3=[]; r_weighted=[]
    for val in df.get('Âge/Sexe', ['']*len(df)):
        m = re.search(r'(\d+)', str(val))
        ages.append(float(m.group(1)) if m else 4.0)
        v = str(val).upper()
        is_female.append(1 if 'F' in v else 0)
    for mus in df.get('Musique', ['']*len(df)):
        a,b,c = music_to_features(mus)
        r_wins.append(a); r_top3.append(b); r_weighted.append(c)

    df['age'] = ages
    df['is_female'] = is_female
    df['recent_wins'] = r_wins
    df['recent_top3'] = r_top3
    df['recent_weighted'] = r_weighted

    df = df[df['odds_numeric'] > 0]
    df = df.reset_index(drop=True)
    return df

# --------------------- Model Manager (Hybrid ML + DL with Auto-Training) ---------------------
@st.cache_resource
class HorseRacingModel:
    def __init__(self):
        # ML models
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

        # DL model placeholders
        self.dl_model = None
        self.dl_input_dim = None

        # paths
        self.model_path = MODEL_PATH
        self.scaler_path = SCALER_PATH
        self.hist_path = HIST_PATH
        self.log_path = LOG_PATH

        # feature columns used for DL and ML
        self.feature_cols = ['odds_numeric','draw_numeric','weight_kg','age','is_female','recent_wins','recent_top3','recent_weighted']

        # try loading existing DL model + scaler
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.dl_model = models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                st.info('✅ Modèle DL et scaler chargés.')
            except Exception as e:
                st.warning(f'⚠️ Erreur chargement modèle existant: {e}')

    def build_dl(self, input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.25),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def update_historique(self, df_new):
        df_copy = df_new.copy()
        df_copy['source_ts'] = datetime.now().isoformat()
        if os.path.exists(self.hist_path):
            try:
                old = pd.read_csv(self.hist_path)
                combined = pd.concat([old, df_copy], ignore_index=True)
            except Exception:
                combined = df_copy
        else:
            combined = df_copy
        combined.to_csv(self.hist_path, index=False)
        st.info(f'🗂️ Historique mis à jour ({len(combined)} lignes).')

    def prepare_Xy_from_historique(self, use_pseudo_target=True):
        if not os.path.exists(self.hist_path):
            return None, None
        df = pd.read_csv(self.hist_path)
        df = prepare_data(df)
        X = df[self.feature_cols].fillna(0)
        if 'placement' in df.columns or 'rank' in df.columns:
            if 'placement' in df.columns:
                y = 1.0/(df['placement'].astype(float)+0.1)
            else:
                y = 1.0/(df['rank'].astype(float)+0.1)
            return X.values, y.values
        if use_pseudo_target:
            y = 0.7*(1.0/(df['odds_numeric']+0.1)) + 0.3*(df['recent_weighted'] / (df['recent_weighted'].max()+1e-6))
            y = y + np.random.normal(0, 0.02, size=len(y))
            return X.values, y.values
        return X.values, None

    def auto_train_dl(self, X_new, y_new=None, epochs=8, batch_size=8, val_split=0.15):
        try:
            X_hist, y_hist = self.prepare_Xy_from_historique(use_pseudo_target=True)
            X_train, y_train = None, None
            if X_hist is not None and y_hist is not None and len(X_hist) >= 4:
                X_train, y_train = X_hist, y_hist
            else:
                if isinstance(X_new, np.ndarray):
                    X_train = X_new
                else:
                    X_train = X_new.values if hasattr(X_new, 'values') else np.array(X_new)
                if y_new is None:
                    y_train = 0.7*(1.0/(X_new[:,0]+0.1)) + 0.3*(X_new[:,7] / (np.max(X_new[:,7]) + 1e-6))
                else:
                    y_train = y_new

            Xs = self.scaler.fit_transform(X_train)

            if self.dl_model is None:
                self.dl_input_dim = Xs.shape[1]
                self.dl_model = self.build_dl(self.dl_input_dim)

            es = callbacks.EarlyStopping(patience=6, restore_best_weights=True)
            hist = self.dl_model.fit(Xs, y_train, validation_split=val_split, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
            loss = float(hist.history['loss'][-1])

            try:
                self.dl_model.save(self.model_path, overwrite=True)
                joblib.dump(self.scaler, self.scaler_path)
            except Exception as e:
                st.warning(f"⚠️ Erreur sauvegarde modèle: {e}")

            with open(self.log_path, 'a') as f:
                f.write(f"{datetime.now().isoformat()},{len(X_train)},{loss:.6f},{self.model_path}\n")

            st.success(f'✅ Auto-entrainement DL terminé (loss={loss:.5f})')
            return hist.history
        except Exception as e:
            st.warning(f'⚠️ Erreur auto_train_dl: {e}')
            return None

    def predict_dl(self, X):
        if self.dl_model is None:
            return np.zeros(len(X))
        Xs = self.scaler.transform(X)
        preds = self.dl_model.predict(Xs).flatten()
        return preds

    def train_ml_models_on_historical(self):
        X_hist, y_hist = self.prepare_Xy_from_historique(use_pseudo_target=True)
        if X_hist is None or y_hist is None or len(X_hist) < 4:
            return None
        try:
            self.rf.fit(X_hist, y_hist)
            self.gb.fit(X_hist, y_hist)
            return True
        except Exception as e:
            st.warning(f'⚠️ Erreur entraînement ML full: {e}')
            return None

model_manager = HorseRacingModel()

# --------------------- Combinations generator ---------------------
from itertools import combinations, permutations

def generate_e_trio(df, n_combinations=35):
    df = df.copy().reset_index(drop=True)
    df['fav_score'] = 1 / (df['odds_numeric'] + 0.1)
    df = df.sort_values('fav_score', ascending=False).reset_index()
    favorites = df.head(max(3, int(len(df)*0.3)))['Nom'].tolist()
    outsiders = df.tail(max(3, int(len(df)*0.3)))['Nom'].tolist()
    pool = list(dict.fromkeys(favorites + outsiders + df['Nom'].tolist()))

    combos = []
    for a in pool:
        for b in pool:
            for c in pool:
                if a!=b and b!=c and a!=c:
                    combos.append(tuple([a,b,c]))
    combos_unique = []
    seen = set()
    for comb in combos:
        key = tuple(sorted(comb))
        if key not in seen:
            seen.add(key)
            combos_unique.append(comb)
    name_to_score = dict(zip(df['Nom'], df['fav_score']))
    def combo_score(comb):
        return sum(name_to_score.get(n, 0) for n in comb)
    combos_unique = sorted(combos_unique, key=combo_score, reverse=True)
    return combos_unique[:min(n_combinations, len(combos_unique))]

# --------------------- Visualizations ---------------------
def create_visualization(df_ranked, feature_importance=None):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('🏆 Scores par Position','📊 Distribution Cotes','⚖️ Poids vs Score','🧠 Features'))
    score_col = 'score_final' if 'score_final' in df_ranked.columns else 'ml_score'
    if score_col in df_ranked.columns:
        fig.add_trace(go.Scatter(x=df_ranked['rang'], y=df_ranked[score_col], mode='markers+lines', text=df_ranked['Nom'], name='Score'), row=1,col=1)
    fig.add_trace(go.Histogram(x=df_ranked['odds_numeric'], nbinsx=8, name='Cotes'), row=1, col=2)
    if score_col in df_ranked.columns:
        fig.add_trace(go.Scatter(x=df_ranked['weight_kg'], y=df_ranked[score_col], mode='markers', text=df_ranked['Nom'], name='Poids vs Score'), row=2, col=1)
    fig.update_layout(height=650, showlegend=True, title_text='📊 Analyse Complète', title_x=0.5)
    return fig

# --------------------- Sample data generator ---------------------
def generate_sample_data(data_type="plat"):
    if data_type == "plat":
        return pd.DataFrame({
            'Nom': ['Thunder Bolt', 'Lightning Star', 'Storm King', 'Rain Dance', 'Wind Walker'],
            'Numéro de corde': ['1', '2', '3', '4', '5'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1'],
            'Poids': ['56.5', '57.0', '58.5', '59.0', '57.5'],
            'Musique': ['1a2a3a', '2a1a4a', '3a3a1a', '1a4a2a', '4a2a5a'],
            'Âge/Sexe': ['4H', '5M', '3F', '6H', '4M']
        })
    else:
        return pd.DataFrame({
            'Nom': ['Ace Impact', 'Torquator Tasso', 'Adayar', 'Tarnawa', 'Chrono Genesis'],
            'Numéro de corde': ['1', '2', '3', '4', '5'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1'],
            'Poids': ['59.5', '59.5', '59.5', '58.5', '58.5'],
            'Musique': ['1a1a2a', '1a3a1a', '2a1a4a', '1a2a1a', '3a1a2a'],
            'Âge/Sexe': ['4H', '5H', '4H', '5F', '5F']
        })

# --------------------- App UI ---------------------

def main():
    st.markdown('<h1 class="main-header">🏇 Analyseur Hippique IA (Auto-DL)</h1>', unsafe_allow_html=True)
    st.markdown('*Application entièrement autonome : fusion ML + DL, auto-entraînement incrémental, génération e-trio.*')

    with st.sidebar:
        st.header('⚙️ Configuration')
        race_type = st.selectbox('🏁 Type de course', ['AUTO','PLAT','ATTELE_AUTOSTART','ATTELE_VOLTE'])
        enable_dl = st.checkbox('✅ Activer Auto Deep Learning', value=True)
        dl_epochs = st.number_input('🏋️‍♂️ Epochs (auto-train each update)', min_value=2, max_value=500, value=8, step=1)
        dl_batch = st.number_input('🧮 Batch size', min_value=2, max_value=128, value=8, step=1)
        ml_confidence = st.slider('🎯 Poids ML dans score final', 0.0, 1.0, 0.6, 0.05)
        num_combos = st.number_input('🔢 Nb combinaisons e-trio', min_value=5, max_value=200, value=35, step=1)
        st.info("ℹ️ L'application entraînera automatiquement le modèle DL à chaque import de nouvelle course (CSV/URL).")

    tab1, tab2, tab3 = st.tabs(['🌐 URL Analysis','📁 Upload CSV','🧪 Test Data'])
    df_final = None

    with tab1:
        st.subheader('🔍 Analyse d\'URL de Course')
        col1, col2 = st.columns([3,1])
        with col1:
            url = st.text_input('🌐 URL de la course:', placeholder='https://...')
        with col2:
            analyze_button = st.button('🔍 Analyser')
        if analyze_button and url:
            with st.spinner('🔄 Extraction...'):
                df, msg = scrape_race_data(url)
                if df is not None:
                    st.success(f'✅ {len(df)} chevaux extraits')
                    st.dataframe(df.head())
                    df_final = df
                else:
                    st.error(f'❌ {msg}')

    with tab2:
        st.subheader('📤 Upload CSV (historique possible)')
        uploaded_file = st.file_uploader('Fichier CSV', type='csv')
        if uploaded_file:
            try:
                df_final = pd.read_csv(uploaded_file)
                st.success(f'✅ {len(df_final)} chevaux chargés')
                st.dataframe(df_final.head())
            except Exception as e:
                st.error(f'❌ Erreur: {e}')

    with tab3:
        st.subheader('🧪 Données de Test')
        c1,c2,c3 = st.columns(3)
        with c1:
            if st.button('🏃 Test Plat'):
                df_final = generate_sample_data('plat')
                st.success('✅ Données PLAT chargées')
        with c2:
            if st.button('🚗 Test Attelé'):
                df_final = generate_sample_data('attele')
                st.success('✅ Données ATTELÉ chargées')
        with c3:
            if st.button('⭐ Test Premium'):
                df_final = generate_sample_data('premium')
                st.success('✅ Données PREMIUM chargées')
        if df_final is not None:
            st.dataframe(df_final)

    # ----- Core processing -----
    if df_final is not None and len(df_final)>0:
        st.markdown('---')
        st.header('🎯 Analyse et Résultats (Auto-DL + ML)')

        df_prep = prepare_data(df_final)
        if len(df_prep)==0:
            st.error('❌ Aucune donnée valide')
            return

        if race_type=='AUTO':
            weight_std = df_prep['weight_kg'].std()
            weight_mean = df_prep['weight_kg'].mean()
            if weight_std>2.5:
                detected='PLAT'
            elif weight_mean>65 and weight_std<1.5:
                detected='ATTELE_AUTOSTART'
            else:
                detected='PLAT'
            st.info(f'🤖 Type détecté: {detected}')
        else:
            detected = race_type
            st.info(f'📋 {CONFIGS[detected]["description"]}')

        # update historique automatically (fusion)
        try:
            model_manager.update_historique(df_final)
        except Exception as e:
            st.warning(f"⚠️ Historique non mis à jour: {e}")

        # prepare features for current race
        X_curr = df_prep[model_manager.feature_cols].fillna(0)

        # If DL enabled -> auto-train on full historique (preferable)
        dl_preds = np.zeros(len(X_curr))
        if enable_dl:
            with st.spinner('🏋️‍♂️ Auto-entrainement DL en cours...'):
                hist = model_manager.auto_train_dl(X_curr, y_new=None, epochs=int(dl_epochs), batch_size=int(dl_batch))
                if hist is not None:
                    dl_preds = model_manager.predict_dl(X_curr.values)
        else:
            st.info('ℹ️ DL désactivé')

        # classical ML heuristic score
        trad = 1.0/(df_prep['odds_numeric']+0.1)
        if trad.max()!=trad.min():
            trad = (trad - trad.min())/(trad.max()-trad.min())

        # also try ML models (train on historique if available)
        ml_score = np.zeros(len(X_curr))
        trained_ml = model_manager.train_ml_models_on_historical()
        if trained_ml:
            try:
                Xc = X_curr.values
                preds_rf = model_manager.rf.predict(Xc)
                preds_gb = model_manager.gb.predict(Xc)
                preds = 0.5*preds_rf + 0.5*preds_gb
                if preds.max()!=preds.min():
                    preds = (preds - preds.min())/(preds.max()-preds.min())
                ml_score = preds
            except Exception as e:
                st.warning(f'⚠️ Erreur prédiction ML: {e}')

        # normalize DL preds
        if dl_preds.max() != dl_preds.min():
            dl_norm = (dl_preds - dl_preds.min())/(dl_preds.max()-dl_preds.min())
        else:
            dl_norm = np.zeros_like(dl_preds)

        # final blended score
        final_score = (1-ml_confidence)*trad + ml_confidence*(0.5*ml_score + 0.5*dl_norm)

        df_prep['ml_score'] = ml_score
        df_prep['dl_score'] = dl_norm
        df_prep['score_final'] = final_score
        df_ranked = df_prep.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_ranked['rang'] = range(1, len(df_ranked)+1)

        # display
        c1,c2 = st.columns([2,1])
        with c1:
            st.subheader('🏆 Classement Final')
            display_cols = ['rang','Nom','Cote','Numéro de corde']
            if 'Poids' in df_ranked.columns:
                display_cols.append('Poids')
            display_cols.append('score_final')
            display_df = df_ranked[display_cols].copy()
            display_df['Score'] = display_df['score_final'].round(3)
            display_df = display_df.drop('score_final', axis=1)
            st.dataframe(display_df, use_container_width=True)

        with c2:
            st.subheader('📊 Métriques')
            st.markdown(f'<div class="metric-card">🧠 DL activé<br><strong>{"Oui" if enable_dl else "Non"}</strong></div>', unsafe_allow_html=True)
            favoris = len(df_ranked[df_ranked['odds_numeric']<5])
            outsiders = len(df_ranked[df_ranked['odds_numeric']>15])
            st.markdown(f'<div class="metric-card">⭐ Favoris<br><strong>{favoris}</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card">🎲 Outsiders<br><strong>{outsiders}</strong></div>', unsafe_allow_html=True)
            st.subheader('🥇 Top 3')
            for i in range(min(3, len(df_ranked))):
                horse = df_ranked.iloc[i]
                st.markdown(f"<div class=\"prediction-box\"><strong>{i+1}. {horse['Nom']}</strong><br>Cote: {horse['Cote']} | Score: {horse['score_final']:.3f}</div>", unsafe_allow_html=True)

        # visuals
        st.subheader('📊 Visualisations')
        fig = create_visualization(df_ranked)
        st.plotly_chart(fig, use_container_width=True)

        # e-trio
        st.subheader('🎲 Générateur e-trio')
        combos = generate_e_trio(df_ranked, n_combinations=int(num_combos))
        for idx, c in enumerate(combos):
            st.markdown(f'{idx+1}. {c[0]} — {c[1]} — {c[2]}')

        # export
        st.subheader('💾 Export')
        colx, coly = st.columns(2)
        with colx:
            csv_data = df_ranked.to_csv(index=False)
            st.download_button('📄 CSV', csv_data, f'pronostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        with coly:
            json_data = df_ranked.to_json(orient='records', indent=2)
            st.download_button('📋 JSON', json_data, f'pronostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    st.markdown('---')
    st.markdown('**Notes**: Le modèle DL s\'entraîne automatiquement à chaque import. Pour un déploiement stable, considérer l\'usage d\'artifacts externes pour stocker les modèles (S3, etc.).')

if __name__ == '__main__':
    main()

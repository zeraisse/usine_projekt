"""Streamlit dashboard principal (placeholder)."""
import streamlit as st

st.title("Dashboard Usine KPI - Prototype")
st.write("Remplacez ce fichier par votre dashboard Streamlit.")
import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('../3_ai_engine'))
try:
    from worker import TRMInferenceWorker
except ImportError:
    st.error("Impossible de trouver le module IA. Vérifiez l'arborescence.")

st.set_page_config(page_title="Investigateur de Logs", layout="wide")

@st.cache_resource
def load_ai():
    return TRMInferenceWorker()

try:
    ia_worker = load_ai()
    ia_ready = True
except Exception as e:
    st.warning(f"L'IA n'est pas prête ou en cours d'entraînement : {e}")
    ia_ready = False

st.title("Dashboard d'Investigation des Logs")
st.markdown("---")

with st.sidebar:
    st.header("Ingestion des données")
    uploaded_file = st.file_uploader("Déposez le fichier de production (.log)", type=["log", "txt"])

if uploaded_file is not None:
    log_content = uploaded_file.getvalue().decode("utf-8")
    lignes_logs = log_content.split('\n')
    
    st.success(f"Fichier chargé avec succès : {len(lignes_logs)} lignes détectées.")
    
    st.subheader("Métriques Rapides")
    col1, col2, col3 = st.columns(3)
    
    nb_erreurs = sum(1 for ligne in lignes_logs if "ERROR" in ligne)
    nb_warnings = sum(1 for ligne in lignes_logs if "WARN" in ligne)
    
    col1.metric("Total Lignes", len(lignes_logs))
    col2.metric("Erreurs Critiques", nb_erreurs, delta_color="inverse")
    col3.metric("Avertissements", nb_warnings, delta_color="inverse")
    
    st.markdown("---")
    
    st.subheader("Analyse Deep Learning (TRM)")
    if ia_ready and st.button("Lancer l'investigation IA", type="primary"):
        with st.spinner("Analyse en cours..."):
            index_coupable = ia_worker.analyze_log_block(log_content)
            
            st.error(f"Alerte : Cause racine détectée à la ligne {index_coupable + 1}")
            st.code(lignes_logs[index_coupable], language="log")
            
    st.subheader("Trace Complète")
    df_logs = pd.DataFrame(lignes_logs, columns=["Message"])
    st.dataframe(df_logs, use_container_width=True, height=400)

else:
    st.info("Veuillez uploader un fichier de log depuis le menu latéral pour commencer l'analyse.")
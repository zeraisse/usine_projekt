import streamlit as st
import pandas as pd
import sys
import os
from collections import Counter

dossier_actuel = os.path.dirname(os.path.abspath(__file__))
dossier_ai = os.path.abspath(os.path.join(dossier_actuel, '../3_ai_engine'))

if dossier_ai not in sys.path:
    sys.path.append(dossier_ai)

try:
    from worker import TRMInferenceWorker
except ImportError as e:
    st.error(f"Erreur critique d'importation du module IA : {e}")
    st.stop() 

st.set_page_config(page_title="Dashboard KPI Usine", layout="wide")

@st.cache_resource
def load_ai():
    return TRMInferenceWorker()

try:
    ia_worker = load_ai()
    ia_ready = True
except Exception as e:
    st.warning(f"L'IA n'est pas prête : {e}")
    ia_ready = False

st.title("Tableau de Bord : Qualité des Releases (KPI)")
st.markdown("---")

with st.sidebar:
    st.header("Ingestion des données")
    uploaded_files = st.file_uploader(
        "Déposez les fichiers de logs (ex: jours ou releases successives)", 
        type=["log", "txt"], 
        accept_multiple_files=True
    )

if uploaded_files:
    st.info(f"Traitement en lot de {len(uploaded_files)} fichier(s) en cours...")
    
    donnees_kpi = []
    toutes_causes_racines = [] # Liste globale pour stocker les noms des pannes
    
    for fichier in uploaded_files:
        nom_fichier = fichier.name
        log_content = fichier.getvalue().decode("utf-8")
        lignes_logs = log_content.split('\n')
        
        # Métriques de base
        total_lignes = len(lignes_logs)
        erreurs_brutes = sum(1 for ligne in lignes_logs if "ERROR" in ligne)
        
        # Analyse IA
        cascades_ia = 0
        if ia_ready:
            index_erreurs = [i for i, ligne in enumerate(lignes_logs) if "ERROR" in ligne]
            dernier_index_analyse = -1
            
            for i in index_erreurs:
                if i <= dernier_index_analyse:
                    continue
                    
                start_idx = max(0, i - 5)
                end_idx = min(len(lignes_logs), start_idx + 15)
                texte_bloc = "\n".join(lignes_logs[start_idx:end_idx])
                
                index_predit = ia_worker.analyze_log_block(texte_bloc)
                
                if index_predit != 15:
                    cascades_ia += 1
                    
                    ligne_cause = lignes_logs[start_idx + index_predit]
                    parts = ligne_cause.split('|')
                    if len(parts) >= 4:
                        nom_erreur = parts[3].strip()
                        toutes_causes_racines.append(nom_erreur)
                
                dernier_index_analyse = end_idx
                
        # Stockage des stats pour ce fichier
        donnees_kpi.append({
            "Release / Fichier": nom_fichier,
            "Volume total (Lignes)": total_lignes,
            "Erreurs brutes (Bruit)": erreurs_brutes,
            "Cascades critiques (IA)": cascades_ia
        })

    # --- AFFICHAGE DES KPIs ---
    df_kpi = pd.DataFrame(donnees_kpi)
    
    st.subheader("Synthèse Globale")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Fichiers Analysés", len(uploaded_files))
    with col2:
        total_cascades = df_kpi["Cascades critiques (IA)"].sum()
        st.metric("Pannes Critiques Détectées (Total)", total_cascades)
        
    st.markdown("---")

    # VALEUR MÉTIER & DIAGNOSTIC ---
    st.subheader("Diagnostic IA & Réduction du Bruit")
    
    col3, col4 = st.columns([1, 2])
    
    with col3:
        total_erreurs_brutes = df_kpi["Erreurs brutes (Bruit)"].sum()
        bruit_supprime = total_erreurs_brutes - total_cascades
        taux_reduction_bruit = (bruit_supprime / total_erreurs_brutes * 100) if total_erreurs_brutes > 0 else 0
        
        st.metric("Bruit Filtré par l'IA", f"{taux_reduction_bruit:.1f} %", delta="Fausses alertes ignorées")

    with col4:
        st.markdown("**Palmarès des Causes Racines Identifiées :**")
        if toutes_causes_racines:
            # On compte les occurrences de chaque type de panne
            compte_causes = Counter(toutes_causes_racines)
            df_causes = pd.DataFrame(compte_causes.items(), columns=["Type de Panne Critique (Root Cause)", "Occurrences"])
            df_causes = df_causes.sort_values(by="Occurrences", ascending=False).reset_index(drop=True)
            
            # Affichage propre d'un petit tableau
            st.dataframe(df_causes, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune panne critique détectée dans ces logs.")

    st.markdown("---")
    
    # --- GRAPHIQUES DE TENDANCE ---
    st.subheader("Évolution de la stabilité par Release")
    
    df_chart = df_kpi.set_index("Release / Fichier")
    st.line_chart(df_chart[["Erreurs brutes (Bruit)", "Cascades critiques (IA)"]])
    
    st.markdown("---")
    
    # --- TABLEAU DE DÉTAIL ---
    st.subheader("Détail des Métriques")
    st.dataframe(df_kpi, use_container_width=True)

else:
    st.info("Veuillez uploader un ou plusieurs fichiers de logs depuis le menu latéral pour générer la synthèse KPI.")
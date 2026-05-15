import torch
import pickle
import os
import sys

# On s'assure que le script trouve model_utils et train_trm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_utils import LogTokenizer
from train_trm import TinyRecursiveModel

class TRMInferenceWorker:
    def __init__(self, model_path="trm_model.pt", tokenizer_path="tokenizer.pkl"):
        """Initialise le worker en chargeant le cerveau gelé en mémoire."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Charger le Dictionnaire
        try:
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
        except FileNotFoundError:
            raise Exception("Erreur : tokenizer.pkl introuvable. As-tu lancé l'entraînement ?")

        # 2. Charger le Modèle (Mêmes dimensions que l'entraînement)
        self.model = TinyRecursiveModel(
            vocab_size=5000, 
            embed_dim=64, 
            hidden_dim=128, 
            max_lines_per_block=10
        ).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            raise Exception("Erreur : trm_model.pt introuvable. As-tu lancé l'entraînement ?")
            
        self.model.eval() 
        print("Worker IA Initialisé et prêt à l'emploi.")

    def analyze_log_block(self, log_text):
        """Reçoit un bloc de logs bruts et renvoie l'index de la cause racine."""
        with torch.no_grad():
            input_tensor = self.tokenizer.encode(log_text).unsqueeze(0).to(self.device)
            
            output = self.model(input_tensor)
            
            predicted_index = torch.argmax(output, dim=1).item()
            
        return predicted_index

# --- TEST EN ISOLÉ ---
if __name__ == "__main__":
    # Ce bloc ne s'exécute que si on lance le worker.py
    worker = TRMInferenceWorker()
    test_log = "11-05-2026-08:28 | LOG | INFO | Cache Refresh\n11-05-2026-08:35 | LOG | ERROR | Dependency Injection Failure\n11-05-2026-08:36 | SYS | ERROR | Pipeline Crash"
    
    resultat = worker.analyze_log_block(test_log)
    print(f"Analyse terminée. La cause racine est estimée à la ligne index : {resultat}")
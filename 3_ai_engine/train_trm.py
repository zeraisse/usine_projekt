import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import pickle
from model_utils import LogTokenizer, LogDataset

# 1. DÉFINITION DE L'ARCHITECTURE
class TinyRecursiveModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_lines_per_block, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.core_logic = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, max_lines_per_block)

    def forward(self, x):
        embedded = self.embedding(x)
        hidden_state = None
        for i in range(self.num_iterations):
            output, hidden_state = self.core_logic(embedded, hidden_state)
        
        final_thought = output[:, -1, :] 
        prediction = self.classifier(final_thought)
        return prediction

# 2. LOGIQUE D'ENTRAÎNEMENT
def train():
    # --- Configuration ---
    VOCAB_SIZE = 5000
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    MAX_LINES = 10   
    MAX_LEN = 128    
    EPOCHS = 10     
    BATCH_SIZE = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    print("Chargement des données...")
    tokenizer = LogTokenizer(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    
    # Charge le CSV pour que le tokenizer apprenne les mots
    raw_df = pd.read_csv("dummy_data.csv")
    tokenizer.fit(raw_df['log_block'].tolist())
    
    # Création du Dataset et du DataLoader
    dataset = LogDataset("dummy_data.csv", tokenizer)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Initialisation du Modèle ---
    model = TinyRecursiveModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, MAX_LINES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() # Idéal pour la classification d'index

    # --- Boucle d'Apprentissage ---
    print("Démarrage de l'entraînement...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Époque [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f}")

    print("Sauvegarde du modèle et du tokenizer...")
    torch.save(model.state_dict(), "trm_model.pt")
    
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
        
    print("Fichiers 'trm_model.pt' et 'tokenizer.pkl' créés avec succès.")

if __name__ == "__main__":
    train()
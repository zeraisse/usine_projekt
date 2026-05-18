import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split 
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
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

# --- Calcul de l'Accuracy ---
def calculate_accuracy(outputs, targets):
    """Calcule le pourcentage d'exactitude (Accuracy) pour un batch."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return (correct / total) * 100

# 2. LOGIQUE D'ENTRAÎNEMENT AMÉLIORÉE
def train():
    # --- Configuration ---
    VOCAB_SIZE = 5000
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    MAX_LINES = 16   
    MAX_LEN = 128    
    EPOCHS = 30     
    BATCH_SIZE = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    # Suivi de l'historique ---
    history = {
        'loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    start_train_time = time.time() # Chronomètre total

    print("Chargement des données...")
    tokenizer = LogTokenizer(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    raw_df = pd.read_csv("training_data.csv")
    tokenizer.fit(raw_df['log_block'].tolist())
    dataset = LogDataset("training_data.csv", tokenizer)

    # Split Train/Validation ---
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Initialisation du Modèle ---
    model = TinyRecursiveModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, MAX_LINES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # --- Boucle d'Apprentissage ---
    print("Démarrage de l'entraînement...")
    for epoch in range(EPOCHS):
        start_epoch_time = time.time() # Chronomètre de l'époque
        
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        total_loss = 0
        total_train_correct = 0
        total_train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            batch_y[batch_y == -1] = 15
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calcul de l'accuracy d'entraînement
            accuracy = calculate_accuracy(outputs, batch_y)
            total_train_correct += (accuracy / 100) * batch_y.size(0)
            total_train_total += batch_y.size(0)
            
        epoch_train_loss = total_loss / len(train_loader)
        epoch_train_acc = (total_train_correct / total_train_total) * 100

        # --- PHASE DE VALIDATION ---
        model.eval()
        total_val_correct = 0
        total_val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_y[batch_y == -1] = 15
                outputs = model(batch_x)
                
                # Calcul de l'accuracy de validation
                accuracy = calculate_accuracy(outputs, batch_y)
                total_val_correct += (accuracy / 100) * batch_y.size(0)
                total_val_total += batch_y.size(0)
                
        epoch_val_acc = (total_val_correct / total_val_total) * 100
        epoch_end_time = time.time() # Fin du chrono de l'époque
        epoch_duration = epoch_end_time - start_epoch_time

        # Enregistrement de l'historique ---
        history['loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['epoch_times'].append(epoch_duration)
            
        print(f"Époque [{epoch+1}/{EPOCHS}] - Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.1f}% - Val Acc: {epoch_val_acc:.1f}% - Temps: {epoch_duration:.2f}s")

    end_train_time = time.time()
    print(f"Entraînement terminé en {(end_train_time - start_train_time):.2f} secondes.")

    # --- SAUVEGARDE DU MODÈLE ET TOKENIZER ---
    print("Sauvegarde du modèle et du tokenizer...")
    torch.save(model.state_dict(), "trm_model.pt")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Fichiers 'trm_model.pt' et 'tokenizer.pkl' créés avec succès.")

    # --- Génération des PNG ---
    print("Génération des graphiques d'entraînement (PNG)...")
    
    # Graphique 1 : Loss et Exactitude (Entraînement vs Validation)
    fig1, ax1_loss = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, EPOCHS + 1)

    # Courbe de la Loss
    ax1_loss.set_xlabel('Époque')
    ax1_loss.set_ylabel('Loss (Entraînement)', color='tab:blue')
    ax1_loss.plot(epochs_range, history['loss'], color='tab:blue', label='Train Loss')
    ax1_loss.tick_params(axis='y', labelcolor='tab:blue')

    # Créer un deuxième axe y pour l'exactitude
    ax1_acc = ax1_loss.twinx()  
    ax1_acc.set_ylabel('Exactitude (%)', color='tab:green')  
    ax1_acc.plot(epochs_range, history['train_acc'], color='tab:green', linestyle='dashed', label='Train Acc')
    ax1_acc.plot(epochs_range, history['val_acc'], color='tab:red', linestyle='dashed', label='Val Acc')
    ax1_acc.tick_params(axis='y', labelcolor='tab:green')
    ax1_acc.set_ylim(0, 105)

    fig1.suptitle('Historique de l\'Entraînement : Loss et Exactitude')
    # Combinaison des légendes
    lines, labels = ax1_loss.get_legend_handles_labels()
    lines2, labels2 = ax1_acc.get_legend_handles_labels()
    ax1_acc.legend(lines + lines2, labels + labels2, loc='upper left')

    fig1.tight_layout()
    fig1.savefig("training_history.png") # SAUVEGARDE DU PNG 1
    plt.close(fig1) # Fermer pour libérer la mémoire


    # Graphique 2 : Temps par époque
    fig2, ax2_time = plt.subplots(figsize=(10, 6))
    ax2_time.set_xlabel('Époque')
    ax2_time.set_ylabel('Temps (secondes)')
    ax2_time.plot(epochs_range, history['epoch_times'], color='tab:purple', marker='o')
    ax2_time.set_title(f'Temps d\'Entraînement par Époque (Total: {end_train_time - start_train_time:.1f}s)')
    ax2_time.grid(axis='y', linestyle='--')

    fig2.tight_layout()
    fig2.savefig("training_times.png") # SAUVEGARDE DU PNG 2
    plt.close(fig2)

    print("Graphiques 'training_history.png' et 'training_times.png' sauvegardés.")

if __name__ == "__main__":
    train()
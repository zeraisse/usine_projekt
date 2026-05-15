import re
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader


class LogTokenizer:
    def __init__(self, vocab_size=5000, max_len=128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
    def _clean_text(self, text):
        parts = text.split('|')
        clean_parts = [p.strip() for p in parts[1:]] 
        text = " ".join(clean_parts).lower()
        
        text = re.sub(r'\S+@\S+\.\S+', '<EMAIL>', text)
        text = re.sub(r'data_\d+\.ext', '<FILE>', text)
        
        text = re.sub(r'[^\w\s]', ' ', text)
        return text

    def fit(self, texts):
        all_words = []
        for t in texts:
            all_words.extend(self._clean_text(t).split())
        
        counts = Counter(all_words)
        most_common = counts.most_common(self.vocab_size - 2)
        for i, (word, _) in enumerate(most_common):
            self.word2idx[word] = i + 2
            self.idx2word[i + 2] = word
            
    def encode(self, text):
        cleaned = self._clean_text(text)
        tokens = cleaned.split()
        encoded = [self.word2idx.get(t, 1) for t in tokens]
        
        # Padding / Truncating
        if len(encoded) < self.max_len:
            encoded += [0] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        return torch.tensor(encoded)

if __name__ == "__main__":
    example_log = "11-05-2026-08:35 | SYS | INFO | User Login | User 'j.doe@domain.com' authenticated via OAuth2 provider."
    tokenizer = LogTokenizer(max_len=64)
    tokenizer.fit([example_log])
    
    vector = tokenizer.encode(example_log)
    print(f"Log original : {example_log}")
    print(f"Vecteur encodé : {vector}")





class LogDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        # On charge le CSV
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. On récupère le texte du bloc de log
        text_block = str(self.df.iloc[idx]['log_block'])
        
        # 2. On récupère le label (l'index de la cause racine)
        label = int(self.df.iloc[idx]['root_cause_index'])
        
        # 3. On encode le texte en tenseur
        tensor_x = self.tokenizer.encode(text_block)
        
        # Le label -1 signifie "Pas d'erreur", on peut le gérer 
        # en décalant les index ou en le traitant comme une classe à part.
        # Pour l'instant, on reste simple :
        tensor_y = torch.tensor(label, dtype=torch.long)
        
        return tensor_x, tensor_y
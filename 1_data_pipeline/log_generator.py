"""Génère un faux log d'usine (placeholder).
Usage: python log_generator.py
"""
import csv
from datetime import datetime
import random

def generate_fake_logs(path, n=100):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","machine","status","value"])
        for i in range(n):
            ts = datetime.utcnow().isoformat()
            machine = f"M{random.randint(1,5)}"
            status = random.choice(["OK","WARN","ERR"])
            value = round(random.uniform(0,100),2)
            writer.writerow([ts, machine, status, value])

if __name__ == "__main__":
    generate_fake_logs("fake_logs.csv", n=200)

"""Parser et nettoyage des logs (placeholder).
Fonctions : read_csv -> nettoyage basique -> DataFrame
"""
import pandas as pd

def parse_logs(path):
    df = pd.read_csv(path)
    # placeholder: conversion timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

if __name__ == "__main__":
    print(parse_logs("fake_logs.csv").head())

import torch

print(f"Version de Torch : {torch.__version__}")
print(f"CUDA disponible  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU détecté      : {torch.cuda.get_device_name(0)}")
    print(f"Nombre de coeurs : {torch.cuda.device_count()}")
    # Test de transfert
    x = torch.rand(3, 3).to("cuda")
    print("Test réussi : Tenseur envoyé sur la RTX 5070 Ti !")
else:
    print("ERREUR : Toujours sur le CPU. Vérifie tes drivers NVIDIA.")
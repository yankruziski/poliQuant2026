import os

folder = 'data/processed'
print(f"Tamanhos dos arquivos em {folder} (MB):\n")
for fname in os.listdir(folder):
    path = os.path.join(folder, fname)
    if os.path.isfile(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"{fname:35s} {size:10.2f} MB")

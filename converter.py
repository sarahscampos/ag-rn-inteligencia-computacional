import pandas as pd

df = pd.read_excel("dados.xlsx")
df.to_csv("dados.csv", sep=";", index=False)
print(f"Convertido: {df.shape[0]} linhas x {df.shape[1]} colunas")

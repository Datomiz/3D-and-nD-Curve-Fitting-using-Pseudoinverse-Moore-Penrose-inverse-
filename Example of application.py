import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Ajuste_curva_3D_r import auto_ml_regressao_multivariavel


# --- SEU PREPARO DE DADOS ORIGINAL ---
df = pd.read_csv("Resultados_simu_19.csv").sample(5000)

colunas_parametros = [
    "Razão dia",
    "Razão noite",
    #"Medicoes Por Dia",
    "Consumo total",
    #"Vazamento FFT",
    # "Vazamento Bins",
    # "Vazamento Noturno",
    # "Vazamento Quantil",
    "Vazamento Media Total"
]

lista_param = []

for col in colunas_parametros:

    lista_param.append(df[col].tolist())

alvo = "Vazamento Real"

trocas_permitidas = [
        "x**-1",
        "x**0.5"
    ]


equacao_achada, metricas_finais = auto_ml_regressao_multivariavel(
    n=2,
    lista_parametros=lista_param,
    resultado = list(df[alvo]),
    lista_troca_permitidas=trocas_permitidas
)


print("================ RESULTADO DE AUTO-ML MULTIVARIÁVEL ================")
print(f"Métricas Alcançadas -> R²: {metricas_finais['R2']*100:.2f}% | MAPE: {metricas_finais['MAPE']:.2f}%")
print(f"\nEquação Matemática Otimizada:\nZ = {equacao_achada}")
print("====================================================================")
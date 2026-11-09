import numpy as np
#import cupy as np
import plotly.graph_objects as go
import itertools
import time
import datetime as dat
from typing import Literal

def ajuste_linear_3D(
        x:np.ndarray,
        y:np.ndarray,
        z:np.ndarray,
):

    # 2. Cálculo Matricial da Regressão (Corrigido o alvo para 'z')
    x0 = np.ones(len(x))
    X = np.column_stack((x0, x, y))

    # Encontra os pesos (w0 = intercepto, w1 = inclinação de x, w2 = inclinação de y)
    # Usando a equação normal ou a pseudoinversa
    w = np.linalg.pinv(X) @ z #mesma coisa do .dot
    lista_resultados = w.tolist()

    # 3. Criando a String da Equação (Seu método mantido)
    str_eq = f"{lista_resultados[0]} + xr * {lista_resultados[1]} + yr * {lista_resultados[2]}"
    #print("Equação gerada:", str_eq)

    # --- GERANDO A SUPERFÍCIE NO PLOTLY ---

    # 4. Criar uma malha (Grid) cobrindo todo o intervalo de X e Y do gráfico
    x_range = np.linspace(x.min(), x.max(), 20)
    y_range = np.linspace(y.min(), y.max(), 20)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    # 5. Calcular os valores de Z para a superfície usando a sua String e eval()
    # Para o eval funcionar com matrizes do numpy, precisamos passar o X_grid como 'x' e Y_grid como 'y'
    xr = X_grid
    yr = Y_grid
    Z_grid = eval(str_eq)

    # 6. Montar o gráfico usando graph_objects (permite sobrepor pontos e superfícies)
    fig = go.Figure()

    # Adiciona os pontos reais (Scatter3d)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=6, color='red', opacity=0.9),
        name='Dados Reais'
    ))

    # Adiciona a superfície do plano de regressão (Surface)
    fig.add_trace(go.Surface(
        x=x_range,
        y=y_range,
        z=Z_grid,
        opacity=0.6,
        colorscale='Blues',
        showscale=False, # Remove a barra de cores lateral se não quiser
        name='Plano de Regressão'
    ))

    # Configurações de layout e eixos
    fig.update_layout(
        title="Clinical trial - Regressão Linear 3D",
        scene=dict(
            xaxis_title="ml",
            yaxis_title="score",
            zaxis_title="Resultado (z)"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return(str_eq,fig)

    #fig.show()


def ajuste_quadrado_3D(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
):
    # 1. Garantir que os dados são arrays do NumPy unidimensionais
    x_dados = np.asarray(x).flatten()
    y_dados = np.asarray(y).flatten()
    z_dados = np.asarray(z).flatten()

    # 2. Montar a Matriz X com os termos não-lineares (Polinômio de 2º Grau)
    x0 = np.ones(len(x_dados))
    x_quad = x_dados ** 2
    y_quad = y_dados ** 2
    xy_interacao = x_dados * y_dados
    
    X = np.column_stack((x0, x_dados, y_dados, x_quad, y_quad, xy_interacao))

    # 3. Cálculo Matricial para encontrar os 6 pesos (w)
    w = np.linalg.pinv(X) @ z_dados
    w_list = w.tolist()

    # --- NOVO: CÁLCULO DE PERFORMANCE (R² e MAPE) ---
    
    # Previsão de Z para os pontos reais usando os pesos encontrados
    z_previsto = X @ w
    
    # Cálculo do R²
    massa_z_real = np.mean(z_dados)
    ss_res = np.sum((z_dados - z_previsto) ** 2)      # Soma dos Quadrados dos Resíduos
    ss_tot = np.sum((z_dados - massa_z_real) ** 2)     # Soma Total dos Quadrados
    r2 = 1 - (ss_res / ss_tot)
    
    # Cálculo do MAPE (com proteção para divisão por zero)
    epsilon = 1e-10  # evita divisão por zero se z_dados tiver valor 0
    mape = np.mean(np.abs((z_dados - z_previsto) / (z_dados + epsilon))) * 100

    # 4. Criando a String da Equação
    str_eq = (f"{w_list[0]} + x * {w_list[1]} + y * {w_list[2]} + "
              f"(x**2) * {w_list[3]} + (y**2) * {w_list[4]} + (x*y) * {w_list[5]}")

    # 5. Criar a malha (Grid) cobrindo o intervalo de X e Y para o gráfico
    x_range = np.linspace(x_dados.min(), x_dados.max(), 30)
    y_range = np.linspace(y_dados.min(), y_dados.max(), 30)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    # 6. Calcular a superfície Z usando a string não-linear no eval()
    x = X_grid
    y = Y_grid
    Z_grid = eval(str_eq)

    # 7. Montar o gráfico no Plotly
    fig = go.Figure()

    # Pontos Reais
    fig.add_trace(go.Scatter3d(
        x=x_dados, y=y_dados, z=z_dados,
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.5),
        name='Dados Reais'
    ))

    # Superfície Curvada
    fig.add_trace(go.Surface(
        x=x_range,
        y=y_range,
        z=Z_grid,
        opacity=0.6,
        colorscale='Viridis',
        showscale=False,
        name='Superfície Polinomial'
    ))

    # Título dinâmico exibindo as métricas formatadas
    titulo_metricas = (
        f"Regressão Polinomial 3D<br>"
        f"<b>R²:</b> {r2:.4f} ({r2*100:.1f}%) | <b>MAPE:</b> {mape:.2f}%"
    )

    fig.update_layout(
        title=titulo_metricas,
        margin=dict(l=0, r=0, b=0, t=60),
        scene=dict(
            xaxis_title="Eixo X",
            yaxis_title="Eixo Y",
            zaxis_title="Eixo Z"
        )
    )

    # Retornamos os valores calculados em um dicionário de métricas para fácil acesso
    metricas = {"R2": r2, "MAPE": mape}

    return str_eq, fig,metricas


def ajuste_de_curva_3D(
        n:int,
        x: np.ndarray | list,
        y: np.ndarray | list,
        z: np.ndarray | list,
    )->tuple[str,go.Figure,dict]:
    
    # 1. Garantir que os dados são arrays do NumPy unidimensionais
    x_dados = np.asarray(x).flatten()
    y_dados = np.asarray(y).flatten()
    z_dados = np.asarray(z).flatten()

    # 2. Montar a Matriz X com os termos não-lineares (Polinômio de 2º Grau)
    x0 = np.ones(len(x_dados))
    xy_interacao = x_dados * y_dados

    tuple_dados = [x0,xy_interacao]

    for i in range(n):
        tuple_dados.append(x_dados ** (i+1))
        tuple_dados.append(y_dados ** (i+1))

    #print(tuple_dados)
    X = np.column_stack(tuple(tuple_dados))

    # 3. Cálculo Matricial para encontrar os 6 pesos (w)
    w = np.linalg.pinv(X) @ z_dados #mesma coisa que dot
    w_list = w.tolist()

    # --- NOVO: CÁLCULO DE PERFORMANCE (R² e MAPE) ---
    
    # Previsão de Z para os pontos reais usando os pesos encontrados
    z_previsto = X @ w
    
    # Cálculo do R²
    massa_z_real = np.mean(z_dados)
    ss_res = np.sum((z_dados - z_previsto) ** 2)      # Soma dos Quadrados dos Resíduos
    ss_tot = np.sum((z_dados - massa_z_real) ** 2)     # Soma Total dos Quadrados
    r2 = 1 - (ss_res / ss_tot)
    
    # Cálculo do MAPE (com proteção para divisão por zero)
    epsilon = 1e-10  # evita divisão por zero se z_dados tiver valor 0
    mape = np.mean(np.abs((z_dados - z_previsto) / (z_dados + epsilon))) * 100

    str_base_x = "xr"
    str_base_y = "yr"
    

    str_eq = f"{w_list[0]} + ({str_base_x}*{str_base_y}) * {w_list[1]} + "

    w_l = w_list[2:]

    for i in range(len(w_l)):
        if i%2 == 0:
            str_eq += f"({str_base_x}**{i+1 - i//2}) * {w_l[i]} + ({str_base_y}**{i+1- i//2}) * {w_l[i+1]} + "
    
    str_eq = str_eq[:-2]

    # 5. Criar a malha (Grid) cobrindo o intervalo de X e Y para o gráfico
    x_range = np.linspace(x_dados.min(), x_dados.max(), 30)
    y_range = np.linspace(y_dados.min(), y_dados.max(), 30)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    # 6. Calcular a superfície Z usando a string não-linear no eval()
    xr = X_grid
    yr = Y_grid
    Z_grid = eval(str_eq)

    # 7. Montar o gráfico no Plotly
    fig = go.Figure()

    # Pontos Reais
    fig.add_trace(go.Scatter3d(
        x=x_dados, y=y_dados, z=z_dados,
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.5),
        name='Dados Reais'
    ))

    # Superfície Curvada
    fig.add_trace(go.Surface(
        x=x_range,
        y=y_range,
        z=Z_grid,
        opacity=0.6,
        colorscale='Viridis',
        showscale=False,
        name='Superfície Polinomial'
    ))

    # Título dinâmico exibindo as métricas formatadas
    titulo_metricas = (
        f"Regressão Polinomial 3D<br>"
        f"<b>R²:</b> {r2:.4f} ({r2*100:.1f}%) | <b>MAPE:</b> {mape:.2f}%"
    )

    fig.update_layout(
        title=titulo_metricas,
        margin=dict(l=0, r=0, b=0, t=60),
        scene=dict(
            xaxis_title="Eixo X",
            yaxis_title="Eixo Y",
            zaxis_title="Eixo Z"
        )
    )

    # Retornamos os valores calculados em um dicionário de métricas para fácil acesso
    metricas = {"R2": r2, "MAPE": mape}

    return(str_eq,fig,metricas)


def track_progress(iterable,nome:str = ""):
    """
    Wraps an iterable to print progress and estimated time of arrival (ETA).
    """
    my_list = list(iterable) # Convert to list to get total length safely
    total = len(my_list)
    tempo_m = []

    com_t = time.time()
    
    print(f"\nComeçando o loop {nome} com {total} itens...\n")
    
    for contador, item in enumerate(my_list, 1):
        com = time.time()
        
        # 1. Yield the item back to your actual loop
        yield item 
        
        # 2. Run the calculation after the loop body executes
        fim = time.time()
        tempo_m.append(fim - com)
        
        tempo_restante = (sum(tempo_m) / len(tempo_m)) * (total - contador)
        segundos = dat.timedelta(seconds=int(tempo_restante))
        termino = dat.datetime.now() + segundos
        
        # Clear the line and print progress
        string_porc = f"\rProgress: {100*contador/total:.2f}% | Iteration: {contador}/{total} | ETA: {termino:%d/%m %H:%M:%S}"
        print(string_porc, end='', flush=True)
    
    fim = time.time()

    tempo_total = fim - com_t

    segundos = dat.timedelta(seconds=int(tempo_total))

    print(f"\nLoop finalizado, tempo total {segundos}")



def ajuste_de_curva_3D_troca(
        n:int,
        vx: np.ndarray | list,
        vy: np.ndarray | list,
        vz: np.ndarray | list,
        dict_decisoes:dict = {}
    )->tuple[str,go.Figure,dict]:


    #dict_decisoes é um dicionário que realiza a substituição de valores simples de x**2 ou y**2 por fatores mais complexos
    #como cos(x)**2 ou 1/y**2 para alcançar R² ou MAPE melhores e gerar funções mais próximas do fenômeno real
    #a estrutura do dicionario deve funcionar da seguinte forma:
    #   key -> valor original da função, por exemplo "x**1" ou "y**2"
    #   value -> valor que deve ser substituido pelo valor da key, algo como "cos(x)" ou "tan(y**2)"

    # 1. Garantir que os dados são arrays do NumPy unidimensionais
    x_dados = np.asarray(vx).flatten()
    y_dados = np.asarray(vy).flatten()
    z_dados = np.asarray(vz).flatten()

    # 2. Listas para construir a Matriz X e a String da Equação de forma sincronizada
    termos_matriz = [np.ones(len(x_dados)), x_dados * y_dados]
    termos_string = ["{w[0]}", "({x}*{y}) * {w[1]}"]
    
    idx_peso = 2  # Controla o índice dinâmico do vetor de pesos 'w' na string
    
    # 3. Loop Inteligente: Varre os graus de 1 até n
    for i in range(1, n + 1):
        # --- Processamento do Eixo X ---
        chave_x = f"x**{i}"
        # Se existir no dicionário, pega a substituição, se não, mantém o padrão "x**i"
        expr_x = dict_decisoes.get(chave_x, chave_x)
        
        # Alimenta a matriz (calcula os dados reais) e a estrutura de texto da equação
        termos_matriz.append(eval(expr_x.replace("x", "x_dados").replace("y","y_dados")))
        termos_string.append(f"({expr_x}) * {{w[{idx_peso}]}}")
        idx_peso += 1

        # --- Processamento do Eixo Y ---
        chave_y = f"y**{i}"
        expr_y = dict_decisoes.get(chave_y, chave_y)
        
        termos_matriz.append(eval(expr_y.replace("y", "y_dados").replace("x", "x_dados")))
        termos_string.append(f"({expr_y}) * {{w[{idx_peso}]}}")
        idx_peso += 1

    # Monta a matriz final X empilhando as colunas calculadas
    X = np.column_stack(termos_matriz)

    # 4. Cálculo Matricial (Pesos w)
    w = np.linalg.pinv(X) @ z_dados #mesma coisa que dot
    w_list = w.tolist()

    # 5. Criação Dinâmica e Perfeita da String da Equação usando .format()
    # Junta os blocos com " + " e injeta os valores reais encontrados em w_list
    str_eq = " + ".join(termos_string).format(w=w_list, x="x", y="y")

    # --- CÁLCULO DE PERFORMANCE (R² e MAPE) ---
    z_previsto = X @ w
    r2 = 1 - (np.sum((z_dados - z_previsto) ** 2) / np.sum((z_dados - np.mean(z_dados)) ** 2))
    mape = np.mean(np.abs((z_dados - z_previsto) / (z_dados + 1e-10))) * 100

    # 6. Malha (Grid) para a superfície Plotly
    x_range = np.linspace(x_dados.min(), x_dados.max(), 30)
    y_range = np.linspace(y_dados.min(), y_dados.max(), 30)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    # Executa o eval injetando as variáveis locais que a string da equação espera encontrar ('x' e 'y')
    x, y = X_grid, Y_grid
    Z_grid = eval(str_eq) # Remove o prefixo se houver para o eval local do grid se preferir, ou mantém

    # 7. Construção do Gráfico Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_dados, y=y_dados, z=z_dados, mode='markers', marker=dict(size=3, color='blue', opacity=0.5), name='Dados Reais'))
    fig.add_trace(go.Surface(x=x_range, y=y_range, z=Z_grid, opacity=0.6, colorscale='Viridis', showscale=False, name='Superfície'))

    titulo_metricas = f"Regressão Customizada 3D<br><b>R²:</b> {r2:.4f} ({r2*100:.1f}%) | <b>MAPE:</b> {mape:.2f}%"
    fig.update_layout(title=titulo_metricas, margin=dict(l=0, r=0, b=0, t=60), scene=dict(xaxis_title="Eixo X", yaxis_title="Eixo Y", zaxis_title="Eixo Z"))

    return str_eq, fig, {"R2": r2, "MAPE": mape}


def ajuste_de_curva_3D_sem_graf(
        n:int,
        vx: np.ndarray | list,
        vy: np.ndarray | list,
        vz: np.ndarray | list,
        dict_decisoes:dict = {}
    )->tuple[str,dict]:


    #dict_decisoes é um dicionário que realiza a substituição de valores simples de x**2 ou y**2 por fatores mais complexos
    #como cos(x)**2 ou 1/y**2 para alcançar R² ou MAPE melhores e gerar funções mais próximas do fenômeno real
    #a estrutura do dicionario deve funcionar da seguinte forma:
    #   key -> valor original da função, por exemplo "x**1" ou "y**2"
    #   value -> valor que deve ser substituido pelo valor da key, algo como "cos(x)" ou "tan(y**2)"

    # 1. Garantir que os dados são arrays do NumPy unidimensionais
    x_dados = np.asarray(vx).flatten()
    y_dados = np.asarray(vy).flatten()
    z_dados = np.asarray(vz).flatten()

    # 2. Listas para construir a Matriz X e a String da Equação de forma sincronizada
    termos_matriz = [np.ones(len(x_dados)), x_dados * y_dados]
    termos_string = ["{w[0]}", "({x}*{y}) * {w[1]}"]
    
    idx_peso = 2  # Controla o índice dinâmico do vetor de pesos 'w' na string
    
    # 3. Loop Inteligente: Varre os graus de 1 até n
    for i in range(1, n + 1):
        # --- Processamento do Eixo X ---
        chave_x = f"x**{i}"
        # Se existir no dicionário, pega a substituição, se não, mantém o padrão "x**i"
        expr_x = dict_decisoes.get(chave_x, chave_x)
        
        # Alimenta a matriz (calcula os dados reais) e a estrutura de texto da equação
        termos_matriz.append(eval(expr_x.replace("x", "x_dados").replace("y","y_dados")))
        termos_string.append(f"({expr_x}) * {{w[{idx_peso}]}}")
        idx_peso += 1

        # --- Processamento do Eixo Y ---
        chave_y = f"y**{i}"
        expr_y = dict_decisoes.get(chave_y, chave_y)
        
        termos_matriz.append(eval(expr_y.replace("y", "y_dados").replace("x", "x_dados")))
        termos_string.append(f"({expr_y}) * {{w[{idx_peso}]}}")
        idx_peso += 1

    # Monta a matriz final X empilhando as colunas calculadas
    X = np.column_stack(termos_matriz)

    # 4. Cálculo Matricial (Pesos w)
    w = np.linalg.pinv(X) @ z_dados #mesma coisa que dot
    w_list = w.tolist()

    # 5. Criação Dinâmica e Perfeita da String da Equação usando .format()
    # Junta os blocos com " + " e injeta os valores reais encontrados em w_list
    str_eq = " + ".join(termos_string).format(w=w_list, x="x", y="y")

    # --- CÁLCULO DE PERFORMANCE (R² e MAPE) ---
    z_previsto = X @ w
    r2 = 1 - (np.sum((z_dados - z_previsto) ** 2) / np.sum((z_dados - np.mean(z_dados)) ** 2))
    mape = np.mean(np.abs((z_dados - z_previsto) / (z_dados + 1e-10))) * 100

    return str_eq, {"R2": r2, "MAPE": mape}

# --- FUNÇÃO AUXILIAR PARA RENDERIZAR O GRÁFICO PLOTLY NO FINAL ---
def plota_superficie_3d(str_eq: str, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, titulo: str) -> go.Figure:
    x_range = np.linspace(vx.min(), vx.max(), 30)
    y_range = np.linspace(vy.min(), vy.max(), 30)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Injeta x e y locais que a string gerada pela regressão aguarda para o eval do Grid
    x, y = X_grid, Y_grid
    Z_grid = eval(str_eq)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=vx, y=vy, z=vz, mode='markers', marker=dict(size=4, color='red', opacity=0.5), name='Reais'))
    fig.add_trace(go.Surface(x=x_range, y=y_range, z=Z_grid, opacity=0.6, colorscale='Viridis', showscale=False, name='Ajuste'))
    fig.update_layout(title=titulo, margin=dict(l=0, r=0, b=0, t=40), scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    return fig

OpcoesTransformacao = Literal[
    "x**1", "x**2", "x**3", "x**(1/2)", "x**(2/3)", "x**-1", "x**-2",
    "x+y", "x/y", "np.exp(-x)", "x*np.exp(-x)", "np.log(x)", "2**x",
    "np.cos(x)", "np.sin(x)", "np.tanh(x)","np.cosh(x)","np.sinh(x)"
]

def otimizar_curva_3d(
    n: int,
    vx: np.ndarray | list,
    vy: np.ndarray | list,
    vz: np.ndarray | list,
    lista_troca_permitidas: list[OpcoesTransformacao] = []
) -> tuple[dict, dict, go.Figure, go.Figure]:
    """
    Otimizador de Curva 3D.
    Mapeia a lista de strings agnósticas de forma independente para X e Y.

    Opções permitidas na lista_troca_permitidas:

    "x**1", "x**2", "x**3", "x**(1/2)", "x**(2/3)", "x**-1", "x**-2",
    "x+y", "x/y", "np.exp(-x)", "x*np.exp(-x)", "np.log(x)", "2**x",
    "np.cos(x)", "np.sin(x)", "np.tanh(x)","np.cosh(x)","np.sinh(x)"

    """
    # 1. Padronização dos dados de entrada
    x_dados = np.asarray(vx).flatten()
    y_dados = np.asarray(vy).flatten()
    z_dados = np.asarray(vz).flatten()
    n_amostras = len(z_dados)

    nomes_variaveis = ["x", "y"]
    
    # 2. Construção do banco de hipóteses dinâmico para X e Y
    banco_hipoteses = {}
    for i in range(1, n + 1):
        for nome in nomes_variaveis:
            chave_termo = f"{nome}**{i}"
            candidatos = [chave_termo]  # Comportamento padrão (polinomial)
            
            for troca in lista_troca_permitidas:
                # Transforma a string genérica do usuário (ex: 'x**-1') para a coluna atual (ex: 'y**-1')
                termo_customizado = troca.replace("x", nome).replace("y", nome)
                candidatos.append(termo_customizado)
                
            banco_hipoteses[chave_termo] = candidatos

    chaves_totais = sorted(list(banco_hipoteses.keys()))
    listas_candidatos = [banco_hipoteses[c] for c in chaves_totais]
    
    # Estruturas para guardar os melhores resultados
    melhor_r2 = {"r2": -float('inf'), "mape": float('inf'), "dict": {}, "eq": ""}
    melhor_mape = {"r2": -float('inf'), "mape": float('inf'), "dict": {}, "eq": ""}
    
    print(f"Iniciando varredura de combinações para n={n}...")
    tentativas = 0
    
    # 3. Varredura Combinatória Global
    with np.errstate(divide='ignore', invalid='ignore'):
        for combinacao in itertools.product(*listas_candidatos):
            dict_rodada = dict(zip(chaves_totais, combinacao))
            
            # Inicializa os termos base do seu pipeline (Intercepto + Interação Cruzada)
            termos_matriz = [np.ones(n_amostras), x_dados * y_dados]
            termos_string = ["{w[0]}", "(x*y) * {w[1]}"]
            idx_peso = 2
            
            escopo_eval = {"x": x_dados, "y": y_dados, "np": np}
            
            try:
                for chave in chaves_totais:
                    expr_atual = dict_rodada[chave]
                    valores_calculados = eval(expr_atual, escopo_eval)
                    
                    termos_matriz.append(valores_calculados)
                    termos_string.append(f"({expr_atual}) * {{w[{idx_peso}]}}")
                    idx_peso += 1
                    
                X = np.column_stack(termos_matriz)
                
                # Resolução direta via álgebra linear (Pseudoinversa)
                w = np.linalg.pinv(X) @ z_dados
                w_list = w.tolist()
                
                # Avaliação de métricas de performance
                z_previsto = X @ w
                ss_res = np.sum((z_dados - z_previsto) ** 2)
                ss_tot = np.sum((z_dados - np.mean(z_dados)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                if np.isnan(r2) or np.isinf(r2) or r2 < 0:
                    continue
                    
                mape = np.mean(np.abs((z_dados - z_previsto) / (z_dados + 1e-10))) * 100
                str_eq = " + ".join(termos_string).format(w=w_list)
                
                tentativas += 1
                
                # Validação do Campeão de R²
                if r2 > melhor_r2["r2"]:
                    melhor_r2 = {"r2": r2, "mape": mape, "dict": dict_rodada, "eq": str_eq}
                    
                # Validação do Campeão de MAPE
                if mape < melhor_mape["mape"]:
                    melhor_mape = {"r2": r2, "mape": mape, "dict": dict_rodada, "eq": str_eq}
                    
            except Exception:
                continue
                
    print(f"Varredura concluída! {tentativas} modelos válidos avaliados.")
    
    # 4. Renderização dos gráficos tridimensionais dos vencedores
    fig_r2 = plota_superficie_3d(melhor_r2["eq"], x_dados, y_dados, z_dados, 
                                 f"Melhor R²: {melhor_r2['r2']*100:.2f}% (MAPE: {melhor_r2['mape']:.2f}%)")
    
    fig_mape = plota_superficie_3d(melhor_mape["eq"], x_dados, y_dados, z_dados, 
                                  f"Melhor MAPE: {melhor_mape['mape']:.2f}% (R²: {melhor_mape['r2']*100:.2f}%)")
    
    return melhor_r2, melhor_mape, fig_r2, fig_mape

def auto_ml_regressao_multivariavel(
    n: int,
    lista_parametros: list[np.ndarray | list],
    resultado: list | np.ndarray,
    lista_troca_permitidas: list = []
) -> tuple[str, dict]:
    """
    Motor de AutoML matricial para Regressão Não-Linear Multivariável.
    Ajusta dinamicamente infinitas variáveis usando a pseudoinversa.
    """
    # 1. Padronização do alvo (variável dependente)
    z_dados = np.asarray(resultado).flatten()
    n_amostras = len(z_dados)
    
    # 2. Padronização das variáveis independentes (Cria nomes v0, v1, v2...)
    variaveis_dados = [np.asarray(p).flatten() for p in lista_parametros]
    n_variaveis = len(variaveis_dados)
    nomes_variaveis = [f"v{i}" for i in range(n_variaveis)]

    # 3. Construção do banco de hipóteses de transformações
    # Se a lista estiver vazia, o sistema assume o comportamento polinomial padrão (v**grau)
    banco_hipoteses = {}
    for i in range(1, n + 1):
        for nome in nomes_variaveis:
            chave_termo = f"{nome}**{i}"
            
            # Opções de candidatos para este termo específico
            candidatos = [chave_termo] # Padrão é ele mesmo
            for troca in lista_troca_permitidas:
                # Adapta a string genérica (ex: 'x**-1') para o nome da variável real (ex: 'v0**-1')
                termo_customizado = troca.replace("x", nome).replace("y", nome)
                candidatos.append(termo_customizado)
                
            banco_hipoteses[chave_termo] = candidatos

    # Ordena as chaves para garantir consistência no itertools
    chaves_termos = sorted(list(banco_hipoteses.keys()))
    listas_para_produto = [banco_hipoteses[c] for c in chaves_termos]

    # Estruturas para armazenar o campeão global
    melhor_r2 = -float('inf')
    campeao_eq = ""
    campeao_performance = {"R2": 0.0, "MAPE": 100.0}

    # 4. Busca Combinatória Global via Força Bruta Matricial
    # Varre todas as combinações possíveis de transformações de variáveis
    with np.errstate(divide='ignore', invalid='ignore'):
        for combinacao in track_progress(itertools.product(*listas_para_produto),"Ajustes"):
            # Mapeia qual expressão substitui qual termo polinomial básico
            mapa_rodada = dict(zip(chaves_termos, combinacao))
            
            # Inicia a matriz de design X com o Intercepto (coluna de 1s)
            termos_matriz = [np.ones(n_amostras)]
            termos_string = ["{w[0]}"]
            idx_peso = 1

            # Monta os escopos locais dinamicamente para o eval rodar com alta velocidade
            escopo_eval = {nomes_variaveis[k]: variaveis_dados[k] for k in range(n_variaveis)}
            escopo_eval['np'] = np # Injeta o numpy caso haja np.cos, np.exp, etc. #type:ignore

            try:
                # Alimenta os termos na Matriz X e na String da Equação de forma sincronizada
                for chave in chaves_termos:
                    expr_atual = mapa_rodada[chave]
                    
                    # Executa o cálculo matemático vetorizado na memória
                    valores_calculados = eval(expr_atual, escopo_eval)
                    
                    termos_matriz.append(valores_calculados)
                    termos_string.append(f"({expr_atual}) * {{w[{idx_peso}]}}")
                    idx_peso += 1
                    
                # Cria a matriz X final empilhando as colunas geradas
                X = np.column_stack(termos_matriz)
                
                # 5. Álgebra Linear Direta (Pseudoinversa)
                w = np.linalg.pinv(X) @ z_dados #mesmo que o dot
                w_list = w.tolist()
                
                # --- VALIDAÇÃO DE PERFORMANCE ---
                z_previsto = X @ w #mesmo que o dot
                
                ss_res = np.sum((z_dados - z_previsto) ** 2)
                ss_tot = np.sum((z_dados - np.mean(z_dados)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Pula combinações numericamente inválidas (NaN ou Infinitos)
                if np.isnan(r2) or np.isinf(r2):
                    continue
                    
                # Se encontrou um R² global melhor, salva a estrutura
                if r2 > melhor_r2:
                    melhor_r2 = r2
                    mape = np.mean(np.abs((z_dados - z_previsto) / (z_dados + 1e-10))) * 100
                    
                    # Formata a string final injetando os coeficientes calculados
                    campeao_eq = " + ".join(termos_string).format(w=w_list)
                    campeao_performance = {"R2": r2, "MAPE": mape}
                    
            except Exception:
                # Captura e ignora erros matemáticos de domínio (ex: log de número negativo)
                continue

    return campeao_eq, campeao_performance

if __name__ == "__main__":

    # # Massa de dados 1: Comportada (Pouco ruído)
    # np.random.seed(42)
    # v_1 = np.random.uniform(30, 90, 40)
    # t_1 = np.random.uniform(190, 240, 40)
    # # Pouquíssimo ruído térmico adicionado (escala 0.3)
    # r_1 = 50 - 0.02*(v_1 - 50)**2 - 0.01*(t_1 - 215)**2 + np.random.normal(0, 0.3, 40)

    # # Massa de dados 2: Caótica (Muito ruído ambiental)
    # v_2 = v_1.copy()
    # t_2 = t_1.copy()
    # # Ruído violento adicionado no teste (escala 8.0)
    # r_2 = 50 - 0.02*(v_2 - 50)**2 - 0.01*(t_2 - 215)**2 + np.random.normal(0, 8.0, 40)

    # # Executando o cenário 1
    # eq1, fig1, met1 = ajuste_de_curva_3D(2,v_1, t_1, r_1)
    # #print("Cenário 1 ->", met1) # Deve mostrar R² muito alto (ex: > 98%) e MAPE baixo

    # # Executando o cenário 2
    # eq2, fig2, met2 = ajuste_de_curva_3D(2,v_2, t_2, r_2)
    # #print("Cenário 2 ->", met2) # Deve mostrar R² bem menor e MAPE maior devido à distância dos pontos até o plano

    # fig1.show()
    # #fig2.show()

    # dict_decisoes = {
    #     #"x**3":"1/x",
    #     #"y**3":"np.exp(1/y)",
        
    # }

    # eq3, fig3, met3 = ajuste_de_curva_3D_troca(
    #     3,
    #     v_1,
    #     t_1,
    #     r_1,
    #     dict_decisoes
    # )

    # print(eq3)

    # fig3.show()

    # # Simulando o laboratório: 1000 pacientes experimentais
    # np.random.seed(42)
    # dias_X = np.random.uniform(1, 10, 1000)     # Dias de tratamento
    # dose_Y = np.random.uniform(1, 100, 1000)   # Dose em mg

    # # A lei biológica real por trás dos dados (Saturação Exponencial)
    # # Eficácia máxima teórica de 100%. Note o uso de termos exponenciais e inversos.
    # Z_eficacia_real = 100 * (1 - np.exp(-0.5 * dias_X)) + (20 * (1 / (10 / dose_Y)))
    # # Adicionando o ruído humano/clínico inevitável nos testes
    # Z_eficacia_real += np.random.normal(0, 2, 1000) 
    # Z_eficacia_real = np.clip(Z_eficacia_real, 0, 100) # Limita a 100%

    # # --- EXPERIMENTO 1: Rodando com Polinômio Puro (Grau 2) ---
    # # Simulando a ausência de termos customizados (dicionário vazio)
    # eq_pura, perf_pura = ajuste_de_curva_3D_sem_graf(5, dias_X, dose_Y, Z_eficacia_real, dict_decisoes={})

    # # --- EXPERIMENTO 2: Rodando com o seu Otimizador (Grau 2) ---
    # # Ele vai testar substituições inteligentes como np.exp(-x) e 1/y
    # res_r2, res_mape, graf_r2, graf_mape = otimizar_curva_3d(2, dias_X, dose_Y, Z_eficacia_real)

    # print("\n================ COMPARAÇÃO DE PERFORMANCE ================")
    # print(f"Polinômio Puro (n=2) -> R²: {perf_pura['R2']*100:.2f}% | MAPE: {perf_pura['MAPE']:.2f}%")
    # print(f"Otimizado com Troca  -> R²: {res_r2['r2']*100:.2f}% | MAPE: {res_r2['mape']:.2f}%")
    # print(f"Combinação Vencedora: {res_r2['dict']}")
    # print("===========================================================")

    # graf_r2.show()
    # graf_mape.show()

    np.random.seed(42)
    
    # Criando 4 variáveis preditoras independentes na memória (Simulando uma tabela real)
    v0_dados = np.random.uniform(1, 10, 50)
    v1_dados = np.random.uniform(1, 10, 50)
    v2_dados = np.random.uniform(1, 10, 50)
    v3_dados = np.random.uniform(1, 10, 50)
    
    # Agrupamos na estrutura de lista exigida pelo parâmetro
    lista_de_recursos = [v0_dados, v1_dados, v2_dados, v3_dados]
    
    # Geramos uma equação alvo complexa oculta com ruído: 
    # Z = 5 + (1/v0)*12 + cos(v1)*3 + (v2**2)*1.5 + v3*0.8
    alvo_Z = 5 + (1 / v0_dados)*12 + np.cos(v1_dados)*3 + (v2_dados**2)*1.5 + v3_dados*0.8 + np.random.normal(0, 0.5, 50)

    # Lista de hipóteses customizadas que o sistema tem permissão para testar
    trocas_permitidas = [
        "x**-1",
        "x**0.5"
    ]#["x**-1", "np.cos(x)", "np.exp(-x)","np.tanh(x)"]

    #trocas_permitidas  = []

    # Executando a busca automática limitando o grau básico em n=2
    equacao_achada, metricas_finais = auto_ml_regressao_multivariavel(
        n=2, 
        lista_parametros=lista_de_recursos, #type:ignore
        resultado=alvo_Z, 
        lista_troca_permitidas=trocas_permitidas
    )

    print("================ RESULTADO DE AUTO-ML MULTIVARIÁVEL ================")
    print(f"Métricas Alcançadas -> R²: {metricas_finais['R2']*100:.2f}% | MAPE: {metricas_finais['MAPE']:.2f}%")
    print(f"\nEquação Matemática Otimizada:\nZ = {equacao_achada}")
    print("====================================================================")
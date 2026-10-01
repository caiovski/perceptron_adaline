"""
=============================================================================
PERCEPTRON — Classificação de Pureza de Óleo
=============================================================================
CEFET-MG Campus VIII – Varginha
Bacharelado em Sistemas de Informação
Disciplina: Lab. Inteligência Artificial
Professor: Lázaro Eduardo da Silva

Descrição:
    Implementação do Perceptron com regra de aprendizagem de Hebb para
    classificação binária de pureza de óleo em duas classes (C1 e C2)
    a partir de três propriedades físico-químicas (x1, x2, x3).

Arquitetura do neurônio:
    x0 = -1 (bias) ──(w0)──┐
    x1 ─────────────(w1)────┤
    x2 ─────────────(w2)────┼──► Σ (soma ponderada) ──► g(.) ──► y
    x3 ─────────────(w3)────┘         u                degrau

Configurações:
    - Taxa de aprendizagem (η): 0.01
    - Função de ativação: Degrau Bipolar (sinal)
    - Entrada de bias (x0): -1
    - Critério de parada: erro total = 0 ou máximo de 1000 épocas
    - Convenção: d = -1 (classe C1) | d = +1 (classe C2)
=============================================================================
"""

import numpy as np
import csv
import os
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica (para salvar imagens)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================
# Estes são os parâmetros que controlam o comportamento do perceptron.
# Foram definidos conforme o enunciado do trabalho.

TAXA_APRENDIZAGEM = 0.01   # η (eta) — quão grande é o ajuste a cada erro
MAX_EPOCAS = 1000           # Limite de segurança para evitar loop infinito
SEEDS = [1, 2, 3, 4, 5]    # Seeds fixas — garante mesmos resultados ao re-executar
BIAS_INPUT = -1             # Valor fixo da entrada x0 (bias do neurônio)

# Caminhos de arquivos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_DIR = os.path.join(SCRIPT_DIR, 'resultados')
DATASET_PATH = os.path.join(SCRIPT_DIR, '..', 'atividades', 'oleo_dataset.csv')

# Criar pasta de resultados se não existir
os.makedirs(RESULTADOS_DIR, exist_ok=True)


# =============================================================================
# AMOSTRAS DE TESTE (fornecidas no enunciado — Item 3)
# =============================================================================
# Estas são as 10 amostras novas que o professor pede para classificar
# APÓS o treinamento do perceptron.

AMOSTRAS_TESTE = [
    [-0.3565,  0.0620, 5.9891],   # Amostra 1
    [-0.7842,  1.1267, 5.5912],   # Amostra 2
    [ 0.3012,  0.5611, 5.8234],   # Amostra 3
    [ 0.7757,  1.0648, 8.0677],   # Amostra 4
    [ 0.1570,  0.8028, 6.3040],   # Amostra 5
    [-0.7014,  1.0316, 3.6005],   # Amostra 6
    [ 0.3748,  0.1536, 6.1537],   # Amostra 7
    [-0.6920,  0.9404, 4.4058],   # Amostra 8
    [-1.3970,  0.7141, 4.9263],   # Amostra 9
    [-1.8842, -0.2805, 1.2548],   # Amostra 10
]


# =============================================================================
# FUNÇÕES DO PERCEPTRON
# =============================================================================

def funcao_degrau(u):
    """
    Função de ativação: Degrau Bipolar (também chamada de função sinal).

    Regra simples de decisão:
        - Se u >= 0  →  retorna +1  (classifica como classe C2)
        - Se u <  0  →  retorna -1  (classifica como classe C1)

    É a função de ativação mais simples possível: apenas verifica
    se o potencial de ativação é positivo ou negativo.

    Parâmetro:
        u (float): Potencial de ativação (soma ponderada das entradas)

    Retorna:
        float: +1.0 ou -1.0
    """
    return 1.0 if u >= 0 else -1.0


def treinar_perceptron(X, d, seed):
    """
    Treina o perceptron usando a Regra de Hebb (aprendizado supervisionado).

    ╔══════════════════════════════════════════════════════════╗
    ║  ALGORITMO DE TREINAMENTO — REGRA DE HEBB              ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  1. Inicializar pesos aleatórios (entre 0 e 1)          ║
    ║                                                          ║
    ║  2. Para cada ÉPOCA:                                     ║
    ║     Para cada AMOSTRA (x, d):                            ║
    ║       a) Calcular potencial: u = Σ(wi * xi)              ║
    ║       b) Calcular saída: y = degrau(u)                   ║
    ║       c) Calcular erro: e = d - y                        ║
    ║       d) Se errou: w = w + η * e * x                     ║
    ║                                                          ║
    ║  3. Parar quando erro_total = 0 (todos acertou)          ║
    ╚══════════════════════════════════════════════════════════╝

    A ideia é simples: se o neurônio ERROU, ajusta os pesos na
    direção que CORRIGE o erro. Se acertou, não muda nada.

    Parâmetros:
        X    : np.array (N, 4) — Entradas já com bias na 1ª coluna
                                  [[x0, x1, x2, x3], ...]
        d    : np.array (N,)   — Saídas desejadas (-1.0 ou +1.0)
        seed : int              — Semente para reprodutibilidade

    Retorna:
        pesos_iniciais  : np.array (4,) — Pesos antes do treinamento
        pesos_finais    : np.array (4,) — Pesos após convergência
        num_epocas      : int           — Quantas épocas foram necessárias
        historico_erros : list[int]     — Nº de classificações erradas por época
    """
    # Reiniciar o gerador de números aleatórios com a seed fornecida.
    # Isso garante que, ao rodar novamente, os pesos iniciais serão os mesmos.
    np.random.seed(seed)

    # PASSO 1: Inicializar os 4 pesos aleatoriamente entre 0 e 1
    # pesos[0] = w0 (peso do bias)
    # pesos[1] = w1 (peso de x1)
    # pesos[2] = w2 (peso de x2)
    # pesos[3] = w3 (peso de x3)
    pesos = np.random.rand(4)
    pesos_iniciais = pesos.copy()  # Guardar cópia antes de treinar

    # Lista para registrar quantas amostras foram classificadas errado em cada época
    historico_erros = []

    num_amostras = len(X)

    # PASSO 2: Loop de treinamento — cada iteração completa = 1 ÉPOCA
    for epoca in range(1, MAX_EPOCAS + 1):
        erros_na_epoca = 0  # Contador de erros nesta época

        # Apresentar TODAS as amostras ao neurônio (uma por vez)
        for i in range(num_amostras):
            # 2a. Calcular o POTENCIAL DE ATIVAÇÃO (soma ponderada)
            #     u = w0*x0 + w1*x1 + w2*x2 + w3*x3
            #     Usando produto escalar (dot product) para simplificar
            u = np.dot(pesos, X[i])

            # 2b. Aplicar a FUNÇÃO DE ATIVAÇÃO (degrau bipolar)
            y = funcao_degrau(u)

            # 2c. Calcular o ERRO (diferença entre desejado e obtido)
            #     Se d=+1 e y=+1 → erro = 0  (acertou!)
            #     Se d=+1 e y=-1 → erro = 2  (errou — deveria ser +1)
            #     Se d=-1 e y=+1 → erro = -2 (errou — deveria ser -1)
            #     Se d=-1 e y=-1 → erro = 0  (acertou!)
            erro = d[i] - y

            # 2d. Se ERROU, ajustar os pesos pela REGRA DE HEBB
            #     w_novo = w_atual + η × erro × entrada
            #
            #     Intuição: se errou para +1, os pesos são aumentados
            #     na direção da entrada. Se errou para -1, são diminuídos.
            if erro != 0.0:
                pesos = pesos + TAXA_APRENDIZAGEM * erro * X[i]
                erros_na_epoca += 1

        # Registrar quantos erros houve nesta época
        historico_erros.append(erros_na_epoca)

        # PASSO 3: Critério de parada
        # Se NENHUMA amostra foi classificada errada → convergiu!
        if erros_na_epoca == 0:
            break

    return pesos_iniciais, pesos.copy(), epoca, historico_erros


def classificar(pesos, X):
    """
    Classifica um conjunto de amostras usando pesos JÁ TREINADOS.

    Para cada amostra:
        1. Calcula u = soma ponderada (w · x)
        2. Aplica degrau bipolar
        3. Retorna +1 (classe C2) ou -1 (classe C1)

    Parâmetros:
        pesos : np.array (4,)   — Vetor de pesos treinado
        X     : np.array (N, 4) — Amostras com bias na 1ª coluna

    Retorna:
        list[float]: Lista com +1.0 ou -1.0 para cada amostra
    """
    resultados = []
    for entrada in X:
        u = np.dot(pesos, entrada)
        y = funcao_degrau(u)
        resultados.append(y)
    return resultados


# =============================================================================
# CARREGAR DADOS DO CSV
# =============================================================================

def carregar_dataset(caminho):
    """
    Carrega o dataset CSV do óleo e prepara os dados para o perceptron.

    O arquivo CSV tem o formato:
        Padrao, x1, x2, x3, d

    Esta função:
        1. Lê cada linha do CSV
        2. Adiciona o bias (x0 = -1) como primeira coluna
        3. Retorna as entradas X e as saídas desejadas d

    Parâmetros:
        caminho (str): Caminho para o arquivo CSV

    Retorna:
        X : np.array (30, 4) — Entradas: [[-1, x1, x2, x3], ...]
        d : np.array (30,)   — Saídas desejadas: [-1, +1, -1, ...]
    """
    X = []
    d = []

    with open(caminho, 'r') as arquivo:
        leitor = csv.DictReader(arquivo)
        for linha in leitor:
            x1 = float(linha['x1'])
            x2 = float(linha['x2'])
            x3 = float(linha['x3'])
            saida_desejada = float(linha['d'])

            # Montar vetor de entrada com bias: [x0, x1, x2, x3]
            # x0 = -1 é o bias (entrada fixa que permite ao neurônio
            # ajustar seu "limiar de decisão")
            X.append([BIAS_INPUT, x1, x2, x3])
            d.append(saida_desejada)

    return np.array(X), np.array(d)


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO (GRÁFICOS)
# =============================================================================

def plotar_evolucao_erros(resultados_treino):
    """
    Gera gráfico com a evolução do número de erros por época
    para cada um dos 5 treinamentos.

    Eixo X: Número da época
    Eixo Y: Quantidade de amostras classificadas incorretamente
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Paleta de cores harmoniosa
    cores = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    for i, r in enumerate(resultados_treino):
        epocas = range(1, len(r['historico_erros']) + 1)
        ax.plot(
            epocas, r['historico_erros'],
            color=cores[i],
            linewidth=2.5,
            marker='o',
            markersize=5,
            markerfacecolor='white',
            markeredgewidth=2,
            markeredgecolor=cores[i],
            label=f"T{r['treinamento']} (seed={r['seed']}) — {r['epocas']} épocas",
            alpha=0.9
        )

    ax.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax.set_ylabel('Número de Classificações Erradas', fontsize=13, fontweight='bold')
    ax.set_title(
        'Perceptron — Evolução dos Erros por Época de Treinamento',
        fontsize=15, fontweight='bold', pad=15
    )
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=-0.5)
    ax.tick_params(labelsize=11)

    # Linha de referência em y=0 (convergência)
    ax.axhline(y=0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(
        ax.get_xlim()[1] * 0.85, 0.3,
        '← convergência (0 erros)',
        fontsize=9, color='green', alpha=0.7
    )

    plt.tight_layout()
    caminho = os.path.join(RESULTADOS_DIR, 'evolucao_erros.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    return caminho


def plotar_dispersao_3d(X_treino, d_treino, resultados_treino):
    """
    Gera gráfico 3D de dispersão das amostras de treinamento
    coloridas por classe, com o hiperplano de separação.

    O hiperplano é definido pela equação:
        w0*(-1) + w1*x1 + w2*x2 + w3*x3 = 0

    Resolvendo para x3:
        x3 = (w0 - w1*x1 - w2*x2) / w3
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Separar amostras por classe (removendo a coluna do bias)
    idx_c1 = d_treino == -1
    idx_c2 = d_treino == 1
    c1 = X_treino[idx_c1][:, 1:]  # x1, x2, x3 da classe C1
    c2 = X_treino[idx_c2][:, 1:]  # x1, x2, x3 da classe C2

    # Plotar pontos das classes
    ax.scatter(
        c1[:, 0], c1[:, 1], c1[:, 2],
        c='#E74C3C', marker='x', s=100, linewidths=2.5,
        label='Classe C1 (d = -1)', depthshade=True
    )
    ax.scatter(
        c2[:, 0], c2[:, 1], c2[:, 2],
        c='#3498DB', marker='o', s=100, edgecolors='#2C3E50', linewidths=1.5,
        label='Classe C2 (d = +1)', depthshade=True
    )

    # Plotar hiperplano de separação (usando pesos do T1)
    w = resultados_treino[0]['w_finais']

    if abs(w[3]) > 1e-10:  # Verificar se w3 ≠ 0 (evitar divisão por zero)
        # Criar grade de pontos x1, x2
        margem = 0.5
        x1_range = np.linspace(
            X_treino[:, 1].min() - margem,
            X_treino[:, 1].max() + margem,
            25
        )
        x2_range = np.linspace(
            X_treino[:, 2].min() - margem,
            X_treino[:, 2].max() + margem,
            25
        )
        X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)

        # Calcular x3 do hiperplano:
        # u = 0 → w0*(-1) + w1*x1 + w2*x2 + w3*x3 = 0
        # x3 = (w0 - w1*x1 - w2*x2) / w3
        X3_mesh = (w[0] - w[1] * X1_mesh - w[2] * X2_mesh) / w[3]

        # Limitar o plano à região dos dados
        x3_min = X_treino[:, 3].min() - 1
        x3_max = X_treino[:, 3].max() + 1
        X3_mesh = np.clip(X3_mesh, x3_min, x3_max)

        # Desenhar a superfície do hiperplano
        ax.plot_surface(
            X1_mesh, X2_mesh, X3_mesh,
            alpha=0.15, color='#2ECC71',
            edgecolor='#27AE60', linewidth=0.3
        )

    ax.set_xlabel('x1', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('x2', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('x3', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(
        'Dispersão 3D das Amostras + Hiperplano de Separação (T1)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend(fontsize=11, loc='upper left')

    # Melhorar ângulo de visualização
    ax.view_init(elev=25, azim=135)

    plt.tight_layout()
    caminho = os.path.join(RESULTADOS_DIR, 'dispersao_3d.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    return caminho


def plotar_comparacao_classificacoes(classificacoes, amostras):
    """
    Gera gráfico de mapa de calor (heatmap) mostrando as classificações
    das 10 amostras de teste por cada um dos 5 treinamentos.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Montar matriz de classificações (10 amostras × 5 treinamentos)
    matriz = np.array(classificacoes).T  # Transpor: linhas=amostras, colunas=treinos

    # Criar mapa de calor
    cores_mapa = plt.cm.colors.ListedColormap(['#E74C3C', '#3498DB'])
    limites = [-1.5, 0, 1.5]
    norma = plt.cm.colors.BoundaryNorm(limites, cores_mapa.N)

    im = ax.imshow(matriz, cmap=cores_mapa, norm=norma, aspect='auto')

    # Rótulos
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'T{i+1}' for i in range(5)], fontsize=12, fontweight='bold')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'Amostra {i+1}' for i in range(10)], fontsize=11)
    ax.set_xlabel('Treinamento', fontsize=13, fontweight='bold')
    ax.set_title(
        'Classificação das Amostras de Teste por Treinamento',
        fontsize=14, fontweight='bold', pad=15
    )

    # Adicionar texto nas células
    for i in range(10):
        for j in range(5):
            val = matriz[i, j]
            classe = 'C1' if val == -1 else 'C2'
            cor_texto = 'white'
            ax.text(j, i, classe, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=cor_texto)

    # Legenda manual
    from matplotlib.patches import Patch
    legenda = [
        Patch(facecolor='#E74C3C', label='C1 (d = -1)'),
        Patch(facecolor='#3498DB', label='C2 (d = +1)')
    ]
    ax.legend(handles=legenda, loc='upper right', fontsize=11,
              bbox_to_anchor=(1.22, 1.0))

    plt.tight_layout()
    caminho = os.path.join(RESULTADOS_DIR, 'classificacao_teste.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    return caminho


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    """
    Função principal que executa todo o trabalho:
        1. Carrega os dados
        2. Treina o perceptron 5 vezes
        3. Classifica amostras de teste
        4. Gera tabelas e gráficos
        5. Apresenta respostas teóricas
    """

    # ─────────────────────────────────────────────────────────────────────
    # CABEÇALHO
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  PERCEPTRON — Classificação de Pureza de Óleo")
    print("  CEFET-MG Campus VIII – Varginha | Lab. Inteligência Artificial")
    print("=" * 72)
    print(f"\n  ⚙ Configurações:")
    print(f"    • Taxa de aprendizagem (η):  {TAXA_APRENDIZAGEM}")
    print(f"    • Bias (x0):                 {BIAS_INPUT}")
    print(f"    • Função de ativação:         Degrau Bipolar")
    print(f"    • Máximo de épocas:           {MAX_EPOCAS}")
    print(f"    • Seeds (reprodutibilidade):  {SEEDS}")

    # ─────────────────────────────────────────────────────────────────────
    # CARREGAR DATASET
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  📂 Carregando dataset...")
    X_treino, d_treino = carregar_dataset(DATASET_PATH)
    n_c1 = int(np.sum(d_treino == -1))
    n_c2 = int(np.sum(d_treino == 1))
    print(f"    ✓ {len(X_treino)} amostras carregadas")
    print(f"    • Classe C1 (d = -1): {n_c1} amostras")
    print(f"    • Classe C2 (d = +1): {n_c2} amostras")

    # Preparar amostras de teste (adicionar bias x0 = -1)
    X_teste = np.array([[BIAS_INPUT] + amostra for amostra in AMOSTRAS_TESTE])

    # ─────────────────────────────────────────────────────────────────────
    # ITENS 1 e 2: EXECUTAR 5 TREINAMENTOS
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  📋 ITENS 1 e 2 — Resultados dos 5 Treinamentos")
    print(f"{'=' * 72}")

    resultados_treino = []

    for i, seed in enumerate(SEEDS):
        # Treinar o perceptron com esta seed
        w_ini, w_fim, epocas, hist = treinar_perceptron(X_treino, d_treino, seed)

        # Guardar resultados
        resultados_treino.append({
            'treinamento': i + 1,
            'seed': seed,
            'w_iniciais': w_ini,
            'w_finais': w_fim,
            'epocas': epocas,
            'historico_erros': hist,
        })

        # Exibir detalhes de cada treinamento
        print(f"\n  ── Treinamento T{i+1} (seed = {seed}) {'─' * 40}")
        print(f"    Pesos Iniciais:  w0 = {w_ini[0]:.4f}   "
              f"w1 = {w_ini[1]:.4f}   w2 = {w_ini[2]:.4f}   w3 = {w_ini[3]:.4f}")
        print(f"    Pesos Finais:    w0 = {w_fim[0]:.4f}   "
              f"w1 = {w_fim[1]:.4f}   w2 = {w_fim[2]:.4f}   w3 = {w_fim[3]:.4f}")
        print(f"    Épocas até convergir: {epocas}")

    # Tabela resumo
    print(f"\n{'─' * 100}")
    print(f"  {'Treino':^8}│{'Pesos Iniciais':^44}│{'Pesos Finais':^44}│{'Épocas':^8}")
    print(f"  {'':^8}│{'w0':^11}{'w1':^11}{'w2':^11}{'w3':^11}│"
          f"{'w0':^11}{'w1':^11}{'w2':^11}{'w3':^11}│{'':^8}")
    print(f"{'─' * 100}")
    for r in resultados_treino:
        wi = r['w_iniciais']
        wf = r['w_finais']
        print(f"  {'T'+str(r['treinamento']):^8}│"
              f"{wi[0]:^11.4f}{wi[1]:^11.4f}{wi[2]:^11.4f}{wi[3]:^11.4f}│"
              f"{wf[0]:^11.4f}{wf[1]:^11.4f}{wf[2]:^11.4f}{wf[3]:^11.4f}│"
              f"{r['epocas']:^8}")
    print(f"{'─' * 100}")

    # ─────────────────────────────────────────────────────────────────────
    # ITEM 3: CLASSIFICAÇÃO DAS AMOSTRAS DE TESTE
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  📋 ITEM 3 — Classificação das Amostras de Teste")
    print(f"{'=' * 72}")

    # Classificar com cada modelo treinado
    classificacoes = []
    for r in resultados_treino:
        cls = classificar(r['w_finais'], X_teste)
        classificacoes.append(cls)

    # Cabeçalho da tabela
    print(f"\n  {'Amostra':^8}│{'x1':^10}{'x2':^10}{'x3':^10}│"
          f"{'T1':^7}{'T2':^7}{'T3':^7}{'T4':^7}{'T5':^7}")
    print(f"{'─' * 85}")

    for j in range(len(AMOSTRAS_TESTE)):
        x1, x2, x3 = AMOSTRAS_TESTE[j]
        linha = f"  {j+1:^8}│{x1:^10.4f}{x2:^10.4f}{x3:^10.4f}│"
        for i in range(5):
            val = classificacoes[i][j]
            classe = "C1" if val == -1 else "C2"
            linha += f"{classe:^7}"
        print(linha)
    print(f"{'─' * 85}")

    # Legenda
    print(f"\n  Legenda: C1 = Classe 1 (d = -1) | C2 = Classe 2 (d = +1)")

    # Verificar concordância entre treinamentos
    print(f"\n  📊 Análise de concordância:")
    for j in range(len(AMOSTRAS_TESTE)):
        votos = [classificacoes[i][j] for i in range(5)]
        if len(set(votos)) == 1:
            classe = "C1" if votos[0] == -1 else "C2"
            print(f"    Amostra {j+1:>2}: UNÂNIME → {classe} (5/5 treinamentos concordam)")
        else:
            n_c1 = votos.count(-1)
            n_c2 = votos.count(1)
            maioria = "C1" if n_c1 > n_c2 else "C2"
            print(f"    Amostra {j+1:>2}: DIVERGÊNCIA → maioria {maioria} "
                  f"(C1: {n_c1}/5, C2: {n_c2}/5)")

    # ─────────────────────────────────────────────────────────────────────
    # ITEM 4: EXPLICAÇÃO TEÓRICA
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  📋 ITEM 4 — Por que o número de épocas varia?")
    print(f"{'=' * 72}")
    print("""
    O número de épocas varia porque os PESOS INICIAIS são diferentes em
    cada treinamento (inicializados aleatoriamente com seeds distintas).

    Os pesos iniciais determinam o PONTO DE PARTIDA do algoritmo no
    espaço de soluções possíveis:

    • Se os pesos iniciais já estão PRÓXIMOS dos valores ideais,
      o perceptron precisa de POUCAS épocas para ajustá-los.

    • Se os pesos iniciais estão DISTANTES, o algoritmo precisa de
      MAIS épocas para gradualmente corrigir os pesos.

    Analogia: É como procurar a saída de um labirinto. Dependendo de
    onde você começa (pesos iniciais), o caminho pode ser mais curto
    ou mais longo, mas a saída (convergência) é a mesma.

    O Teorema da Convergência do Perceptron garante que, para dados
    linearmente separáveis, o algoritmo SEMPRE converge — independente
    dos pesos iniciais — mas não garante em QUANTAS épocas.
    """)

    # Comparação dos treinamentos
    epocas_lista = [r['epocas'] for r in resultados_treino]
    print(f"    Neste experimento:")
    for r in resultados_treino:
        barra = "█" * (r['epocas'] * 2)
        print(f"      T{r['treinamento']} (seed={r['seed']}): {r['epocas']:>3} épocas  {barra}")
    print(f"\n    Variação: {min(epocas_lista)} a {max(epocas_lista)} épocas "
          f"(diferença de {max(epocas_lista) - min(epocas_lista)} épocas)")

    # ─────────────────────────────────────────────────────────────────────
    # ITEM 5: LIMITAÇÃO DO PERCEPTRON
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  📋 ITEM 5 — Principal limitação do Perceptron")
    print(f"{'=' * 72}")
    print("""
    A principal limitação do Perceptron é que ele SOMENTE consegue
    resolver problemas de classificação LINEARMENTE SEPARÁVEIS.

    ┌─────────────────────────────────────────────────────────────┐
    │  LINEARMENTE SEPARÁVEL          NÃO LINEARMENTE SEPARÁVEL  │
    │                                                             │
    │       ✕ ✕ ✕ │ ● ● ●                  ✕ ● ✕                │
    │     ✕ ✕     │   ● ●                ● ✕ ● ✕               │
    │       ✕     │ ● ●                    ● ✕ ●                │
    │             │                                               │
    │   Uma linha separa as       Nenhuma linha reta consegue     │
    │   duas classes ✓            separar as classes ✗            │
    └─────────────────────────────────────────────────────────────┘

    Exemplo clássico: o problema XOR (OU-Exclusivo)

        x1  x2  │  Saída            Não existe nenhuma linha
        ────────┼──────            reta que separe as saídas
         0   0  │   0  (✕)          0 e 1 neste problema.
         0   1  │   1  (●)
         1   0  │   1  (●)          O perceptron ficaria em
         1   1  │   0  (✕)          loop INFINITO tentando.

    Para resolver problemas não linearmente separáveis, são
    necessárias redes com MÚLTIPLAS CAMADAS, como o MLP
    (Multilayer Perceptron), que introduz camadas ocultas
    com funções de ativação não-lineares.
    """)

    # ─────────────────────────────────────────────────────────────────────
    # GRÁFICOS
    # ─────────────────────────────────────────────────────────────────────
    print(f"{'=' * 72}")
    print("  📊 Gerando gráficos...")
    print(f"{'=' * 72}")

    # Gráfico 1: Evolução dos erros
    caminho1 = plotar_evolucao_erros(resultados_treino)
    print(f"    ✓ Evolução dos erros:       {caminho1}")

    # Gráfico 2: Dispersão 3D
    caminho2 = plotar_dispersao_3d(X_treino, d_treino, resultados_treino)
    print(f"    ✓ Dispersão 3D:             {caminho2}")

    # Gráfico 3: Classificação das amostras de teste
    caminho3 = plotar_comparacao_classificacoes(classificacoes, AMOSTRAS_TESTE)
    print(f"    ✓ Classificação de teste:   {caminho3}")

    # ─────────────────────────────────────────────────────────────────────
    # SALVAR RELATÓRIO EM ARQUIVO TEXTO
    # ─────────────────────────────────────────────────────────────────────
    relatorio_path = os.path.join(RESULTADOS_DIR, 'relatorio.txt')
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write("  RELATÓRIO — PERCEPTRON — Classificação de Pureza de Óleo\n")
        f.write("  CEFET-MG Campus VIII – Varginha\n")
        f.write("  Lab. Inteligência Artificial — Prof. Lázaro Eduardo da Silva\n")
        f.write("=" * 72 + "\n\n")

        f.write("CONFIGURAÇÕES:\n")
        f.write(f"  Taxa de aprendizagem (η): {TAXA_APRENDIZAGEM}\n")
        f.write(f"  Bias (x0): {BIAS_INPUT}\n")
        f.write(f"  Função de ativação: Degrau Bipolar\n")
        f.write(f"  Seeds: {SEEDS}\n")
        f.write(f"  Dataset: {len(X_treino)} amostras (C1: {n_c1}, C2: {n_c2})\n\n")

        # Tabela de treinamentos
        f.write("─" * 100 + "\n")
        f.write("ITENS 1 e 2 — Resultados dos 5 Treinamentos\n")
        f.write("─" * 100 + "\n\n")

        for r in resultados_treino:
            wi = r['w_iniciais']
            wf = r['w_finais']
            f.write(f"  Treinamento T{r['treinamento']} (seed={r['seed']}):\n")
            f.write(f"    Pesos Iniciais: w0={wi[0]:.4f}  w1={wi[1]:.4f}  "
                    f"w2={wi[2]:.4f}  w3={wi[3]:.4f}\n")
            f.write(f"    Pesos Finais:   w0={wf[0]:.4f}  w1={wf[1]:.4f}  "
                    f"w2={wf[2]:.4f}  w3={wf[3]:.4f}\n")
            f.write(f"    Épocas: {r['epocas']}\n\n")

        # Tabela de classificação
        f.write("─" * 85 + "\n")
        f.write("ITEM 3 — Classificação das Amostras de Teste\n")
        f.write("─" * 85 + "\n\n")

        f.write(f"  {'Amostra':^8} {'x1':^10} {'x2':^10} {'x3':^10} "
                f"{'T1':^6} {'T2':^6} {'T3':^6} {'T4':^6} {'T5':^6}\n")
        for j in range(len(AMOSTRAS_TESTE)):
            x1, x2, x3 = AMOSTRAS_TESTE[j]
            linha = f"  {j+1:^8} {x1:^10.4f} {x2:^10.4f} {x3:^10.4f} "
            for i in range(5):
                val = classificacoes[i][j]
                classe = "C1" if val == -1 else "C2"
                linha += f"{classe:^6} "
            f.write(linha + "\n")

        f.write(f"\n\n{'─' * 72}\n")
        f.write("ITEM 4 — Por que o número de épocas varia?\n")
        f.write("─" * 72 + "\n")
        f.write("O número de épocas varia porque os pesos iniciais são diferentes\n")
        f.write("em cada treinamento. Pesos iniciais diferentes significam pontos\n")
        f.write("de partida diferentes no espaço de busca, resultando em caminhos\n")
        f.write("mais curtos ou mais longos até a convergência.\n\n")

        f.write(f"{'─' * 72}\n")
        f.write("ITEM 5 — Principal limitação do Perceptron\n")
        f.write("─" * 72 + "\n")
        f.write("O perceptron só classifica padrões linearmente separáveis.\n")
        f.write("Para problemas não linearmente separáveis (como XOR), o perceptron\n")
        f.write("nunca converge. São necessárias redes multicamadas (MLP).\n")

    print(f"\n    ✓ Relatório salvo:          {relatorio_path}")

    # ─────────────────────────────────────────────────────────────────────
    # FINALIZAÇÃO
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  ✅ Execução concluída com sucesso!")
    print(f"{'=' * 72}")
    print(f"\n  Arquivos gerados em: perceptron/resultados/")
    print(f"    • evolucao_erros.png      — Gráfico de erros por época")
    print(f"    • dispersao_3d.png        — Dispersão 3D com hiperplano")
    print(f"    • classificacao_teste.png — Mapa de classificações")
    print(f"    • relatorio.txt           — Relatório completo em texto")
    print()


# Ponto de entrada do programa
if __name__ == '__main__':
    main()

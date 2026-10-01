"""
=============================================================================
ADALINE — Classificação de Sinais para Válvulas A e B
=============================================================================
CEFET-MG Campus VIII – Varginha
Bacharelado em Sistemas de Informação
Disciplina: Lab. Inteligência Artificial
Professor: Lázaro Eduardo da Silva

Descrição:
    Implementação da rede ADALINE (Adaptive Linear Element) com a Regra
    Delta (LMS - Least Mean Squares) para classificação de sinais
    ruidosos que devem ser encaminhados para a válvula A ou B.

Arquitetura do neurônio:
    x0 = -1 (bias) ──(w0)──┐
    x1 ─────────────(w1)────┤
    x2 ─────────────(w2)────┼──► Σ ──► u ──► g(.) ──► y
    x3 ─────────────(w3)────┤         ↑
    x4 ─────────────(w4)────┘         │
                                      │    +
                                      └──── d (desejado)
                                      erro = d - u
                                      (ANTES da ativação!)

Diferença fundamental em relação ao Perceptron:
    ┌───────────────────────────────────────────────────────────────┐
    │  PERCEPTRON: erro = d - y (DEPOIS do degrau → discreto)      │
    │  ADALINE:    erro = d - u (ANTES do degrau → contínuo)       │
    │                                                               │
    │  Consequência: a superfície de erro da Adaline é PARABÓLICA  │
    │  com um ÚNICO mínimo global, garantindo que os pesos finais  │
    │  sejam SEMPRE os mesmos, independente da inicialização.      │
    └───────────────────────────────────────────────────────────────┘

Configurações:
    - Taxa de aprendizagem (η): 0.0025
    - Precisão (ε): 10⁻⁶
    - Função de ativação: Degrau Bipolar (apenas na operação)
    - Entrada de bias (x0): -1
    - Convenção: d = -1 (válvula A) | d = +1 (válvula B)
=============================================================================
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica (para salvar imagens)
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================

TAXA_APRENDIZAGEM = 0.0025  # η (eta) — menor que no Perceptron para estabilidade
PRECISAO = 1e-6             # ε (epsilon) — critério de parada baseado no EQM
MAX_EPOCAS = 50000          # Limite de segurança (Adaline pode precisar de muitas épocas)
SEEDS = [1, 2, 3, 4, 5]    # Seeds fixas para reprodutibilidade
BIAS_INPUT = -1             # Valor fixo da entrada x0 (bias)

# Caminhos de arquivos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_DIR = os.path.join(SCRIPT_DIR, 'resultados')
os.makedirs(RESULTADOS_DIR, exist_ok=True)


# =============================================================================
# CONJUNTO DE TREINAMENTO (35 amostras — do anexo do enunciado)
# =============================================================================
# Cada linha: [x1, x2, x3, x4, d]
# d = -1 → sinal para válvula A
# d = +1 → sinal para válvula B

DADOS_TREINAMENTO = [
    [ 0.4329, -1.3719,  0.7022, -0.8535,  1.0],
    [ 0.3024,  0.2286,  0.8630,  2.7909, -1.0],
    [ 0.1349, -0.6445,  1.0530,  0.5687, -1.0],
    [ 0.3374, -1.7163,  0.3670, -0.6283, -1.0],
    [ 1.1434, -0.0485,  0.6637,  1.2606,  1.0],
    [ 1.3749, -0.5071,  0.4464,  1.3009,  1.0],
    [ 0.7221, -0.7587,  0.7681, -0.5592,  1.0],
    [ 0.4403, -0.8072,  0.5154, -0.3129,  1.0],
    [-0.5231,  0.3548,  0.2538,  1.5776, -1.0],
    [ 0.3255, -2.0000,  0.7112, -1.1209,  1.0],
    [ 0.5824,  1.3915, -0.2291,  4.1735, -1.0],
    [ 0.1340,  0.6081,  0.4450,  3.2230, -1.0],
    [ 0.1480, -0.2988,  0.4778,  0.8649,  1.0],
    [ 0.7359,  0.1869, -0.0872,  2.3584,  1.0],
    [ 0.7115, -1.1469,  0.3394,  0.9573, -1.0],
    [ 0.8251, -1.2840,  0.8452,  1.2382, -1.0],
    [ 0.1569,  0.3712,  0.8825,  1.7633,  1.0],
    [ 0.0033,  0.6835,  0.5389,  2.8249, -1.0],
    [ 0.4243,  0.8313,  0.2634,  3.5855, -1.0],
    [ 1.0490,  0.1326,  0.9138,  1.9792,  1.0],
    [ 1.4276,  0.5331, -0.0145,  3.7286,  1.0],
    [ 0.5971,  1.4865,  0.2904,  4.6069, -1.0],
    [ 0.8475,  2.1479,  0.3179,  5.8235, -1.0],
    [ 1.3967, -0.4171,  0.6443,  1.3927,  1.0],
    [ 0.0044,  1.5378,  0.6099,  4.7755, -1.0],
    [ 0.2201, -0.5668,  0.0515,  0.7829,  1.0],
    [ 0.6300, -1.2480,  0.8591,  0.8093, -1.0],
    [-0.2479,  0.8960,  0.0547,  1.7381,  1.0],
    [-0.3088, -0.0929,  0.8659,  1.5483, -1.0],
    [-0.5180,  1.4974,  0.5453,  2.3993,  1.0],
    [ 0.6833,  0.8266,  0.0829,  2.8864,  1.0],
    [ 0.4353, -1.4066,  0.4207, -0.4879,  1.0],
    [-0.1069, -3.2329,  0.1856, -2.4572, -1.0],
    [ 0.4662,  0.6261,  0.7304,  3.4370, -1.0],
    [ 0.8298, -1.4089,  0.3119,  1.3235, -1.0],
]


# =============================================================================
# AMOSTRAS DE TESTE (15 amostras — fornecidas no enunciado, Item 4)
# =============================================================================

AMOSTRAS_TESTE = [
    [ 0.9694,  0.6909,  0.4334,  3.4965],   # Amostra 1
    [ 0.5427,  1.3832,  0.6390,  4.0352],   # Amostra 2
    [ 0.6081, -0.9196,  0.5925,  0.1016],   # Amostra 3
    [-0.1618,  0.4694,  0.2030,  3.0117],   # Amostra 4
    [ 0.1870, -0.2578,  0.6124,  1.7749],   # Amostra 5
    [ 0.4891, -0.5276,  0.4378,  0.6439],   # Amostra 6
    [ 0.3777,  2.0149,  0.7423,  3.3932],   # Amostra 7
    [ 1.1498, -0.4067,  0.2469,  1.5866],   # Amostra 8
    [ 0.9325,  1.0950,  1.0359,  3.3591],   # Amostra 9
    [ 0.5060,  1.3317,  0.9222,  3.7174],   # Amostra 10
    [ 0.0497, -2.0656,  0.6124, -0.6585],   # Amostra 11
    [ 0.4004,  3.5369,  0.9766,  5.3532],   # Amostra 12
    [-0.1874,  1.3343,  0.5374,  3.2189],   # Amostra 13
    [ 0.5060,  1.3317,  0.9222,  3.7174],   # Amostra 14
    [ 1.6375, -0.7911,  0.7537,  0.5515],   # Amostra 15
]


# =============================================================================
# FUNÇÕES DA ADALINE
# =============================================================================

def funcao_degrau(u):
    """
    Função de ativação: Degrau Bipolar.

    IMPORTANTE: Na Adaline, esta função é usada SOMENTE na fase de
    OPERAÇÃO (classificação), NÃO durante o treinamento!

    Durante o treinamento, o erro é calculado usando 'u' diretamente
    (saída linear contínua), não 'y' (saída após o degrau).

    Parâmetro:
        u (float): Potencial de ativação (soma ponderada)

    Retorna:
        float: +1.0 (válvula B) ou -1.0 (válvula A)
    """
    return 1.0 if u >= 0 else -1.0


def treinar_adaline(X, d, seed):
    """
    Treina a rede ADALINE usando a Regra Delta (LMS).

    ╔══════════════════════════════════════════════════════════════╗
    ║  DIFERENÇA FUNDAMENTAL EM RELAÇÃO AO PERCEPTRON             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Perceptron:  erro = d - y    (y = saída APÓS degrau)       ║
    ║  Adaline:     erro = d - u    (u = saída ANTES do degrau)   ║
    ║                                                              ║
    ║  Isso faz toda a diferença! O erro contínuo permite que      ║
    ║  a superfície de erro seja uma PARÁBOLA, com um ÚNICO        ║
    ║  mínimo global. O Gradiente Descendente caminha "morro       ║
    ║  abaixo" nesta parábola até encontrar o mínimo.              ║
    ╚══════════════════════════════════════════════════════════════╝

    Algoritmo:
        1. Inicializar pesos aleatórios
        2. Para cada época:
           a) Para cada amostra:
              - Calcular u = w · x (soma ponderada)
              - Calcular erro = d - u (ANTES da ativação!)
              - Atualizar pesos: w = w + η · erro · x
           b) Calcular EQM da época
        3. Parar quando |EQM_atual - EQM_anterior| ≤ ε

    Parâmetros:
        X    : np.array (N, 5) — Entradas com bias na 1ª coluna
        d    : np.array (N,)   — Saídas desejadas (-1 ou +1)
        seed : int              — Semente para reprodutibilidade

    Retorna:
        pesos_iniciais  : np.array (5,)  — Pesos antes do treinamento
        pesos_finais    : np.array (5,)  — Pesos após convergência
        num_epocas      : int            — Quantas épocas foram necessárias
        historico_eqm   : list[float]    — EQM de cada época
    """
    np.random.seed(seed)

    # PASSO 1: Inicializar 5 pesos aleatórios entre 0 e 1
    # pesos[0] = w0 (bias), pesos[1] = w1, ..., pesos[4] = w4
    pesos = np.random.rand(5)
    pesos_iniciais = pesos.copy()

    num_amostras = len(X)
    historico_eqm = []

    # EQM da "época anterior" — iniciar com valor grande
    eqm_anterior = float('inf')

    # PASSO 2: Loop de treinamento
    for epoca in range(1, MAX_EPOCAS + 1):
        soma_erros_quadrados = 0.0

        # Apresentar TODAS as amostras ao neurônio
        for i in range(num_amostras):
            # 2a. Calcular o POTENCIAL DE ATIVAÇÃO (saída linear)
            #     u = w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4
            u = np.dot(pesos, X[i])

            # 2b. Calcular o ERRO (ANTES da função de ativação!)
            #     Esta é a diferença fundamental da Adaline:
            #     usamos 'u' (contínuo), não 'y' (discreto)
            erro = d[i] - u

            # 2c. Atualizar pesos pela REGRA DELTA (LMS)
            #     w = w + η · (d - u) · x
            #
            #     Como o erro é contínuo, o ajuste é proporcional
            #     à distância do valor desejado — ajustes mais
            #     finos que no Perceptron.
            pesos = pesos + TAXA_APRENDIZAGEM * erro * X[i]

            # Acumular erro quadrático para calcular EQM
            soma_erros_quadrados += erro ** 2

        # 2d. Calcular o ERRO QUADRÁTICO MÉDIO (EQM) desta época
        #     EQM = (1/N) × Σ(d_k - u_k)²
        #
        #     O EQM mede o quão longe, em média, as saídas da rede
        #     estão dos valores desejados.
        eqm_atual = soma_erros_quadrados / num_amostras
        historico_eqm.append(eqm_atual)

        # PASSO 3: Critério de parada
        #     |EQM_atual - EQM_anterior| ≤ ε (precisão de 10⁻⁶)
        #
        #     Diferente do Perceptron (que para com erro ZERO),
        #     a Adaline para quando a VARIAÇÃO do EQM é tão
        #     pequena que não vale mais a pena continuar.
        #
        #     Isso é necessário porque trabalhamos com valores
        #     contínuos — nunca chegaríamos a erro EXATAMENTE zero.
        if abs(eqm_atual - eqm_anterior) <= PRECISAO:
            break

        eqm_anterior = eqm_atual

    return pesos_iniciais, pesos.copy(), epoca, historico_eqm


def classificar(pesos, X):
    """
    Classifica amostras usando pesos JÁ TREINADOS.

    Na fase de OPERAÇÃO da Adaline:
        1. Calcula u = w · x (soma ponderada)
        2. Aplica degrau bipolar → y = +1 ou -1
        3. Retorna a classe

    Nota: o degrau SÓ é usado aqui, na classificação final.
    Durante o treinamento, usa-se apenas 'u' (valor contínuo).

    Parâmetros:
        pesos : np.array (5,)   — Vetor de pesos treinado
        X     : np.array (N, 5) — Amostras com bias na 1ª coluna

    Retorna:
        list[float]: +1.0 (válvula B) ou -1.0 (válvula A) para cada amostra
    """
    resultados = []
    for entrada in X:
        u = np.dot(pesos, entrada)
        y = funcao_degrau(u)
        resultados.append(y)
    return resultados


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO (GRÁFICOS)
# =============================================================================

def plotar_eqm_dois_treinamentos(resultado_t1, resultado_t2):
    """
    Gera gráfico com curvas de EQM × Épocas para os 2 primeiros
    treinamentos, plotados na MESMA figura (conforme o enunciado).

    A curva de EQM mostra como o erro vai diminuindo ao longo do
    treinamento — a rede vai "aprendendo" gradualmente.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    # --- Treinamento 1 ---
    epocas_t1 = range(1, len(resultado_t1['historico_eqm']) + 1)
    ax1.plot(
        epocas_t1, resultado_t1['historico_eqm'],
        color='#E74C3C', linewidth=2, alpha=0.9
    )
    ax1.set_ylabel('EQM', fontsize=12, fontweight='bold')
    ax1.set_title(
        f"Treinamento T1 (seed={resultado_t1['seed']}) — "
        f"{resultado_t1['epocas']} épocas",
        fontsize=13, fontweight='bold'
    )
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=10)

    # Anotar EQM final
    eqm_final_t1 = resultado_t1['historico_eqm'][-1]
    ax1.axhline(y=eqm_final_t1, color='#E74C3C', linestyle=':', alpha=0.4)
    ax1.text(
        len(resultado_t1['historico_eqm']) * 0.7, eqm_final_t1 * 1.3,
        f'EQM final = {eqm_final_t1:.6f}',
        fontsize=10, color='#E74C3C', fontweight='bold'
    )

    # --- Treinamento 2 ---
    epocas_t2 = range(1, len(resultado_t2['historico_eqm']) + 1)
    ax2.plot(
        epocas_t2, resultado_t2['historico_eqm'],
        color='#3498DB', linewidth=2, alpha=0.9
    )
    ax2.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax2.set_ylabel('EQM', fontsize=12, fontweight='bold')
    ax2.set_title(
        f"Treinamento T2 (seed={resultado_t2['seed']}) — "
        f"{resultado_t2['epocas']} épocas",
        fontsize=13, fontweight='bold'
    )
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=10)

    # Anotar EQM final
    eqm_final_t2 = resultado_t2['historico_eqm'][-1]
    ax2.axhline(y=eqm_final_t2, color='#3498DB', linestyle=':', alpha=0.4)
    ax2.text(
        len(resultado_t2['historico_eqm']) * 0.7, eqm_final_t2 * 1.3,
        f'EQM final = {eqm_final_t2:.6f}',
        fontsize=10, color='#3498DB', fontweight='bold'
    )

    fig.suptitle(
        'ADALINE — Erro Quadrático Médio (EQM) × Épocas',
        fontsize=15, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    caminho = os.path.join(RESULTADOS_DIR, 'eqm_treinamentos.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    return caminho


def plotar_eqm_todos(resultados_treino):
    """
    Gráfico adicional: EQM de todos os 5 treinamentos sobrepostos.
    Mostra visualmente que todos convergem para o MESMO valor de EQM.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    cores = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    for i, r in enumerate(resultados_treino):
        epocas = range(1, len(r['historico_eqm']) + 1)
        ax.plot(
            epocas, r['historico_eqm'],
            color=cores[i], linewidth=2, alpha=0.8,
            label=f"T{r['treinamento']} (seed={r['seed']}) — {r['epocas']} épocas"
        )

    ax.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax.set_ylabel('EQM', fontsize=13, fontweight='bold')
    ax.set_title(
        'ADALINE — Convergência do EQM nos 5 Treinamentos',
        fontsize=15, fontweight='bold', pad=15
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    caminho = os.path.join(RESULTADOS_DIR, 'eqm_todos_treinamentos.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    return caminho


def plotar_comparacao_classificacoes(classificacoes, amostras):
    """
    Gera heatmap mostrando as classificações das 15 amostras de teste
    por cada um dos 5 treinamentos.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Montar matriz (15 amostras × 5 treinamentos)
    matriz = np.array(classificacoes).T

    cores_mapa = plt.cm.colors.ListedColormap(['#E74C3C', '#3498DB'])
    limites = [-1.5, 0, 1.5]
    norma = plt.cm.colors.BoundaryNorm(limites, cores_mapa.N)

    im = ax.imshow(matriz, cmap=cores_mapa, norm=norma, aspect='auto')

    ax.set_xticks(range(5))
    ax.set_xticklabels([f'T{i+1}' for i in range(5)], fontsize=12, fontweight='bold')
    ax.set_yticks(range(15))
    ax.set_yticklabels([f'Amostra {i+1}' for i in range(15)], fontsize=10)
    ax.set_xlabel('Treinamento', fontsize=13, fontweight='bold')
    ax.set_title(
        'Classificação das Amostras de Teste por Treinamento',
        fontsize=14, fontweight='bold', pad=15
    )

    # Texto nas células
    for i in range(15):
        for j in range(5):
            val = matriz[i, j]
            classe = 'A' if val == -1 else 'B'
            ax.text(j, i, classe, ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')

    # Legenda
    from matplotlib.patches import Patch
    legenda = [
        Patch(facecolor='#E74C3C', label='Válvula A (d = -1)'),
        Patch(facecolor='#3498DB', label='Válvula B (d = +1)')
    ]
    ax.legend(handles=legenda, loc='upper right', fontsize=11,
              bbox_to_anchor=(1.28, 1.0))

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
    Função principal que executa todo o trabalho da Adaline:
        1. Prepara os dados
        2. Treina a Adaline 5 vezes
        3. Gera gráficos de EQM
        4. Classifica amostras de teste
        5. Apresenta respostas teóricas
    """

    # ─────────────────────────────────────────────────────────────────────
    # CABEÇALHO
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  ADALINE — Classificação de Sinais para Válvulas A e B")
    print("  CEFET-MG Campus VIII – Varginha | Lab. Inteligência Artificial")
    print("=" * 72)
    print(f"\n   Configurações:")
    print(f"    • Taxa de aprendizagem (η):  {TAXA_APRENDIZAGEM}")
    print(f"    • Precisão (ε):              {PRECISAO}")
    print(f"    • Bias (x0):                 {BIAS_INPUT}")
    print(f"    • Função de ativação:         Degrau Bipolar (somente na operação)")
    print(f"    • Máximo de épocas:           {MAX_EPOCAS}")
    print(f"    • Seeds (reprodutibilidade):  {SEEDS}")

    # ─────────────────────────────────────────────────────────────────────
    # PREPARAR DADOS
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("   Preparando dados...")

    # Montar conjunto de treinamento com bias
    dados = np.array(DADOS_TREINAMENTO)
    X_treino = np.column_stack([
        np.full(len(dados), BIAS_INPUT),  # x0 = -1 (bias)
        dados[:, :4]                       # x1, x2, x3, x4
    ])
    d_treino = dados[:, 4]  # Saída desejada

    n_valvA = int(np.sum(d_treino == -1))
    n_valvB = int(np.sum(d_treino == 1))
    print(f"     {len(X_treino)} amostras de treinamento")
    print(f"    • Válvula A (d = -1): {n_valvA} amostras")
    print(f"    • Válvula B (d = +1): {n_valvB} amostras")

    # Preparar amostras de teste com bias
    X_teste = np.array([[BIAS_INPUT] + amostra for amostra in AMOSTRAS_TESTE])

    # ─────────────────────────────────────────────────────────────────────
    # ITENS 1 e 2: EXECUTAR 5 TREINAMENTOS
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("   ITENS 1 e 2 — Resultados dos 5 Treinamentos")
    print(f"{'=' * 72}")

    resultados_treino = []

    for i, seed in enumerate(SEEDS):
        w_ini, w_fim, epocas, hist_eqm = treinar_adaline(X_treino, d_treino, seed)

        resultados_treino.append({
            'treinamento': i + 1,
            'seed': seed,
            'w_iniciais': w_ini,
            'w_finais': w_fim,
            'epocas': epocas,
            'historico_eqm': hist_eqm,
        })

        print(f"\n  ── Treinamento T{i+1} (seed = {seed}) {'─' * 40}")
        print(f"    Pesos Iniciais:  w0={w_ini[0]:.4f}  w1={w_ini[1]:.4f}  "
              f"w2={w_ini[2]:.4f}  w3={w_ini[3]:.4f}  w4={w_ini[4]:.4f}")
        print(f"    Pesos Finais:    w0={w_fim[0]:.4f}  w1={w_fim[1]:.4f}  "
              f"w2={w_fim[2]:.4f}  w3={w_fim[3]:.4f}  w4={w_fim[4]:.4f}")
        print(f"    Épocas: {epocas}")
        print(f"    EQM final: {hist_eqm[-1]:.8f}")

    # Tabela resumo
    print(f"\n{'─' * 120}")
    print(f"  {'Treino':^8}│{'Pesos Iniciais':^55}│{'Pesos Finais':^55}│{'Épocas':^8}")
    print(f"  {'':^8}│{'w0':^11}{'w1':^11}{'w2':^11}{'w3':^11}{'w4':^11}│"
          f"{'w0':^11}{'w1':^11}{'w2':^11}{'w3':^11}{'w4':^11}│{'':^8}")
    print(f"{'─' * 120}")
    for r in resultados_treino:
        wi = r['w_iniciais']
        wf = r['w_finais']
        print(f"  {'T'+str(r['treinamento']):^8}│"
              f"{wi[0]:^11.4f}{wi[1]:^11.4f}{wi[2]:^11.4f}{wi[3]:^11.4f}{wi[4]:^11.4f}│"
              f"{wf[0]:^11.4f}{wf[1]:^11.4f}{wf[2]:^11.4f}{wf[3]:^11.4f}{wf[4]:^11.4f}│"
              f"{r['epocas']:^8}")
    print(f"{'─' * 120}")

    # ─────────────────────────────────────────────────────────────────────
    # ITEM 3: GRÁFICOS DE EQM (T1 e T2 na mesma folha)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("   ITEM 3 — Gráficos de EQM × Épocas")
    print(f"{'=' * 72}")

    caminho1 = plotar_eqm_dois_treinamentos(resultados_treino[0], resultados_treino[1])
    print(f"     EQM (T1 e T2):            {caminho1}")

    caminho2 = plotar_eqm_todos(resultados_treino)
    print(f"     EQM (todos sobrepostos):   {caminho2}")

    # ─────────────────────────────────────────────────────────────────────
    # ITEM 4: CLASSIFICAÇÃO DAS AMOSTRAS DE TESTE
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("   ITEM 4 — Classificação das Amostras de Teste")
    print(f"{'=' * 72}")

    classificacoes = []
    for r in resultados_treino:
        cls = classificar(r['w_finais'], X_teste)
        classificacoes.append(cls)

    # Cabeçalho da tabela
    print(f"\n  {'Amostra':^8}│{'x1':^10}{'x2':^10}{'x3':^10}{'x4':^10}│"
          f"{'T1':^7}{'T2':^7}{'T3':^7}{'T4':^7}{'T5':^7}")
    print(f"{'─' * 95}")

    for j in range(len(AMOSTRAS_TESTE)):
        x1, x2, x3, x4 = AMOSTRAS_TESTE[j]
        linha = f"  {j+1:^8}│{x1:^10.4f}{x2:^10.4f}{x3:^10.4f}{x4:^10.4f}│"
        for i in range(5):
            val = classificacoes[i][j]
            valvula = "A" if val == -1 else "B"
            linha += f"{valvula:^7}"
        print(linha)
    print(f"{'─' * 95}")

    print(f"\n  Legenda: A = Válvula A (d = -1) | B = Válvula B (d = +1)")

    # Análise de concordância
    print(f"\n   Análise de concordância:")
    for j in range(len(AMOSTRAS_TESTE)):
        votos = [classificacoes[i][j] for i in range(5)]
        if len(set(votos)) == 1:
            valvula = "A" if votos[0] == -1 else "B"
            print(f"    Amostra {j+1:>2}: UNÂNIME → Válvula {valvula} "
                  f"(5/5 treinamentos concordam)")
        else:
            n_a = votos.count(-1)
            n_b = votos.count(1)
            maioria = "A" if n_a > n_b else "B"
            print(f"    Amostra {j+1:>2}: DIVERGÊNCIA → maioria Válvula {maioria} "
                  f"(A: {n_a}/5, B: {n_b}/5)")

    # Gráfico de classificação
    caminho3 = plotar_comparacao_classificacoes(classificacoes, AMOSTRAS_TESTE)
    print(f"\n     Heatmap classificações:    {caminho3}")

    # ─────────────────────────────────────────────────────────────────────
    # ITEM 5: EXPLICAÇÃO TEÓRICA
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("   ITEM 5 — Por que os pesos finais são praticamente iguais?")
    print(f"{'=' * 72}")
    print("""
    Embora o número de épocas varie entre os treinamentos (porque os
    pesos iniciais são diferentes), os PESOS FINAIS são praticamente
    IDÊNTICOS. Isso acontece por causa da natureza matemática da Adaline:

    ┌─────────────────────────────────────────────────────────────────┐
    │  A Adaline minimiza o Erro Quadrático Médio (EQM) usando o     │
    │  GRADIENTE DESCENDENTE. A função de custo EQM é QUADRÁTICA     │
    │  em relação aos pesos, formando uma superfície PARABÓLICA      │
    │  (um parabolóide em dimensões superiores).                     │
    │                                                                 │
    │  Uma parábola tem um ÚNICO PONTO DE MÍNIMO GLOBAL.            │
    │                                                                 │
    │        EQM                                                     │
    │        ▲                                                       │
    │       ╱ ╲         Não importa de onde se parte (pesos          │
    │      ╱   ╲        iniciais), o gradiente SEMPRE leva           │
    │     ╱  ●  ╲       ao mesmo ponto de mínimo.                   │
    │    ╱       ╲                                                   │
    │   ╱    ▼    ╲     ● = mínimo global (único!)                  │
    │  ╱─────●─────╲                                                │
    │ ╱              ╲                                               │
    │╱________________╲▶ Pesos (w)                                  │
    └─────────────────────────────────────────────────────────────────┘

    Diferente do Perceptron (que para na PRIMEIRA solução que encontra,
    podendo ser qualquer reta válida), a Adaline converge para A MESMA
    solução ótima — a que MINIMIZA o erro quadrático.

    A pequena variação no número de épocas acontece porque caminhos
    diferentes na superfície de erro (partindo de pontos iniciais
    diferentes) têm comprimentos diferentes, mas TODOS chegam ao
    mesmo destino: o fundo da parábola.

    Isso torna a Adaline mais ROBUSTA que o Perceptron, pois seu
    hiperplano de separação fica posicionado no CENTRO da região
    de separabilidade, equidistante das duas classes.
    """)

    # Comparação numérica dos pesos finais
    print("     Comparação numérica dos pesos finais:")
    print(f"    {'Treino':^8} {'w0':^12} {'w1':^12} {'w2':^12} {'w3':^12} {'w4':^12}")
    print(f"    {'─' * 68}")
    for r in resultados_treino:
        wf = r['w_finais']
        print(f"    {'T'+str(r['treinamento']):^8} "
              f"{wf[0]:^12.6f} {wf[1]:^12.6f} {wf[2]:^12.6f} "
              f"{wf[3]:^12.6f} {wf[4]:^12.6f}")
    print(f"    {'─' * 68}")

    # Calcular variação máxima em cada peso
    print(f"\n    Variação máxima entre treinamentos:")
    pesos_todos = np.array([r['w_finais'] for r in resultados_treino])
    for j in range(5):
        var = pesos_todos[:, j].max() - pesos_todos[:, j].min()
        print(f"      w{j}: variação = {var:.6f}")

    # ─────────────────────────────────────────────────────────────────────
    # SALVAR RELATÓRIO EM ARQUIVO TEXTO
    # ─────────────────────────────────────────────────────────────────────
    relatorio_path = os.path.join(RESULTADOS_DIR, 'relatorio.txt')
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write("  RELATÓRIO — ADALINE — Classificação de Sinais (Válvulas A/B)\n")
        f.write("  CEFET-MG Campus VIII – Varginha\n")
        f.write("  Lab. Inteligência Artificial — Prof. Lázaro Eduardo da Silva\n")
        f.write("=" * 72 + "\n\n")

        f.write("CONFIGURAÇÕES:\n")
        f.write(f"  Taxa de aprendizagem (η): {TAXA_APRENDIZAGEM}\n")
        f.write(f"  Precisão (ε): {PRECISAO}\n")
        f.write(f"  Bias (x0): {BIAS_INPUT}\n")
        f.write(f"  Função de ativação: Degrau Bipolar (apenas na operação)\n")
        f.write(f"  Seeds: {SEEDS}\n")
        f.write(f"  Dataset: {len(X_treino)} amostras "
                f"(Válv.A: {n_valvA}, Válv.B: {n_valvB})\n\n")

        # Tabela de treinamentos
        f.write("─" * 100 + "\n")
        f.write("ITENS 1 e 2 — Resultados dos 5 Treinamentos\n")
        f.write("─" * 100 + "\n\n")

        for r in resultados_treino:
            wi = r['w_iniciais']
            wf = r['w_finais']
            f.write(f"  Treinamento T{r['treinamento']} (seed={r['seed']}):\n")
            f.write(f"    Pesos Iniciais: w0={wi[0]:.4f}  w1={wi[1]:.4f}  "
                    f"w2={wi[2]:.4f}  w3={wi[3]:.4f}  w4={wi[4]:.4f}\n")
            f.write(f"    Pesos Finais:   w0={wf[0]:.4f}  w1={wf[1]:.4f}  "
                    f"w2={wf[2]:.4f}  w3={wf[3]:.4f}  w4={wf[4]:.4f}\n")
            f.write(f"    Épocas: {r['epocas']}\n")
            f.write(f"    EQM final: {r['historico_eqm'][-1]:.8f}\n\n")

        # Tabela de classificação
        f.write("─" * 95 + "\n")
        f.write("ITEM 4 — Classificação das Amostras de Teste\n")
        f.write("─" * 95 + "\n\n")

        f.write(f"  {'Amostra':^8} {'x1':^10} {'x2':^10} {'x3':^10} {'x4':^10} "
                f"{'T1':^6} {'T2':^6} {'T3':^6} {'T4':^6} {'T5':^6}\n")
        for j in range(len(AMOSTRAS_TESTE)):
            x1, x2, x3, x4 = AMOSTRAS_TESTE[j]
            linha = f"  {j+1:^8} {x1:^10.4f} {x2:^10.4f} {x3:^10.4f} {x4:^10.4f} "
            for i in range(5):
                val = classificacoes[i][j]
                valvula = "A" if val == -1 else "B"
                linha += f"{valvula:^6} "
            f.write(linha + "\n")

        f.write(f"\n\n{'─' * 72}\n")
        f.write("ITEM 5 — Por que os pesos finais são praticamente iguais?\n")
        f.write("─" * 72 + "\n")
        f.write("A Adaline minimiza o EQM via Gradiente Descendente. A superfície\n")
        f.write("de erro é parabólica com um ÚNICO mínimo global. Independente dos\n")
        f.write("pesos iniciais, o algoritmo sempre converge para os MESMOS pesos\n")
        f.write("finais. A variação no número de épocas ocorre porque caminhos\n")
        f.write("diferentes na superfície de erro têm comprimentos diferentes,\n")
        f.write("mas todos chegam ao mesmo ponto de mínimo.\n")

    print(f"\n     Relatório salvo: {relatorio_path}")

    # ─────────────────────────────────────────────────────────────────────
    # FINALIZAÇÃO
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("   Execução concluída com sucesso!")
    print(f"{'=' * 72}")
    print(f"\n  Arquivos gerados em: adaline/resultados/")
    print(f"    • eqm_treinamentos.png      — EQM de T1 e T2 (mesma folha)")
    print(f"    • eqm_todos_treinamentos.png — EQM dos 5 treinamentos")
    print(f"    • classificacao_teste.png    — Mapa de classificações")
    print(f"    • relatorio.txt              — Relatório completo em texto")
    print()


# Ponto de entrada do programa
if __name__ == '__main__':
    main()

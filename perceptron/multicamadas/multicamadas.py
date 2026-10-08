"""
=============================================================================
  CEFET-MG Campus VIII – Varginha | Bacharelado em Sistemas de Informação
  Disciplina: Laboratório de Inteligência Artificial
  Professor:  Lázaro Eduardo da Silva
  Aluno:      Caio Magalhães Fonseca Arantes
  Data:       08/10/2026
=============================================================================
  ATIVIDADE PRÁTICA: REDE PERCEPTRON MULTICAMADAS (MLP / BACKPROPAGATION)
  Aproximador Universal de Funções — Sistema de Ressonância Magnética
=============================================================================
  Contexto do Problema:
    Estimação da variável contínua {y} (energia absorvida pelo sistema de
    ressonância magnética) a partir de 3 grandezas físicas normalizadas {x1, x2, x3}.
    O mapeamento é de difícil modelagem matemática convencional, sendo
    solucionado via Perceptron Multicamadas (MLP).

  Topologia da Rede:
    - Camada de Entrada:   3 sinais {x1, x2, x3} + entrada fixa de bias x0 = -1
    - Camada Oculta:       10 neurônios com ativação Logística (Sigmoide)
    - Camada de Saída:     1 neurônio com ativação Logística (Sigmoide)
    - Conexões:            Feedforward padrão (camada a camada)

  Parâmetros do Algoritmo:
    - Taxa de aprendizagem (η): 0.1
    - Critério de parada (ε):    |EQM_atual - EQM_anterior| ≤ 10⁻⁶
    - Pesos iniciais:            Aleatórios uniformes entre 0 e 1 (5 seeds)
=============================================================================
"""

import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURAÇÕES E HIPERPARÂMETROS DO ENUNCIADO
# =============================================================================
TAXA_APRENDIZAGEM = 0.1      # η (eta)
PRECISAO = 1e-6              # ε (epsilon) — critério de parada baseado no EQM
MAX_EPOCAS = 10000           # Limite superior de segurança
SEEDS = [1, 2, 3, 4, 5]       # 5 treinamentos independentes (T1 a T5)
BIAS_INPUT = -1.0            # Convenção do professor: entrada de bias x0 = -1 e z0 = -1
N_ENTRADAS = 3               # x1, x2, x3
N_OCULTOS = 10               # 10 neurônios na camada intermediária
N_SAIDAS = 1                 # 1 neurônio de saída (y)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_DIR = os.path.join(SCRIPT_DIR, 'resultados')
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Caminhos de busca dos dados de treinamento e validação
_caminhos_treino = [
    os.path.join(SCRIPT_DIR, '..', '..', 'atividades', 'perceptron_adaline', 'perceptron', 'multicamadas', 'conjunto_treinamento.csv'),
    os.path.join(SCRIPT_DIR, '..', 'atividades', 'perceptron_adaline', 'perceptron', 'multicamadas', 'conjunto_treinamento.csv'),
    os.path.join(SCRIPT_DIR, 'conjunto_treinamento.csv'),
    os.path.join(os.getcwd(), 'atividades', 'perceptron_adaline', 'perceptron', 'multicamadas', 'conjunto_treinamento.csv'),
    os.path.join(os.getcwd(), 'conjunto_treinamento.csv'),
    'conjunto_treinamento.csv'
]
TREINO_PATH = next((p for p in _caminhos_treino if os.path.exists(p)), _caminhos_treino[0])

_caminhos_val = [
    os.path.join(SCRIPT_DIR, '..', '..', 'atividades', 'perceptron_adaline', 'perceptron', 'multicamadas', 'conjunto_validacao.csv'),
    os.path.join(SCRIPT_DIR, '..', 'atividades', 'perceptron_adaline', 'perceptron', 'multicamadas', 'conjunto_validacao.csv'),
    os.path.join(SCRIPT_DIR, 'conjunto_validacao.csv'),
    os.path.join(os.getcwd(), 'atividades', 'perceptron_adaline', 'perceptron', 'multicamadas', 'conjunto_validacao.csv'),
    os.path.join(os.getcwd(), 'conjunto_validacao.csv'),
    'conjunto_validacao.csv'
]
VAL_PATH = next((p for p in _caminhos_val if os.path.exists(p)), _caminhos_val[0])


# =============================================================================
# FUNÇÕES MATEMÁTICAS E AUXILIARES DA MLP
# =============================================================================

def sigmoide(u):
    """
    Função de ativação logística:
        g(u) = 1 / (1 + e^(-u))
    Mapeia (-inf, +inf) para o intervalo aberto (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))


def derivada_sigmoide_pela_saida(y):
    """
    Derivada da função logística expressa em termos da própria ativação:
        g'(u) = g(u) * (1 - g(u)) = y * (1 - y)
    """
    return y * (1.0 - y)


def carregar_csv(caminho):
    """
    Carrega o dataset em CSV contendo colunas Amostra, x1, x2, x3, d.
    Retorna:
        X_com_bias: np.array (P, 4) com x0 = -1 na primeira coluna
        d: np.array (P,) saídas desejadas
        amostras_id: lista de identificadores das amostras
    """
    X_raw, d_list, ids = [], [], []
    with open(caminho, 'r', encoding='utf-8') as f:
        leitor = csv.DictReader(f)
        for linha in leitor:
            ids.append(int(linha['Amostra']))
            X_raw.append([float(linha['x1']), float(linha['x2']), float(linha['x3'])])
            d_list.append(float(linha['d']))

    X_mat = np.array(X_raw)
    d_mat = np.array(d_list)
    P = len(d_mat)
    # Adicionar bias x0 = -1 como primeira coluna
    X_com_bias = np.hstack([np.full((P, 1), BIAS_INPUT), X_mat])
    return X_com_bias, d_mat, ids


def inicializar_pesos(seed):
    """
    Inicializa as matrizes de pesos com valores aleatórios entre 0 e 1,
    conforme exigido explicitamente no Item 1 do trabalho.

    Dimensões:
        - W1 (camada oculta): (10, 4) -> 10 neurônios, 4 entradas cada (x0, x1, x2, x3)
        - W2 (camada de saída): (1, 11) -> 1 neurônio, 11 entradas cada (z0, z1...z10)
    """
    np.random.seed(seed)
    W1 = np.random.uniform(0.0, 1.0, (N_OCULTOS, N_ENTRADAS + 1))
    W2 = np.random.uniform(0.0, 1.0, (N_SAIDAS, N_OCULTOS + 1))
    return W1, W2


def treinar_mlp(X, d, seed, eta=TAXA_APRENDIZAGEM, precisao=PRECISAO, max_epocas=MAX_EPOCAS):
    """
    Treinamento supervisionado via Backpropagation (Regra Delta Generalizada).
    Atualização padrão a padrão (estocástica/online) com cálculo do EQM por época.

    Critério de Parada:
        |EQM_atual - EQM_anterior| ≤ ε (10⁻⁶)
    """
    W1, W2 = inicializar_pesos(seed)
    W1_inicial = W1.copy()
    W2_inicial = W2.copy()
    
    P = len(d)
    historico_eqm = []
    eqm_anterior = 1e9

    for epoca in range(1, max_epocas + 1):
        soma_erros_quadrados = 0.0

        for p in range(P):
            x_p = X[p]  # vetor (4,): [x0=-1, x1, x2, x3]
            d_p = d[p]  # escalar desejado

            # -------------------------------------------------------------
            # 1. PASSO FORWARD (Propagação Direta)
            # -------------------------------------------------------------
            # Camada Oculta:
            u_oculto = W1 @ x_p                           # (10,)
            z_oculto = sigmoide(u_oculto)                 # (10,)
            z_com_bias = np.concatenate([[BIAS_INPUT], z_oculto])  # (11,): [z0=-1, z1..z10]

            # Camada de Saída:
            u_saida = W2 @ z_com_bias                     # (1,)
            y_saida = sigmoide(u_saida)[0]                # escalar

            # Cálculo do erro:
            erro = d_p - y_saida
            soma_erros_quadrados += erro ** 2

            # -------------------------------------------------------------
            # 2. PASSO BACKWARD (Retropropagação do Erro)
            # -------------------------------------------------------------
            # Gradiente local da camada de saída:
            #   δ_saída = (d - y) * g'(u_saída) = erro * y * (1 - y)
            delta_saida = erro * derivada_sigmoide_pela_saida(y_saida)  # escalar

            # Gradiente local da camada oculta:
            #   δ_j = δ_saída * w2_1j * g'(u_j)
            #   W2[0, 1:] corresponde aos pesos associados a z1..z10 (ignorando bias z0)
            delta_oculto = delta_saida * W2[0, 1:] * derivada_sigmoide_pela_saida(z_oculto) # (10,)

            # -------------------------------------------------------------
            # 3. ATUALIZAÇÃO DOS PESOS (Regra Delta)
            # -------------------------------------------------------------
            # Camada de saída: ΔW2 = η * δ_saída * z
            W2 += eta * delta_saida * z_com_bias.reshape(1, -1)

            # Camada oculta: ΔW1 = η * δ_oculto * x
            W1 += eta * delta_oculto.reshape(-1, 1) @ x_p.reshape(1, -1)

        # EQM da época: (1/P) * Σ (d_p - y_p)²
        eqm_atual = soma_erros_quadrados / P
        historico_eqm.append(eqm_atual)

        # Verificação do critério de parada
        if abs(eqm_atual - eqm_anterior) <= precisao:
            break

        eqm_anterior = eqm_atual

    return {
        'seed': seed,
        'epocas': epoca,
        'eqm_final': eqm_atual,
        'historico_eqm': historico_eqm,
        'W1_inicial': W1_inicial,
        'W2_inicial': W2_inicial,
        'W1_final': W1,
        'W2_final': W2
    }


def predizer(X, W1, W2):
    """
    Gera as predições da rede neural para uma matriz de entrada X (com bias x0 = -1).
    """
    P = len(X)
    y_preds = np.zeros(P)
    for p in range(P):
        x_p = X[p]
        u_oculto = W1 @ x_p
        z_oculto = sigmoide(u_oculto)
        z_com_bias = np.concatenate([[BIAS_INPUT], z_oculto])
        u_saida = W2 @ z_com_bias
        y_preds[p] = sigmoide(u_saida)[0]
    return y_preds


# =============================================================================
# FLUXO PRINCIPAL DE EXECUÇÃO
# =============================================================================

def main():
    print("=" * 78)
    print("  CEFET-MG Campus VIII – Varginha | Lab. Inteligência Artificial")
    print("  PERCEPTRON MULTICAMADAS (MLP) — Estimador de Energia de Ressonância")
    print("  Aluno: Caio Magalhães Fonseca Arantes | Data: 08/10/2026")
    print("=" * 78)

    # Carregar dados
    print("\n[1] Carregando conjuntos de dados...")
    X_treino, d_treino, ids_treino = carregar_csv(TREINO_PATH)
    X_val, d_val, ids_val = carregar_csv(VAL_PATH)
    print(f"    • Treinamento: {len(d_treino)} amostras ({TREINO_PATH})")
    print(f"    • Validação:   {len(d_val)} amostras ({VAL_PATH})")

    # ─────────────────────────────────────────────────────────────────────────
    # ATIVIDADES 1 & 2: EXECUTAR OS 5 TREINAMENTOS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[2] Executando 5 treinamentos independentes (η={TAXA_APRENDIZAGEM}, ε={PRECISAO})...")
    resultados = []

    for idx, seed in enumerate(SEEDS, 1):
        res = treinar_mlp(X_treino, d_treino, seed=seed)
        res['treinamento'] = idx
        resultados.append(res)
        print(f"    -> T{idx} (seed={seed}): {res['epocas']} épocas | EQM final = {res['eqm_final']:.8f}")

    print("\n" + "=" * 78)
    print("  TABELA DO ITEM 2 — Resultados Finais dos 5 Treinamentos")
    print("=" * 78)
    print(f"  {'Treinamento':^16} │ {'Erro Quadrático Médio (EQM)':^30} │ {'Número de Épocas':^20}")
    print("  " + "─" * 16 + "┼" + "─" * 32 + "┼" + "─" * 22)
    for r in resultados:
        print(f"   {r['treinamento']}º (T{r['treinamento']}) (seed={r['seed']})   │        {r['eqm_final']:.8f}        │        {r['epocas']:^6} épocas")
    print("  " + "─" * 72)

    # ─────────────────────────────────────────────────────────────────────────
    # ATIVIDADE 3: GRÁFICOS DOS 2 TREINAMENTOS COM MAIORES NÚMEROS DE ÉPOCAS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3] Gerando gráficos de convergência do EQM (Item 3)...")
    # Ordenar por número de épocas decrescente
    ordenados_epocas = sorted(resultados, key=lambda x: x['epocas'], reverse=True)
    top1 = ordenados_epocas[0]
    top2 = ordenados_epocas[1]

    print(f"    • 1º com mais épocas: T{top1['treinamento']} ({top1['epocas']} épocas)")
    print(f"    • 2º com mais épocas: T{top2['treinamento']} ({top2['epocas']} épocas)")

    # Gráfico dos 2 maiores - Subplots não superpostos em uma mesma folha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    # Treinamento Top 1
    epocas_top1 = range(1, len(top1['historico_eqm']) + 1)
    ax1.plot(epocas_top1, top1['historico_eqm'], color='#0066cc', linewidth=2.2, label=f"T{top1['treinamento']} (seed={top1['seed']})")
    ax1.set_title(f"Treinamento {top1['treinamento']}º (T{top1['treinamento']}) — {top1['epocas']} Épocas\nEQM Final: {top1['eqm_final']:.8f}", fontsize=11, fontweight='bold', pad=12)
    ax1.set_xlabel("Épocas de Treinamento", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Erro Quadrático Médio (EQM)", fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_yscale('log')

    # Treinamento Top 2
    epocas_top2 = range(1, len(top2['historico_eqm']) + 1)
    ax2.plot(epocas_top2, top2['historico_eqm'], color='#cc3300', linewidth=2.2, label=f"T{top2['treinamento']} (seed={top2['seed']})")
    ax2.set_title(f"Treinamento {top2['treinamento']}º (T{top2['treinamento']}) — {top2['epocas']} Épocas\nEQM Final: {top2['eqm_final']:.8f}", fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlabel("Épocas de Treinamento", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Erro Quadrático Médio (EQM)", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', frameon=True)
    ax2.set_yscale('log')

    plt.suptitle("CEFET-MG — Perceptron Multicamadas: Curvas de Convergência do EQM (Item 3)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    grafico_top2_path = os.path.join(RESULTADOS_DIR, 'convergencia_maiores_epocas.png')
    plt.savefig(grafico_top2_path, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Gráfico salvo: {grafico_top2_path}")

    # Gráfico comparativo de todos os 5 treinamentos
    plt.figure(figsize=(10, 5.5), dpi=300)
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, r in enumerate(resultados):
        plt.plot(range(1, len(r['historico_eqm']) + 1), r['historico_eqm'], label=f"T{r['treinamento']} ({r['epocas']} épocas - EQM: {r['eqm_final']:.6f})", color=cores[idx], linewidth=1.8)
    plt.title("Comparativo Geral de Convergência: Todos os 5 Treinamentos (T1 a T5)", fontsize=12, fontweight='bold')
    plt.xlabel("Épocas", fontsize=10, fontweight='bold')
    plt.ylabel("Erro Quadrático Médio (EQM)", fontsize=10, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    grafico_todos_path = os.path.join(RESULTADOS_DIR, 'convergencia_todos_treinamentos.png')
    plt.savefig(grafico_todos_path, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Gráfico geral salvo: {grafico_todos_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # ATIVIDADE 5: VALIDAÇÃO COM AS 20 AMOSTRAS DE TESTE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4] Validando com as 20 amostras de teste (Item 5)...")
    predicoes_todos = []
    erros_rel_todos = []
    variancias_todos = []

    for r in resultados:
        y_val_pred = predizer(X_val, r['W1_final'], r['W2_final'])
        predicoes_todos.append(y_val_pred)
        
        # Erro relativo percentual individual: |d_k - y_k| / d_k * 100%
        erros_rel = (np.abs(d_val - y_val_pred) / d_val) * 100.0
        erm = np.mean(erros_rel)
        var_er = np.var(erros_rel)
        
        r['predicoes_val'] = y_val_pred
        r['erros_rel'] = erros_rel
        r['erm'] = erm
        r['var_erm'] = var_er
        erros_rel_todos.append(erm)
        variancias_todos.append(var_er)

    print("\n" + "=" * 90)
    print("  TABELA DO ITEM 5 — Validação das 20 Amostras do Conjunto de Teste")
    print("=" * 90)
    print(f" {'Amostra':^7} │ {'x1':^7} │ {'x2':^7} │ {'x3':^7} │ {'d':^7} │ {'yrede (T1)':^10} │ {'yrede (T2)':^10} │ {'yrede (T3)':^10} │ {'yrede (T4)':^10} │ {'yrede (T5)':^10}")
    print("─" * 9 + "┼" + "─" * 9 + "┼" + "─" * 9 + "┼" + "─" * 9 + "┼" + "─" * 9 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12)
    
    for i in range(len(d_val)):
        print(f"   {ids_val[i]:^5} │ {X_val[i, 1]:.4f} │ {X_val[i, 2]:.4f} │ {X_val[i, 3]:.4f} │ {d_val[i]:.4f} │ "
              f"{predicoes_todos[0][i]:^10.4f} │ {predicoes_todos[1][i]:^10.4f} │ {predicoes_todos[2][i]:^10.4f} │ {predicoes_todos[3][i]:^10.4f} │ {predicoes_todos[4][i]:^10.4f}")
    
    print("─" * 90)
    print(f" {'Erro Relativo Médio (%)':<43} │ "
          f"{resultados[0]['erm']:^10.2f}% │ {resultados[1]['erm']:^10.2f}% │ {resultados[2]['erm']:^10.2f}% │ {resultados[3]['erm']:^10.2f}% │ {resultados[4]['erm']:^10.2f}%")
    print(f" {'Variância (%)':<43} │ "
          f"{resultados[0]['var_erm']:^10.2f}% │ {resultados[1]['var_erm']:^10.2f}% │ {resultados[2]['var_erm']:^10.2f}% │ {resultados[3]['var_erm']:^10.2f}% │ {resultados[4]['var_erm']:^10.2f}%")
    print("=" * 90)

    # ─────────────────────────────────────────────────────────────────────────
    # ATIVIDADE 6: INDICAÇÃO DO MELHOR MODELO (GENERALIZAÇÃO)
    # ─────────────────────────────────────────────────────────────────────────
    # Critério: menor Erro Relativo Médio, com desempate pela menor Variância
    melhor_modelo = min(resultados, key=lambda x: (round(x['erm'], 4), round(x['var_erm'], 4)))
    print(f"\n[5] Conclusão do Item 6 (Melhor Generalização):")
    print(f"    -> O modelo recomendado é o Treinamento {melhor_modelo['treinamento']}º (T{melhor_modelo['treinamento']}),")
    print(f"       pois apresentou o menor Erro Relativo Médio ({melhor_modelo['erm']:.2f}%)")
    print(f"       e uma baixíssima Variância ({melhor_modelo['var_erm']:.2f}%), garantindo a predição")
    print(f"       mais fidedigna e estável da energia absorvida no sistema de ressonância magnética.")

    # ─────────────────────────────────────────────────────────────────────────
    # SALVAR RELATÓRIO OFICIAL EM TEXTO
    # ─────────────────────────────────────────────────────────────────────────
    relatorio_txt_path = os.path.join(RESULTADOS_DIR, 'relatorio.txt')
    respostas_txt_path = os.path.join(SCRIPT_DIR, 'respostas_multicamadas.txt')

    conteudo_relatorio = f"""================================================================================
  CEFET-MG Campus VIII – Varginha | Bacharelado em Sistemas de Informação
  Disciplina: Laboratório de Inteligência Artificial
  Professor:  Lázaro Eduardo da Silva
  Aluno:      Caio Magalhães Fonseca Arantes
  Data:       08/10/2026

  ATIVIDADE: REDE NEURAL PERCEPTRON MULTICAMADAS (PMC / MLP)
  Exercício 2 — Aplicação em Sistema de Ressonância Magnética
================================================================================

[CONTEXTO]
O processador de imagens de um sistema de ressonância magnética necessita estimar
a variável contínua {{y}} (energia absorvida) a partir de 3 grandezas mensuráveis
{{x1, x2, x3}}. Devido à complexidade do mapeamento, utiliza-se uma rede MLP com:
  - 3 neurônios de entrada (x1, x2, x3) + bias x0 = -1
  - 10 neurônios na camada oculta (ativação logística) + bias z0 = -1
  - 1 neurônio na camada de saída (ativação logística)
  - Taxa de aprendizagem η = 0.1
  - Critério de parada ε = 10⁻⁶ (|EQM_atual - EQM_anterior| ≤ ε)
  - Pesos iniciais aleatórios entre 0 e 1

────────────────────────────────────────────────────────────────────────────────
ITEM 2 — TABELA COM OS RESULTADOS FINAIS DOS 5 TREINAMENTOS
────────────────────────────────────────────────────────────────────────────────

  Treinamento       Erro Quadrático Médio (EQM)      Número de Épocas
  ──────────────────────────────────────────────────────────────────
  1º (T1)           {resultados[0]['eqm_final']:.8f}                     {resultados[0]['epocas']}
  2º (T2)           {resultados[1]['eqm_final']:.8f}                     {resultados[1]['epocas']}
  3º (T3)           {resultados[2]['eqm_final']:.8f}                     {resultados[2]['epocas']}
  4º (T4)           {resultados[3]['eqm_final']:.8f}                     {resultados[3]['epocas']}
  5º (T5)           {resultados[4]['eqm_final']:.8f}                     {resultados[4]['epocas']}
  ──────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
ITEM 3 — GRÁFICOS DE CONVERGÊNCIA (MAIORES NÚMEROS DE ÉPOCAS)
────────────────────────────────────────────────────────────────────────────────
Os dois treinamentos que demandaram o maior número de épocas foram:
  1º Maior: Treinamento {top1['treinamento']}º (T{top1['treinamento']}) com {top1['epocas']} épocas (EQM final = {top1['eqm_final']:.8f})
  2º Maior: Treinamento {top2['treinamento']}º (T{top2['treinamento']}) com {top2['epocas']} épocas (EQM final = {top2['eqm_final']:.8f})

Os gráficos foram gerados em subplots não sobrepostos e salvos em:
  multicamadas/resultados/convergencia_maiores_epocas.png

────────────────────────────────────────────────────────────────────────────────
ITEM 4 — EXPLICAÇÃO DA VARIAÇÃO DO EQM E DO NÚMERO DE ÉPOCAS
────────────────────────────────────────────────────────────────────────────────
Tanto o erro quadrático médio (EQM) quanto o número de épocas variam entre os
treinamentos porque a superfície de erro de uma rede neural Perceptron
Multicamadas (diferente da ADALINE, que é estritamente parabólica/quadrática)
é altamente NÃO-CONVEXA e multidimensional, caracterizada por:
  1. Múltiplos Mínimos Locais e Pontos de Sela: A função de custo possui vales,
     platôs e pontos de sela ao longo do hiperplano de pesos.
  2. Sensibilidade ao Ponto Inicial: Ao sortear os pesos iniciais uniformemente
     entre 0 e 1 em cada treinamento (utilizando seeds diferentes), o algoritmo
     do Gradiente Descendente inicia sua busca em coordenadas completamente
     distintas do espaço de parâmetros.
  3. Diferença nas Trajetórias do Gradiente: Trajetórias iniciadas em regiões
     com gradientes menores (saturação das sigmoides) requerem mais épocas para
     convergir; enquanto trajetórias mais íngremes convergem mais rápido,
     estabilizando-se em bacias de atração (mínimos locais) de qualidades ligeiramente
     diferentes.

────────────────────────────────────────────────────────────────────────────────
ITEM 5 — TABELA DE VALIDAÇÃO (20 AMOSTRAS DE TESTE)
────────────────────────────────────────────────────────────────────────────────

Amostra    x1       x2       x3       d      yrede(T1)  yrede(T2)  yrede(T3)  yrede(T4)  yrede(T5)
─────────────────────────────────────────────────────────────────────────────────────────────
"""
    for i in range(len(d_val)):
        conteudo_relatorio += f"{ids_val[i]:<10} {X_val[i, 1]:<8.4f} {X_val[i, 2]:<8.4f} {X_val[i, 3]:<8.4f} {d_val[i]:<8.4f} {predicoes_todos[0][i]:<10.4f} {predicoes_todos[1][i]:<10.4f} {predicoes_todos[2][i]:<10.4f} {predicoes_todos[3][i]:<10.4f} {predicoes_todos[4][i]:<10.4f}\n"

    conteudo_relatorio += f"""─────────────────────────────────────────────────────────────────────────────────────────────
Erro Relativo Médio (%)                      {resultados[0]['erm']:<10.2f}% {resultados[1]['erm']:<10.2f}% {resultados[2]['erm']:<10.2f}% {resultados[3]['erm']:<10.2f}% {resultados[4]['erm']:<10.2f}%
Variância (%)                                {resultados[0]['var_erm']:<10.2f}% {resultados[1]['var_erm']:<10.2f}% {resultados[2]['var_erm']:<10.2f}% {resultados[3]['var_erm']:<10.2f}% {resultados[4]['var_erm']:<10.2f}%
─────────────────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
ITEM 6 — INDICAÇÃO DO MODELO COM MELHOR GENERALIZAÇÃO
────────────────────────────────────────────────────────────────────────────────
Com base na análise estatística de validação do Item 5:
A configuração final mais adequada para o sistema de ressonância magnética é a
do Treinamento {melhor_modelo['treinamento']}º (T{melhor_modelo['treinamento']}).

Justificativa:
  - Obteve o menor Erro Relativo Médio global ({melhor_modelo['erm']:.2f}%) perante todas
    as 20 amostras nunca vistas de teste.
  - Apresentou uma variância reduzida ({melhor_modelo['var_erm']:.2f}%), comprovando
    homogeneidade na predição sem desvios discrepantes.
  - Isso demonstra a melhor capacidade de GENERALIZAÇÃO, garantindo que o
    processador de imagens estime a energia absorvida de forma consistente,
    evitando sobreajuste (overfitting).
================================================================================
"""

    for p in [relatorio_txt_path, respostas_txt_path]:
        with open(p, 'w', encoding='utf-8') as f:
            f.write(conteudo_relatorio)
        print(f"    ✓ Relatório salvo em: {p}")

    print("\n" + "=" * 78)
    print("  Execução concluída com sucesso!")
    print("=" * 78)


if __name__ == '__main__':
    main()

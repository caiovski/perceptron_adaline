# Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG)
## Campus VIII – Varginha | Bacharelado em Sistemas de Informação
**Disciplina:** Laboratório de Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Aluno:** Caio Magalhães Fonseca Arantes  
**Data:** 08/10/2026  

---

# RELATÓRIO TÉCNICO — EXERCÍCIO 1
## Demonstração Analítica do Algoritmo Backpropagation com Conexões Diretas (*Skip Connections*)

---

## 1. Contexto e Topologia da Rede Neural

A equipe de engenheiros e cientistas projetou uma **Rede Neural Artificial do tipo Perceptron Multicamadas (PMC)** constituída de três camadas, com conexões diretas entre a camada de entrada e a camada neural de saída.

```
                       ┌────────────────────────────────────────┐
                       │               Matriz W3                │
                       │   (Conexão direta entrada -> saída)    │
                       ▼                                        │
           ┌───────────────────────┐                            │
           │   Camada 1: ENTRADA   │                            │
           │ (X1, X2, ..., XN, X0) │                            │
           └──────────┬────────────┘                            │
                      │                                         │
                      │ Matriz W1                               │
                      ▼                                         │
           ┌───────────────────────┐                            │
           │  Camada 2: ESCONDIDA  │                            │
           │   (Neurônios 1 a N1)  │                            │
           └──────────┬────────────┘                            │
                      │                                         │
                      │ Matriz W2                               │
                      ▼                                         │
           ┌───────────────────────┐ ◀──────────────────────────┘
           │   Camada 3: SAÍDA     │
           │   (Apenas 1 neurônio) │ ────▶ Saída da Rede (y)
           └───────────────────────┘
```

### Especificações da Topologia:
1. **Camada de Entrada (1ª Camada):** Constituída de $N$ sinais de entrada:
   $$\mathbf{X} = [x_1, x_2, \dots, x_N]^T$$
   Estendida com o sinal de bias fixo $x_0 = -1$ (convenção adotada):
   $$\mathbf{X}_{\text{ext}} = [x_0, x_1, x_2, \dots, x_N]^T \in \mathbb{R}^{N+1}$$

2. **Camada Neural Escondida (2ª Camada):** Constituída de $N_1$ neurônios artificiais ($j = 1, 2, \dots, N_1$). Cada neurônio $j$ possui potencial de ativação $u_j^{(2)}$ e sinal de saída $z_j = g(u_j^{(2)})$. Esta camada também possui sinal de bias associado $z_0 = -1$ para a conexão subsequente.

3. **Camada Neural de Saída (3ª Camada):** Constituída de apenas $1$ neurônio ($k = 1$), com potencial de ativação $u^{(3)}$ e sinal de saída final $y = f(u^{(3)})$.

4. **Conjunto de Padrões de Treinamento:** Composto por $P$ pares de treinamento:
   $$\{(\mathbf{X}_p, d_p)\}_{p=1}^P, \quad \text{onde } d_p \text{ é a saída desejada para o padrão } p.$$

### Convenção das Matrizes de Pesos:
* **Matriz $W_1 \in \mathbb{R}^{N_1 \times (N+1)}$:** Pesos sinápticos entre a 1ª camada (entrada) e a 2ª camada (escondida). O elemento $w_{1, ji}$ representa a sinapse conectando a entrada $x_i$ ao neurônio escondido $j$ ($i = 0, \dots, N$ e $j = 1, \dots, N_1$).
* **Matriz $W_2 \in \mathbb{R}^{1 \times (N_1+1)}$:** Pesos sinápticos entre a 2ª camada (escondida) e a 3ª camada (saída). O elemento $w_{2, 1j}$ representa a sinapse conectando o neurônio escondido $z_j$ ao neurônio de saída ($j = 0, \dots, N_1$).
* **Matriz $W_3 \in \mathbb{R}^{1 \times (N+1)}$:** Pesos sinápticos entre a 1ª camada (entrada) e a 3ª camada (saída) — **conexão direta de atalho**. O elemento $w_{3, 1i}$ representa a sinapse conectando a entrada $x_i$ diretamente ao neurônio de saída ($i = 0, \dots, N$).

---

## 2. Passo Forward (Propagação Direta dos Sinais)

Para a apresentação de um dado padrão de treinamento $p$:

### 2.1. Propagação para a Camada Escondida (2ª Camada)
Cada neurônio $j \in \{1, 2, \dots, N_1\}$ calcula o potencial de ativação através do produto escalar entre os pesos de $W_1$ e o vetor de entrada estendido:
$$u_j^{(2)} = \sum_{i=0}^N w_{1, ji} \, x_i = w_{1, j0} \, x_0 + \sum_{i=1}^N w_{1, ji} \, x_i$$
Aplicando a função de ativação diferenciável $g(\cdot)$ (tipicamente a logística/sigmoide):
$$z_j = g\left(u_j^{(2)}\right)$$
O vetor de saídas intermediárias estendido com o bias $z_0 = -1$ é expresso por:
$$\mathbf{Z} = [z_0, z_1, z_2, \dots, z_{N_1}]^T$$

### 2.2. Propagação para a Camada de Saída (3ª Camada)
O neurônio de saída recebe simultaneamente duas contribuições:
1. A soma ponderada das saídas da camada intermediária via matriz $W_2$: $\sum_{j=0}^{N_1} w_{2, 1j} \, z_j$.
2. A soma ponderada dos sinais diretos da entrada via matriz $W_3$: $\sum_{i=0}^N w_{3, 1i} \, x_i$.

Portanto, o potencial de ativação total $u^{(3)}$ do neurônio de saída é a combinação linear aditiva de ambas as matrizes:
$$u^{(3)} = \sum_{j=0}^{N_1} w_{2, 1j} \, z_j + \sum_{i=0}^N w_{3, 1i} \, x_i$$

A saída final estimada pela rede é dada por:
$$y = f\left(u^{(3)}\right)$$
onde $f(\cdot)$ é a função de ativação do neurônio de saída (diferenciável).

---

## 3. Função de Custo e Critério de Otimização

O erro instantâneo de predição $e$ associado ao padrão de treinamento apresentado é:
$$e = d - y$$
A energia instantânea do erro (função de perda quadrática clássica da Regra Delta Generalizada) é definida como:
$$E = \frac{1}{2} e^2 = \frac{1}{2} (d - y)^2$$

O objetivo do algoritmo Backpropagation é minimizar $E$ ajustando os parâmetros através do **Método do Gradiente Descendente**:
$$\Delta w = -\eta \, \frac{\partial E}{\partial w}$$
onde $\eta > 0$ é a taxa de aprendizagem.

---

## 4. Passo Backward (Retropropagação e Ajuste das Matrizes)

Aplicamos a **Regra da Cadeia do Cálculo Diferencial** para calcular as derivadas parciais de $E$ em relação a cada peso das matrizes $W_1, W_2$ e $W_3$.

### 4.1. Definição do Gradiente Local da Camada de Saída ($\delta^{(3)}$)
A derivada do erro $E$ em relação ao potencial $u^{(3)}$ da camada de saída é:
$$\frac{\partial E}{\partial u^{(3)}} = \frac{\partial E}{\partial e} \cdot \frac{\partial e}{\partial y} \cdot \frac{\partial y}{\partial u^{(3)}}$$
Calculando cada termo:
* $\frac{\partial E}{\partial e} = \frac{\partial}{\partial e} \left( \frac{1}{2} e^2 \right) = e$
* $\frac{\partial e}{\partial y} = \frac{\partial}{\partial y} (d - y) = -1$
* $\frac{\partial y}{\partial u^{(3)}} = f'\left(u^{(3)}\right)$

Logo:
$$\frac{\partial E}{\partial u^{(3)}} = (e) \cdot (-1) \cdot f'\left(u^{(3)}\right) = - (d - y) \, f'\left(u^{(3)}\right)$$

Definimos o **gradiente local do neurônio de saída** ($\delta^{(3)}$) como o negativo desta derivada parcial:
$$\delta^{(3)} \equiv -\frac{\partial E}{\partial u^{(3)}} = (d - y) \, f'\left(u^{(3)}\right)$$

---

### 4.2. Ajuste da Matriz $W_2$ (Conexão Camada 2 $\to$ Camada 3)
A derivada de $E$ em relação a cada peso $w_{2, 1j}$ (para $j = 0, \dots, N_1$) é:
$$\frac{\partial E}{\partial w_{2, 1j}} = \frac{\partial E}{\partial u^{(3)}} \cdot \frac{\partial u^{(3)}}{\partial w_{2, 1j}}$$
Como $u^{(3)} = \sum_{j=0}^{N_1} w_{2, 1j} z_j + \sum_{i=0}^N w_{3, 1i} x_i$, temos:
$$\frac{\partial u^{(3)}}{\partial w_{2, 1j}} = z_j$$
Substituindo a definição de $\delta^{(3)}$:
$$\frac{\partial E}{\partial w_{2, 1j}} = -\delta^{(3)} \cdot z_j$$
Pela regra de atualização do gradiente descendente $\Delta w_{2, 1j} = -\eta \, \frac{\partial E}{\partial w_{2, 1j}}$:
$$\mathbf{\Delta w_{2, 1j} = \eta \cdot \delta^{(3)} \cdot z_j}$$
$$w_{2, 1j}^{(\text{novo})} = w_{2, 1j}^{(\text{atual})} + \Delta w_{2, 1j}, \quad \forall j \in \{0, 1, \dots, N_1\}$$

---

### 4.3. Ajuste da Matriz $W_3$ (Conexão Direta Camada 1 $\to$ Camada 3)
A derivada de $E$ em relação a cada peso $w_{3, 1i}$ (para $i = 0, \dots, N$) é:
$$\frac{\partial E}{\partial w_{3, 1i}} = \frac{\partial E}{\partial u^{(3)}} \cdot \frac{\partial u^{(3)}}{\partial w_{3, 1i}}$$
Diferenciando o potencial $u^{(3)}$ em relação a $w_{3, 1i}$:
$$\frac{\partial u^{(3)}}{\partial w_{3, 1i}} = x_i$$
Substituindo o gradiente local $\delta^{(3)}$:
$$\frac{\partial E}{\partial w_{3, 1i}} = -\delta^{(3)} \cdot x_i$$
Portanto, a regra de atualização dos pesos da matriz $W_3$ é dada por:
$$\mathbf{\Delta w_{3, 1i} = \eta \cdot \delta^{(3)} \cdot x_i}$$
$$w_{3, 1i}^{(\text{novo})} = w_{3, 1i}^{(\text{atual})} + \Delta w_{3, 1i}, \quad \forall i \in \{0, 1, \dots, N\}$$

> **Interpretação Teórica Notável:** Como os pesos de $W_3$ incidem diretamente sobre o neurônio de saída, seu ajuste é modulado diretamente pelo gradiente de saída $\delta^{(3)}$ e pelo sinal de entrada $x_i$, sem necessidade de retropropagar o erro pela camada intermediária!

---

### 4.4. Ajuste da Matriz $W_1$ (Conexão Camada 1 $\to$ Camada 2)
Para os pesos $w_{1, ji}$ conectando a entrada $i$ ao neurônio escondido $j$, a dependência de $E$ se dá através de $u^{(3)}$ e subsequentemente através de $z_j$ e $u_j^{(2)}$:
$$\frac{\partial E}{\partial w_{1, ji}} = \frac{\partial E}{\partial u^{(3)}} \cdot \frac{\partial u^{(3)}}{\partial z_j} \cdot \frac{\partial z_j}{\partial u_j^{(2)}} \cdot \frac{\partial u_j^{(2)}}{\partial w_{1, ji}}$$

Avaliando termo a termo:
1. $\frac{\partial E}{\partial u^{(3)}} = -\delta^{(3)}$
2. $\frac{\partial u^{(3)}}{\partial z_j} = w_{2, 1j}$ (o erro retropropaga através do peso de $W_2$ conectado a esse neurônio)
3. $\frac{\partial z_j}{\partial u_j^{(2)}} = g'\left(u_j^{(2)}\right)$
4. $\frac{\partial u_j^{(2)}}{\partial w_{1, ji}} = x_i$

Reunindo os fatores:
$$\frac{\partial E}{\partial w_{1, ji}} = -\delta^{(3)} \cdot w_{2, 1j} \cdot g'\left(u_j^{(2)}\right) \cdot x_i$$

Definimos o **gradiente local do neurônio escondido $j$** ($\delta_j^{(2)}$) como:
$$\delta_j^{(2)} \equiv -\frac{\partial E}{\partial u_j^{(2)}} = \delta^{(3)} \cdot w_{2, 1j} \cdot g'\left(u_j^{(2)}\right)$$

Assim, a derivada de erro em relação ao peso $w_{1, ji}$ é:
$$\frac{\partial E}{\partial w_{1, ji}} = -\delta_j^{(2)} \cdot x_i$$

Aplicando a regra de atualização do gradiente descendente:
$$\mathbf{\Delta w_{1, ji} = \eta \cdot \delta_j^{(2)} \cdot x_i}$$
$$w_{1, ji}^{(\text{novo})} = w_{1, ji}^{(\text{atual})} + \Delta w_{1, ji}, \quad \forall j \in \{1, \dots, N_1\}, \; \forall i \in \{0, \dots, N\}$$

---

## 5. Resumo Executivo das Equações de Ajuste

| Matriz de Pesos | Camadas Conectadas | Gradiente Local | Equação de Atualização ($\Delta w$) |
| :---: | :---: | :---: | :---: |
| **$W_2$** | Oculta $\to$ Saída | $\delta^{(3)} = (d - y) \, f'(u^{(3)})$ | $\Delta w_{2, 1j} = \eta \cdot \delta^{(3)} \cdot z_j$ |
| **$W_3$** | Entrada $\to$ Saída (Direta) | $\delta^{(3)} = (d - y) \, f'(u^{(3)})$ | $\Delta w_{3, 1i} = \eta \cdot \delta^{(3)} \cdot x_i$ |
| **$W_1$** | Entrada $\to$ Oculta | $\delta_j^{(2)} = \delta^{(3)} \cdot w_{2, 1j} \cdot g'(u_j^{(2)})$ | $\Delta w_{1, ji} = \eta \cdot \delta_j^{(2)} \cdot x_i$ |

---

## 6. Algoritmo Passo a Passo (Pseudocódigo Estruturado)

```text
Para cada época k = 1 até MAX_EPOCAS:
    Para cada padrão p = 1 até P:
        1. PASSO FORWARD:
           a) Para cada neurônio j = 1 até N1:
              u_j^(2) = ∑_{i=0}^N w1_ji * x_i
              z_j = g(u_j^(2))
           b) Neurônio de saída:
              u^(3) = ∑_{j=0}^N1 w2_1j * z_j + ∑_{i=0}^N w3_1i * x_i
              y = f(u^(3))
              e = d_p - y

        2. PASSO BACKWARD:
           a) Calcular gradiente de saída:
              δ^(3) = (d_p - y) * f'(u^(3))
           b) Para cada neurônio j = 1 até N1:
              δ_j^(2) = δ^(3) * w2_1j * g'(u_j^(2))

        3. AJUSTE DAS MATRIZES:
           a) Atualizar W2:  w2_1j = w2_1j + η * δ^(3) * z_j       (j = 0..N1)
           b) Atualizar W3:  w3_1i = w3_1i + η * δ^(3) * x_i       (i = 0..N)
           c) Atualizar W1:  w1_ji = w1_ji + η * δ_j^(2) * x_i     (j = 1..N1, i = 0..N)

    4. CRITÉRIO DE PARADA:
       Calcular EQM da época: EQM = (1/P) * ∑ (d_p - y_p)²
       Se |EQM_atual - EQM_anterior| ≤ ε: Interromper treinamento.
```

---
*Demonstração concluída em conformidade com as diretrizes do CEFET-MG.*

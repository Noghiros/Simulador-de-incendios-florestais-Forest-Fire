# 🌲 Forest Fire — Projeto Paralelo e Distribuído

**EC48A — Sistemas Distribuídos (UTFPR-CP) - C81**
<p> Projeto final desenvolvido para a disciplina de Sistemas Distribuídos, aplicando técnicas de programação sequencial, paralela e distribuída para simular o clássico modelo de propagação de incêndios florestais (Forest Fire Model). </p>

---

## 🎯 Objetivo

Implementar e comparar três versões do modelo de incêndios florestais (Forest Fire):

* Versão Sequencial;
* Versão Paralela (Threads Python);
* Versão Distribuída (Sockets TCP).

O objetivo central é analisar:

* 🔥 Desempenho;
* 🔥 Speedup;
* 🔥 Escalabilidade;
* 🔥 Custos de comunicação;
* 🔥 Limitações da paralelização e distribuição.

---

## 🌱 Conceitos Envolvidos

### 📌 Modelo Forest Fire (Stanford / Drossel–Schwabl)

Um autômato celular probabilístico onde cada célula pode estar em um de três estados:

* 0 – Vazio;
* 1 – Árvore;
* 2 – Queimando.

A evolução segue as regras probabilísticas clássicas:

1. Uma árvore pega fogo se algum vizinho estiver queimando;
2. Uma árvore pode acender sozinha com probabilidade **f** (raios);
3. Uma célula vazia pode crescer uma árvore com probabilidade **p**;
4. Uma célula queimando vira vazia.

Esse modelo é amplamente usado para estudar sistemas críticos, dinâmica de propagação e autômatos celulares.

### 📌 Técnicas Computacionais

* Execução sequencial pura;
* Paralelismo com Threads (GIL-bound, mas útil para I/O e simulação)**;
* Domínio dividido (decomposition);
* Comunicação via Sockets TCP em topologia linear;
* Sincronização de fronteiras (ghost rows);
* Medição de tempos com `time.perf_counter`;
* Plotagem e análise com matplotlib + CSV.

---

# 👥 Integrantes e suas contribuições (Grupo 2)

* [<img src="https://i.imgur.com/6wtHdzd.png" width="30">](https://github.com/felipebataglini) **Felipe de Oliveira Guimarães Bataglini**
  - Implementação versão distribuída;
  - Comunicação via sockets;
  - Testes e validação.
  
* [<img src="https://i.imgur.com/fA4JpJg.png" width="30">](https://github.com/JoaoVBLaneiro) **João Vitor Briganti Laneiro**
  - Implementação versão paralela com threads;
  - Sistema de logging e CSV;
  - Documentação técnica.
  
* [<img src="https://i.imgur.com/0ldubtT.png" width="30">](https://github.com/Noghiros) **Stefano Calheiros Stringhini**
  - Implementação versão sequencial;
  - Desenvolvimento do sistema de benchmark;
  - Análise de resultados.

---

## ⚙️ Dependências

```sh
pip install numpy matplotlib
```

* Python **3.8+**;
* `numpy` — matrizes e atualização do autômato;
* `matplotlib` — geração de gráficos comparativos;
* (`socket`, `threading`, `csv` → nativos do Python).

---

## 🧪 Metodologia de Testes

Foram executadas baterias de testes variando:

* Tamanho da grade:
  `50×50`, `100×100`, `200×200`, ...
* Número de iterações:
  `10`, `30`, `50`, `100`, ...
* Número de threads / processos distribuídos:
  `2`, `4`, `8`, ...

Cada execução foi repetida de 3 a 5 vezes, com cálculo de:

* Tempo médio;
* Speedup;
* Eficiência;
* Tempo gasto em comunicação (versão distribuída).

Ferramentas de medição:

* `perf_counter()`;
* arquivos `.csv` gerados automaticamente;
* `plotar_graficos.py` para gerar gráficos.

## 💻 Ambiente de Testes

### Hardware
- **CPU**: [Intel Core i3-7100U CPU @ 2.40GHz - 2 Núcleos Físicos / 4 Threads Lógicas (Virtual Cores)];
- **RAM**: [12,0 GB DDR4];
- **Rede**: [Localhost (127.0.0.1) — Custo Zero de Latência Real, mas o overhead de serialização/socket é mensurado.];
- **Sistema**: [Windows 10 Home].

### Software
- **Python**: 3.12.1;
- **Bibliotecas**: numpy 2.2.6, matplotlib 3.1.0.7.

---

## ▶️ Etapas para Execução

### **1. Versão Sequencial**

```sh
python sequencial/forest_fire_sequencial.py
```

### **2. Versão Paralela com Threads**

```sh
python paralelo/forest_fire_paralelo.py <num_threads>
```

### **3. Versão Distribuída (Local)**

```sh
python run_distributed_local.py <num_procs> <base_port> Nx Ny nsteps p f d0
```

Exemplo real:

```sh
python run_distributed_local.py 4 9000 50 50 30 0.01 0.0001 0.6
```

### **4. Gerar gráficos**

```sh
python plotar_graficos.py
```

---

# 📊 Resultados Obtidos

### Tabela Comparativa de Tempos (segundos)

| Tamanho    | Sequencial | 2 Threads | 4 Threads | 2 Workers | 4 Workers |
|------------|------------|-----------|-----------|-----------|-----------|
| 256×256    | 0.96s      | 43.96s    | 44.37s    | 29.33s    | 24.07s    |
| 512×512    | 3.26s      | 183.19s   | 182.33s   | 112.83s   | 90.23s    |
| 1024×1024  | 9.30s      | 728.02s   | 771.74s   | 560.37s   | 357.11s   |

### Gráficos de Desempenho
<img width="800" height="500" alt="Gráfico Comparativo" src="https://github.com/user-attachments/assets/8b725016-9a0e-4603-ab0c-d92cd23b870e" />

### Análise de Speedup

| Configuração  | 256×256 | 512×512 | 1024×1024 | Eficiência |
|---------------|---------|---------|-----------|------------|
| 2 Threads     | 0.02x   | 0.02x   | 0.01x     | 1.0%       |
| 4 Threads     | 0.02x   | 0.02x   | 0.01x     | 0.5%       |
| 2 Workers     | 0.03x   | 0.03x   | 0.02x     | 1.5%       |
| 4 Workers     | 0.04x   | 0.04x   | 0.03x     | 0.8%       |

## ⚠️ Limitações Identificadas

Os resultados mostram que as versões paralela e distribuída apresentaram desempenho inferior à versão sequencial, indicando **overhead excessivo** nas implementações atuais. 
Isso se deve principalmente à sincronização frequente e custos de comunicação que superam os ganhos da paralelização, conforme detalhado abaixo:

* **Paralela (Threads): Slowdown Causado pelo GIL**
  A simulação é classificada como CPU-bound (limitada pela capacidade de cálculo da CPU). O ganho de desempenho foi nulo devido ao Python Global Interpreter Lock (GIL).
  - **Problema:** O GIL garante que apenas uma thread execute código Python por vez, impedindo o paralelismo real em CPUs multi-core. 
  - **Consequência:** A implementação com threading incorre no alto custo de gerenciamento e sincronização de 4 threads, mas sem o benefício da execução simultânea, resultando em um enorme slowdown.
  - **Resultado:** O tempo foi de 771.74s (4 Threads) contra a baseline sequencial de 9.30s.

* **Distribuída (Sockets): Penalidade do Overhead de Comunicação**
  - **Alto Custo para Granularidade Fina:** A troca de Ghost Rows em Sockets TCP a cada um dos 200 passos da simulação exige repetida serialização (numpy para bytes) e desserialização
  - **Sincronização Excessiva:** Sincronização total e frequente entre todos os workers
  - **Resultado:** Overhead de comunicação superior ao tempo de processamento sequencial

## 🚀 Melhorias Propostas

🔧 **Para Versão Paralela**:
- Usar o módulo _multiprocessing_ (em vez de _threading_) para criar processos separados, cada um com sua própria instância do interpretador Python, permitindo a execução simultânea em múltiplos núcleos físicos.

🔧 **Para Versão Distribuída**:
- Usar bibliotecas otimizadas como MPI (Message Passing Interface) em vez de Sockets puros. MPI é projetado para transferência de grandes volumes de dados de forma eficiente em computação paralela e distribuída.
- Buscar uma arquitetura de granularidade mais grossa (se o problema permitir), reduzindo a frequência de comunicação.

---

# 📚 Referências

### 📄 Modelo Forest Fire

* **Drossel, B. & Schwabl, F.** (1992). *Self-organized critical forest-fire model*. Physical Review Letters.
* **Stanford CS** – Cellular Automata: Forest Fire Model (modelo clássico utilizado amplamente em cursos e implementações)
* **Karafyllidis & Thanailakis** (1997). *A cellular automata wildfire spread model*.

### 📄 Autômatos e dinâmica do fogo

* **Schenk et al.** (2001). *Forest fire models on large scales*. arXiv.
* **Ghosh, R.** (2024). *Probabilistic Cellular Automata Fire Spread Model*. [arXiv](https://arxiv.org/pdf/2403.08817).

### 💬 Créditos de desenvolvimento

* Google, StackOverflow, Documentação oficial de Python
* Interação com ChatGPT, DeepSeek e Gemini durante desenvolvimento
* E como sempre: fé, café, força de vontade e todas as ferramentas ao nosso alcance

# 🌲 Forest Fire — Projeto Paralelo e Distribuído

**EC48A — Sistemas Distribuídos (UTFPR-CP) - C81**
Projeto final desenvolvido para a disciplina de Sistemas Distribuídos, aplicando técnicas de programação sequencial, paralela e distribuída para simular o clássico modelo de *propagação de incêndios florestais* (*Forest Fire Model*).

---

## 🎯 Objetivo

Implementar e comparar três versões do modelo de incêndios florestais (Forest Fire):

* Versão Sequencial
* Versão Paralela (Threads Python)
* Versão Distribuída (Sockets TCP)

O objetivo central é analisar:

* 🔥 Desempenho
* 🔥 Speedup
* 🔥 Escalabilidade
* 🔥 Custos de comunicação
* 🔥 Limitações da paralelização e distribuição

---

## 🌱 Conceitos Envolvidos

### 📌 Modelo Forest Fire (Stanford / Drossel–Schwabl)

Um autômato celular probabilístico onde cada célula pode estar em um de três estados:

* 0 – Vazio
* 1 – Árvore
* 2 – Queimando

A evolução segue as regras probabilísticas clássicas:

1. Uma árvore pega fogo se algum vizinho estiver queimando
2. Uma árvore pode acender sozinha com probabilidade **f** (raios)
3. Uma célula vazia pode crescer uma árvore com probabilidade **p**
4. Uma célula queimando vira vazia

Esse modelo é amplamente usado para estudar sistemas críticos, dinâmica de propagação e autômatos celulares.

### 📌 Técnicas Computacionais

* Execução sequencial pura
* Paralelismo com Threads (GIL-bound, mas útil para I/O e simulação)**
* Domínio dividido (decomposition)
* Comunicação via Sockets TCP em topologia linear
* Sincronização de fronteiras (ghost rows)
* Medição de tempos com `time.perf_counter`
* Plotagem e análise com matplotlib + CSV

---

## 👥 Integrantes (Grupo 2)

* [<img src="https://i.imgur.com/6wtHdzd.png" width="30">](https://github.com/felipebataglini) **Felipe de Oliveira Guimarães Bataglini**
* [<img src="https://i.imgur.com/fA4JpJg.png" width="30">](https://github.com/JoaoVBLaneiro) **João Vitor Briganti Laneiro**
* [<img src="https://i.imgur.com/0ldubtT.png" width="30">](https://github.com/Noghiros) **Stefano Calheiros Stringhini**

---

## ⚙️ Dependências

```sh
pip install numpy matplotlib
```

* Python **3.8+**
* `numpy` — matrizes e atualização do autômato
* `matplotlib` — geração de gráficos comparativos
* (`socket`, `threading`, `csv` → nativos do Python)

---

## 🧪 Metodologia de Testes

Foram executadas baterias de testes variando:

* Tamanho da grade:
  `50×50`, `100×100`, `200×200`, ...
* Número de iterações:
  `10`, `30`, `50`, `100`...
* Número de threads / processos distribuídos:
  `2`, `4`, `8`...

Cada execução foi repetida de 3 a 5 vezes, com cálculo de:

* Tempo médio
* Speedup
* Eficiência
* Tempo gasto em comunicação (versão distribuída)

Ferramentas de medição:

* `perf_counter()`
* arquivos `.csv` gerados automaticamente
* `plotar_graficos.py` para gerar gráficos

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

## 📚 Referências

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

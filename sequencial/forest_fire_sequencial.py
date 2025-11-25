import sys, time, csv
import numpy as np

# Estados do autômato celular
EMPTY, TREE, BURN = 0, 1, 2   # vazio, árvore, pegando fogo


def step(grid, p, f):
    """
    Executa UM passo da simulação do modelo Forest Fire
    (versão totalmente sequencial, sem paralelismo).

    Regras:
    - Células queimando viram vazias.
    - Árvores pegam fogo se houver vizinho queimando.
    - Árvores podem pegar fogo espontaneamente (prob. f).
    - Células vazias podem crescer árvores (prob. p).
    """
    Nx, Ny = grid.shape

    # new = cópia da grid atual onde aplicaremos as atualizações
    new = grid.copy()

    # 1. Células queimando viram vazias
    new[grid == BURN] = EMPTY

    # 2. Criar máscara booleana de vizinhos queimando (vizinhança de Moore)
    g = (grid == BURN)  # matriz True/False indicando onde está queimando
    mask = np.zeros_like(g, dtype=bool)

    # Adjacent (cima, baixo, esquerda, direita)
    mask[:-1, :] |= g[1:, :]      # vizinho de baixo
    mask[1:, :]  |= g[:-1, :]     # vizinho de cima
    mask[:, :-1] |= g[:, 1:]      # vizinho da direita
    mask[:, 1:]  |= g[:, :-1]     # vizinho da esquerda

    # Diagonais
    mask[:-1, :-1] |= g[1:, 1:]
    mask[:-1, 1:]  |= g[1:, :-1]
    mask[1:, :-1]  |= g[:-1, 1:]
    mask[1:, 1:]   |= g[:-1, :-1]

    # 3. Máscara de células que são árvores
    tree_mask = (grid == TREE)

    # 4. Árvores que devem pegar fogo por vizinhança
    catch = tree_mask & mask

    # 5. Árvores que pegam fogo espontaneamente (raio)
    ignite = (np.random.random(size=grid.shape) < f) & tree_mask

    # 6. Células vazias onde nasce árvore
    grow = (np.random.random(size=grid.shape) < p) & (grid == EMPTY)

    # Atualizações finais
    new[catch] = BURN
    new[ignite] = BURN
    new[grow] = TREE

    return new



def run(Nx=512, Ny=512, nsteps=500, p=0.01, f=0.0001, d0=0.6,
        out_csv='result_seq.csv'):
    """
    Executa a versão sequencial completa do modelo Forest Fire.

    - Inicializa grade com densidade inicial d0.
    - Roda nsteps passos chamando 'step'.
    - Mede tempo total.
    - Salva estatísticas em CSV.
    """

    # Inicializa grid aleatória: árvore com prob d0
    grid = np.where(np.random.random((Nx, Ny)) < d0, TREE, EMPTY).astype(np.int8)

    burn_counts = []  # para armazenar quantas células queimando por passo

    t0 = time.perf_counter()

    # Loop principal da simulação
    for s in range(nsteps):
        grid = step(grid, p, f)
        burn_counts.append(int((grid == BURN).sum()))  # contagem do fogo

    t1 = time.perf_counter()
    total = t1 - t0

    print(f'[SEQ] Nx={Nx} Ny={Ny} steps={nsteps} total_time={total:.6f}s')

    # Salvar resultados no CSV
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode','Nx','Ny','nsteps','p','f','total_time','avg_burn'])
        w.writerow([
            'seq', Nx, Ny, nsteps, p, f,
            f'{total:.6f}',
            sum(burn_counts)/len(burn_counts) if burn_counts else 0.0
        ])



if __name__ == '__main__':
    # Permite rodar pela linha de comando:
    # python sequencial.py Nx Ny nsteps p f d0
    args = sys.argv[1:]

    if len(args) >= 6:
        Nx = int(args[0])
        Ny = int(args[1])
        nsteps = int(args[2])
        p = float(args[3])
        f = float(args[4])
        d0 = float(args[5])
    else:
        # Valores padrão para testes
        Nx, Ny, nsteps, p, f, d0 = 512, 512, 500, 0.01, 0.0001, 0.6

    run(Nx, Ny, nsteps, p, f, d0)

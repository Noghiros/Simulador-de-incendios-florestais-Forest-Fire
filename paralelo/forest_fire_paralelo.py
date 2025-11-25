import sys, time, csv, threading
import numpy as np

# Estados possíveis no autômato celular:
EMPTY, TREE, BURN = 0, 1, 2   # vazio, árvore, queimando


class Worker(threading.Thread):
    """
    Cada Worker é uma thread responsável por atualizar
    UM BLOCO de linhas da grade do autômato celular.
    """

    def __init__(self, tid, grid, new_grid, i0, i1, nsteps, p, f, barrier):
        super().__init__()
        self.tid = tid                   # ID da thread
        self.grid = grid                 # grade atual (compartilhada entre threads)
        self.new_grid = new_grid         # grade onde é escrita a atualização
        self.i0 = i0                     # linha inicial que esta thread processa
        self.i1 = i1                     # linha final (não inclusiva)
        self.nsteps = nsteps             # número total de steps de simulação
        self.p = p                       # probabilidade de nascer árvore
        self.f = f                       # probabilidade de queda de raio (árvore vira fogo)
        self.barrier = barrier           # barreira de sincronização entre threads


    def run(self):
        """
        Loop principal da thread.
        Cada iteração executa um passo da simulação:
          1. Atualiza suas linhas atribuídas
          2. Espera todas as outras na barrier (sincronização)
          3. Espera o thread principal copiar new_grid → grid
        """
        Nx, Ny = self.grid.shape

        for step in range(self.nsteps):

            # --- PROCESSAMENTO DAS LINHAS DESTAS THREAD ---
            for i in range(self.i0, self.i1):
                for j in range(Ny):

                    s = int(self.grid[i, j])  # estado atual da célula

                    # Regras do modelo Forest Fire:

                    # 1. Célula queimando → vira vazia
                    if s == BURN:
                        self.new_grid[i, j] = EMPTY

                    # 2. Célula com árvore
                    elif s == TREE:

                        # Verifica se algum vizinho está queimando
                        burning = False
                        for di in (-1, 0, 1):
                            ni = i + di
                            if ni < 0 or ni >= Nx:
                                continue
                            for dj in (-1, 0, 1):
                                nj = j + dj
                                if di == 0 and dj == 0:
                                    continue
                                if nj < 0 or nj >= Ny:
                                    continue
                                if self.grid[ni, nj] == BURN:
                                    burning = True
                                    break
                            if burning:
                                break

                        # Se vizinho queimando → pega fogo
                        if burning:
                            self.new_grid[i, j] = BURN

                        # Raio espontâneo (probabilidade f)
                        elif np.random.random() < self.f:
                            self.new_grid[i, j] = BURN

                        # Continua como árvore
                        else:
                            self.new_grid[i, j] = TREE

                    # 3. Célula vazia pode nascer árvore
                    else:  # s == EMPTY
                        if np.random.random() < self.p:
                            self.new_grid[i, j] = TREE
                        else:
                            self.new_grid[i, j] = EMPTY

            # --- SINCRONIZAÇÃO ---
            # Espera todas as threads terminarem o cálculo deste step
            self.barrier.wait()

            # Espera a thread principal copiar new_grid → grid
            self.barrier.wait()



def run(Nx=512, Ny=512, nsteps=300, nthreads=4, p=0.01, f=0.0001, d0=0.6,
        out_csv='result_threads.csv'):
    """
    Função principal da versão com Threads.
    Inicializa a grade, cria as threads, executa o loop de steps
    e mede o tempo de execução total.
    """

    # Inicialização da matriz com densidade d0
    grid = np.where(np.random.random((Nx, Ny)) < d0, TREE, EMPTY).astype(np.int8)
    new_grid = grid.copy()

    # Barreira: nthreads + 1 (para incluir a thread principal)
    barrier = threading.Barrier(nthreads + 1)

    # Divide linhas igualmente entre as threads
    rows_per = Nx // nthreads
    threads = []

    for t in range(nthreads):
        i0 = t * rows_per
        i1 = Nx if t == nthreads - 1 else (t + 1) * rows_per

        th = Worker(t, grid, new_grid, i0, i1, nsteps, p, f, barrier)
        threads.append(th)
        th.start()

    # Cronômetro
    t0 = time.perf_counter()

    # Loop global: coordena a troca grid <-> new_grid
    for step in range(nsteps):
        barrier.wait()            # espera as threads processarem
        grid[:, :] = new_grid     # copia new_grid → grid (swap sem criar novo array)
        barrier.wait()            # libera as threads para próximo step

    # Espera todas terminarem
    for th in threads:
        th.join()

    t1 = time.perf_counter()
    total = t1 - t0

    print(f'[THREADS] Nx={Nx} Ny={Ny} steps={nsteps} threads={nthreads} total_time={total:.6f}s')

    # Registro em CSV
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'Nx', 'Ny', 'nsteps', 'threads', 'p', 'f', 'total_time'])
        w.writerow(['threads', Nx, Ny, nsteps, nthreads, p, f, f'{total:.6f}'])



if __name__ == '__main__':
    # Permite rodar pela linha de comando:
    # python threads.py Nx Ny nsteps nthreads p f d0
    args = sys.argv[1:]

    if len(args) >= 7:
        Nx = int(args[0])
        Ny = int(args[1])
        nsteps = int(args[2])
        nthreads = int(args[3])
        p = float(args[4])
        f = float(args[5])
        d0 = float(args[6])
    else:
        # Valores padrão
        Nx, Ny, nsteps, nthreads, p, f, d0 = 512, 512, 300, 4, 0.01, 0.0001, 0.6

    run(Nx, Ny, nsteps, nthreads, p, f, d0)

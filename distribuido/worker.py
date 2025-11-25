# -------------------------------------------------------------
# worker.py
# Um WORKER do sistema distribuído de incêndio florestal.
# Cada processo simula apenas um pedaço da matriz total,
# comunicando-se com os vizinhos via TCP.
# -------------------------------------------------------------

import sys, socket, time, numpy as np, csv
from forest_fire_simulacao import update_block

# -------------------------------------------------------------
# Funções auxiliares para comunicação binária eficiente
# -------------------------------------------------------------

def recv_exact(sock, n):
    """Recebe *exatamente* n bytes via socket."""
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket fechado inesperadamente.")
        data += chunk
    return data

def send_row(sock, row):
    """Envia uma linha inteira de células como bytes."""
    sock.sendall(row.tobytes())

def recv_row(sock, Ny):
    """Recebe uma linha de tamanho Ny e converte em vetor numpy."""
    data = recv_exact(sock, Ny)
    return np.frombuffer(data, dtype=np.int8).copy()

# -------------------------------------------------------------
# Função principal do worker
# -------------------------------------------------------------

def worker_main(rank, num_procs, base_port, Nx, Ny, nsteps, p, f, d0):
    """
    Cada worker:
      - cria um bloco local da matriz
      - abre um servidor TCP na porta base_port + rank
      - conecta-se aos vizinhos (rank-1 e rank+1)
      - troca ghost-rows a cada passo
      - atualiza a simulação usando update_block()
    """

    # Divide as linhas entre os workers
    rows_per = Nx // num_procs
    i0 = rank * rows_per
    i1 = Nx if rank == num_procs - 1 else (rank + 1) * rows_per
    my_rows = i1 - i0

    # Inicializa o bloco local
    grid = np.where(np.random.random((my_rows, Ny)) < d0, 1, 0).astype(np.int8)
    new_grid = grid.copy()

    # Cria servidor para aceitar conexão DO vizinho abaixo
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    my_port = base_port + rank
    server.bind(('127.0.0.1', my_port))
    server.listen(2)

    up_sock = None
    down_sock = None
    accept_conn = {}

    # Aceitação assíncrona (apenas se existe vizinho abaixo)
    def accept_once(name):
        conn, addr = server.accept()
        accept_conn[name] = conn

    # Worker que NÃO é o último vai aceitar conexão do vizinho abaixo
    if rank < num_procs - 1:
        import threading
        thr = threading.Thread(target=accept_once, args=('down',), daemon=True)
        thr.start()

    # Conecta com o vizinho ACIMA
    if rank > 0:
        up_port = base_port + (rank - 1)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Tenta conectar repetidamente
        while True:
            try:
                s.connect(('127.0.0.1', up_port))
                up_sock = s
                break
            except Exception:
                time.sleep(0.05)

    # Espera vizinho inferior conectar
    if rank < num_procs - 1:
        while 'down' not in accept_conn:
            time.sleep(0.01)
        down_sock = accept_conn['down']

    # Medição de tempos
    total_comm = 0.0
    total_comp = 0.0

    # ---------------------------------------------------------
    # Loop principal da simulação
    # ---------------------------------------------------------
    for step in range(nsteps):

        top_row = grid[0, :]
        bottom_row = grid[-1, :]

        # ----------------------------------------------
        # Comunicação: troca ghost-rows
        # ----------------------------------------------
        s_comm = time.perf_counter()

        if up_sock:
            send_row(up_sock, top_row)
            topGhost = recv_row(up_sock, Ny)
        else:
            topGhost = None

        if down_sock:
            send_row(down_sock, bottom_row)
            bottomGhost = recv_row(down_sock, Ny)
        else:
            bottomGhost = None

        e_comm = time.perf_counter()
        total_comm += (e_comm - s_comm)

        # ----------------------------------------------
        # Cálculo local
        # ----------------------------------------------
        s_comp = time.perf_counter()
        new_grid = update_block(grid, topGhost, bottomGhost, p, f)
        e_comp = time.perf_counter()
        total_comp += (e_comp - s_comp)

        # Copia resultado
        grid[:, :] = new_grid

    # Fecha conexões
    if up_sock: up_sock.close()
    if down_sock: down_sock.close()
    server.close()

    # Log do worker
    print(f"[WORKER {rank}] comp={total_comp:.6f}s comm={total_comm:.6f}s total={total_comp+total_comm:.6f}s")

    # Salva CSV
    with open(f"result_worker_{rank}.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'rows', 'Nx_total', 'Ny', 'nsteps', 'p', 'f',
                    'comp_time', 'comm_time', 'total'])
        w.writerow([
            rank, f"{i0}-{i1-1}", Nx, Ny, nsteps, p, f,
            f"{total_comp:.6f}", f"{total_comm:.6f}", f"{total_comp+total_comm:.6f}"
        ])

# -------------------------------------------------------------
# Execução via terminal
# -------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 10:
        print("Usage: python worker.py <rank> <num_procs> <base_port> Nx Ny nsteps p f d0")
        sys.exit(1)

    rank = int(sys.argv[1])
    num_procs = int(sys.argv[2])
    base_port = int(sys.argv[3])
    Nx = int(sys.argv[4])
    Ny = int(sys.argv[5])
    nsteps = int(sys.argv[6])
    p = float(sys.argv[7])
    f = float(sys.argv[8])
    d0 = float(sys.argv[9])

    worker_main(rank, num_procs, base_port, Nx, Ny, nsteps, p, f, d0)

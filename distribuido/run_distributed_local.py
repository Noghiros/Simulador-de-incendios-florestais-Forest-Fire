# -------------------------------------------------------------
# run_distributed_local.py
# Script que inicia vários WORKERS locais (processos separados)
# para executar a versão DISTRIBUÍDA usando TCP.
#
# Este arquivo NÃO faz simulação — apenas lança processos,
# cada um rodando worker.py com parâmetros corretos.
#
# Execução:
#   python run_distributed_local.py <num_procs> <base_port> Nx Ny nsteps p f d0
#
# Exemplo:
#   python run_distributed_local.py 4 9000 256 256 300 0.01 0.001 0.5
# -------------------------------------------------------------

import sys, subprocess, time

# -------------------------------------------------------------
# Verificação de argumentos
# -------------------------------------------------------------
if len(sys.argv) < 9:
    print("Usage: python run_distributed_local.py <num_procs> <base_port> Nx Ny nsteps p f d0")
    sys.exit(1)

# -------------------------------------------------------------
# Parâmetros de execução
# -------------------------------------------------------------
num_procs = int(sys.argv[1])   # quantidade de workers
base_port = int(sys.argv[2])   # porta inicial
Nx = int(sys.argv[3])          # altura total da matriz
Ny = int(sys.argv[4])          # largura total da matriz
nsteps = int(sys.argv[5])      # número de passos da simulação
p = float(sys.argv[6])         # probabilidade de crescimento
f = float(sys.argv[7])         # probabilidade de ignição espontânea
d0 = float(sys.argv[8])        # densidade inicial de árvores

# Lista de processos iniciados
procs = []

print("\n=== Iniciando simulação DISTRIBUÍDA local ===\n")

# -------------------------------------------------------------
# Cria e inicia cada WORKER em um processo separado
# -------------------------------------------------------------
for rank in range(num_procs):

    cmd = [
        sys.executable,      # usa o Python do ambiente virtual
        "worker.py",
        str(rank),
        str(num_procs),
        str(base_port),
        str(Nx),
        str(Ny),
        str(nsteps),
        str(p),
        str(f),
        str(d0)
    ]

    print("Iniciando:", " ".join(cmd))

    # IMPORTANTE: cwd precisa ser "distribuido"
    p_handle = subprocess.Popen(cmd, cwd="distribuido")

    procs.append(p_handle)
    time.sleep(0.05)  # pequena pausa para evitar sobrecarga no servidor TCP

# -------------------------------------------------------------
# Espera todos os workers terminarem
# -------------------------------------------------------------
for p_handle in procs:
    p_handle.wait()

print("\n=== Todos os workers finalizaram ===")

# ---------------------------------------------------------------
# forest_fire_simulacao.py
# Funções auxiliares usadas pelos WORKERS na versão distribuída.
# Aqui cada worker atualiza apenas seu bloco local da matriz,
# usando "ghost rows" recebidas dos vizinhos para evitar bordas
# incorretas na simulação.
# ---------------------------------------------------------------

import numpy as np

# Estados possíveis de cada célula
EMPTY, TREE, BURN = 0, 1, 2

def update_block(block, top_ghost, bottom_ghost, p, f):
    """
    Atualiza um bloco local da simulação de incêndio florestal.
    Cada worker opera apenas sobre um pedaço da matriz original.

    Parâmetros:
        block        -> bloco local (matriz menor do worker)
        top_ghost    -> linha fantasma superior recebida do vizinho acima
        bottom_ghost -> linha fantasma inferior recebida do vizinho abaixo
        p            -> probabilidade de crescimento de árvore
        f            -> probabilidade de ignição espontânea

    Retorna:
        new -> bloco atualizado
    """

    rows, Ny = block.shape

    # Se não existe vizinho acima/abaixo, cria uma borda vazia (EMPTY)
    pad_top = np.full((1, Ny), EMPTY, dtype=np.int8) if top_ghost is None else top_ghost.reshape(1, Ny)
    pad_bottom = np.full((1, Ny), EMPTY, dtype=np.int8) if bottom_ghost is None else bottom_ghost.reshape(1, Ny)

    # Junta: [linha fantasma superior] + [bloco real] + [linha fantasma inferior]
    padded = np.vstack([pad_top, block, pad_bottom])

    # Copia o bloco para construir o novo estado
    new = block.copy()

    # Atualização célula a célula
    for i in range(rows):
        for j in range(Ny):
            s = padded[i+1, j]  # deslocamento +1 por causa da linha fantasma

            # Se queimou, vira EMPTY
            if s == BURN:
                new[i, j] = EMPTY
                continue

            # Se tem árvore
            if s == TREE:
                burning = False

                # Verifica vizinhos Moore (8 direções)
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue  # ignora o próprio pixel

                        ni = i + 1 + di
                        nj = j + dj

                        # Bordas laterais
                        if nj < 0 or nj >= Ny:
                            continue

                        # Se algum vizinho está queimando
                        if padded[ni, nj] == BURN:
                            burning = True
                            break
                    if burning:
                        break

                # Se vizinho queimando, pega fogo
                if burning:
                    new[i, j] = BURN

                # Chance de ignição espontânea
                elif np.random.random() < f:
                    new[i, j] = BURN

                else:
                    new[i, j] = TREE

            else:
                # EMPTY -> pode crescer árvore com probabilidade p
                if np.random.random() < p:
                    new[i, j] = TREE

                else:
                    new[i, j] = EMPTY

    return new

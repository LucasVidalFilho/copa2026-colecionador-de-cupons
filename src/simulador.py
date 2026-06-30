"""
Simulador do problema do colecionador de cupons aplicado ao álbum
da Copa do Mundo FIFA 2026.

Referências
-----------
Flajolet, P. & Sedgewick, R. (2009). Analytic Combinatorics, Cap. IV.
https://en.wikipedia.org/wiki/Coupon_collector%27s_problem
"""

import numpy as np

__all__ = [
    "um_album_simulado",
    "simular_monte_carlo_em_lote",
    "numero_harmonico",
    "valor_esperado_teorico",
    "variancia_teorica",
    "media_acumulada_welford",
]


# ---------------------------------------------------------------------
# Simulação — um colecionador por vez (loop simples)
# ---------------------------------------------------------------------

def um_album_simulado(
    total_figurinhas_no_album: int,
    total_no_pacote: int,
    rng: np.random.Generator,
) -> int:
    """
    Simula um colecionador comprando pacotes até completar o álbum.

    A cada pacote, sorteiam-se `total_no_pacote` figurinhas com reposição
    (podem repetir inclusive dentro do mesmo pacote) dentre as
    `total_figurinhas_no_album` possíveis, até que todas tenham sido
    obtidas ao menos uma vez.

    Parameters
    ----------
    total_figurinhas_no_album : int
        Número total de figurinhas distintas do álbum (N).
    total_no_pacote : int
        Número de figurinhas sorteadas por pacote.
    rng : np.random.Generator
        Gerador de números aleatórios injetado pelo chamador, para que a
        reprodutibilidade seja controlada de fora da função (ver notebook).

    Returns
    -------
    int
        Número de pacotes comprados até completar o álbum.
    """
    album = np.zeros(total_figurinhas_no_album, dtype=bool)
    pacotes_comprados = 0
    figurinhas_unicas = 0

    while figurinhas_unicas < total_figurinhas_no_album:
        pacotes_comprados += 1

        pacote = rng.integers(0, total_figurinhas_no_album, size=total_no_pacote)
        album[pacote] = True

        figurinhas_unicas = album.sum()

    return pacotes_comprados


# ---------------------------------------------------------------------
# Simulação — todos os colecionadores em paralelo (vetorizado)
# ---------------------------------------------------------------------

def simular_monte_carlo_em_lote(
    N: int,
    total_no_pacote: int,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simula B colecionadores em paralelo (vetorizado), em vez de chamar
    `um_album_simulado` B vezes em um loop Python.

    A cada iteração, todos os B álbuns compram um pacote simultaneamente
    (um sorteio (B, total_no_pacote) de uma vez), o que reduz o número de
    chamadas ao gerador aleatório de ~B * E[pacotes] para apenas
    max(pacotes por álbum) — tipicamente uma ordem de grandeza mais rápido.

    Parameters
    ----------
    N : int
        Número total de figurinhas do álbum.
    total_no_pacote : int
        Número de figurinhas por pacote.
    B : int
        Número de colecionadores (réplicas) simulados em paralelo.
    rng : np.random.Generator
        Gerador de números aleatórios.

    Returns
    -------
    np.ndarray
        Array de shape (B,) com o número de pacotes comprados por cada
        um dos B colecionadores simulados.
    """
    album = np.zeros((B, N), dtype=bool)
    pacotes_comprados = np.zeros(B, dtype=np.int64)
    completo = np.zeros(B, dtype=bool)

    while not completo.all():
        # sorteia um pacote (total_no_pacote figurinhas) para cada um dos B álbuns, de uma vez
        sorteio = rng.integers(0, N, size=(B, total_no_pacote))

        # "fancy indexing" 2D: linhas repete cada índice de álbum total_no_pacote vezes,
        # para casar posição a posição com sorteio.ravel() (achatado em ordem de linha)
        linhas = np.repeat(np.arange(B), total_no_pacote)
        album[linhas, sorteio.ravel()] = True

        # só soma +1 pra quem ainda não tinha completado NESTA rodada
        # (True == 1, False == 0 — soma vetorizada, sem if por álbum)
        pacotes_comprados += ~completo

        completo = album.all(axis=1)

    return pacotes_comprados


# ---------------------------------------------------------------------
# Resultados teóricos — Problema do Colecionador de Cupons
# ---------------------------------------------------------------------

def numero_harmonico(N: int) -> float:
    """
    Calcula o N-ésimo número harmônico, H_N = sum(1/k) para k = 1..N.

    Parameters
    ----------
    N : int
        Limite superior da soma.

    Returns
    -------
    float
        Valor de H_N.
    """
    k_vec = np.arange(1, N + 1)
    return float(np.sum(1 / k_vec))


def valor_esperado_teorico(N: int) -> float:
    """
    Valor esperado teórico de T (figurinhas até completar o álbum):
    E[T] = N * H_N.

    Parameters
    ----------
    N : int
        Número total de figurinhas do álbum.

    Returns
    -------
    float
        E[T].
    """
    return N * numero_harmonico(N)


def variancia_teorica(N: int) -> float:
    """
    Variância teórica de T:
    Var(T) = N^2 * sum(1/k^2) - N * H_N, para k = 1..N.

    Parameters
    ----------
    N : int
        Número total de figurinhas do álbum.

    Returns
    -------
    float
        Var(T).
    """
    k_vec = np.arange(1, N + 1)
    return float(N**2 * np.sum(1 / k_vec**2) - N * numero_harmonico(N))


# ---------------------------------------------------------------------
# Estatística online — algoritmo de Welford
# ---------------------------------------------------------------------

def media_acumulada_welford(amostra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula a média e o erro-padrão acumulados ao longo de uma amostra,
    usando o algoritmo online de Welford: O(n) no total, em vez de
    recalcular a variância do zero em cada prefixo (O(n^2)).

    Usado para visualizar a Lei dos Grandes Números: a média acumulada
    converge para E[T] conforme o número de simulações cresce, com o
    erro-padrão encolhendo a uma taxa O(1/sqrt(n)).

    Parameters
    ----------
    amostra : np.ndarray
        Array 1D com as B observações simuladas (ex.: T_sim).

    Returns
    -------
    media_acum : np.ndarray
        Média acumulada após cada nova observação, shape (B,).
    erro_padrao_acum : np.ndarray
        Erro-padrão da média acumulada (ddof=1), shape (B,).
        O primeiro elemento é 0 (variância indefinida com 1 ponto).
    """
    B = len(amostra)
    media_acum = np.zeros(B)
    erro_padrao_acum = np.zeros(B)

    mu = 0.0
    m2 = 0.0  # soma dos quadrados dos desvios (Welford)

    for i, x in enumerate(amostra):
        n = i + 1
        delta = x - mu
        mu += delta / n
        delta2 = x - mu
        m2 += delta * delta2

        media_acum[i] = mu
        if n > 1:
            variancia = m2 / (n - 1)  # ddof=1
            erro_padrao_acum[i] = np.sqrt(variancia / n)

    return media_acum, erro_padrao_acum


# ---------------------------------------------------------------------
# Execução direta do módulo: smoke test rápido
# ---------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    N, k, B = 980, 7, 200

    pacotes = simular_monte_carlo_em_lote(N, k, B, rng)
    T_sim = pacotes * k

    print(f"E[T] teórico   : {valor_esperado_teorico(N):,.1f}")
    print(f"Média simulada : {T_sim.mean():,.1f}  (B={B})")
    print("✓ simulador.py executado sem erros")
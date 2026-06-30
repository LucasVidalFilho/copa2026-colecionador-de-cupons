"""
Simulador do problema do colecionador de cupons aplicado ao álbum
da Copa do Mundo FIFA 2026.
"""

import numpy as np


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
"""
Camada de calibracao (Platt scaling) aplicada DEPOIS da agregacao.

Por que isso existe: a sintese de 11 analises do Metaculus mede ganho de
~0,016 no Brier aplicando Platt scaling sobre a saida do bot. E o ajuste
mais barato que existe: nao gasta token nenhum, so transforma o numero final.

Formato: p' = sigmoid(A * logit(p) + B)
  A > 1  -> empurra para os extremos (bot subconfiante)
  A < 1  -> puxa para 50% (bot superconfiante)
  B != 0 -> corrige vies sistematico para Yes/No

Padrao de fabrica: A=1, B=0, ou seja identidade. Nao invente coeficiente:
rode o bot, junte perguntas resolvidas e ajuste com `fit_platt`.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass

# Metaculus nao aceita 0% nem 100%. O template oficial usa estes limites.
FLOOR = 0.01
CEIL = 0.99

_EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def apply_platt(p: float, a: float = 1.0, b: float = 0.0) -> float:
    """Aplica a transformacao e recorta em [FLOOR, CEIL]."""
    if a == 1.0 and b == 0.0:
        return min(max(p, FLOOR), CEIL)
    return min(max(sigmoid(a * logit(p) + b), FLOOR), CEIL)


# ----------------------------------------------------------------------------
# Ajuste dos coeficientes a partir de perguntas ja resolvidas
# ----------------------------------------------------------------------------


@dataclass
class FitResult:
    a: float
    b: float
    log_loss_before: float
    log_loss_after: float
    brier_before: float
    brier_after: float
    n: int

    def improved(self) -> bool:
        return self.brier_after < self.brier_before


def _log_loss(pairs: list[tuple[float, int]], a: float, b: float) -> float:
    total = 0.0
    for p, y in pairs:
        q = apply_platt(p, a, b)
        total -= math.log(q) if y == 1 else math.log(1 - q)
    return total / len(pairs)


def _brier(pairs: list[tuple[float, int]], a: float, b: float) -> float:
    return sum((apply_platt(p, a, b) - y) ** 2 for p, y in pairs) / len(pairs)


def fit_platt(pairs: list[tuple[float, int]]) -> FitResult:
    """
    Ajusta A e B minimizando log loss por busca em grade com refinamento.

    pairs: lista de (probabilidade_prevista, resultado) com resultado 0 ou 1.

    Busca em grade em vez de gradiente porque o espaco e de 2 dimensoes e
    pequeno: e exato o bastante e nao adiciona dependencia.
    """
    if len(pairs) < 20:
        raise ValueError(
            f"Amostra pequena demais para calibrar: {len(pairs)} perguntas. "
            "Com menos de ~50 resolvidas o ajuste vira ruido. Continue rodando."
        )

    best = (1.0, 0.0, _log_loss(pairs, 1.0, 0.0))
    a_lo, a_hi, b_lo, b_hi = 0.3, 3.0, -1.5, 1.5

    for _ in range(4):  # 4 rodadas de refinamento
        step_a = (a_hi - a_lo) / 20
        step_b = (b_hi - b_lo) / 20
        for i in range(21):
            a = a_lo + i * step_a
            for j in range(21):
                b = b_lo + j * step_b
                loss = _log_loss(pairs, a, b)
                if loss < best[2]:
                    best = (a, b, loss)
        a_lo, a_hi = best[0] - step_a, best[0] + step_a
        b_lo, b_hi = best[1] - step_b, best[1] + step_b

    a, b, loss_after = best
    return FitResult(
        a=round(a, 4),
        b=round(b, 4),
        log_loss_before=_log_loss(pairs, 1.0, 0.0),
        log_loss_after=loss_after,
        brier_before=_brier(pairs, 1.0, 0.0),
        brier_after=_brier(pairs, a, b),
        n=len(pairs),
    )


def load_pairs_from_csv(path: str) -> list[tuple[float, int]]:
    """
    Le um CSV com colunas `prediction` (0-1) e `outcome` (0 ou 1).

    Uma linha por pergunta binaria ja resolvida em que o bot opinou.
    """
    pairs: list[tuple[float, int]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p = float(row["prediction"])
            y = int(row["outcome"])
            if y not in (0, 1):
                raise ValueError(f"outcome deve ser 0 ou 1, veio {y!r}")
            pairs.append((p, y))
    if not pairs:
        raise ValueError(f"Nenhuma linha lida de {path}")
    return pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ajusta os coeficientes de calibracao a partir de perguntas resolvidas."
    )
    parser.add_argument(
        "csv", help="CSV com colunas prediction (0-1) e outcome (0/1)"
    )
    args = parser.parse_args()

    result = fit_platt(load_pairs_from_csv(args.csv))
    print(f"Perguntas resolvidas usadas : {result.n}")
    print(f"Brier sem calibracao        : {result.brier_before:.4f}")
    print(f"Brier com calibracao        : {result.brier_after:.4f}")
    print(f"Log loss sem calibracao     : {result.log_loss_before:.4f}")
    print(f"Log loss com calibracao     : {result.log_loss_after:.4f}")
    print()
    if result.improved():
        print("Coloque no .env:")
        print(f"  CALIBRATION_A={result.a}")
        print(f"  CALIBRATION_B={result.b}")
        if result.a > 1.15:
            print("\nLeitura: o bot esta subconfiante, as previsoes precisam ir mais para os extremos.")
        elif result.a < 0.85:
            print("\nLeitura: o bot esta superconfiante, as previsoes precisam ir mais para 50%.")
        if abs(result.b) > 0.2:
            lado = "Yes" if result.b > 0 else "No"
            print(f"Leitura: ha vies sistematico contra {lado}, B corrige isso.")
    else:
        print("O ajuste nao melhorou o Brier. Mantenha A=1 e B=0 e junte mais dados.")

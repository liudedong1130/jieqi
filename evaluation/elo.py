from __future__ import annotations

INITIAL_ELO: float = 1000.0
K_FACTOR: float = 32.0
ELO_SCALE: float = 400.0


def expected_score(elo_a: float, elo_b: float) -> float:
    """Probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / ELO_SCALE))


def update_elo(
    elo_a: float, elo_b: float, score_a: float, k: float = K_FACTOR
) -> tuple[float, float]:
    """Update Elo ratings after a match.

    Parameters
    ----------
    elo_a, elo_b : float
        Current ratings.
    score_a : float
        Outcome from A's perspective: 1.0 = win, 0.5 = draw, 0.0 = loss.
    k : float
        K-factor (default 32).

    Returns
    -------
    (new_elo_a, new_elo_b)
    """
    ea = expected_score(elo_a, elo_b)
    eb = 1.0 - ea
    new_a = elo_a + k * (score_a - ea)
    new_b = elo_b + k * ((1.0 - score_a) - eb)
    return new_a, new_b

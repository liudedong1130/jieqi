from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.belief_mcts_agent import BeliefMCTSAgent, BeliefState, sample_determinization
from agents.greedy_agent import HIDDEN_ESTIMATE, PIECE_VALUE
from agents.musesfish_agent import MusesfishAgent
from agents.musesfish_cpp_agent import MusesfishCppAgent
from jieqi.env import JieqiEnv
from jieqi.move import pos_to_rc
from rl.ismcts import ISMCTSAgent


_MUSESFISH_CPP_LOCK = threading.Lock()
_MUSESFISH_CPP_AGENT: MusesfishCppAgent | None = None
_MUSESFISH_CPP_CONFIG: tuple[float, int, int] | None = None


@dataclass
class Recommendation:
    move: str = ""
    action: int = 0
    score: float = 0.0
    mean_score: float = 0.0
    p10_score: float = 0.0
    uncertainty: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "move": self.move, "action": self.action,
            "score": round(self.score, 2),
            "mean_score": round(self.mean_score, 2),
            "p10_score": round(self.p10_score, 2),
            "uncertainty": round(self.uncertainty, 2),
            "reasons": self.reasons,
        }


def _format_move(action: int) -> str:
    f, t = action // 90, action % 90
    fr, fc = pos_to_rc(f)
    tr, tc = pos_to_rc(t)
    return f"{fc + 1}路{fr + 1}线→{tc + 1}路{tr + 1}线"


def _check_capture(env: JieqiEnv, action: int) -> str | None:
    to_pos = action % 90
    r, c = pos_to_rc(to_pos)
    obs = env.observation()
    # Channels 26+ are metadata, not board occupancy.
    if obs[:26, r, c].sum() > 0.5:
        return "吃子"
    return None


def _check_reveal(env: JieqiEnv, action: int) -> str | None:
    from_pos = action // 90
    for pos, p in enumerate(env.board.cells):
        if pos == from_pos and p is not None and not p.revealed:
            return "揭开暗子"
    return None


def _check_moves_into_danger(env: JieqiEnv, action: int) -> str | None:
    # Simple heuristic: if action moves piece to a position attacked by opponent
    # For v1, skip detailed check
    return None


def _public_target_value(env: JieqiEnv, action: int) -> int:
    """Estimate the captured piece value from public observation only."""
    to_pos = action % 90
    r, c = pos_to_rc(to_pos)
    obs = env.observation()

    for ch in range(14):
        if obs[ch, r, c] > 0.5:
            return PIECE_VALUE.get(ch % 7, 0)

    for ch in range(14, 26):
        if obs[ch, r, c] > 0.5:
            return HIDDEN_ESTIMATE

    return 0


def _score_greedy_actions(env: JieqiEnv, actions: list[int]) -> list[dict]:
    scored = []
    for a in actions:
        score = float(_public_target_value(env, a))
        scored.append({"action": a, "scores": [score]})
    return scored


def _score_musesfish_actions(env: JieqiEnv, actions: list[int]) -> list[dict]:
    agent = MusesfishAgent(seed=0)
    return [{"action": a, "scores": [agent.score_action(env, a)]} for a in actions]


def _score_musesfish_search_actions(
    env: JieqiEnv,
    actions: list[int],
    *,
    think_time: float,
    search_min_depth: int,
    search_max_depth: int,
) -> list[dict]:
    """Rank actions with the C++ Musesfish search best move first.

    The C++ engine exposes one principal move rather than a full ranked
    policy.  We use its searched best move as the top recommendation and keep
    the lightweight evaluator to order the rest of the list.
    """
    scorer = MusesfishAgent(seed=0, use_original_search=False)
    scored = [{"action": a, "scores": [scorer.score_action(env, a)]} for a in actions]

    global _MUSESFISH_CPP_AGENT, _MUSESFISH_CPP_CONFIG
    cfg = (think_time, search_min_depth, search_max_depth)
    with _MUSESFISH_CPP_LOCK:
        if _MUSESFISH_CPP_AGENT is None or _MUSESFISH_CPP_CONFIG != cfg:
            if _MUSESFISH_CPP_AGENT is not None:
                _MUSESFISH_CPP_AGENT.close()
            _MUSESFISH_CPP_AGENT = MusesfishCppAgent(
                seed=0,
                timeout=think_time,
                min_depth=search_min_depth,
                max_depth=search_max_depth,
                persistent=True,
                fallback=True,
            )
            _MUSESFISH_CPP_CONFIG = cfg
        best = _MUSESFISH_CPP_AGENT.select_action(env)

    if best in actions:
        baseline = max(item["scores"][0] for item in scored) if scored else 0.0
        for item in scored:
            if item["action"] == best:
                item["scores"] = [baseline + 10000.0]
                item["cpp_searched_best"] = True
                break
    return scored


def generate_reasons(env: JieqiEnv, action: int, scores: list[float]) -> list[str]:
    reasons = []
    r = _check_capture(env, action)
    if r:
        reasons.append(r)
    r = _check_reveal(env, action)
    if r:
        reasons.append(r)
    arr = np.array(scores)
    if arr.std() > 50:
        reasons.append("高不确定性")
    elif arr.std() < 10:
        reasons.append("高置信度")
    return reasons


def generate_recommendations(
    env: JieqiEnv,
    agent_type: str = "ismcts",
    top_k: int = 5,
    checkpoint: str | None = None,
    musesfish_search: bool = False,
    musesfish_think_time: float = 3.0,
    musesfish_search_min_depth: int = 5,
    musesfish_search_max_depth: int = 6,
) -> list[Recommendation]:
    """Generate top-k move recommendations with explanations.

    Uses ISMCTS or BeliefMCTS to score each candidate action across
    multiple determinizations, then ranks by composite score and
    attaches human-readable reasons.
    """
    actions = env.legal_actions()
    if len(actions) <= 1:
        a = actions[0] if actions else 0
        return [Recommendation(move=_format_move(a), action=a, reasons=["唯一合法招"])]

    scored: list[dict] = []

    if agent_type == "ismcts":
        agent = ISMCTSAgent(num_simulations=100, max_depth=5, seed=0)
        if hasattr(agent, "get_policy"):
            policy, _ = agent.get_policy(env)
            for a in actions:
                scored.append({"action": a, "scores": [float(policy[a]) * 100]})
        else:
            scored = [{"action": a, "scores": [0.0]} for a in actions]
    elif agent_type == "belief_mcts":
        agent = BeliefMCTSAgent(num_samples=20, seed=0)
        belief = BeliefState.from_env(env)
        import random as _random
        rng = _random.Random(0)
        samples = [sample_determinization(belief, rng) for _ in range(20)]
        player = env.current_player()
        for a in actions:
            s = []
            for bd in samples:
                from agents.belief_mcts_agent import _simulate_action, evaluate_determinized
                board = agent._board
                sim = _simulate_action(bd, a // 90, a % 90)
                s.append(evaluate_determinized(sim, player, board))
            scored.append({"action": a, "scores": s})
    elif agent_type == "greedy":
        scored = _score_greedy_actions(env, actions)
    elif agent_type == "musesfish":
        if musesfish_search:
            scored = _score_musesfish_search_actions(
                env,
                actions,
                think_time=musesfish_think_time,
                search_min_depth=musesfish_search_min_depth,
                search_max_depth=musesfish_search_max_depth,
            )
        else:
            scored = _score_musesfish_actions(env, actions)
    else:
        scored = [{"action": a, "scores": [0.0]} for a in actions]

    results = []
    for item in scored:
        a = item["action"]
        s = item["scores"]
        arr = np.array(s, dtype=np.float64)
        mean_s = float(arr.mean())
        p10 = float(np.percentile(arr, 10))
        std_s = float(arr.std())
        composite = 0.6 * mean_s + 0.3 * p10 - 0.1 * std_s
        results.append(Recommendation(
            move=_format_move(a), action=a,
            score=composite, mean_score=mean_s,
            p10_score=p10, uncertainty=std_s,
            reasons=(["C++搜索首选"] if item.get("cpp_searched_best") else []) + generate_reasons(env, a, s),
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


def recommendations_to_json(recs: list[Recommendation]) -> str:
    return json.dumps([r.to_dict() for r in recs], indent=2, ensure_ascii=False)


def recommendations_to_text(recs: list[Recommendation]) -> str:
    lines = [f"{'Rank':<5} {'Move':<18} {'Score':>8} {'Mean':>8} {'P10':>8} {'Std':>8}  Reasons"]
    lines.append("-" * 80)
    for i, r in enumerate(recs, 1):
        reasons = ", ".join(r.reasons) if r.reasons else "-"
        lines.append(
            f"{i:<5} {r.move:<18} {r.score:>8.1f} {r.mean_score:>8.1f} "
            f"{r.p10_score:>8.1f} {r.uncertainty:>8.1f}  {reasons}"
        )
    return "\n".join(lines)

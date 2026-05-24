from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch

from agents.belief_mcts_agent import (
    BeliefState,
    PIECE_VALUES,
    _apply_to_board,
    _clear_board,
    _hot_positions,
    evaluate_determinized,
    sample_determinization,
)
from jieqi.board import Board
from jieqi.env import JieqiEnv
from jieqi.move import pos_to_rc, rc_to_pos
from jieqi.rules import generate_piece_moves


class ISMCTSNode:
    __slots__ = ("visit_count", "total_value", "children", "prior")

    def __init__(self, prior: float = 0.0) -> None:
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.children: dict[int, ISMCTSNode] = {}
        self.prior: float = prior

    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


def _get_material_evaluator():
    """Return a function that evaluates a board state by material balance."""

    def evaluate(board_state: dict[int, dict], player: int, _board: Board) -> float:
        own = 0.0
        opp = 0.0
        for info in board_state.values():
            v = PIECE_VALUES.get(info["ptype"], 0)
            if info["color"] == player:
                own += v
            else:
                opp += v
        return float(own - opp)

    return evaluate


def _get_policy_value_evaluator(model: Any, device: str):
    """Return a function that evaluates using a policy-value network."""

    def evaluate(board_state: dict[int, dict], player: int, board_obj: Board) -> float:
        _apply_to_board(board_state, board_obj)
        obs = JieqiEnv._encode_observation(board_obj)
        _clear_board(board_obj)
        t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = model(t)
        return float(value.item())

    return evaluate


class ISMCTSAgent:
    """Information-Set MCTS agent with PUCT selection.

    Parameters
    ----------
    num_simulations : int
        MCTS simulations per move (default 100).
    num_determinizations : int
        Determinizations per simulation (default 1 — one det per sim).
    c_puct : float
        Exploration constant for PUCT (default 2.0).
    max_depth : int
        Maximum search depth (default 10).
    evaluator : str
        ``"material"`` or ``"policy_value"``.
    policy_checkpoint : str | None
        Path to policy checkpoint (required for ``"policy_value"`` evaluator).
    temperature : float
        Temperature for action selection from visit counts (default 1.0).
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        num_simulations: int = 200,
        num_determinizations: int = 1,
        c_puct: float = 2.0,
        max_depth: int = 10,
        evaluator: str = "material",
        policy_checkpoint: str | None = None,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.num_simulations = num_simulations
        self.num_determinizations = num_determinizations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.temperature = temperature
        self._rng = random.Random(seed)
        self._board = Board()

        # Evaluator
        if evaluator == "policy_value":
            if policy_checkpoint is None:
                raise ValueError("policy_checkpoint required for policy_value evaluator")
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            ckpt = torch.load(policy_checkpoint, map_location=self.device, weights_only=False)
            from rl.model import _model_from_config
            cfg = ckpt.get("model_config", {"type": "simple_cnn"})
            self._eval_model = _model_from_config(cfg).to(self.device)
            self._eval_model.load_state_dict(ckpt["model"])
            self._eval_model.eval()
            self._evaluator_fn = _get_policy_value_evaluator(self._eval_model, self.device)
            # Also use model for policy prior
            self._use_policy_prior = True
            self._prior_model = self._eval_model
        else:
            self._eval_model = None
            self._evaluator_fn = _get_material_evaluator()
            self._use_policy_prior = False
            self._prior_model = None

    def get_policy(self, env: JieqiEnv) -> tuple[np.ndarray, int]:
        """Return (visit_count_policy, chosen_action).

        visit_count_policy is a (8100,) array summing to 1.
        This is the target distribution for AlphaZero-style training.
        """
        actions = env.legal_actions()
        if len(actions) == 1:
            policy = np.zeros(8100, dtype=np.float32)
            policy[actions[0]] = 1.0
            return policy, actions[0]

        belief = BeliefState.from_env(env)
        obs = env.observation()
        player = env.current_player()

        prior = self._get_prior(obs, env.legal_action_mask()) if self._use_policy_prior else None

        root = ISMCTSNode()
        root.visit_count = 1

        for a in actions:
            p = float(prior[a]) if prior is not None else 1.0 / len(actions)
            root.children[a] = ISMCTSNode(prior=p)

        for _ in range(self.num_simulations):
            det_board = sample_determinization(belief, self._rng)
            self._simulate(root, det_board, actions, belief, player, depth=0)

        counts = np.zeros(8100, dtype=np.float64)
        for a in actions:
            counts[a] = root.children[a].visit_count

        counts = counts ** (1.0 / max(self.temperature, 0.1))
        total = counts.sum()
        probs = counts / total if total > 0 else np.ones(8100) / 8100

        r = self._rng.random()
        cum = 0.0
        chosen = actions[-1]
        for a in actions:
            cum += probs[a]
            if r <= cum:
                chosen = a
                break

        return probs.astype(np.float32), chosen

    def select_action(self, env: JieqiEnv) -> int:
        actions = env.legal_actions()
        if len(actions) == 1:
            return actions[0]

        belief = BeliefState.from_env(env)
        obs = env.observation()
        player = env.current_player()

        # Get policy prior if available
        prior = self._get_prior(obs, env.legal_action_mask()) if self._use_policy_prior else None

        root = ISMCTSNode()
        root.visit_count = 1

        # Pre-expand children for legal actions
        for a in actions:
            p = float(prior[a]) if prior is not None else 1.0 / len(actions)
            root.children[a] = ISMCTSNode(prior=p)

        for sim in range(self.num_simulations):
            det_board = sample_determinization(belief, self._rng)
            self._simulate(root, det_board, actions, belief, player, depth=0)

        # Build visit-count policy
        counts = np.zeros(8100, dtype=np.float64)
        for a in actions:
            counts[a] = root.children[a].visit_count

        if self.temperature == 0:
            best = int(np.argmax(counts))
            return best if counts[best] > 0 else self._rng.choice(actions)

        counts = counts ** (1.0 / self.temperature)
        total = counts.sum()
        if total <= 0:
            return self._rng.choice(actions)
        probs = counts / total

        # Sample from distribution
        r = self._rng.random()
        cum = 0.0
        for a in actions:
            cum += probs[a]
            if r <= cum:
                return a
        return actions[-1]

    def _simulate(
        self,
        node: ISMCTSNode,
        det_board: dict[int, dict],
        legal_actions: list[int],
        belief: BeliefState,
        player: int,
        depth: int,
    ) -> float:
        """Run one simulation from *node* on *det_board*."""
        if depth >= self.max_depth or not legal_actions:
            return self._evaluate(det_board, player)

        # Selection: PUCT
        sqrt_N = math.sqrt(max(node.visit_count, 1))
        best_action = legal_actions[0]
        best_score = -float("inf")
        for a in legal_actions:
            child = node.children.get(a)
            if child is None:
                child = ISMCTSNode(prior=1.0 / max(len(legal_actions), 1))
                node.children[a] = child
            q = child.q()
            u = self.c_puct * child.prior * sqrt_N / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = a

        # Apply action on determinized board
        next_board = _simulate_action(det_board, best_action)

        # Get legal actions for next player in the determinized board
        _apply_to_board(next_board, self._board)
        self._board._turn = self._board._turn.opposite() if hasattr(self._board, '_turn') else None
        # Actually, generate moves for the next player
        # For simplicity in ISMCTS, estimate next legal actions
        next_player = 1 - player
        next_legal = self._approx_legal_actions(next_board, next_player)
        _clear_board(self._board)

        # Recursively simulate
        child_node = node.children[best_action]
        value = self._simulate(child_node, next_board, next_legal, belief, next_player, depth + 1)

        # Backup (from this node's player's perspective)
        # value is from next_player's perspective, negate for current player
        value_for_this = -value
        node.visit_count += 1
        node.total_value += value_for_this
        child_node.visit_count += 1
        child_node.total_value += value_for_this
        return value_for_this

    def _evaluate(self, det_board: dict[int, dict], player: int) -> float:
        return self._evaluator_fn(det_board, player, self._board)

    def _get_prior(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        if self._prior_model is None:
            return None
        t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self._prior_model(t)
        logits = logits[0].cpu().numpy()
        logits[mask == 0] = -1e9
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    def _approx_legal_actions(self, board_state: dict[int, dict], player: int) -> list[int]:
        """Estimate legal actions on the determinized board."""
        actions = []
        for pos, info in board_state.items():
            if info["color"] != player:
                continue
            r, c = pos_to_rc(pos)
            moves = generate_piece_moves(self._board, r, c)
            for mv in moves:
                a = mv.from_pos * 90 + mv.to_pos
                if a < 8100:
                    actions.append(a)
        return actions if actions else [0]  # fallback


def _simulate_action(board: dict[int, dict], action: int) -> dict[int, dict]:
    """Return new board after executing action on determinized board."""
    from_pos = action // 90
    to_pos = action % 90
    sim = {pos: dict(info) for pos, info in board.items()}
    mover = sim.pop(from_pos, None)
    if mover is None:
        return sim
    mover = dict(mover)
    mover["revealed"] = True
    sim.pop(to_pos, None)
    sim[to_pos] = mover
    return sim

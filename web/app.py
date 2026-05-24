"""Minimal FastAPI backend for Jieqi board analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analysis.recommendation import generate_recommendations
from jieqi.env import JieqiEnv
from jieqi.move import pos_to_rc
from vision.adapter import VisionBoardState, game_state_to_vision_state, vision_state_to_game_state

app = FastAPI(title="Jieqi Web UI")
import random as _random

_env = JieqiEnv(max_steps=500)
_env.reset(seed=_random.randint(0, 999999))

# Chinese piece labels
_CN = {
    0: "帅", 1: "仕", 2: "相", 3: "馬", 4: "車", 5: "炮", 6: "兵",  # 帅仕相馬車炮兵
}
_CN_B = {
    0: "将", 1: "士", 2: "象", 3: "马", 4: "车", 5: "炮", 6: "卒",  # 将士象马车炮卒
}


class AnalyzeRequest(BaseModel):
    agent: str = "ismcts"
    top_k: int = 5


class MoveRequest(BaseModel):
    action: int


# ---------------------------------------------------------------------------
#  API — use persistent _env (no reset, preserves true_type)
# ---------------------------------------------------------------------------


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    recs = generate_recommendations(_env, agent_type=req.agent, top_k=req.top_k)
    return {"recommendations": [r.to_dict() for r in recs]}


@app.get("/api/legal_moves")
async def legal_moves() -> dict[str, Any]:
    moves = []
    for a in _env.legal_actions():
        f, t = a // 90, a % 90
        fr, fc = pos_to_rc(f)
        tr, tc = pos_to_rc(t)
        moves.append({"action": a, "from": [fr, fc], "to": [tr, tc]})
    return {"legal_moves": moves}


@app.post("/api/apply_move")
async def apply_move(req: MoveRequest) -> dict[str, Any]:
    if req.action not in _env.legal_actions():
        raise HTTPException(400, f"Illegal action: {req.action}")
    _env.step(req.action)
    return _board_dict()


@app.get("/api/board_state")
async def board_state(debug: bool = Query(False)) -> dict[str, Any]:
    d = _board_dict()
    if debug:
        for c in d["cells"]:
            pos = c["row"] * 9 + c["col"]
            p = _env.board[pos]
            if p is not None and p.revealed:
                c["true_type"] = int(p.true_type)
    return d


@app.post("/api/load_fen")
async def load_fen(req: dict) -> dict[str, Any]:
    from engine.jieqi_fen import parse_jieqi_fen
    try:
        state = parse_jieqi_fen(req["fen"])
    except Exception as exc:
        raise HTTPException(400, f"Invalid FEN: {exc}") from exc
    vs_cells = []
    for p in state["pieces"]:
        st = ("red_open" if p["revealed"] else "red_hidden") if p["color"] == 0 else ("black_open" if p["revealed"] else "black_hidden")
        vs_cells.append({"row": p["pos"] // 9, "col": p["pos"] % 9, "state": st, "piece_type": p["type"]})
    vs = VisionBoardState(cells=vs_cells, current_player=state["current_player"])
    _env.reset()
    try:
        vision_state_to_game_state(vs, _env, seed=_random.randint(0, 999999))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _board_dict()


@app.post("/api/load_state")
async def load_state(req: dict) -> dict[str, Any]:
    vs = VisionBoardState.from_dict(req)
    _env.reset()
    try:
        vision_state_to_game_state(vs, _env, seed=_random.randint(0, 999999))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _board_dict()


@app.post("/api/reset")
async def reset() -> dict[str, Any]:
    _env.reset(seed=_random.randint(0, 999999))
    return _board_dict()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _board_dict() -> dict[str, Any]:
    vs = game_state_to_vision_state(_env)
    d = vs.to_dict()
    for c in d["cells"]:
        pt = c["piece_type"]
        st = c["state"]
        if st == "red_open":
            c["label"] = _CN.get(pt, "?")
        elif st == "black_open":
            c["label"] = _CN_B.get(pt, "?")
        elif "hidden" in st:
            c["label"] = "暗"
    # Captured pieces (revealed, so capturer can see them)
    captured = []
    for p in _env.board.captured:
        if p.revealed:
            label = _CN.get(int(p.true_type), "?") if p.color == 0 else _CN_B.get(int(p.true_type), "?")
        else:
            label = "暗"
        captured.append({
            "color": int(p.color), "true_type": int(p.true_type),
            "revealed": p.revealed, "label": label,
        })
    d["captured"] = captured
    return d


# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

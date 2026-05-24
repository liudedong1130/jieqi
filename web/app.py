"""Minimal FastAPI backend for Jieqi board analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analysis.recommendation import generate_recommendations, recommendations_to_json
from jieqi.env import JieqiEnv
from jieqi.move import pos_to_rc
from vision.adapter import VisionBoardState, game_state_to_vision_state, vision_state_to_game_state

app = FastAPI(title="Jieqi Web UI")
_env = JieqiEnv(max_steps=500)


# ---------------------------------------------------------------------------
#  Models
# ---------------------------------------------------------------------------


class BoardCell(BaseModel):
    row: int
    col: int
    state: str = "empty"
    piece_type: int = 0


class BoardState(BaseModel):
    cells: list[BoardCell] = []
    current_player: int = 0


class AnalyzeRequest(BaseModel):
    board: BoardState
    agent: str = "ismcts"
    top_k: int = 5


class MoveRequest(BaseModel):
    board: BoardState
    action: int


# ---------------------------------------------------------------------------
#  API endpoints
# ---------------------------------------------------------------------------


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    vs = _to_vision_state(req.board)
    _apply_vision(vs)
    recs = generate_recommendations(_env, agent_type=req.agent, top_k=req.top_k)
    return {"recommendations": [r.to_dict() for r in recs]}


@app.post("/api/legal_moves")
async def legal_moves(req: BoardState) -> dict[str, Any]:
    vs = _to_vision_state(req)
    _apply_vision(vs)
    actions = _env.legal_actions()
    moves = []
    for a in actions:
        f, t = a // 90, a % 90
        fr, fc = pos_to_rc(f)
        tr, tc = pos_to_rc(t)
        moves.append({"action": a, "from": [fr, fc], "to": [tr, tc]})
    return {"legal_moves": moves}


@app.post("/api/apply_move")
async def apply_move(req: MoveRequest) -> dict[str, Any]:
    vs = _to_vision_state(req.board)
    _apply_vision(vs)
    if req.action not in _env.legal_actions():
        raise HTTPException(400, f"Illegal action: {req.action}")
    _env.step(req.action)
    vs_out = game_state_to_vision_state(_env)
    return {"board": vs_out.to_dict()}


@app.get("/api/board_state")
async def board_state(debug: bool = Query(False)) -> dict[str, Any]:
    vs = game_state_to_vision_state(_env)
    result: dict[str, Any] = vs.to_dict()
    if debug:
        # Debug mode: include true_type for revealed pieces only
        for c in result["cells"]:
            pos = c["row"] * 9 + c["col"]
            p = _env.board[pos]
            if p is not None and p.revealed:
                c["true_type"] = int(p.true_type)
    return result


@app.post("/api/load_fen")
async def load_fen(req: dict) -> dict[str, Any]:
    from engine.jieqi_fen import parse_jieqi_fen
    state = parse_jieqi_fen(req["fen"])
    vs_cells = []
    for p in state["pieces"]:
        st = ("red_open" if p["revealed"] else "red_hidden") if p["color"] == 0 else ("black_open" if p["revealed"] else "black_hidden")
        vs_cells.append(BoardCell(row=p["pos"]//9, col=p["pos"]%9, state=st, piece_type=p["type"]))
    vs = VisionBoardState(cells=[c.model_dump() for c in vs_cells], current_player=state["current_player"])
    _apply_vision(vs)
    return {"status": "ok"}


@app.post("/api/reset")
async def reset() -> dict[str, Any]:
    _env.reset(seed=42)
    vs = game_state_to_vision_state(_env)
    return {"board": vs.to_dict()}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _to_vision_state(board: BoardState) -> VisionBoardState:
    return VisionBoardState(
        cells=[c.model_dump() for c in board.cells],
        current_player=board.current_player,
    )


def _apply_vision(vs: VisionBoardState) -> None:
    _env.reset()
    vision_state_to_game_state(vs, _env)


# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

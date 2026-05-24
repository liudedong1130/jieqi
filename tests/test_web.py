from __future__ import annotations

import json

from fastapi.testclient import TestClient

from web.app import app

client = TestClient(app)


class TestWebAPI:
    def test_analyze_returns_recommendations(self) -> None:
        client.post("/api/reset")
        resp = client.post("/api/analyze", json={"agent": "ismcts", "top_k": 3})
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        assert len(recs) <= 3
        assert "move" in recs[0]

    def test_greedy_analyze_scores_public_captures(self) -> None:
        client.post("/api/load_state", json={
            "current_player": 0,
            "cells": [
                {"row": 9, "col": 4, "state": "red_open", "piece_type": 0},
                {"row": 0, "col": 4, "state": "black_open", "piece_type": 0},
                {"row": 4, "col": 4, "state": "red_open", "piece_type": 4},
                {"row": 6, "col": 0, "state": "red_open", "piece_type": 6},
                {"row": 5, "col": 0, "state": "black_open", "piece_type": 6},
            ],
        })
        resp = client.post("/api/analyze", json={"agent": "greedy", "top_k": 3})
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        assert recs[0]["score"] > 0
        assert "吃子" in recs[0]["reasons"]

    def test_legal_moves(self) -> None:
        client.post("/api/reset")
        resp = client.get("/api/legal_moves")
        assert resp.status_code == 200
        assert len(resp.json()["legal_moves"]) > 0

    def test_apply_move(self) -> None:
        client.post("/api/reset")
        resp = client.get("/api/legal_moves")
        action = resp.json()["legal_moves"][0]["action"]
        resp = client.post("/api/apply_move", json={"action": action})
        assert resp.status_code == 200
        # Board should have Chinese labels
        cells = resp.json()["cells"]
        assert any("label" in c for c in cells)

    def test_illegal_move_rejected(self) -> None:
        client.post("/api/reset")
        resp = client.post("/api/apply_move", json={"action": 9999})
        assert resp.status_code == 400

    def test_debug_false_no_true_type(self) -> None:
        client.post("/api/reset")
        resp = client.get("/api/board_state?debug=false")
        for c in resp.json()["cells"]:
            assert "true_type" not in c

    def test_debug_true_includes_true_type(self) -> None:
        client.post("/api/reset")
        resp = client.get("/api/board_state?debug=true")
        has_true = any("true_type" in c for c in resp.json()["cells"])
        assert has_true

    def test_frontend_served(self) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "揭棋" in resp.text

    def test_load_state_updates_board(self) -> None:
        resp = client.post("/api/load_state", json={
            "current_player": 0,
            "cells": [
                {"row": 9, "col": 4, "state": "red_open", "piece_type": 0},
                {"row": 0, "col": 4, "state": "black_open", "piece_type": 0},
                {"row": 6, "col": 0, "state": "red_hidden", "piece_type": 6},
            ],
        })

        assert resp.status_code == 200
        cells = resp.json()["cells"]
        assert any(c["row"] == 6 and c["col"] == 0 and c["state"] == "red_hidden" for c in cells)

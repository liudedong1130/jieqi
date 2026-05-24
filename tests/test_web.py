from __future__ import annotations

import json

from fastapi.testclient import TestClient

from web.app import app

client = TestClient(app)


class TestWebAPI:
    def test_analyze_returns_recommendations(self) -> None:
        # First reset
        resp = client.post("/api/reset")
        data = resp.json()
        board = data["board"]

        resp = client.post("/api/analyze", json={
            "board": board, "agent": "ismcts", "top_k": 3,
        })
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        assert len(recs) <= 3
        assert "move" in recs[0]
        assert "score" in recs[0]

    def test_legal_moves(self) -> None:
        resp = client.post("/api/reset")
        board = resp.json()["board"]
        resp = client.post("/api/legal_moves", json=board)
        assert resp.status_code == 200
        assert len(resp.json()["legal_moves"]) > 0

    def test_apply_move(self) -> None:
        resp = client.post("/api/reset")
        board = resp.json()["board"]
        resp = client.post("/api/legal_moves", json=board)
        actions = resp.json()["legal_moves"]
        action = actions[0]["action"]

        resp = client.post("/api/apply_move", json={"board": board, "action": action})
        assert resp.status_code == 200

    def test_illegal_move_rejected(self) -> None:
        resp = client.post("/api/reset")
        board = resp.json()["board"]
        resp = client.post("/api/apply_move", json={"board": board, "action": 9999})
        assert resp.status_code in (400, 422)

    def test_debug_false_no_true_type(self) -> None:
        resp = client.get("/api/board_state?debug=false")
        assert resp.status_code == 200
        for c in resp.json()["cells"]:
            assert "true_type" not in c

    def test_debug_true_includes_true_type(self) -> None:
        resp = client.get("/api/board_state?debug=true")
        assert resp.status_code == 200
        # At least revealed kings should have true_type
        has_true = any("true_type" in c for c in resp.json()["cells"])
        assert has_true

    def test_frontend_served(self) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Jieqi" in resp.text

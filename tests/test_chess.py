import json
from pathlib import Path

import chess
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from lidar_video.chess_recognition import features, probabilities, recognize, save_forest
from lidar_video.chess_scene import (
    board_from_observation,
    sample_legal_scenarios,
    scene_graph,
    transition_graph,
)


def observation(board):
    return {"symbols": [board.piece_at(s).symbol() if board.piece_at(s) else "." for s in chess.SQUARES]}


CONTEXT = dict(turn="w", castling="KQkq", en_passant="-", halfmove=0, fullmove=1)


def test_visual_placement_requires_explicit_history():
    obs = observation(chess.Board())
    with pytest.raises(ValueError, match="images do not establish history"):
        board_from_observation(obs, None)
    assert board_from_observation(obs, CONTEXT).fen() == chess.STARTING_FEN
    black = board_from_observation(obs, dict(CONTEXT, turn="b"))
    assert not black.turn


@pytest.mark.parametrize("bad", ["?", "", "PN", "z", None])
def test_unresolved_or_malformed_squares_cannot_generate_game_state(bad):
    obs = observation(chess.Board())
    obs["symbols"][0] = bad
    with pytest.raises(ValueError, match="64 squares"):
        board_from_observation(obs, CONTEXT)


def test_impossible_observation_not_silently_repaired():
    obs = observation(chess.Board())
    obs["symbols"][chess.E1] = "."
    with pytest.raises(ValueError, match="illegal"):
        board_from_observation(obs, CONTEXT)


def test_legal_transition_updates_turn_and_keeps_piece_id():
    graph = scene_graph(chess.Board())
    successor = transition_graph(graph, "e2e4")
    assert successor["transition"]["mover"] == "piece_e2"
    assert next(n for n in successor["nodes"] if n["id"] == "piece_e2")["square"] == "e4"
    assert chess.Board(successor["chess"]["fen"]).turn == chess.BLACK
    with pytest.raises(ValueError, match="Illegal move"):
        transition_graph(graph, "e2e5")


@pytest.mark.parametrize(
    "fen,move,removed,mover,kind,extra",
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", None, "piece_e1", "king", ("piece_h1", "f1")),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6", "piece_d5", "piece_e5", "pawn", None),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", None, "piece_a7", "queen", None),
        ("4k3/8/8/8/8/8/r7/R3K3 w Q - 0 1", "a1a2", "piece_a2", "piece_a1", "rook", None),
    ],
)
def test_special_moves_preserve_identity_and_correct_occupancy(fen, move, removed, mover, kind, extra):
    successor = transition_graph(scene_graph(chess.Board(fen)), move)
    by_id = {n["id"]: n for n in successor["nodes"]}
    assert removed not in by_id
    assert by_id[mover]["semantic_type"] == kind
    if extra:
        assert by_id[extra[0]]["square"] == extra[1]
    expected = chess.Board(fen)
    expected.push_uci(move)
    assert successor["chess"]["fen"] == expected.fen(en_passant="fen")
    occupancy = [n["square"] for n in by_id.values() if n["kind"] == "object"]
    assert len(occupancy) == len(set(occupancy)) == len(expected.piece_map())


def test_scenic_samples_are_legal_and_piece_bounds_stay_inside_squares(tmp_path):
    board = chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")
    report = sample_legal_scenarios(scene_graph(board), tmp_path / "scenarios", count=4, seed=43)
    for record in report["samples"]:
        assert chess.Move.from_uci(record["move"]) in board.legal_moves
        graph = json.loads((tmp_path / "scenarios" / record["graph"]).read_text())
        nodes = {n["id"]: n for n in graph["nodes"]}
        for node in nodes.values():
            if node["kind"] == "object":
                square = nodes[node["square"]]
                assert np.all(
                    np.abs(np.array(node["position_m"][:2]) - square["position_m"][:2])
                    + np.array(node["size_m"][:2]) / 2
                    <= 0.0225
                )


def test_numeric_model_export_matches_sklearn_on_unseen_features(tmp_path):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(100, 5))
    y = np.where(x[:, 0] > 0, "P", "p")
    forest = RandomForestClassifier(n_estimators=11, random_state=7).fit(x, y)
    path = tmp_path / "forest.npz"
    save_forest(forest, path)
    heldout = rng.normal(size=(40, 5))
    scores, classes = probabilities(path, heldout)
    assert np.array_equal(classes, forest.classes_)
    assert np.allclose(scores, forest.predict_proba(heldout))


def test_colab_notebook_cells_compile_and_have_no_claimed_execution():
    path = Path(__file__).parents[1] / "notebooks/roboskills_gpu_stages.ipynb"
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), path.name, "exec")
            assert cell["execution_count"] is None and not cell["outputs"]


def test_unobserved_board_abstains_even_with_strong_classifier_votes(tmp_path):
    forest = RandomForestClassifier(n_estimators=3, random_state=7).fit(
        np.array([np.zeros(160), np.ones(160)]), np.array([".", "P"])
    )
    model = tmp_path / "model.npz"
    save_forest(forest, model)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[:, 0, 3] = 10  # valid sensor data, but the board is outside its field of view
    capture = dict(
        rgb=np.zeros((2, 8, 8, 3), np.uint8),
        depth=np.ones((2, 8, 8), np.float32),
        confidence=np.full((2, 8, 8), 2, np.uint8),
        K=np.array([[8, 0, 3.5], [0, 8, 3.5], [0, 0, 1.0]]),
        world_from_camera=poses,
        rgb_time=np.array([0.0, 0.1]),
        depth_time=np.array([0.0, 0.1]),
    )
    calibration = dict(square_size_m=0.045, board_from_world=np.eye(4).tolist())
    result = recognize(capture, calibration, model)
    assert result["unknown_count"] == 64 and set(result["symbols"]) == {"?"}
    calibration["board_from_world"][0][0] = 2
    with pytest.raises(ValueError, match="proper rigid transform"):
        features(capture, calibration)

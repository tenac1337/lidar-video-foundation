"""Recognized chess placement -> rule-checked state -> graph -> two-stage Scenic.

Visual placement cannot recover side to move, castling rights, en-passant or clocks.
Those fields MUST be supplied; they are never silently inferred from an image.
"""

import copy
import json
import random
from pathlib import Path

import chess
import numpy as np
import scenic

from lidar_video.capture import write_json
from lidar_video.chess_data import HEIGHTS, SQUARE, SYMBOLS, render_board, square_xy
from lidar_video.chess_recognition import recognize
from lidar_video.scenic_export import export_scenic, sample_scenic


def board_from_observation(observation, context):
    if not observation.get("board_orientation", {}).get("resolved", True):
        raise ValueError("Resolve the physical a1 orientation before creating chess game state")
    symbols = observation["symbols"]
    if len(symbols) != 64 or any(not isinstance(s, str) or len(s) != 1 or s not in SYMBOLS for s in symbols):
        raise ValueError("All 64 squares must be recognized; unresolved squares cannot become a legal board")
    fields = {"turn", "castling", "en_passant", "halfmove", "fullmove"}
    if context is None or fields - context.keys():
        raise ValueError(
            "Supply turn, castling, en_passant, halfmove and fullmove; images do not establish history"
        )
    if context["turn"] not in ("w", "b"):
        raise ValueError("turn must be w or b")
    board = chess.Board(None)
    for square, symbol in enumerate(symbols):
        if symbol != ".":
            board.set_piece_at(square, chess.Piece.from_symbol(symbol))
    fen = " ".join(
        [
            board.board_fen(),
            *[str(context[k]) for k in ("turn", "castling", "en_passant", "halfmove", "fullmove")],
        ]
    )
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"Recognized board plus supplied context is illegal: status={board.status()}")
    return board


def scene_graph(board, evidence="rule-checked state", ids=None):
    if not board.is_valid():
        raise ValueError("Cannot export an illegal chess board")
    nodes = [
        dict(
            id="board",
            kind="support",
            position_m=[0, 0, -0.0075],
            size_m=[8 * SQUARE, 8 * SQUARE, 0.015],
            color_rgb=[0.55, 0.45, 0.3],
            evidence="supplied board calibration",
        )
    ]
    edges = []
    for square in chess.SQUARES:
        name = chess.square_name(square)
        nodes.append(
            dict(
                id=name,
                kind="square",
                position_m=[*square_xy(square), 0],
                file=chess.square_file(square),
                rank=chess.square_rank(square) + 1,
            )
        )
        edges.append(dict(source=name, relation="part_of", target="board", evidence="board definition"))
    for square, piece in board.piece_map().items():
        name = chess.square_name(square)
        symbol = piece.symbol()
        height = HEIGHTS[symbol.upper()]
        ident = ids[square] if ids is not None else f"piece_{name}"
        nodes.append(
            dict(
                id=ident,
                kind="object",
                semantic_type=chess.piece_name(piece.piece_type),
                symbol=symbol,
                color="white" if piece.color else "black",
                square=name,
                position_m=[*square_xy(square), height / 2],
                size_m=[0.028, 0.028, height],
                color_rgb=[0.78] * 3 if piece.color else [0.15] * 3,
                evidence=evidence,
                geometry="class-conditioned box proxy, not a recovered mesh",
            )
        )
        edges.append(dict(source=ident, relation="occupies", target=name, evidence=evidence))
        edges.append(
            dict(source=ident, relation="on", target="board", evidence="resting-on-board hypothesis")
        )
    return dict(
        schema_version=1,
        source=evidence,
        time_s=0.0,
        frame=0,
        tracks=[],
        nodes=nodes,
        edges=edges,
        chess=dict(
            fen=board.fen(en_passant="fen"),
            legal_moves=sorted(m.uci() for m in board.legal_moves),
            context_source="explicit supplied history, not inferred from RGB-D",
        ),
    )


def transition_graph(graph, uci):
    board = chess.Board(graph["chess"]["fen"])
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move: {uci}")
    ids = {chess.parse_square(n["square"]): n["id"] for n in graph["nodes"] if n["kind"] == "object"}
    mover = ids.pop(move.from_square)
    captured_square = move.to_square
    if board.is_en_passant(move):
        captured_square += -8 if board.turn else 8
    captured = ids.pop(captured_square, None)
    ids[move.to_square] = mover
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        ids[rook_to] = ids.pop(rook_from)
    before = board.fen(en_passant="fen")
    board.push(move)
    result = scene_graph(board, evidence="sampled legal successor", ids=ids)
    result["transition"] = dict(
        uci=uci,
        mover=mover,
        captured=captured,
        promotion=move.promotion,
        before_fen=before,
        after_fen=board.fen(en_passant="fen"),
    )
    return result


def spatial_program(graph, path):
    # The spatial stage uses proxies only; square nodes remain in the semantic graph.
    proxies = copy.deepcopy(graph)
    proxies["nodes"] = [n for n in graph["nodes"] if n["kind"] in ("object", "support")]
    export_scenic(proxies, path, jitter=0.002)
    text = (
        Path(path)
        .read_text()
        .replace(
            "# Box proxies and fixed support; no chess legality or dynamic behaviors.",
            "# Box proxies after legal successor selection; each piece stays inside its assigned square.",
        )
    )
    for i, node in enumerate(proxies["nodes"]):
        if node["kind"] != "object":
            continue
        x, y = square_xy(chess.parse_square(node["square"]))
        radius = node["size_m"][0] / 2
        for axis, center in (("x", x), ("y", y)):
            text += f"\nrequire obj_{i}.position.{axis} >= {float(center - SQUARE / 2 + radius)!r}"
            text += f"\nrequire obj_{i}.position.{axis} <= {float(center + SQUARE / 2 - radius)!r}"
    Path(path).write_text(text + "\n")
    return proxies


def sample_legal_scenarios(graph, out, count=20, seed=7):
    if count < 1:
        raise ValueError("count must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    board = chess.Board(graph["chess"]["fen"])
    moves = sorted(m.uci() for m in board.legal_moves)
    if not moves:
        raise ValueError("Terminal board has no legal successors")
    # Discrete legal choice then continuous spatial placement. Two stages support
    # captures, promotions and castling without phantom/off-board captured objects.
    selector = out / "legal_moves.scenic"
    selector.write_text(
        "# Uniform legal-move variation, NOT a chess-playing policy.\n"
        "# Stage 2 materializes the successor pieces and samples within-square offsets.\n"
        f"param initialFEN = {board.fen(en_passant='fen')!r}\n"
        f"param move = Options({moves!r})\n"
        "ego = new Object at (0, 0, -0.0075), with width 0.36, with length 0.36, with height 0.015, with requireVisible False\n"
    )
    random.seed(seed)
    np.random.seed(seed)
    scenario = scenic.scenarioFromFile(str(selector.resolve()))
    choices = [scenario.generate(maxIterations=100)[0].params["move"] for _ in range(count)]
    records = []
    for i, uci in enumerate(choices):
        successor = transition_graph(graph, uci)
        program = out / f"successor-{i:03d}.scenic"
        proxies = spatial_program(successor, program)
        sampled = sample_scenic(program, proxies, out / f"spatial-{i:03d}.json", count=1, seed=seed + i)[0]
        positions = {n["id"]: n["position_m"] for n in sampled["nodes"]}
        for node in successor["nodes"]:
            if node["id"] in positions:
                node["position_m"] = positions[node["id"]]
        successor["sampling"] = dict(
            seed=seed + i, xy_jitter_m=0.002, evidence="scenario variation, not sensor posterior"
        )
        write_json(out / f"graph-{i:03d}.json", successor)
        records.append(
            dict(
                index=i,
                move=uci,
                after_fen=successor["chess"]["fen"],
                graph=f"graph-{i:03d}.json",
                spatial_program=program.name,
            )
        )
    result = dict(
        count=count,
        seed=seed,
        distinct_moves=len(set(choices)),
        legal_choices=moves,
        samples=records,
        method="Scenic discrete legal move then Scenic continuous within-square placement",
    )
    write_json(out / "samples.json", result)
    return result


def chess_demo(model_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    # Truth is used only to render and evaluate; recognition receives capture+calibration.
    board = chess.Board()
    truth = [board.piece_at(s).symbol() if board.piece_at(s) else "." for s in chess.SQUARES]
    capture = render_board(out / "input", truth, seed=77123)
    calibration = dict(square_size_m=SQUARE, board_from_world=np.eye(4).tolist())
    prediction = recognize(capture, calibration, model_path)
    write_json(out / "recognition.json", prediction)
    context = dict(turn="w", castling="KQkq", en_passant="-", halfmove=0, fullmove=1)
    write_json(out / "supplied_game_context.json", context)
    result = dict(
        correct_squares=sum(a == b for a, b in zip(prediction["symbols"], truth)),
        unknown_squares=prediction["unknown_count"],
        context="explicit fixture history",
        full_board_correct=prediction["symbols"] == truth,
    )
    try:
        observed = board_from_observation(prediction, context)
    except ValueError as error:
        result.update(status="recognition_needs_review", reason=str(error))
        write_json(out / "verification.json", result)
        return result
    graph = scene_graph(observed, evidence=prediction["evidence"])
    write_json(out / "scene_graph.json", graph)
    samples = sample_legal_scenarios(graph, out / "scenarios")
    from lidar_video.simulation import simulate

    selected = json.loads((out / "scenarios/graph-000.json").read_text())
    selected["nodes"] = [n for n in selected["nodes"] if n["kind"] != "square"]
    selected["edges"] = [e for e in selected["edges"] if e["relation"] == "on"]
    simulate(selected, out / "sample_simulation", frames=30)
    result.update(
        status="completed",
        sampled_legal_scenarios=samples["count"],
        recognized_fen=observed.fen(),
        distinct_moves=samples["distinct_moves"],
        simulation="first successor, native passive box-proxy physics",
    )
    write_json(out / "verification.json", result)
    return result

# Constructive scene programs

`scene-program` compiles an attributed scene graph to an executable Scenic program. It is an opt-in compiler for physical, axis-aligned box proxies. The existing reconstruction/chess exporters remain separate; this command does not silently discard their `square`, `occupies`, or `part_of` records.

```sh
lidar-video scene-program graph.json --out runs/replay --samples 1
lidar-video scene-program graph.json --out runs/variation --randomize --samples 20 --seed 7
```

Both commands write `scene.scenic` and `samples.json`. The latter executes the emitted Scenic program, with `maxIterations=1` for every scene. The Python API is `compile_scene_program`, `export_constructive_scenic`, and `run_scene_program` in `lidar_video.scene_program`. `as_networkx` returns a deep-copied `MultiDiGraph`, retaining multiple attributed relationships between the same nodes. JSON remains the interchange format.

## Input contract

Use `nodes` and `edges`, and `units: "metres"` if units are declared. Each physical node has a unique string `id`, `kind` (`object` or `support`), world XYZ `position_m`, and positive `size_m`. Other metadata is retained. Rotation fields are rejected: supply explicit world-aligned box proxies, not oriented boxes or articulated robots presented as if this compiler handled their geometry.

Each edge contains `source`, `target`, and `relation`:

- `on`: source footprint fully contained on target; source bottom equals target top plus optional nonnegative `clearance_m` (default zero).
- `inside`: source box fully contained within target, optionally inset by `gap_m`. A containment ancestor is a container volume, not a solid collision obstacle for its contained descendants. Other pairs still require non-overlap.
- `left_of`, `right_of`, `behind`, `in_front_of`, `below`, `above`: corresponding box-face inequalities, with optional nonnegative `gap_m`. +X is right; +Y is front; +Z is above.

`fixed: true` locks a node's recorded center. Root nodes of kind `support` default to fixed; other nodes default to independent center bounds of +/- `--jitter` (default 4 mm) before joint constraints are applied. An explicit `placement_bounds_m: [[xmin,ymin,zmin],[xmax,ymax,zmax]]` overrides that default for a non-fixed node. These are world-space center bounds; choose sufficient Z bounds when supports can move vertically. For a moving root support, set `fixed: false` explicitly. A node without a placement parent is a root, not a declaration that an ordinary object physically floats. Fixed arm bases can be represented as obstacles; joint configurations, reachability and motion are not implemented.

Reference positions must already satisfy the declared constraints. Replay mode preserves those positions exactly and never adds support clearance or snaps a pose. An invalid reconstruction is reported as invalid rather than silently repaired. Replay is of the reconstructed graph, not a claim of perfect fidelity to its source recording. Dimensions and metadata are retained in sampled graphs; Scenic itself displays box proxies, not reconstructed textured meshes.

## Construction and guarantee

Support/containment edges define a DAG. The compiler places parents before children, preferring larger volumes among currently eligible nodes. At each coordinate it solves two linear programs to find the full feasible interval *given the previous choices and all remaining constraints*. A draw from this projected interval therefore still admits a completion. This avoids the greedy failure in which an early parent placement makes a later object impossible to place. No scene retry, clipping or repair is used. Infeasibility or numerical failure stops execution.

All physical pairs except containment ancestors use a separating axis witnessed in the recorded layout. This compiles collision avoidance into linear constraints and preserves that separation branch. It is deliberately conservative: it excludes arrangements requiring a different separating axis. The result is an order-dependent sequence of uniform conditional-coordinate draws, **not** a uniform distribution over all valid scenes or a learned uncertainty posterior. Linear-program solver tolerances apply; final residuals must be within 1e-8 m. Costs grow with scene size (up to two LP solves per variable plus feasibility checks).

The generated Scenic objects use `allowCollisions True` because the joint distribution already enforces the declared box constraints, including intentional containment. No Scenic `require` rejection loop is used. The command checks inventory, dimensions and all compiled constraints on each actual Scenic output. This guarantee covers only the stated box contract; it is not a guarantee of mesh-level collision freedom, stability, force closure, articulation or dynamic task feasibility.

Duplicate IDs, missing endpoints, multiple placement parents, support cycles, unknown relations, unsupported rotations, overlapping reference boxes, contradictory constraints and invalid reference support are errors. Future nonlinear constraints or multiple separation branches require an explicit extension; they must not fall back to hidden rejection sampling.

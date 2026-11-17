# Native transfer of a support-frame shape proposal

`lidar_video.native_shape.refine_native_assets(graph_file, output, transforms,
support_tolerance=0.0)` writes a separate native scene variant. `transforms` maps
explicit graph object IDs to 4x4 affine matrices in the native scene's metric
support coordinates. Supply a new output directory outside the source scene.
This is an opt-in transfer stage, not automatic promotion of a visual proposal.

The adapter accepts horizontal yaw up to 30 degrees, equal horizontal scale and
independent vertical scale within 15%, no shear or reflection, no base-height
change, and at most 2 cm of horizontal displacement. It applies the same map to
the visual mesh and every convex collision part, recenters their common envelope,
and updates the graph center, dimensions and native body position together.
OBJ face indices, UVs and material statements are retained; normals use the
inverse-transpose transform. Mass and friction remain unchanged priors. The
simulator must recompute inertia from the changed collision geometry.

Supported native templates have explicit asset paths and a direct, unrotated
free mesh body for each selected node. Shared mesh files, transformed mesh geoms,
implicit defaults, includes, keyframes and explicit body inertia are rejected.
This intentionally narrow contract avoids silently changing joints, bindings or
unrelated bodies. Source files remain unchanged, unmodified copied files are
hash checked, and semantic edges are retained. The actual serialized geometry
is rechecked against support, region and non-overlap constraints before the
new directory is published.

A successful export does not establish physical stability. Compile the native
model, verify its world-space vertices against the visual candidate, and compare
passive trajectories with the unchanged source under identical simulator settings.
Record translation, total rotation, tilt relative to gravity, contact behavior,
and all other movable objects. Yaw drift is not tipping; success for one object
does not excuse failures elsewhere. Freeze construction/validation roles before
fitting; repeatedly inspected development cameras are not independent evaluation.

Actual Scenic replay and randomized generation should be checked separately.
The current compiler samples fixed-orientation conservative boxes from conditional
feasible intervals, with no scene retries. This is not arbitrary mesh-orientation
sampling or a uniform distribution over all valid scenes. An empirical successful
sample batch does not prove physical validity after dynamics.

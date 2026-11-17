# Compound-mesh mass and contact configuration

Native mesh exports allocate each object's total mass in proportion to its
collision components' convex hull volumes. Equal mass per component makes the
center of mass depend on arbitrary decomposition size. Volume weighting is a
uniform-density approximation, not a measurement: overlapping components count
overlap more than once, and hollow or nonuniform objects need identified inertial
properties. Invalid or degenerate components fail instead of silently falling
back. Reports record component volumes and masses. `equal_parts` retains the old
allocation for reproduction.

The optional `legacy_mesh_support_v1` contact policy targets explicit rigid-mesh
MJCF scenes. It enables legacy multi-contact CCD, six-dimensional contacts using
the existing sliding/torsional/rolling friction coefficients, and three NoSlip
iterations. It preserves mesh geometry, free joints, total mass and friction
coefficients, but changes the effective contact model. It does not pin objects or
guarantee stability. The policy rejects unresolved defaults, includes and explicit
contact-pair overrides rather than silently applying inconsistent settings.

```sh
lidar-video scene-program-assets /path/to/scene_graph.json \
  --out /path/to/new-run --simulate --frames 1800 \
  --mass-distribution collision_volume \
  --contact-policy legacy_mesh_support_v1
```

Python callers can pass the same `mass_distribution` and `contact_policy` keyword
arguments to `simulate` or `run_asset_scene_program`. The policy is opt-in;
`contact_policy=None` keeps the builder's contact settings. Simulation reports
retain the applied options and prior settings alongside mass allocation and
passive-motion diagnostics. Initial Scenic placement validity does not establish
dynamic validity: check every sampled scene at the intended horizon and apply a
real-force control to check that contacts have not effectively immobilized it.

This profile was checked with MuJoCo 3.5.0 on CPU. Revalidate with other versions,
engines, materials and tasks. Legacy CCD can be slower; NoSlip has inverse-dynamics
limitations and is not a validated robot-control training configuration. See the
[MuJoCo modeling guide](https://mujoco.readthedocs.io/en/3.5.0/modeling.html#preventing-slip)
and [XML reference](https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html).

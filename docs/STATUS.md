# Foundation status

This snapshot preserves the working core and reproducible synthetic fixtures. Real recordings and model artifacts remain local and are not in Git.

## Exercised

- Synthetic tabletop and chess RGB-D reconstruction, calibration, shared scene graph / Scenic export, and native RoboSkills MuJoCo execution.
- An actual Stray Scanner iPhone capture was imported and processed with annotation-assisted calibration and ten selected static views.
- On A100, SAM 3 found 37 raw instances; 14 unambiguous instances proceeded to SAM 3D and all 14 inference jobs completed.
- None of those real-capture meshes passed all geometric checks. The fuller fallback scene contains 30 coarse depth-derived objects, with two visibly occupied cells omitted for insufficient depth.
- Twenty Scenic samples and exact recorded-state physics replay passed for the saved real-capture fallback. Those results do not establish shape accuracy.

## Geometry diagnosis

The median of per-object median surface distances was 3.96 mm against each object's own reference-mask depth, versus 14.35 mm against the fused square clouds. These data sets differ in masks, height filtering, resolution and weighting; this is not a pure estimate of the effect of adding views.

Several visible pieces had only 2.8–5.6 mm median recorded height in their eroded mask interiors. The empty board was much more consistent with its fitted plane (0.45–1.47 mm median absolute height error across selected views). Small pixel shifts did not remove the low-height symptom. Sensor confidence alone did not identify the problem.

Raw-file and transfer checks passed. Denser mesh sampling changed fused median errors by less than 0.2 mm. Rigid fitting improved held-out depth agreement for 8 of 10 eligible objects, but worsened two; four had inadequate split coverage. The fitted candidates were not accepted into the simulation.

Next work: score depth reliability per instance/view, select better supported reference views, establish cross-view object associations, and refine pose with board-support and silhouette checks. Device-level smoothing, material response, registration and timing effects remain unresolved.

## Limits

- Record3D import is fixture-tested, not validated against a real Record3D export.
- Absolute real-world scale and shape accuracy have no independent ground truth.
- Real chess-piece classification and physical a1 orientation remain unresolved.
- The recognizer is trained on procedural synthetic appearances, not arbitrary chess sets.
- Moving pieces, hand actions and real robot manipulation are not validated.
- OMAR / multi-agent orchestration is deferred.
- Profiling scripts target the saved 256×192 native / 960×720 model-input contract and require externally supplied run artifacts.

Original local reports retain the detailed run evidence. No raw capture, diagnostic image containing the recording, neural mesh, checkpoint or executed notebook is included in this repository.

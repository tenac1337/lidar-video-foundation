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

The reliability fix implements per-instance/view depth gates, corroborated reference selection, and scale-preserving pose refinement with board support and held-out depth/silhouette checks. The initial depth-gate replay on the saved 14 inference inputs blocked six (h8, d2, d8, g8, c7, f1); the other eight were candidates, not accepted meshes.

A fresh A100 80 GB run segmented three views (37, 37 and 43 detections), selected e8 from frame 2, and completed its SAM 3D job. The mesh failed held-out validation: 25.24 mm median depth error and 0.009 silhouette IoU. No learned mesh was accepted.

That run also exposed a fallback bug: one instance was split into two square-based objects. The follow-up preserves instance identity in fallback and requires at least 90% of trusted native samples to fall within the assigned square. Only 57.86% of e8’s frame-2 samples did.

Both local replay and a second fresh A100 segmentation run with this gate select no objects. The final GPU audit matches replay decisions and counts: per-view candidate counts are 1, 0 and 0. The pipeline reports `abstained`, launches zero SAM 3D jobs and exports no new scene. The earlier two-object fallback is diagnostic evidence of the fixed bug, not a corrected result. That revision had 78 passing tests. Device-level smoothing, material response, registration and timing effects remain unresolved.

## Native-ray and support-alignment fix

The subsequent ten-view A100 audit produced saved b8 and f8 candidate meshes, neither accepted. Local profiling found that resized depth was backprojected along RGB-grid rays instead of its original sensor rays, and an unconstrained orientation fit could improve the reference while destroying held-out silhouette agreement.

The fix preserves native intrinsics and sample IDs through normalization, geometry and the Colab model handoff. Local replay restored metadata from the original capture after checking source frames, poses, depth and confidence. Corrected XYZ matches native capture XYZ exactly across all ten views (maximum difference 0 m).

Staged support alignment retains the original orientation when possible and regularizes rotation when needed. With identical native geometry, f8's held-out silhouette IoU improves from approximately 0.331 under the previous rigid fit to 0.681. Its held-out median depth error is still 5.612 mm versus 3.252 mm before support correction: a 2.360 mm degradation, exceeding the unchanged 1 mm guard. b8 also fails. Neither saved mesh is accepted, and no new simulation, scene graph or Scenic export is produced by this replay.

The local suite has 89 passing tests. The new SAM 3D point-map adapter is fixture-tested; neural inference has not been rerun with it. A fresh A100 inference run using corrected native rays is the next test of learned shape quality, with the same acceptance thresholds.

## Limits

- Record3D import is fixture-tested, not validated against a real Record3D export.
- Absolute real-world scale and shape accuracy have no independent ground truth.
- Real chess-piece classification and physical a1 orientation remain unresolved.
- The recognizer is trained on procedural synthetic appearances, not arbitrary chess sets.
- Moving pieces, hand actions and real robot manipulation are not validated.
- OMAR / multi-agent orchestration is deferred.
- Profiling scripts target the saved 256×192 native / 960×720 model-input contract and require externally supplied run artifacts.

Original local reports retain the detailed run evidence. No raw capture, diagnostic image containing the recording, neural mesh, checkpoint or executed notebook is included in this repository.

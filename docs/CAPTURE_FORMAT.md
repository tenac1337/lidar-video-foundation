# Capture contract v1

The input is a non-pickled NumPy `.npz`. All seven fields are required:

- `rgb`: uint8 `[N,H,W,3]`, RGB order; rectified and registered to the depth pixel grid.
- `depth`: floating point `[N,H,W]`; **camera-axis Z in metres**, not radial range. Zero, NaN, or infinity means missing. Negative values are rejected.
- `confidence`: `[N,H,W]`, integer levels 0 (low/invalid), 1 (medium), 2 (high). Default processing excludes 0; no dense filling is performed.
- `K`: `[3,3]` or `[N,3,3]`; pinhole intrinsics in the stored image grid, with zero skew, positive focal lengths, and last row `[0,0,1]`. Pixel centers have integer coordinates. Input must already be undistorted.
- `world_from_camera`: `[N,4,4]`; camera-to-world rigid transforms, not their inverses. Camera axes: x right, y down, z forward. World is right-handed with z up. Translation is metres.
- `rgb_time`: `[N]`, strictly increasing seconds on a common monotonic clock.
- `depth_time`: `[N]`, associated depth-frame times on that same clock. A corresponding pair must differ by no more than 20 ms.

All rows correspond to one explicitly associated frame pair. There is no implicit nearest-time matching. Arrays must have the same frame count. Single-frame captures can validate, but video export requires at least two frames. The video path currently requires a uniform frame interval to within 1 ms; variable-rate recordings must be explicitly resampled first.

To preserve source data, import into a new file. Record app version, camera selection, image orientation, units, pose convention, and timestamp semantics in a sidecar JSON. Do not invent calibration or timing from nominal phone specifications.

## Normalization

Per-frame K is mapped to the first frame's pixel grid using pinhole ray remapping. RGB uses bilinear interpolation; depth/confidence use nearest neighbors and invalid borders. Poses and times remain unchanged. Optional integer `depth_sample_id[N,H,W]` maps each stored pixel to its original native-depth sample (row-major index; -1 for invalid borders). Normalization carries this map through the identical nearest-neighbor transform. `native_depth_shape=[height,width]` declares the original sensor grid. Without explicit IDs, legacy upsampled inputs must follow this project’s origin-aligned mapping `native_pixel=round(stored_pixel/scale)`, clipped at the native boundary; other resamplers must supply IDs. Repeated RGB-sized pixels never count as independent sensor samples. This assumes RGB and depth are already spatially registered. Separate sensor extrinsics/distortion need a prior registration stage and cannot be ignored.

### Native sensor rays

Normalization now also writes `native_K`, the original pinhole intrinsics in the native depth grid (`[3,3]` or `[N,3,3]`). Explicit `depth_sample_id` requires `native_K`; `native_depth_shape` identifies the grid in which these IDs and intrinsics are defined. A usable depth value cannot have ID -1.

Geometry backprojects the original sample coordinates with `native_K`, not the resized RGB pixel with `K`. Repeated copies of a sensor sample therefore have identical XYZ. Stored `K` remains the correct matrix for projecting geometry into masks and RGB. Image-corner calibration continues to interpolate local depth onto the observed image landmark; those interpolated landmarks are distinct from measured sensor samples.

**Migration:** older normalized archives with explicit IDs but no `native_K` now fail validation. Re-import from the original capture, or restore original per-frame native intrinsics after checking source-frame identity, poses, copied depths and confidence against that capture. Do not infer the original per-frame intrinsics from the normalized target `K`. Legacy inputs without IDs still support the documented origin-aligned mapping above.

The measured-depth handoff includes all three native metadata fields. `scripts/colab_sam3d.py` runs `scripts/sam3d_native_rays.py`, which supplies exact native point maps to the pinned worker for both model conditioning and pose scoring. Direct upstream launchers do not consume this extension automatically.

## Record3D reader scope

The experimental importer reads a ZIP `.r3d` containing `metadata` JSON and `rgbd/<integer>.jpg`, `.depth`, `.conf`. Depth is little-endian float32 and confidence uint8; both are LZFSE-compressed. It expects metadata `cameraType=1`, `w`, `h`, `dw`, `dh`, column-major `K`, `poses` as `[qx,qy,qz,qw,tx,ty,tz]`, and `frameTimestamps`.

The importer converts the ARKit/OpenGL camera convention to OpenCV and rotates ARKit's y-up world into z-up. RGB is downsampled to the depth grid with matching pixel-center intrinsics. Original `.r3d` remains untouched.

Archives containing `perFrameIntrinsicCoeffs` are refused until their exact schema is verified. Archives without frame timestamps or confidence are refused. The associated-frame timestamp is used for both streams; it does not provide independent RGB and LiDAR exposure times. The importer records that limitation explicitly.

This schema has been tested with constructed archives only. A real iPhone 15 Pro export is still required to verify the app's version-specific intrinsics, orientation, pose indexing, and timestamp semantics.

## Output provenance

`scene_graph.json` contains inferred geometry and hypotheses. `simulation/simulated_scene_graph.json` contains state/contact evidence from the simulator. `scenic_samples.json` contains generated variations, not sensor measurements. Synthetic fixture truth is saved separately and is read only by the evaluation function.

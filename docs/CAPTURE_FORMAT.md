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

Per-frame K is mapped to the first frame's pixel grid using pinhole ray remapping. RGB uses bilinear interpolation; depth/confidence use nearest neighbors and invalid borders. Poses and times remain unchanged. This assumes RGB and depth are already spatially registered. Separate sensor extrinsics/distortion need a prior registration stage and cannot be ignored.

## Record3D reader scope

The experimental importer reads a ZIP `.r3d` containing `metadata` JSON and `rgbd/<integer>.jpg`, `.depth`, `.conf`. Depth is little-endian float32 and confidence uint8; both are LZFSE-compressed. It expects metadata `cameraType=1`, `w`, `h`, `dw`, `dh`, column-major `K`, `poses` as `[qx,qy,qz,qw,tx,ty,tz]`, and `frameTimestamps`.

The importer converts the ARKit/OpenGL camera convention to OpenCV and rotates ARKit's y-up world into z-up. RGB is downsampled to the depth grid with matching pixel-center intrinsics. Original `.r3d` remains untouched.

Archives containing `perFrameIntrinsicCoeffs` are refused until their exact schema is verified. Archives without frame timestamps or confidence are refused. The associated-frame timestamp is used for both streams; it does not provide independent RGB and LiDAR exposure times. The importer records that limitation explicitly.

This schema has been tested with constructed archives only. A real iPhone 15 Pro export is still required to verify the app's version-specific intrinsics, orientation, pose indexing, and timestamp semantics.

## Output provenance

`scene_graph.json` contains inferred geometry and hypotheses. `simulation/simulated_scene_graph.json` contains state/contact evidence from the simulator. `scenic_samples.json` contains generated variations, not sensor measurements. Synthetic fixture truth is saved separately and is read only by the evaluation function.

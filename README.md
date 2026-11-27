# Trackwright

Planar motion tracking in the spirit of Mocha Pro: draw a shape around any flat surface (a screen, a sign, a wall, a tabletop) and Trackwright follows it through the shot, solving a full perspective corner pin you can send to Reelforge, Forma3D, Tidefire, Nuke or After Effects.

Built in Rust with Tauri 2 and a Leptos UI.

© 2026 Scott Hambach. All rights reserved. Trackwright is a trademark of Scott Hambach.

---

## Milestone 1: planar tracker (this release)

* **Footage:** video (mp4, mov, mkv, avi, webm, mxf… decoded via ffmpeg), numbered image sequences (`plate.1001.png` …), folders of images, single images. Open with the buttons, by dragging onto the window, or with `trackwright <path>`.
* **Layers:** any number of tracked layers, each with its own shape, surface, colour and settings.
* **Shapes:** straight-edged polygons or smooth spline outlines; edit points at any tracked frame.
* **Surface:** four-corner plane (Mocha's "surface") that can be adjusted on any tracked frame; the change is carried back to the reference frame so the whole track updates.
* **Tracking:** forward and backward, whole clip or one frame at a time, cancellable, live progress.
  * Motion model per layer: translation, + scale/rotation, + shear, + perspective.
  * Engine: Shi-Tomasi features inside the shape, pyramidal Lucas-Kanade with a forward-backward check, RANSAC model fit, and **anti-drift**: every frame is re-solved against the reference frame warped by the current estimate, so error does not build up over long shots.
* **Checks:** grid insert overlay warped onto the surface, inlier points, and a **quality graph** on the timeline (error line, point-count bars, red/orange/purple marks on suspect frames) with **⚠ Next problem (N)**.

### Tracking quality

* **Exclusion layers:** tick other layers in *Exclusions* and the tracker ignores those areas (and a margin around them), in the current frame and in the reference. Track or keyframe the excluded layer so it follows the occluder.
* **Shape keyframes:** *◆ Shape Key* animates the tracking region; edits on other frames add keys, and points added or removed apply to every key.
* **Adjust Track (A):** drag corners on a bad frame to correct them. Corrections are keyed, interpolated between keys and held outside them; *◆ Lock Here* pins good frames. The raw track is shown as an orange ghost. Exports, inserts and linked layers use the corrected corners.
* **In / Out (I, O, X):** tracking stops at the Out (forward) or In (backward) point. **Re-track In→Out** re-solves just that section and blends it into the existing track so it joins seamlessly at Out.
* **Lighting:** per-patch brightness/contrast compensation handles fades, flicker and moving light.
* **Motion blur / hard frames:** a failed frame is retried with a wider, more forgiving search (flagged purple on the graph).
* **Speed:** multi-core tracking, cropped anti-drift solve, background decoding of the next frame, and an automatic half-resolution first pass on 2.5K+ footage (full-res refinement keeps the precision).
* **Export** (Save to Desktop / Send to app / Save As…):

| Format | Use |
| --- | --- |
| Reelforge track `.rftrack.json` | corners, homography, position/rotation/scale per frame |
| Forma3D / Tidefire planar track `.f3dtrack.json` / `.tftrack.json` | corners in pixels **and** normalised device coords (-1..1, y up), homography |
| Nuke `CornerPin2D` `.nk` | paste into Nuke |
| After Effects Corner Pin `.txt` | copy the file contents, select the layer in AE, paste |
| Fusion / DaVinci Resolve CornerPositioner `.setting` | copy the contents and paste into the Fusion node view (Fusion page in Resolve). Points are normalised; time starts at 0 |
| Blender plane-track script `.py` | run in Blender's Text Editor: loads the footage as a Movie Clip, adds four corner tracks and a plane track |
| Silhouette / Shake 4-point tracker `.shake.txt` | Silhouette: import as Shake trackers (it also reads the AE corner pin and Nuke exports). Trackers are numbered TL, TR, BR, BL |
| Trackwright JSON / CSV | everything / spreadsheet |

"Send to Reelforge / Forma3D / Tidefire" writes into each app's import folder (default `Documents/<App>/Imports/Tracks`, change it in **App Folders**).

* **Render** (Render panel, uses In/Out when set):
  * *Insert composited into the footage*: the pinned image/clip rendered into every frame, with the layer's exclusion areas kept in front.
  * *Stabilize: smooth the motion* (smoothing window in frames) or *lock to the surface*, with zoom to hide the edges.
  * Output is a PNG sequence, plus an H.264 MP4 when ffmpeg is installed.
* **Reelforge import:** the `trackwright-import` crate (`crates/trackwright-import`) reads Trackwright tracks and gives corners, transforms or a corner-pin homography at any time. See its README for wiring it into a Reelforge effect.
* **Projects:** `.twproj` (JSON) save/load, **Recent** menu, autosave every 30 s with crash recovery on the next launch, and a Save / Don't Save / Cancel prompt before closing, starting a new project or opening another one with unsaved changes.

### Workflow

* **Undo / redo** for every edit, including whole tracking runs (Ctrl+Z, Ctrl+Shift+Z / Ctrl+Y).
* **Insert preview:** pin an image, video or image sequence onto a layer's surface with perspective, opacity, start frame and loop, so you can judge the track the way you would in Mocha.
* **Proxy playback:** footage larger than 1280 px gets low-res JPEG proxies built in the background. Viewer resolution is Auto (proxies while playing, scrubbing or tracking), Proxy or Full. Tracking always uses full resolution.
* **Shape editing (E):** drag points, click an edge to add a point, drag inside to move the whole shape, Delete removes the selected point. **Surface (S):** drag corners, or drag inside to move the whole surface.
* **Layers:** duplicate (Ctrl+D), double-click to rename, and **Motion source** to make a layer follow another layer's track (for example, a small screen inside a tracked wall).

### Measured accuracy

Synthetic plates with a textured plane moving over a different textured background, where the true motion is known exactly:

| Test | Worst corner error |
| --- | --- |
| 24 frames, rotation + scale + perspective (unit test) | 0.09 px |
| 48 frames, 27° rotation, 23 % scale, perspective (unit test) | 0.07 px |
| 48-frame 960×540 plate tracked through the app UI and exported | 0.014 px |
| Textured box sliding across the plane, with an exclusion layer (through the app UI) | 0.032 px (without exclusion the track follows the box) |
| Brightness flicker ±45 % gain, ±40 bias (unit test) | 0.08 px (track is lost without compensation) |
| Corrupted 9-frame section re-tracked with In→Out blend (unit test) | 0.08 px, later frames untouched |
| 4K, 2 cores: full-res pass → auto half-res pass | 209 → 116 ms/frame |

## Camera tracking, lens, removal and mattes

* **Camera Solve** (right panel): **1 · Track Points** finds and follows up to 500 corner features across the In/Out range (or whole clip), skipping layers ticked under *Ignore* (people, cars, anything moving on its own). **2 · Solve Camera** runs structure from motion: keyframes, two-view start (or a rotation-only chain for tripod shots), resection, triangulation and bundle adjustment that also refines focal length and, optionally, lens distortion. Detects moving vs tripod cameras automatically; focal length can be solved or entered.
* **Orient scene:** pick a planar layer that lies on the ground (floor, table, road), optionally type its real width, and **Set Ground**. The surface becomes the XZ ground plane with the origin at its centre, X along its top edge and Y up, scaled to real units.
* **Overlays:** camera point trails coloured by solve error, solved 3D points projected back as crosses, and the oriented ground grid with X/Y/Z axes.
* **Camera export:** Forma3D (camera + point cloud), Tidefire, **FBX** (binary 7.4: animated camera + points mesh, for Maya, Cinema 4D, Unreal, Blender), **Alembic** (.abc: camera + point cloud, for Houdini, Maya, Blender), Blender (Python script), Nuke (.nk), Collada (.dae), point cloud (.obj) and .chan (Nuke's ZXY order, which Reelforge's *Import a camera…* reads). Save to Desktop, send to Forma3D/Tidefire, or Save As. FBX and Alembic were checked by importing them into Blender 5.3: camera path, rotation and focal length match the solve. Blender's FBX importer shifts animation by one frame unless its *Animation Offset* is set to 0; other apps place frame 1001 at frame 1001.
* **Lens distortion:** k1/k2 radial model. Set the values, let the camera solve find them, or press **L** and click along edges that should be straight, then **Calibrate from Lines**. *Show lens grid* draws the bending; Render › *Undistort footage* writes straightened frames.
* **Remove object / clean plate:** choose the object layer and the background layer that tracks the surface behind it. Each frame's hole is filled with the background seen in nearby frames (±search), aligned through the background track, with optional lighting match; areas never revealed are filled smoothly. *Clean plate* writes one PNG of the background with the object taken out (median of up to 15 frames).
* **Rotoscoping / mattes:** every layer's shape is a roto mask with a *Mask feather* setting. Render a black-and-white **Matte** or a **Cut-out** (footage with alpha) for one layer or all visible layers, or export **Reelforge mask (.rfmask.json)** for an editable, animated mask in Reelforge.
* **FFmpeg bundled:** the setup .exe / .msi install `ffmpeg.exe` and `ffprobe.exe` next to Trackwright. The portable zip includes `get-ffmpeg.bat` to fetch them.

Measured on a ray-traced 640×360 room with a moving camera and k1 = −0.06, tracked and solved from the images alone: focal length within 2.6 %, k1 −0.060, reprojection error 0.25 px, camera path error 0.3 % of its length.

## Using it

1. **Open Footage** (or drag a file onto the window).
2. Press **D** (Draw Shape) and click around a flat, textured area. Click the first point or press **Enter** to close. Right-click removes the last point.
3. The blue **surface** appears at the shape's bounds. With **S** (Surface) drag its corners onto the real corners of the plane (e.g. the edges of a screen). It doesn't need to be inside the shape.
4. **Track ▶▶** (or Shift + → for one frame). Track backward from the reference frame with **◀◀ Track**.
5. Scrub the timeline and turn on **Grid** to check the lock. If the track slips, stop, fix the shape or surface at that frame and continue.
6. Pick a format and **Save to Desktop** or **Send to …**.

Tips: track areas with texture, and leave out things that move differently (people walking in front, reflections). Set the reference frame (**Set Reference Here**) to the frame where the surface is most clearly visible.

## Building

Requirements: Rust (stable) with the `wasm32-unknown-unknown` target, [Trunk](https://trunkrs.dev), and the Tauri CLI (`cargo install tauri-cli --version "^2"` or `npx @tauri-apps/cli`). For video import, [ffmpeg](https://ffmpeg.org) on PATH (`winget install ffmpeg`) or `ffmpeg.exe` + `ffprobe.exe` next to `trackwright.exe`.

```sh
rustup target add wasm32-unknown-unknown
cargo tauri dev            # run with hot reload
cargo tauri build          # Windows .exe + NSIS/MSI installers in target/release/bundle
cargo test -p trackwright-core --release   # tracking accuracy tests
```

GitHub Actions (`.github/workflows/build.yml`) runs the tests and builds the Windows installers on every push. Download them from the run's **Artifacts**; tag `v*` to attach them to a release.

## Layout

```
crates/trackwright-core   tracking engine (no UI deps): image pyramids, features, KLT,
                          motion models + RANSAC, planar tracker, point tracker, camera solve
                          (epipolar, sfm, bundle adjustment), lens model, render, exporters
src-tauri                 desktop backend: footage import, frame cache, background tracking, export
ui                        Leptos front end: viewport, timeline, layers, export panel
```

## Roadmap

* Auto-update from the latest GitHub release.
* Crash and error log with a "copy diagnostics" button.
* Code signing (needs a certificate), so Windows SmartScreen stops warning.

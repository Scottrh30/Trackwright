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
* **Checks:** grid insert overlay warped onto the surface, inlier points, and a **quality graph** on the timeline (error line, point-count bars, red/orange/purple marks on suspect frames, cyan on predicted ones) with **⚠ Next problem (N)**.

### Tracking quality

* **Exclusion layers:** tick other layers in *Exclusions* and the tracker ignores those areas (and a margin around them), in the current frame and in the reference. Track or keyframe the excluded layer so it follows the occluder.
* **Shape keyframes:** *◆ Shape Key* animates the tracking region; edits on other frames add keys, and points added or removed apply to every key.
* **Adjust Track (A):** drag corners on a bad frame to correct them. Corrections are keyed, interpolated between keys and held outside them; *◆ Lock Here* pins good frames. The raw track is shown as an orange ghost. Exports, inserts and linked layers use the corrected corners.
* **In / Out (I, O, X):** tracking stops at the Out (forward) or In (backward) point. **Re-track In→Out** re-solves just that section and blends it into the existing track so it joins seamlessly at Out.
* **Lighting:** per-patch brightness/contrast compensation handles fades, flicker and moving light.
* **Motion blur / hard frames:** a failed frame is retried with a wider, more forgiving search (flagged purple on the graph).
* **Occlusion solver** (Track Settings, on by default): when the surface is covered (an eyelid over a pupil, a hand or car passing in front) the track carries on instead of stopping or quietly sliding off with the cover.
  * *Noticing the cover:* tracking fails, or the shape suddenly stops looking like it did a frame ago (a lid coming down), or it falls well below how it usually looks (a slower cover). Looks are compared as fine detail, not brightness, so soft shading and lighting changes don't count. If the last frame or two before the cover were already partly covered, the solver backs up to the last clean one.
  * *Predicting hidden frames:* from the area around the shape (a ring 2.5× its size: face, skin, background) when that still tracks, limited to the layer's motion model; otherwise from the surface's own recent motion, slowing down. A fast head move during a blink blurs the surroundings and loses many points, but the group that still agrees is used, so the guesses keep up with the face.
  * *Finding it again:* every hidden frame it searches near the prediction for the target's look, using the frame just before the cover and a memory of up to six clean frames from along the track, each turned and sized the way the surroundings moved. A match far from the prediction has to be much stronger than one close to it. It is accepted when point tracking confirms it, or when the search finds it in the same place two frames running and it looks about as much like itself as clean frames usually do (or clearly more than it did while hidden), and only if it still looks right a frame later. That stops a shut eyelid with lashes being taken for the eye coming back. The predicted stretch is then corrected to join up seamlessly.
  * Predicted frames are cyan on the timeline. *Bridge up to* (default 48 frames) limits the gap; past it the guesses are dropped and tracking stops with a message.
  * *Tested on real footage.* A 4K, 981-frame talking-head clip with about 15 blinks, a long eye closure, fast head moves, head tilts and speech: the whole-eye shapes (left and right) bridge every blink and stay on the eye for the whole clip, including a blink during a fast upward head jerk. An iris box bridges the blinks too, but its look changes a lot as the eye turns, so some open-eye frames get marked as guessed (cyan) even though the box stays on the iris. A mouth shape on the same clip stays on the mouth throughout, but speech changes its look so much that about 12% of frames are marked as guessed; for a talking mouth, turn the occlusion solver off or include the chin and cheeks in the shape. An earlier 4K, 355-frame portrait (three blinks, eye rolls and a head turn during a blink): an iris-only box (Translation model) and a whole-eye shape (Similarity or Perspective) both carry through all three blinks and stay on the eye for the whole clip, with no false alarms on the iris box. For a pupil or iris, use the **Translation** model; scale and perspective drift on something that small and round. Tracking backwards through an eye that slowly reopens works with the whole-eye shape; a tiny iris box can slide off with the lid in that direction. Synthetic tests: a pupil covered for 6 frames stays within 0.2 px, one that moves 16 px while hidden is found where it went, and guesses through a blink with an 80 px motion-blurred head jerk stay within 1 px.
  * The *few good points* warning now scales to the layer: a small target that always has 10–20 points isn't flagged on every frame.
* **Mesh (warp) tracking** (Track Settings › Mesh): a 4×4 to 12×12 grid over the surface that bends with it, for faces, mouths, cheeks and cloth that don't stay flat. Each grid point is tracked locally (from the previous frame, checked against the reference so it doesn't drift), then smoothed so neighbours move together (*Mesh stiffness*). Points that can't be tracked follow their neighbours. Inserts (preview and render) bend with the mesh, the Reelforge / Forma3D / Tidefire exports include the grid per frame, and Adjust Track corrections move the mesh too. Test: on a surface that bulges by 23 px, one flat plane is up to 17 px off; the mesh averages 0.7 px.
* **Eye / pupil mode**: set a layer's *Target* to *Eye / pupil* and track the whole eye. Inside it, Trackwright finds the pupil on every frame as the dark round spot in a lighter ring (green ring in the viewer), to sub-pixel accuracy, and exports it with the track (`pupil: {x, y, radius}` per frame). It is found even on frames where the eye box itself is guessed, and left out when the eye is shut. Steadier than tracking a tiny box on the pupil, which loses it when the eye turns.
* **Rotation** (Track Settings): *Its own* (from the surface's points), *Follow surroundings* (turns as much as the area around it, so an eye turns with the head instead of slowly tilting on its own), or *Locked* (never turns). Choosing *Eye / pupil* switches on *Follow surroundings* and *Find the pupil*. On the 981-frame test clip, the whole-eye box ended 20° tilted on its own; with *Follow surroundings*, 6°.
* **Hand fixes carry into the next run**: when you re-track through frames you fixed with Adjust Track, the track snaps to your corners there and eases the frames before onto them, so one fix corrects the run instead of just being layered on top. The fix becomes part of the track (its Adjust offset goes to zero, the key stays to mark it as checked), frames outside the re-tracked stretch keep exactly where they were, and the fixed frames are used as extra references for finding the target again after a cover. What the fixes teach is learned first.
* **Learning from your manual tracking** (right panel › Learning): Trackwright gets better at the kinds of targets you track the more you fix its tracks by hand.
  * Set each layer's *Target* in Track Settings (General, Eye / pupil, Mouth / lips, Face / skin, Screen / sign, Object / vehicle). Each kind learns separately.
  * While tracking, the solver notes what it judged each frame by (how much the surface changed since the last frame, how it compares with its usual look, why it called a frame covered or found it again). When you fix frames with **Adjust Track**, your corners show which of those calls were right. On **Save** and **Export** (or **Learn From Layer**), those frames become examples, and the thresholds for that kind of target are re-fitted to make the fewest mistakes on them. A missed cover counts double a false alarm. A few examples only nudge the defaults; it takes consistent evidence to move them far.
  * It learns only from frames inside the span of your Adjust keys (frames you worked on), never from its own unchecked output. To mark a stretch as right without changing it, add an Adjust key at its start and end.
  * Learning from the same layer again replaces its earlier examples instead of counting them twice.
  * **Learn from my manual tracking** switches learning on or off. **Use what it has learned when tracking** switches between the learned thresholds and the built-in defaults, so you can compare.
  * **Reset** (one kind) or **Reset All** forgets what was learned (click twice to confirm); tracking goes back to the defaults and it starts learning again from your next fixes. **Back Up…** / **Load…** save and restore it as a `.twlearn` file (Load replaces the current learning).
  * Everything stays on your computer, in `learning.json` in Trackwright's settings folder.
  * Test: a target that changes look every few frames (like a mouth talking) gets 14 false "covered" flags with the defaults; after learning from one hand-fixed shot, a new shot of it gets none, and a real 5-frame cover is still caught.
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

## Testing on real footage

`crates/trackwright-core/examples/track_clip.rs` runs the tracker on a folder of frames outside the app and writes every frame's corners, predicted flag and appearance scores to JSON:

```
cargo run --release -p trackwright-core --features parallel --example track_clip -- \
    FRAMES_DIR START END x0,y0,x1,y1 out.json model=translation bridge=1
```

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

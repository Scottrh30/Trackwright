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
* **Checks:** grid insert overlay warped onto the surface, inlier points, per-frame error.
* **Export** (Save to Desktop / Send to app / Save As…):

| Format | Use |
| --- | --- |
| Reelforge track `.rftrack.json` | corners, homography, position/rotation/scale per frame |
| Forma3D / Tidefire planar track `.f3dtrack.json` / `.tftrack.json` | corners in pixels **and** normalised device coords (-1..1, y up), homography |
| Nuke `CornerPin2D` `.nk` | paste into Nuke |
| After Effects Corner Pin `.txt` | copy the file contents, select the layer in AE, paste |
| Trackwright JSON / CSV | everything / spreadsheet |

"Send to Reelforge / Forma3D / Tidefire" writes into each app's import folder (default `Documents/<App>/Imports/Tracks`, change it in **App Folders**).

* **Projects:** `.twproj` (JSON) save/load.

### Measured accuracy

Synthetic plates with a textured plane moving over a different textured background, where the true motion is known exactly:

| Test | Worst corner error |
| --- | --- |
| 24 frames, rotation + scale + perspective (unit test) | 0.09 px |
| 48 frames, 27° rotation, 23 % scale, perspective (unit test) | 0.07 px |
| 48-frame 960×540 plate tracked through the app UI and exported | 0.014 px |

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
                          motion models + RANSAC, planar tracker, project model, exporters
src-tauri                 desktop backend: footage import, frame cache, background tracking, export
ui                        Leptos front end: viewport, timeline, layers, export panel
```

## Roadmap

* **Milestone 2: camera tracking.** Automatic point tracks → camera solve (structure from motion) with focal length/lens solve, ground plane and scene orientation. Export camera + point cloud to Forma3D and Tidefire, plus FBX/Alembic/Nuke.
* Mocha-style extras: per-frame shape keyframes, exclusion layers (holes), linked/grouped layers, lens distortion, stabilise and remove (clean plate) modules, direct Reelforge plugin.

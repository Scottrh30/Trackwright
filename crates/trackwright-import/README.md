# trackwright-import

Read Trackwright planar tracks and evaluate them at any frame or time. Made for **Reelforge** (also usable in Forma3D and Tidefire). It depends only on `serde` and `serde_json`.

Reads `.rftrack.json` (Reelforge export) and `.f3dtrack.json` / `.tftrack.json` (planar exports).

## Add to Reelforge

```toml
[dependencies]
trackwright-import = { git = "https://github.com/Scottrh30/Trackwright" }
```

## Use

```rust
use trackwright_import::{apply, invert, CornerPinTrack};

let track = CornerPinTrack::load(path)?           // any Trackwright track
    .rescaled(timeline_w as f64, timeline_h as f64); // if the clip is shown at another size

// Each frame of the effect:
let corners = track.corners_at_time(t_seconds);           // TL, TR, BR, BL in pixels (interpolated)
let xf = track.transform_at(track.frame_at_time(t_seconds)); // position / rotation / scale
let h = track.pin_layer_at_time(t_seconds, layer_w, layer_h).unwrap(); // layer -> screen
let hi = invert(&h).unwrap();                             // screen -> layer, for inverse mapping
// For each output pixel inside `corners`, sample the layer at apply(&hi, [x + 0.5, y + 0.5]).
// On the GPU, pass `hi` as a mat3 and divide by w in the fragment shader.
```

* Frame numbers use the footage numbering (`first_frame`, e.g. 1001). `frame_at_time(0.0)` is the first frame.
* Between tracked frames the corners are interpolated; outside the tracked range they hold.
* `examples/corner_pin.rs` is a complete CPU corner pin of one frame:

```
cargo run -p trackwright-import --example corner_pin -- shot.rftrack.json frame.png logo.png 1012 out.png
```

## Suggested Reelforge effect

A "Trackwright Corner Pin" effect plugin with parameters *Track file*, *Layer/clip to pin*, *Opacity* and *Time offset*. Per frame it calls `pin_layer_at_time` and draws the layer through the inverse homography, as in the example above. The same track can also drive a "Transform" effect from `transform_at`, or a mask from `corners_at`.

© Scott Hambach. All rights reserved.

//! Corner-pin a picture onto one frame of footage using a Trackwright track.
//!
//! cargo run -p trackwright-import --example corner_pin -- shot.rftrack.json frame.png logo.png 1012 out.png
//!
//! This is the core of what a Reelforge "Trackwright corner pin" effect does each frame:
//! ask the track for a homography at the current time, then inverse-map every output
//! pixel inside the surface back into the layer being pinned.

use image::{Rgba, RgbaImage};
use trackwright_import::{apply, invert, CornerPinTrack};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!(
            "usage: corner_pin <track.json> <frame.png> <layer.png> <frame number> <out.png>"
        );
        std::process::exit(2);
    }
    let track = CornerPinTrack::load(&args[1]).expect("track");
    let mut frame = image::open(&args[2]).expect("frame").to_rgba8();
    let layer = image::open(&args[3]).expect("layer").to_rgba8();
    let n: f64 = args[4].parse().expect("frame number");

    // If the frame isn't the size that was tracked, rescale the track to match.
    let track = track.rescaled(frame.width() as f64, frame.height() as f64);
    let h = track
        .pin_layer_at(n, layer.width() as f64, layer.height() as f64)
        .expect("degenerate surface");
    let hi = invert(&h).expect("not invertible");
    let (lw, lh) = (layer.width() as f64, layer.height() as f64);
    for y in 0..frame.height() {
        for x in 0..frame.width() {
            let p = apply(&hi, [x as f64 + 0.5, y as f64 + 0.5]);
            if p[0] < 0.0 || p[1] < 0.0 || p[0] >= lw || p[1] >= lh {
                continue;
            }
            let s = layer.get_pixel(p[0] as u32, p[1] as u32);
            let a = s[3] as f32 / 255.0;
            let d = frame.get_pixel(x, y);
            let mix = |k: usize| (s[k] as f32 * a + d[k] as f32 * (1.0 - a)) as u8;
            frame.put_pixel(x, y, Rgba([mix(0), mix(1), mix(2), 255]));
        }
    }
    save(&frame, &args[5]);
}

fn save(img: &RgbaImage, path: &str) {
    img.save(path).expect("write output");
    println!("wrote {path}");
}

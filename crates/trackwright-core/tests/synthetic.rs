//! End-to-end: a textured plane moves under known homographies over a static
//! textured background; the tracker must recover the plane's corners.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use trackwright_core::export::{export, ExportFormat};
use trackwright_core::geometry::{apply, invert, Mat3, XorShift};
use trackwright_core::*;

const W: usize = 640;
const H: usize = 480;

/// Smooth multi-octave value noise.
fn noise_texture(seed: u64, w: usize, h: usize) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    let mut rng = XorShift::new(seed);
    for (cell, amp) in [(48usize, 90.0f32), (16, 60.0), (6, 40.0), (3, 25.0)] {
        let gw = w / cell + 2;
        let gh = h / cell + 2;
        let grid: Vec<f32> = (0..gw * gh)
            .map(|_| (rng.below(1000) as f32 / 1000.0) - 0.5)
            .collect();
        let g = GrayImage {
            width: gw,
            height: gh,
            data: grid,
        };
        for y in 0..h {
            for x in 0..w {
                img.data[y * w + x] +=
                    amp * g.sample(x as f32 / cell as f32, y as f32 / cell as f32);
            }
        }
    }
    for v in &mut img.data {
        *v = (*v + 128.0).clamp(0.0, 255.0);
    }
    img
}

struct Synthetic {
    frames: Vec<Arc<Pyramid>>,
}

impl FrameSource for Synthetic {
    fn frame_count(&self) -> usize {
        self.frames.len()
    }
    fn pyramid(&mut self, index: usize, _levels: usize) -> Result<Arc<Pyramid>, String> {
        self.frames
            .get(index)
            .cloned()
            .ok_or_else(|| "out of range".into())
    }
}

fn truth(k: usize) -> Mat3 {
    let t = k as f64;
    let ang = 0.012 * t;
    let s = 1.0 + 0.006 * t;
    let (c, sn) = (ang.cos() * s, ang.sin() * s);
    // Rotate/scale about the plane centre, translate, and add perspective.
    let (cx, cy) = (320.0, 240.0);
    let tx = 4.0 * t;
    let ty = -2.5 * t;
    let p = 1.5e-5 * t;
    [
        c,
        -sn,
        cx - c * cx + sn * cy + tx,
        sn,
        c,
        cy - sn * cx - c * cy + ty,
        p,
        -0.5 * p,
        1.0 - p * cx + 0.5 * p * cy,
    ]
}

fn render(plane: &GrayImage, bg: &GrayImage, rect: [f64; 4], h: &Mat3) -> GrayImage {
    let hi = invert(h).unwrap();
    let mut out = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let s = apply(&hi, [x as f64, y as f64]);
            let v = if s[0] >= rect[0] && s[0] <= rect[2] && s[1] >= rect[1] && s[1] <= rect[3] {
                plane.sample(s[0] as f32, s[1] as f32)
            } else {
                bg.get(x, y)
            };
            out.data[y * W + x] = v;
        }
    }
    out
}

#[test]
fn recovers_moving_plane() {
    let plane = noise_texture(11, W, H);
    let bg = noise_texture(99, W, H);
    let rect = [200.0, 150.0, 440.0, 330.0];
    let n = 24;
    let frames: Vec<Arc<Pyramid>> = (0..n)
        .map(|k| Arc::new(Pyramid::build(render(&plane, &bg, rect, &truth(k)), 4)))
        .collect();
    let mut src = Synthetic { frames };

    // Shape slightly inside the plane so the background is excluded.
    let shape = Shape::rect(
        rect[0] + 8.0,
        rect[1] + 8.0,
        rect[2] - rect[0] - 16.0,
        rect[3] - rect[1] - 16.0,
    );
    let mut project = Project::default();
    let id = project.add_layer(0, shape);
    let layer = project.layer_mut(id).unwrap();
    layer.surface = [
        [rect[0], rect[1]],
        [rect[2], rect[1]],
        [rect[2], rect[3]],
        [rect[0], rect[3]],
    ];
    let cancel = AtomicBool::new(false);
    let mut seen = 0;
    let r = track_layer(&mut src, layer, 0, n - 1, &cancel, |_, _| seen += 1)
        .expect("tracking succeeds");
    assert_eq!(r, StopReason::Finished);
    assert_eq!(seen, n - 1);

    let mut worst: f64 = 0.0;
    for k in 0..n {
        let est = layer.surface_at(k).unwrap();
        for (i, c) in layer.surface.iter().enumerate() {
            let t = apply(&truth(k), *c);
            let e = ((t[0] - est[i][0]).powi(2) + (t[1] - est[i][1]).powi(2)).sqrt();
            worst = worst.max(e);
        }
    }
    println!("worst corner error over {n} frames: {worst:.3}px");
    assert!(worst < 0.75, "corner error too large: {worst}");

    // Backward tracking from a later reference frame also works.
    let mut layer2 = Layer::new(2, n - 1, layer.shape.transformed(&truth(n - 1)));
    let r = track_layer(&mut src, &mut layer2, n - 1, 0, &cancel, |_, _| {}).unwrap();
    assert_eq!(r, StopReason::Finished);
    assert_eq!(layer2.tracks.len(), n);

    // Exporters produce well-formed output.
    let footage = Footage {
        source: "/tmp/plate.mov".into(),
        frames: vec![String::new(); n],
        width: W as u32,
        height: H as u32,
        fps: 24.0,
        first_frame: 1001,
    };
    for f in ExportFormat::ALL {
        let s = export(f, &footage, layer);
        assert!(!s.is_empty());
        if f.extension().ends_with("json") {
            let v: serde_json::Value = serde_json::from_str(&s).unwrap();
            assert!(v.get("format").is_some());
        }
    }
    let nk = export(ExportFormat::NukeCornerPin, &footage, layer);
    assert!(nk.contains("CornerPin2D") && nk.contains("x1001"));
    let ae = export(ExportFormat::AfterEffectsCornerPin, &footage, layer);
    assert!(
        ae.starts_with("Adobe After Effects 8.0 Keyframe Data") && ae.contains("Corner Pin-0004")
    );

    // Project round-trips through JSON.
    let json = project.to_json();
    assert_eq!(Project::from_json(&json).unwrap(), project);
}

fn truth_hard(k: usize) -> Mat3 {
    let t = k as f64;
    let ang = 0.01 * t;
    let s = 1.0 + 0.005 * t;
    let (c, sn) = (ang.cos() * s, ang.sin() * s);
    let (cx, cy) = (320.0, 240.0);
    let p = 1.2e-5 * t;
    [
        c,
        -sn,
        cx - c * cx + sn * cy + 2.0 * t,
        sn,
        c,
        cy - sn * cx - c * cy - 1.2 * t,
        p,
        -0.5 * p,
        1.0 - p * cx + 0.5 * p * cy,
    ]
}

/// Long shot with large accumulated rotation: anti-drift must keep error flat.
#[test]
fn no_drift_under_large_rotation() {
    let plane = noise_texture(5, W, H);
    let bg = noise_texture(77, W, H);
    let rect = [220.0, 160.0, 420.0, 320.0];
    let n = 48;
    let frames: Vec<Arc<Pyramid>> = (0..n)
        .map(|k| Arc::new(Pyramid::build(render(&plane, &bg, rect, &truth_hard(k)), 4)))
        .collect();
    let mut src = Synthetic { frames };
    let mut layer = Layer::new(
        1,
        0,
        Shape::rect(
            rect[0] + 6.0,
            rect[1] + 6.0,
            rect[2] - rect[0] - 12.0,
            rect[3] - rect[1] - 12.0,
        ),
    );
    layer.surface = [
        [rect[0], rect[1]],
        [rect[2], rect[1]],
        [rect[2], rect[3]],
        [rect[0], rect[3]],
    ];
    let cancel = AtomicBool::new(false);
    track_layer(&mut src, &mut layer, 0, n - 1, &cancel, |_, _| {}).expect("tracks");
    let mut worst: f64 = 0.0;
    for k in 0..n {
        let est = layer.surface_at(k).unwrap();
        for (i, c) in layer.surface.iter().enumerate() {
            let t = apply(&truth_hard(k), *c);
            worst = worst.max(((t[0] - est[i][0]).powi(2) + (t[1] - est[i][1]).powi(2)).sqrt());
        }
    }
    println!("large-rotation worst corner error over {n} frames: {worst:.3}px");
    assert!(worst < 0.75, "drift too large: {worst}");
}

#[test]
fn rebase_keeps_world_positions() {
    let shape = Shape::rect(10.0, 10.0, 50.0, 40.0);
    let mut l = Layer::new(1, 0, shape);
    l.tracks.insert(
        5,
        FrameTrack {
            h_from_ref: truth(5),
            ..FrameTrack::reference()
        },
    );
    let before = l.surface_at(5).unwrap();
    let before0 = l.surface_at(0).unwrap();
    assert!(l.rebase(5));
    let after = l.surface_at(5).unwrap();
    let after0 = l.surface_at(0).unwrap();
    for i in 0..4 {
        assert!(
            (before[i][0] - after[i][0]).abs() < 1e-6
                && (before0[i][1] - after0[i][1]).abs() < 1e-6
        );
    }
}

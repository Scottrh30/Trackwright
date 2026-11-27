//! End-to-end: a textured plane moves under known homographies over a static
//! textured background; the tracker must recover the plane's corners.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use trackwright_core::export::{export, ExportFormat};
use trackwright_core::geometry::{apply, invert, Mat3, Point, XorShift};
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
    let r = track_layer(
        &mut src,
        layer,
        0,
        n - 1,
        &no_exclusions,
        &cancel,
        |_, _| seen += 1,
    )
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
    let r = track_layer(
        &mut src,
        &mut layer2,
        n - 1,
        0,
        &no_exclusions,
        &cancel,
        |_, _| {},
    )
    .unwrap();
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
    track_layer(
        &mut src,
        &mut layer,
        0,
        n - 1,
        &no_exclusions,
        &cancel,
        |_, _| {},
    )
    .expect("tracks");
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

fn worst_error(layer: &Layer, n: usize, truth_fn: &dyn Fn(usize) -> Mat3) -> f64 {
    let mut worst: f64 = 0.0;
    for k in 0..n {
        let Some(est) = layer.surface_at(k) else {
            return f64::INFINITY;
        };
        for (i, c) in layer.surface.iter().enumerate() {
            let t = apply(&truth_fn(k), *c);
            worst = worst.max(((t[0] - est[i][0]).powi(2) + (t[1] - est[i][1]).powi(2)).sqrt());
        }
    }
    worst
}

fn track_or_inf(
    src: &mut Synthetic,
    layer: &mut Layer,
    n: usize,
    excl: &dyn Fn(usize) -> Vec<Vec<Point>>,
    truth_fn: &dyn Fn(usize) -> Mat3,
) -> f64 {
    let cancel = AtomicBool::new(false);
    match track_layer(src, layer, 0, n - 1, excl, &cancel, |_, _| {}) {
        Ok(_) => worst_error(layer, n, truth_fn),
        Err(e) => {
            println!("  (tracking failed: {e})");
            f64::INFINITY
        }
    }
}

fn plane_layer(rect: [f64; 4]) -> Layer {
    let mut l = Layer::new(
        1,
        0,
        Shape::rect(
            rect[0] + 6.0,
            rect[1] + 6.0,
            rect[2] - rect[0] - 12.0,
            rect[3] - rect[1] - 12.0,
        ),
    );
    l.surface = [
        [rect[0], rect[1]],
        [rect[2], rect[1]],
        [rect[2], rect[3]],
        [rect[0], rect[3]],
    ];
    l
}

/// A textured "hand" slides across the plane in the other direction. With the hand
/// set as an exclusion layer the plane must still track accurately.
#[test]
fn exclusion_layer_ignores_occluder() {
    let plane = noise_texture(21, W, H);
    let bg = noise_texture(22, W, H);
    let hand = noise_texture(23, W, H);
    let rect = [200.0, 150.0, 440.0, 330.0];
    let n = 30;
    // Occluder: 110x150 box moving right-to-left over the plane.
    let occ_pos = |k: usize| [430.0 - 9.0 * k as f64, 160.0 + 2.0 * k as f64];
    let frames: Vec<Arc<Pyramid>> = (0..n)
        .map(|k| {
            let mut img = render(&plane, &bg, rect, &truth(k));
            let o = occ_pos(k);
            for y in 0..150usize {
                for x in 0..110usize {
                    let (fx, fy) = (o[0] as usize + x, o[1] as usize + y);
                    if fx < W && fy < H {
                        img.data[fy * W + fx] = hand.get(x + 300, y + 200);
                    }
                }
            }
            Arc::new(Pyramid::build(img, 4))
        })
        .collect();
    let mut src = Synthetic { frames };
    let occ_outline = move |k: usize| {
        let o = occ_pos(k);
        vec![vec![
            [o[0] - 4.0, o[1] - 4.0],
            [o[0] + 114.0, o[1] - 4.0],
            [o[0] + 114.0, o[1] + 154.0],
            [o[0] - 4.0, o[1] + 154.0],
        ]]
    };

    let mut without = plane_layer(rect);
    without.settings.refine_against_reference = false;
    let e_without = track_or_inf(&mut src, &mut without, n, &no_exclusions, &truth);
    let mut with = plane_layer(rect);
    let e_with = track_or_inf(&mut src, &mut with, n, &occ_outline, &truth);
    println!("occluder: worst error without exclusion {e_without:.3}px (no anti-drift), with exclusion {e_with:.3}px");
    assert!(e_with < 0.75, "exclusion should keep the track: {e_with}");
}

/// Strong lighting flicker and a fade: gain 0.55..1.45 and bias -40..+40 per frame.
#[test]
fn lighting_changes_are_compensated() {
    let plane = noise_texture(31, W, H);
    let bg = noise_texture(32, W, H);
    let rect = [200.0, 150.0, 440.0, 330.0];
    let n = 24;
    let frames: Vec<Arc<Pyramid>> = (0..n)
        .map(|k| {
            let mut img = render(&plane, &bg, rect, &truth(k));
            let gain = 1.0 + 0.45 * ((k as f32) * 1.7).sin();
            let bias = 40.0 * ((k as f32) * 0.9).cos();
            for v in &mut img.data {
                *v = (*v * gain + bias).clamp(0.0, 255.0);
            }
            Arc::new(Pyramid::build(img, 4))
        })
        .collect();
    let mut src = Synthetic { frames };
    let mut off = plane_layer(rect);
    off.settings.illumination = false;
    off.settings.retry_hard_frames = false;
    let e_off = track_or_inf(&mut src, &mut off, n, &no_exclusions, &truth);
    let mut on = plane_layer(rect);
    let e_on = track_or_inf(&mut src, &mut on, n, &no_exclusions, &truth);
    println!("flicker: worst error without compensation {e_off:.3}px, with {e_on:.3}px");
    assert!(
        e_on < 0.75,
        "lighting compensation should hold the track: {e_on}"
    );
}

/// Corrupt a section of a good track, re-track just that section, and check it
/// is fixed and joins the untouched frames after it.
#[test]
fn retrack_section_blends_into_existing_track() {
    let plane = noise_texture(41, W, H);
    let bg = noise_texture(42, W, H);
    let rect = [200.0, 150.0, 440.0, 330.0];
    let n = 30;
    let frames: Vec<Arc<Pyramid>> = (0..n)
        .map(|k| Arc::new(Pyramid::build(render(&plane, &bg, rect, &truth(k)), 4)))
        .collect();
    let mut src = Synthetic { frames };
    let mut l = plane_layer(rect);
    assert!(track_or_inf(&mut src, &mut l, n, &no_exclusions, &truth) < 0.75);
    let untouched: Vec<_> = (21..n).map(|f| l.tracks[&f].h_from_ref).collect();
    // Simulate a bad patch: frames 11..=19 drift up to 9px.
    for f in 11..20usize {
        let t = l.tracks.get_mut(&f).unwrap();
        let d = (f - 10) as f64;
        t.h_from_ref = trackwright_core::geometry::mul(
            &[1.0, 0.0, d, 0.0, 1.0, -d, 0.0, 0.0, 1.0],
            &t.h_from_ref,
        );
    }
    assert!(worst_error(&l, n, &truth) > 5.0);
    let cancel = AtomicBool::new(false);
    retrack_section(&mut src, &mut l, 10, 20, &no_exclusions, &cancel, |_, _| {}).unwrap();
    let e = worst_error(&l, n, &truth);
    println!("retrack section: worst error after fix {e:.3}px");
    assert!(e < 0.75);
    let after: Vec<_> = (21..n).map(|f| l.tracks[&f].h_from_ref).collect();
    assert_eq!(untouched, after, "frames after the section must not change");
}

/// Timing on a 4K plate (run with: cargo test --release --features parallel -- --ignored --nocapture).
#[test]
#[ignore]
fn bench_4k() {
    let (w, h) = (3840usize, 2160usize);
    let plane = noise_texture(51, w, h);
    let bg = noise_texture(52, w, h);
    let rect = [1300.0, 700.0, 2500.0, 1500.0];
    let n = 12;
    let s4 = 6.0;
    let truth4k = |k: usize| {
        let t = k as f64;
        let ang = 0.008 * t;
        let (c, sn) = (ang.cos(), ang.sin());
        let (cx, cy) = (1900.0, 1100.0);
        [
            c,
            -sn,
            cx - c * cx + sn * cy + s4 * t,
            sn,
            c,
            cy - sn * cx - c * cy - 3.0 * t,
            0.0,
            0.0,
            1.0,
        ]
    };
    let t0 = std::time::Instant::now();
    let frames: Vec<Arc<Pyramid>> = (0..n)
        .map(|k| {
            let hi = invert(&truth4k(k)).unwrap();
            let mut img = GrayImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let s = apply(&hi, [x as f64, y as f64]);
                    img.data[y * w + x] =
                        if s[0] >= rect[0] && s[0] <= rect[2] && s[1] >= rect[1] && s[1] <= rect[3]
                        {
                            plane.sample(s[0] as f32, s[1] as f32)
                        } else {
                            bg.get(x, y)
                        };
                }
            }
            Arc::new(Pyramid::build(img, 6))
        })
        .collect();
    println!(
        "rendered + pyramids for {n} 4K frames in {:.2}s",
        t0.elapsed().as_secs_f64()
    );
    let mut src = Synthetic { frames };
    for (label, level) in [("full-res pass", Some(0)), ("auto (half-res pass)", None)] {
        let mut l = plane_layer(rect);
        l.settings.detail_level = level;
        let t = std::time::Instant::now();
        let e = track_or_inf(&mut src, &mut l, n, &no_exclusions, &truth4k);
        println!(
            "4K {label}: {:.0} ms/frame, worst error {e:.3}px",
            t.elapsed().as_secs_f64() * 1000.0 / (n - 1) as f64
        );
    }
}

#[test]
fn new_export_formats_are_well_formed() {
    let mut l = Layer::new(1, 0, Shape::rect(100.0, 100.0, 200.0, 100.0));
    for k in 1..5 {
        l.tracks.insert(
            k,
            FrameTrack {
                h_from_ref: truth(k),
                ..FrameTrack::reference()
            },
        );
    }
    let footage = Footage {
        source: "C:\\shots\\plate.mov".into(),
        frames: vec![String::new(); 5],
        width: 640,
        height: 480,
        fps: 24.0,
        first_frame: 1001,
    };
    let fu = export(ExportFormat::FusionCornerPin, &footage, &l);
    assert!(
        fu.contains("CornerPositioner")
            && fu.contains("Trackwright_Layer_1TopLeftX = BezierSpline")
            && fu.contains("[4] = {")
    );
    assert_eq!(
        fu.matches('{').count(),
        fu.matches('}').count(),
        "balanced braces"
    );
    // Top-left of the reference surface is (100,100) -> normalised (0.15625, 1-0.2083).
    assert!(fu.contains("[0] = { 0.156250,") && fu.contains("[0] = { 0.791667,"));
    let py = export(ExportFormat::Blender, &footage, &l);
    assert!(
        py.contains("bpy.data.movieclips.load")
            && py.contains(r#"FOOTAGE = r"C:\shots\plate.mov""#)
            && py.contains("    1: ((100.0000, 100.0000)")
    );
    let sh = export(ExportFormat::SilhouetteShake, &footage, &l);
    assert_eq!(sh.matches("TrackName").count(), 4);
    assert!(sh.contains("   1.00   100.000   380.000   1.000"));
}

#[test]
#[ignore]
fn dump_new_formats() {
    let mut l = Layer::new(1, 0, Shape::rect(100.0, 100.0, 200.0, 100.0));
    for k in 1..3 {
        l.tracks.insert(
            k,
            FrameTrack {
                h_from_ref: truth(k),
                ..FrameTrack::reference()
            },
        );
    }
    let footage = Footage {
        source: "/tmp/plate.mov".into(),
        frames: vec![String::new(); 3],
        width: 640,
        height: 480,
        fps: 24.0,
        first_frame: 1001,
    };
    for f in [
        ExportFormat::FusionCornerPin,
        ExportFormat::Blender,
        ExportFormat::SilhouetteShake,
    ] {
        std::fs::write(
            format!("/tmp/twexport.{}", f.extension()),
            export(f, &footage, &l),
        )
        .unwrap();
    }
}

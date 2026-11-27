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

/// Face motion for the blink tests: sway, drift down, slow turn.
fn truth_face(k: usize) -> Mat3 {
    let t = k as f64;
    let ang = 0.004 * t;
    let (c, sn) = (ang.cos(), ang.sin());
    let (cx, cy) = (320.0, 240.0);
    let tx = 30.0 * (t * 0.15).sin();
    let ty = 2.5 * t;
    [
        c,
        -sn,
        cx - c * cx + sn * cy + tx,
        sn,
        c,
        cy - sn * cx - c * cy + ty,
        0.0,
        0.0,
        1.0,
    ]
}

/// A "pupil" patch on a moving "face"; an "eyelid" (smooth skin) closes over it
/// for `closed` frames (half-closed on the frames either side), then opens.
fn blink_footage(
    n: usize,
    closed: std::ops::Range<usize>,
    face_textured: bool,
) -> (Vec<Arc<Pyramid>>, [f64; 4]) {
    blink_footage_moving(n, closed, face_textured, &|_| [0.0, 0.0])
}

/// As `blink_footage`, with the pupil shifted by `gaze(k)` (face pixels) inside the
/// eye on frame k: an eye that looks somewhere else.
fn blink_footage_moving(
    n: usize,
    closed: std::ops::Range<usize>,
    face_textured: bool,
    gaze: &dyn Fn(usize) -> [f64; 2],
) -> (Vec<Arc<Pyramid>>, [f64; 4]) {
    let face = if face_textured {
        noise_texture(61, W, H)
    } else {
        GrayImage {
            width: W,
            height: H,
            data: vec![128.0; W * H],
        }
    };
    let pupil = noise_texture(62, W, H);
    let lid = {
        let mut l = noise_texture(63, W, H);
        for v in &mut l.data {
            *v = 150.0 + (*v - 128.0) * 0.08; // nearly flat skin
        }
        l
    };
    let rect = [280.0, 200.0, 370.0, 265.0];
    let frames = (0..n)
        .map(|k| {
            let h = truth_face(k);
            let hi = invert(&h).unwrap();
            // How far the lid is down (0 open .. 1 closed).
            let down = if closed.contains(&k) {
                1.0
            } else if k + 1 == closed.start || k == closed.end {
                0.5
            } else {
                0.0
            };
            let gz = gaze(k);
            let mut img = GrayImage::new(W, H);
            for y in 0..H {
                for x in 0..W {
                    let s = apply(&hi, [x as f64, y as f64]);
                    let in_eye = s[0] >= rect[0] - 6.0
                        && s[0] <= rect[2] + 6.0
                        && s[1] >= rect[1] - 6.0
                        && s[1] <= rect[3] + 6.0;
                    let lid_edge = rect[1] - 6.0 + down * (rect[3] - rect[1] + 12.0);
                    img.data[y * W + x] = if in_eye && s[1] <= lid_edge && down > 0.0 {
                        lid.sample(s[0] as f32, s[1] as f32)
                    } else if s[0] >= rect[0] + gz[0]
                        && s[0] <= rect[2] + gz[0]
                        && s[1] >= rect[1] + gz[1]
                        && s[1] <= rect[3] + gz[1]
                    {
                        pupil.sample((s[0] - gz[0]) as f32, (s[1] - gz[1]) as f32)
                    } else {
                        face.sample(s[0] as f32, s[1] as f32)
                    };
                }
            }
            Arc::new(Pyramid::build(img, 4))
        })
        .collect();
    (frames, rect)
}

/// Worst corner error over `frames`, ignoring frames with no track.
fn worst_over(layer: &Layer, frames: impl Iterator<Item = usize>) -> f64 {
    frames
        .map(|k| {
            let Some(est) = layer.surface_at(k) else {
                return f64::INFINITY;
            };
            layer
                .surface
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let t = apply(&truth_face(k), *c);
                    ((t[0] - est[i][0]).powi(2) + (t[1] - est[i][1]).powi(2)).sqrt()
                })
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

/// A blink: the pupil is covered for 6 frames. The occlusion solver carries it
/// with the face around it, finds it again when the eye opens, and joins up.
#[test]
fn occlusion_solver_tracks_through_a_blink() {
    let n = 36;
    let closed = 14..20;
    let (frames, rect) = blink_footage(n, closed.clone(), true);
    // TW_DUMP_BLINK=<dir> writes the footage as PGM for trying it in the app.
    if let Ok(dir) = std::env::var("TW_DUMP_BLINK") {
        std::fs::create_dir_all(&dir).unwrap();
        for (k, f) in frames.iter().enumerate() {
            let mut out = format!("P5\n{W} {H}\n255\n").into_bytes();
            out.extend(
                f.levels[0]
                    .data
                    .iter()
                    .map(|v| v.round().clamp(0.0, 255.0) as u8),
            );
            std::fs::write(format!("{dir}/blink_{k:04}.pgm"), out).unwrap();
        }
    }
    let mut src = Synthetic { frames };
    let cancel = AtomicBool::new(false);

    let mut off = plane_layer(rect);
    off.settings.bridge_occlusions = false;
    let without = track_layer(
        &mut src,
        &mut off,
        0,
        n - 1,
        &no_exclusions,
        &cancel,
        |_, _| {},
    );
    let off_err = worst_over(&off, closed.end + 2..n);
    println!(
        "blink without solver: {:?}, error after {off_err:.2}px",
        without.as_ref().map_err(|e| e.to_string())
    );

    let mut on = plane_layer(rect);
    let r = track_layer(
        &mut src,
        &mut on,
        0,
        n - 1,
        &no_exclusions,
        &cancel,
        |_, _| {},
    );
    assert_eq!(r, Ok(StopReason::Finished), "{r:?}");
    let predicted: Vec<usize> = on
        .tracks
        .iter()
        .filter(|(_, t)| t.predicted)
        .map(|(f, _)| *f)
        .collect();
    let before = worst_over(&on, 0..closed.start - 1);
    let during = worst_over(&on, closed.clone());
    let after = worst_over(&on, closed.end + 2..n);
    println!("blink with solver: predicted {predicted:?}; error before {before:.2}px, while covered {during:.2}px, after {after:.2}px");
    assert!(
        !predicted.is_empty(),
        "the covered frames should be marked predicted"
    );
    assert!(
        predicted
            .iter()
            .all(|f| *f + 2 >= closed.start && *f <= closed.end + 1),
        "only covered frames are predicted: {predicted:?}"
    );
    assert!(before < 0.75 && after < 0.75, "seen frames stay accurate");
    assert!(during < 3.0, "covered frames follow the face: {during}");
}

/// Nothing to follow around it (a flat face): the prediction carries on with
/// the pupil's own recent motion, and the track still picks up after the blink.
#[test]
fn occlusion_solver_predicts_from_motion_alone() {
    let n = 30;
    let closed = 12..16;
    let (frames, rect) = blink_footage(n, closed.clone(), false);
    let mut src = Synthetic { frames };
    let cancel = AtomicBool::new(false);
    let mut on = plane_layer(rect);
    on.settings.model = MotionModel::Similarity;
    let r = track_layer(
        &mut src,
        &mut on,
        0,
        n - 1,
        &no_exclusions,
        &cancel,
        |_, _| {},
    );
    assert_eq!(r, Ok(StopReason::Finished), "{r:?}");
    let during = worst_over(&on, closed.clone());
    let after = worst_over(&on, closed.end + 2..n);
    println!("motion-only blink: while covered {during:.2}px, after {after:.2}px");
    assert!(after < 1.0, "picks up again after the blink: {after}");
    assert!(during < 12.0, "prediction stays near: {during}");
}

/// Covered for longer than "Bridge up to": the guesses are dropped and the
/// track stops with a clear message instead of wandering off.
#[test]
fn occlusion_solver_gives_up_after_the_limit() {
    let n = 30;
    let (frames, rect) = blink_footage(n, 10..30, true);
    let mut src = Synthetic { frames };
    let cancel = AtomicBool::new(false);
    let mut l = plane_layer(rect);
    l.settings.max_bridge_frames = 6;
    let r = track_layer(
        &mut src,
        &mut l,
        0,
        n - 1,
        &no_exclusions,
        &cancel,
        |_, _| {},
    );
    println!("long cover: {:?}", r.as_ref().map_err(|e| e.to_string()));
    assert!(
        matches!(r, Err(TrackError::Occluded { frames: 6, .. })),
        "{r:?}"
    );
    assert!(
        l.tracks.values().all(|t| !t.predicted),
        "guesses are removed"
    );
}

/// The eye turns while it's shut: the pupil comes back 16 px to the right of where
/// it was. The solver searches near the prediction, finds it there, and the track
/// follows the pupil (not just the face) afterwards.
#[test]
fn occlusion_solver_finds_a_target_that_moved_while_hidden() {
    let n = 32;
    let closed = 12..18;
    let gaze = |k: usize| if k >= 15 { [16.0, 0.0] } else { [0.0, 0.0] };
    let (frames, rect) = blink_footage_moving(n, closed.clone(), true, &gaze);
    let mut src = Synthetic { frames };
    let cancel = AtomicBool::new(false);
    let mut l = plane_layer(rect);
    l.settings.model = MotionModel::Translation;
    let r = track_layer(
        &mut src,
        &mut l,
        0,
        n - 1,
        &no_exclusions,
        &cancel,
        |_, _| {},
    );
    assert_eq!(r, Ok(StopReason::Finished), "{r:?}");
    // Error of the centre (what a pupil track is for; a translation-only box can't
    // turn with the face, so its corners can't match) against where the pupil really
    // is: the face's motion plus the gaze shift.
    let centre = |p: &[Point; 4]| {
        [
            (p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4.0,
            (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4.0,
        ]
    };
    let err = |k: usize| {
        let e = centre(&l.surface_at(k).unwrap());
        let (mid, g) = (centre(&l.surface), gaze(k));
        let t = apply(&truth_face(k), [mid[0] + g[0], mid[1] + g[1]]);
        ((t[0] - e[0]).powi(2) + (t[1] - e[1]).powi(2)).sqrt()
    };
    let before = (0..closed.start - 1).map(err).fold(0.0, f64::max);
    let after = (closed.end + 2..n).map(err).fold(0.0, f64::max);
    let predicted: Vec<usize> = l
        .tracks
        .iter()
        .filter(|(_, t)| t.predicted)
        .map(|(f, _)| *f)
        .collect();
    println!("moved while hidden: predicted {predicted:?}, centre error before {before:.2}px, after {after:.2}px");
    assert!(after < 1.5, "follows the pupil to where it went: {after}");
}

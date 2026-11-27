//! End to end: ray-trace a textured room seen by a moving camera with lens distortion,
//! auto-track points in the images, solve the camera and compare with the truth.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use trackwright_core::geometry::XorShift;
use trackwright_core::lens::Lens;
use trackwright_core::linalg::{
    add, log_rotation, mat_vec, norm, rodrigues, scale, sub, transpose, V3,
};
use trackwright_core::points::{auto_track, AutoTrackSettings};
use trackwright_core::sfm::{solve_camera, FocalMode, Pose, SolveMode, SolveOptions};
use trackwright_core::*;

const W: usize = 640;
const H: usize = 360;

fn texture(seed: u64) -> GrayImage {
    let n = 512;
    let mut img = GrayImage::new(n, n);
    let mut rng = XorShift::new(seed);
    for (cell, amp) in [(64usize, 80.0f32), (16, 60.0), (6, 45.0), (2, 25.0)] {
        let g = n / cell;
        let grid: Vec<f32> = (0..g * g)
            .map(|_| rng.below(1000) as f32 / 1000.0 - 0.5)
            .collect();
        for y in 0..n {
            for x in 0..n {
                // Wrapping bilinear value noise so the texture tiles.
                let (fx, fy) = (x as f32 / cell as f32, y as f32 / cell as f32);
                let (x0, y0) = (fx.floor() as usize % g, fy.floor() as usize % g);
                let (x1, y1) = ((x0 + 1) % g, (y0 + 1) % g);
                let (tx, ty) = (fx.fract(), fy.fract());
                let v = |a: usize, b: usize| grid[b * g + a];
                let top = v(x0, y0) + (v(x1, y0) - v(x0, y0)) * tx;
                let bot = v(x0, y1) + (v(x1, y1) - v(x0, y1)) * tx;
                img.data[y * n + x] += amp * (top + (bot - top) * ty);
            }
        }
    }
    for v in &mut img.data {
        *v = (*v + 128.0).clamp(0.0, 255.0);
    }
    img
}

fn sample_wrap(t: &GrayImage, u: f64, v: f64) -> f32 {
    let n = t.width as f64;
    t.sample(u.rem_euclid(n - 1.0) as f32, v.rem_euclid(n - 1.0) as f32)
}

fn pose_at(k: usize, n: usize) -> Pose {
    let t = k as f64 / n as f64;
    let r = rodrigues([0.04 * t - 0.05, 0.3 * t - 0.12, 0.01]);
    let c = [2.5 * t - 0.8, -0.2 * t, 1.2 * t];
    Pose {
        r: log_rotation(&r),
        t: scale(mat_vec(&r, c), -1.0),
    }
}

fn render(pose: &Pose, focal: f64, lens: &Lens, tex: &[GrayImage; 3]) -> GrayImage {
    let mut img = GrayImage::new(W, H);
    let rt = transpose(&pose.rot());
    let c = pose.center();
    for y in 0..H {
        for x in 0..W {
            let ideal = lens.undistort([x as f64 + 0.5, y as f64 + 0.5], W as f64, H as f64);
            let d = mat_vec(
                &rt,
                [
                    (ideal[0] - W as f64 / 2.0) / focal,
                    (ideal[1] - H as f64 / 2.0) / focal,
                    1.0,
                ],
            );
            // Floor y = 1.6 (y points down), back wall z = 18, left wall x = -6.
            let mut best = (f64::INFINITY, 0.0f32);
            let mut hit = |t: f64, val: f32| {
                if t > 0.0 && t < best.0 {
                    best = (t, val);
                }
            };
            if d[1].abs() > 1e-9 {
                let t = (1.6 - c[1]) / d[1];
                let p = add(c, scale(d, t));
                hit(t, sample_wrap(&tex[0], p[0] * 40.0, p[2] * 40.0));
            }
            if d[2].abs() > 1e-9 {
                let t = (18.0 - c[2]) / d[2];
                let p = add(c, scale(d, t));
                hit(t, sample_wrap(&tex[1], p[0] * 30.0, p[1] * 30.0));
            }
            if d[0].abs() > 1e-9 {
                let t = (-6.0 - c[0]) / d[0];
                let p = add(c, scale(d, t));
                hit(t, sample_wrap(&tex[2], p[2] * 35.0, p[1] * 35.0));
            }
            img.data[y * W + x] = if best.0.is_finite() { best.1 } else { 40.0 };
        }
    }
    img
}

struct Frames(Vec<Arc<Pyramid>>);

impl FrameSource for Frames {
    fn frame_count(&self) -> usize {
        self.0.len()
    }
    fn pyramid(&mut self, i: usize, _: usize) -> Result<Arc<Pyramid>, String> {
        self.0.get(i).cloned().ok_or_else(|| "out of range".into())
    }
}

#[test]
fn rendered_room_solve() {
    let n = 40;
    let focal = 520.0;
    let lens = Lens { k1: -0.06, k2: 0.0 };
    let tex = [texture(1), texture(2), texture(3)];
    let truth: Vec<Pose> = (0..n).map(|k| pose_at(k, n)).collect();
    let frames: Vec<Arc<Pyramid>> = truth
        .iter()
        .map(|p| Arc::new(Pyramid::build(render(p, focal, &lens, &tex), 4)))
        .collect();
    // TW_DUMP=<dir> writes the rendered frames as PGM for manual testing in the app.
    if let Ok(dir) = std::env::var("TW_DUMP") {
        std::fs::create_dir_all(&dir).unwrap();
        for (k, p) in truth.iter().enumerate() {
            let img = render(p, focal, &lens, &tex);
            let mut out = format!("P5\n{W} {H}\n255\n").into_bytes();
            out.extend(img.data.iter().map(|v| v.round().clamp(0.0, 255.0) as u8));
            std::fs::write(format!("{dir}/room_{k:04}.pgm"), out).unwrap();
        }
    }
    let mut src = Frames(frames);
    let cancel = AtomicBool::new(false);
    let settings = AutoTrackSettings::for_width(W, 400, 30.0);
    let (tracks, _) = auto_track(
        &mut src,
        0,
        n - 1,
        &settings,
        &no_exclusions,
        &cancel,
        |_, _| {},
    )
    .unwrap();
    let long = tracks.iter().filter(|t| t.points.len() >= 10).count();
    println!(
        "auto-tracked {} tracks ({} spanning 10+ frames)",
        tracks.len(),
        long
    );
    assert!(tracks.len() > 150);

    let opts = SolveOptions {
        width: W as f64,
        height: H as f64,
        focal: FocalMode::Auto,
        sensor_mm: 36.0,
        mode: SolveMode::Auto,
        solve_distortion: true,
        lens: Lens::default(),
    };
    let frames_idx: Vec<usize> = (0..n).collect();
    let solve = solve_camera(&tracks, &frames_idx, &opts, &cancel, |_, _| {}).expect("solve");

    // Compare camera centres after a similarity alignment.
    let a: Vec<V3> = (0..n).map(|k| solve.poses[&k].center()).collect();
    let b: Vec<V3> = truth.iter().map(|p| p.center()).collect();
    let ca = scale(a.iter().fold([0.0; 3], |s, p| add(s, *p)), 1.0 / n as f64);
    let cb = scale(b.iter().fold([0.0; 3], |s, p| add(s, *p)), 1.0 / n as f64);
    let mut cov = [0.0; 9];
    let mut va = 0.0;
    for (p, q) in a.iter().zip(&b) {
        let (da, db) = (sub(*p, ca), sub(*q, cb));
        va += da.iter().map(|v| v * v).sum::<f64>();
        for i in 0..3 {
            for j in 0..3 {
                cov[i * 3 + j] += db[i] * da[j];
            }
        }
    }
    let r = trackwright_core::linalg::nearest_rotation(&cov);
    let (_, sv, _) = trackwright_core::linalg::svd3(&cov);
    let sc = (sv[0] + sv[1] + sv[2]) / va;
    let len: f64 = b.windows(2).map(|w| norm(sub(w[1], w[0]))).sum();
    let worst = a
        .iter()
        .zip(&b)
        .map(|(p, q)| norm(sub(add(scale(mat_vec(&r, sub(*p, ca)), sc), cb), *q)))
        .fold(0.0, f64::max);
    println!(
        "solve: focal {:.1} (true {focal}), k1 {:.4} (true {}), rms {:.3}px, {} points, camera path error {:.2}% of its length",
        solve.focal_px,
        solve.lens.k1,
        lens.k1,
        solve.rms_px,
        solve.points.len(),
        100.0 * worst / len
    );
    // Exported camera convention: project a solved point through the Y-up OpenGL camera.
    for k in [0usize, 20, 39] {
        let (pos, rot) = solve.world.camera(&solve.poses[&k]);
        for sp in solve.points.iter().take(20) {
            let q = solve.world.to_user(sp.pos);
            let d = sub(q, pos);
            let c = mat_vec(&transpose(&rot), d); // camera-space, looking down -Z
            let px = [
                W as f64 / 2.0 + solve.focal_px * c[0] / -c[2],
                H as f64 / 2.0 - solve.focal_px * c[1] / -c[2],
            ];
            let ideal = solve.project_ideal(&solve.poses[&k], sp.pos).unwrap();
            assert!((px[0] - ideal[0]).abs() < 1e-6 && (px[1] - ideal[1]).abs() < 1e-6);
        }
    }
    let footage = Footage {
        source: "/tmp/room.mov".into(),
        frames: vec![String::new(); n],
        width: W as u32,
        height: H as u32,
        fps: 24.0,
        first_frame: 1001,
    };
    for fmt in trackwright_core::camera_export::CameraFormat::ALL {
        let text =
            trackwright_core::camera_export::export_camera(fmt, &footage, &solve, 36.0, "Room cam");
        std::fs::write(format!("/tmp/twcam.{}", fmt.extension()), text).unwrap();
    }
    assert!((solve.focal_px - focal).abs() / focal < 0.03);
    assert!((solve.lens.k1 - lens.k1).abs() < 0.02);
    assert!(solve.rms_px < 0.5);
    assert!(worst / len < 0.02);
}

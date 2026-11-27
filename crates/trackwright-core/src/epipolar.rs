//! Two-view and resection geometry: fundamental/essential matrices, triangulation
//! and perspective-n-point.

use crate::geometry::{mul, Mat3, Point, XorShift};
use crate::linalg::{self, mat_vec, null_vector, transpose, V3};

/// Hartley normalisation for 2D points: returns (normalised points, T) with p' = T p.
fn normalise(pts: &[Point]) -> (Vec<Point>, Mat3) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let d = pts
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    let s = if d > 1e-12 {
        std::f64::consts::SQRT_2 / d
    } else {
        1.0
    };
    (
        pts.iter()
            .map(|p| [(p[0] - cx) * s, (p[1] - cy) * s])
            .collect(),
        [s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0],
    )
}

/// Normalised 8-point fundamental matrix with rank 2 enforced. `x2ᵀ F x1 = 0`.
pub fn fundamental_8pt(p1: &[Point], p2: &[Point]) -> Option<Mat3> {
    if p1.len() < 8 {
        return None;
    }
    let (a, t1) = normalise(p1);
    let (b, t2) = normalise(p2);
    let rows: Vec<Vec<f64>> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| {
            vec![
                y[0] * x[0],
                y[0] * x[1],
                y[0],
                y[1] * x[0],
                y[1] * x[1],
                y[1],
                x[0],
                x[1],
                1.0,
            ]
        })
        .collect();
    let f = null_vector(&rows, 9);
    let fm: Mat3 = [f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]];
    let (u, s, v) = linalg::svd3(&fm);
    let us = [
        u[0] * s[0],
        u[1] * s[1],
        0.0,
        u[3] * s[0],
        u[4] * s[1],
        0.0,
        u[6] * s[0],
        u[7] * s[1],
        0.0,
    ];
    let f2 = mul(&us, &transpose(&v));
    let full = mul(&transpose(&t2), &mul(&f2, &t1));
    let n = full.iter().map(|v| v * v).sum::<f64>().sqrt();
    (n > 1e-300).then(|| full.map(|v| v / n))
}

/// Sampson distance (squared, pixels²) of a correspondence to F.
pub fn sampson(f: &Mat3, a: Point, b: Point) -> f64 {
    let x1 = [a[0], a[1], 1.0];
    let x2 = [b[0], b[1], 1.0];
    let fx1 = mat_vec(f, x1);
    let ftx2 = mat_vec(&transpose(f), x2);
    let e = linalg::dot(x2, fx1);
    let den = fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1];
    if den < 1e-300 {
        return f64::INFINITY;
    }
    e * e / den
}

/// RANSAC fundamental matrix. Returns F and the inlier mask.
pub fn ransac_fundamental(
    p1: &[Point],
    p2: &[Point],
    threshold: f64,
    seed: u64,
) -> Option<(Mat3, Vec<bool>)> {
    let n = p1.len();
    if n < 8 {
        return None;
    }
    let t2 = threshold * threshold;
    let mut rng = XorShift::new(seed ^ 0x51F1);
    let mut best: Option<(Mat3, usize)> = None;
    let mut iters = 500usize;
    let mut i = 0;
    let mut idx = [0usize; 8];
    while i < iters {
        i += 1;
        let mut k = 0;
        let mut guard = 0;
        while k < 8 && guard < 200 {
            guard += 1;
            let c = rng.below(n);
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        if k < 8 {
            continue;
        }
        let a: Vec<Point> = idx.iter().map(|&j| p1[j]).collect();
        let b: Vec<Point> = idx.iter().map(|&j| p2[j]).collect();
        let Some(f) = fundamental_8pt(&a, &b) else {
            continue;
        };
        let count = (0..n).filter(|&j| sampson(&f, p1[j], p2[j]) < t2).count();
        if best.as_ref().map_or(true, |b| count > b.1) {
            best = Some((f, count));
            let w = count as f64 / n as f64;
            let p = w.powi(8);
            if p > 1e-9 {
                let need = ((1.0 - 0.995f64).ln() / (1.0 - p).max(1e-12).ln()).ceil();
                iters = iters.min(need.max(30.0) as usize);
            }
        }
    }
    let (mut f, _) = best?;
    // Refit on all inliers twice.
    for _ in 0..2 {
        let inl: Vec<usize> = (0..n).filter(|&j| sampson(&f, p1[j], p2[j]) < t2).collect();
        if inl.len() < 8 {
            break;
        }
        let a: Vec<Point> = inl.iter().map(|&j| p1[j]).collect();
        let b: Vec<Point> = inl.iter().map(|&j| p2[j]).collect();
        if let Some(r) = fundamental_8pt(&a, &b) {
            f = r;
        }
    }
    let mask = (0..n).map(|j| sampson(&f, p1[j], p2[j]) < t2).collect();
    Some((f, mask))
}

/// How far `E = Kᵀ F K` is from a valid essential matrix (two equal singular values).
/// Used to estimate focal length from a fundamental matrix.
pub fn essential_cost(f: &Mat3, focal: f64, cx: f64, cy: f64) -> f64 {
    let k = [focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0];
    let e = mul(&transpose(&k), &mul(f, &k));
    let (_, s, _) = linalg::svd3(&e);
    (s[0] - s[1]) / (s[0] + s[1]).max(1e-300)
}

/// Focal length (pixels) that makes F most consistent with a calibrated camera.
pub fn focal_from_fundamental(f: &Mat3, width: f64, height: f64) -> f64 {
    let (cx, cy) = (width / 2.0, height / 2.0);
    let (lo, hi) = ((0.3 * width).ln(), (5.0 * width).ln());
    let steps = 240;
    let mut best = (f64::INFINITY, width);
    for i in 0..=steps {
        let fl = (lo + (hi - lo) * i as f64 / steps as f64).exp();
        let c = essential_cost(f, fl, cx, cy);
        if c < best.0 {
            best = (c, fl);
        }
    }
    // Golden-section refinement around the best sample.
    let ratio = ((hi - lo) / steps as f64).exp();
    let (mut a, mut b) = (best.1 / ratio, best.1 * ratio);
    let g = 0.618_033_988_75;
    for _ in 0..40 {
        let c1 = b - g * (b - a);
        let c2 = a + g * (b - a);
        if essential_cost(f, c1, cx, cy) < essential_cost(f, c2, cx, cy) {
            b = c2;
        } else {
            a = c1;
        }
    }
    (a + b) / 2.0
}

/// The four (R, t) decompositions of an essential matrix.
pub fn decompose_essential(e: &Mat3) -> [(Mat3, V3); 4] {
    let (mut u, _, mut v) = linalg::svd3(e);
    if linalg::det3(&u) < 0.0 {
        for r in 0..3 {
            u[r * 3 + 2] = -u[r * 3 + 2];
        }
    }
    if linalg::det3(&v) < 0.0 {
        for r in 0..3 {
            v[r * 3 + 2] = -v[r * 3 + 2];
        }
    }
    let w = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let r1 = mul(&u, &mul(&w, &transpose(&v)));
    let r2 = mul(&u, &mul(&transpose(&w), &transpose(&v)));
    let t = [u[2], u[5], u[8]];
    let nt = [-t[0], -t[1], -t[2]];
    [(r1, t), (r1, nt), (r2, t), (r2, nt)]
}

/// Linear triangulation from normalised image coordinates `(x, y)` (i.e. K⁻¹ applied)
/// and camera poses `(R, t)` (world -> camera). Returns `None` for points at infinity.
pub fn triangulate(views: &[(Mat3, V3, Point)]) -> Option<V3> {
    if views.len() < 2 {
        return None;
    }
    let mut rows = Vec::with_capacity(views.len() * 2);
    for (r, t, x) in views {
        let p = [
            [r[0], r[1], r[2], t[0]],
            [r[3], r[4], r[5], t[1]],
            [r[6], r[7], r[8], t[2]],
        ];
        rows.push(
            (0..4)
                .map(|k| x[0] * p[2][k] - p[0][k])
                .collect::<Vec<f64>>(),
        );
        rows.push(
            (0..4)
                .map(|k| x[1] * p[2][k] - p[1][k])
                .collect::<Vec<f64>>(),
        );
    }
    let h = null_vector(&rows, 4);
    if h[3].abs() < 1e-12 {
        return None;
    }
    Some([h[0] / h[3], h[1] / h[3], h[2] / h[3]])
}

/// DLT perspective-n-point from ≥ 6 correspondences (3D point, normalised image point).
pub fn pnp_dlt(world: &[V3], img: &[Point]) -> Option<(Mat3, V3)> {
    let n = world.len();
    if n < 6 {
        return None;
    }
    let c = world.iter().fold([0.0; 3], |a, p| linalg::add(a, *p));
    let c = linalg::scale(c, 1.0 / n as f64);
    let sigma = world
        .iter()
        .map(|p| linalg::norm(linalg::sub(*p, c)))
        .sum::<f64>()
        / n as f64;
    let sigma = sigma.max(1e-12);
    let mut rows = Vec::with_capacity(2 * n);
    for (p, x) in world.iter().zip(img) {
        let q = linalg::scale(linalg::sub(*p, c), 1.0 / sigma);
        rows.push(vec![
            q[0],
            q[1],
            q[2],
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -x[0] * q[0],
            -x[0] * q[1],
            -x[0] * q[2],
            -x[0],
        ]);
        rows.push(vec![
            0.0,
            0.0,
            0.0,
            0.0,
            q[0],
            q[1],
            q[2],
            1.0,
            -x[1] * q[0],
            -x[1] * q[1],
            -x[1] * q[2],
            -x[1],
        ]);
    }
    let mut p = null_vector(&rows, 12);
    let m = |p: &[f64]| -> Mat3 { [p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]] };
    if linalg::det3(&m(&p)) < 0.0 {
        p.iter_mut().for_each(|v| *v = -*v);
    }
    let mm = m(&p);
    let (_, s, _) = linalg::svd3(&mm);
    let lam_sigma = (s[0] + s[1] + s[2]) / 3.0;
    if lam_sigma < 1e-12 {
        return None;
    }
    let r = linalg::nearest_rotation(&mm);
    let lam = lam_sigma / sigma;
    let t = linalg::sub([p[3] / lam, p[7] / lam, p[11] / lam], mat_vec(&r, c));
    Some((r, t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::rodrigues;

    fn project(r: &Mat3, t: V3, x: V3) -> Point {
        let c = linalg::add(mat_vec(r, x), t);
        [c[0] / c[2], c[1] / c[2]]
    }

    #[test]
    fn essential_pose_and_triangulation() {
        let mut rng = XorShift::new(3);
        let pts: Vec<V3> = (0..60)
            .map(|_| {
                [
                    rng.below(1000) as f64 / 250.0 - 2.0,
                    rng.below(1000) as f64 / 400.0 - 1.2,
                    5.0 + rng.below(1000) as f64 / 200.0,
                ]
            })
            .collect();
        let r = rodrigues([0.02, -0.15, 0.03]);
        let t = [0.8, 0.05, -0.1];
        let i3 = rodrigues([0.0; 3]);
        let a: Vec<Point> = pts.iter().map(|p| project(&i3, [0.0; 3], *p)).collect();
        let b: Vec<Point> = pts.iter().map(|p| project(&r, t, *p)).collect();
        // Fundamental on pixel coordinates with K, then focal recovery.
        let (fl, cx, cy) = (1100.0, 960.0, 540.0);
        let px = |p: &Point| [p[0] * fl + cx, p[1] * fl + cy];
        let a_px: Vec<Point> = a.iter().map(px).collect();
        let b_px: Vec<Point> = b.iter().map(px).collect();
        let (f, inl) = ransac_fundamental(&a_px, &b_px, 0.5, 1).unwrap();
        assert!(inl.iter().all(|&i| i));
        let est = focal_from_fundamental(&f, 1920.0, 1080.0);
        assert!((est - fl).abs() / fl < 0.02, "focal {est}");
        // Essential decomposition: one of the four matches the truth (t up to scale).
        let k = [fl, 0.0, cx, 0.0, fl, cy, 0.0, 0.0, 1.0];
        let e = mul(&transpose(&k), &mul(&f, &k));
        let tn = linalg::normalized(t);
        let found = decompose_essential(&e).iter().any(|(rr, tt)| {
            (0..9).all(|i| (rr[i] - r[i]).abs() < 1e-3)
                && (0..3).all(|i| (tt[i] - tn[i]).abs() < 1e-3)
        });
        assert!(found);
        let x = triangulate(&[(i3, [0.0; 3], a[0]), (r, t, b[0])]).unwrap();
        assert!(linalg::norm(linalg::sub(x, pts[0])) < 1e-6);
        let (r2, t2) = pnp_dlt(&pts, &b).unwrap();
        assert!(
            (0..9).all(|i| (r2[i] - r[i]).abs() < 1e-6)
                && (0..3).all(|i| (t2[i] - t[i]).abs() < 1e-6)
        );
    }
}

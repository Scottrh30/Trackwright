//! 3x3 transforms, least-squares motion model fitting and RANSAC.

use serde::{Deserialize, Serialize};

pub type Point = [f64; 2];
/// Row-major 3x3 matrix.
pub type Mat3 = [f64; 9];

pub const IDENTITY: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    r
}

pub fn invert(m: &Mat3) -> Option<Mat3> {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let co0 = e * i - f * h;
    let co1 = -(d * i - f * g);
    let co2 = d * h - e * g;
    let det = a * co0 + b * co1 + c * co2;
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        co0 * inv,
        -(b * i - c * h) * inv,
        (b * f - c * e) * inv,
        co1 * inv,
        (a * i - c * g) * inv,
        -(a * f - c * d) * inv,
        co2 * inv,
        -(a * h - b * g) * inv,
        (a * e - b * d) * inv,
    ])
}

#[inline]
pub fn apply(m: &Mat3, p: Point) -> Point {
    let w = m[6] * p[0] + m[7] * p[1] + m[8];
    let w = if w.abs() < 1e-12 { 1e-12 } else { w };
    [
        (m[0] * p[0] + m[1] * p[1] + m[2]) / w,
        (m[3] * p[0] + m[4] * p[1] + m[5]) / w,
    ]
}

/// Scale so the last element is 1 (when possible).
pub fn normalize(m: &Mat3) -> Mat3 {
    if m[8].abs() < 1e-12 {
        return *m;
    }
    let s = 1.0 / m[8];
    m.map(|v| v * s)
}

/// Homography mapping the four `src` corners exactly onto `dst`.
pub fn homography_from_corners(src: &[Point; 4], dst: &[Point; 4]) -> Option<Mat3> {
    fit_model(MotionModel::Perspective, src, dst)
}

/// How much freedom the planar solve is allowed. Mirrors Mocha's
/// translation / scale / rotation / shear / perspective toggles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MotionModel {
    Translation,
    /// Translation + uniform scale + rotation.
    Similarity,
    /// Adds non-uniform scale and shear.
    Affine,
    /// Full planar perspective (homography).
    #[default]
    Perspective,
}

impl MotionModel {
    pub fn min_samples(self) -> usize {
        match self {
            MotionModel::Translation => 1,
            MotionModel::Similarity => 2,
            MotionModel::Affine => 3,
            MotionModel::Perspective => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MotionModel::Translation => "Translation",
            MotionModel::Similarity => "Translation + Scale + Rotation",
            MotionModel::Affine => "+ Shear",
            MotionModel::Perspective => "+ Perspective (full planar)",
        }
    }

    pub const ALL: [MotionModel; 4] = [
        MotionModel::Translation,
        MotionModel::Similarity,
        MotionModel::Affine,
        MotionModel::Perspective,
    ];
}

/// Solve the dense n x n system `a x = b` by Gaussian elimination with partial pivoting.
pub fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[pivot * n + col].abs() < 1e-12 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(pivot * n + k, col * n + k);
            }
            b.swap(pivot, col);
        }
        let diag = a[col * n + col];
        for row in col + 1..n {
            let f = a[row * n + col] / diag;
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in row + 1..n {
            s -= a[row * n + k] * x[k];
        }
        x[row] = s / a[row * n + row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Accumulates normal equations for a linear least-squares problem.
struct Normal {
    n: usize,
    ata: Vec<f64>,
    atb: Vec<f64>,
}

impl Normal {
    fn new(n: usize) -> Self {
        Self {
            n,
            ata: vec![0.0; n * n],
            atb: vec![0.0; n],
        }
    }
    fn row(&mut self, coeffs: &[f64], rhs: f64) {
        for i in 0..self.n {
            if coeffs[i] == 0.0 {
                continue;
            }
            for j in 0..self.n {
                self.ata[i * self.n + j] += coeffs[i] * coeffs[j];
            }
            self.atb[i] += coeffs[i] * rhs;
        }
    }
    fn solve(self) -> Option<Vec<f64>> {
        solve(self.ata, self.atb, self.n)
    }
}

/// Hartley normalisation: centroid to origin, mean distance sqrt(2).
fn normalizer(pts: &[Point]) -> (Mat3, Mat3) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let mean_d = pts
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    let s = if mean_d > 1e-9 {
        std::f64::consts::SQRT_2 / mean_d
    } else {
        1.0
    };
    let t = [s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0];
    let t_inv = [1.0 / s, 0.0, cx, 0.0, 1.0 / s, cy, 0.0, 0.0, 1.0];
    (t, t_inv)
}

/// Least-squares fit of `model` mapping `src[i]` onto `dst[i]`.
pub fn fit_model(model: MotionModel, src: &[Point], dst: &[Point]) -> Option<Mat3> {
    assert_eq!(src.len(), dst.len());
    if src.len() < model.min_samples() {
        return None;
    }
    let (ts, _) = normalizer(src);
    let (td, td_inv) = normalizer(dst);
    let s: Vec<Point> = src.iter().map(|&p| apply(&ts, p)).collect();
    let d: Vec<Point> = dst.iter().map(|&p| apply(&td, p)).collect();

    let hn: Mat3 = match model {
        MotionModel::Translation => {
            let n = s.len() as f64;
            let tx = s.iter().zip(&d).map(|(a, b)| b[0] - a[0]).sum::<f64>() / n;
            let ty = s.iter().zip(&d).map(|(a, b)| b[1] - a[1]).sum::<f64>() / n;
            [1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0]
        }
        MotionModel::Similarity => {
            let mut ne = Normal::new(4);
            for (p, q) in s.iter().zip(&d) {
                ne.row(&[p[0], -p[1], 1.0, 0.0], q[0]);
                ne.row(&[p[1], p[0], 0.0, 1.0], q[1]);
            }
            let x = ne.solve()?;
            [x[0], -x[1], x[2], x[1], x[0], x[3], 0.0, 0.0, 1.0]
        }
        MotionModel::Affine => {
            let mut ne = Normal::new(6);
            for (p, q) in s.iter().zip(&d) {
                ne.row(&[p[0], p[1], 1.0, 0.0, 0.0, 0.0], q[0]);
                ne.row(&[0.0, 0.0, 0.0, p[0], p[1], 1.0], q[1]);
            }
            let x = ne.solve()?;
            [x[0], x[1], x[2], x[3], x[4], x[5], 0.0, 0.0, 1.0]
        }
        MotionModel::Perspective => {
            let mut ne = Normal::new(8);
            for (p, q) in s.iter().zip(&d) {
                let (x, y, u, v) = (p[0], p[1], q[0], q[1]);
                ne.row(&[x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u], u);
                ne.row(&[0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v], v);
            }
            let h = ne.solve()?;
            [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]
        }
    };
    let h = normalize(&mul(&td_inv, &mul(&hn, &ts)));
    h.iter().all(|v| v.is_finite()).then_some(h)
}

/// Small deterministic PRNG so tracking results are reproducible.
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Clone, Debug)]
pub struct RansacResult {
    pub model: Mat3,
    pub inliers: Vec<bool>,
    pub inlier_count: usize,
    pub rms_error: f64,
}

fn triangle_area(a: Point, b: Point, c: Point) -> f64 {
    ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs() * 0.5
}

fn sample_is_degenerate(model: MotionModel, pts: &[Point]) -> bool {
    match model {
        MotionModel::Translation => false,
        MotionModel::Similarity => {
            let d = (pts[0][0] - pts[1][0]).powi(2) + (pts[0][1] - pts[1][1]).powi(2);
            d < 4.0
        }
        _ => {
            // Any three collinear points make the fit unstable.
            let n = pts.len();
            for i in 0..n {
                for j in i + 1..n {
                    for k in j + 1..n {
                        if triangle_area(pts[i], pts[j], pts[k]) < 2.0 {
                            return true;
                        }
                    }
                }
            }
            false
        }
    }
}

fn score(model: &Mat3, src: &[Point], dst: &[Point], thresh2: f64) -> (Vec<bool>, usize, f64) {
    let mut inl = vec![false; src.len()];
    let mut count = 0;
    let mut err = 0.0;
    for i in 0..src.len() {
        let p = apply(model, src[i]);
        let e = (p[0] - dst[i][0]).powi(2) + (p[1] - dst[i][1]).powi(2);
        if e < thresh2 {
            inl[i] = true;
            count += 1;
            err += e;
        }
    }
    let rms = if count > 0 {
        (err / count as f64).sqrt()
    } else {
        f64::INFINITY
    };
    (inl, count, rms)
}

/// Robustly fit `model` with RANSAC, then refine on all inliers.
pub fn ransac(
    model: MotionModel,
    src: &[Point],
    dst: &[Point],
    threshold: f64,
    seed: u64,
) -> Option<RansacResult> {
    let n = src.len();
    let k = model.min_samples();
    if n < k {
        return None;
    }
    let thresh2 = threshold * threshold;
    let mut rng = XorShift::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut best: Option<(Mat3, usize, f64)> = None;
    let mut max_iters = 2000usize;
    let mut iter = 0;
    let mut idx = vec![0usize; k];
    let mut s = vec![[0.0; 2]; k];
    let mut d = vec![[0.0; 2]; k];
    while iter < max_iters {
        iter += 1;
        // Draw k distinct indices.
        let mut filled = 0;
        let mut guard = 0;
        while filled < k && guard < 100 {
            guard += 1;
            let c = rng.below(n);
            if !idx[..filled].contains(&c) {
                idx[filled] = c;
                filled += 1;
            }
        }
        if filled < k {
            continue;
        }
        for j in 0..k {
            s[j] = src[idx[j]];
            d[j] = dst[idx[j]];
        }
        if sample_is_degenerate(model, &s) || sample_is_degenerate(model, &d) {
            continue;
        }
        let Some(h) = fit_model(model, &s, &d) else {
            continue;
        };
        let (_, count, rms) = score(&h, src, dst, thresh2);
        let better = match best {
            None => true,
            Some((_, bc, br)) => count > bc || (count == bc && rms < br),
        };
        if better {
            best = Some((h, count, rms));
            // Adaptive termination at 99.5% confidence.
            let w = count as f64 / n as f64;
            let p_good = w.powi(k as i32);
            if p_good > 1e-9 {
                let needed = ((1.0 - 0.995f64).ln() / (1.0 - p_good).max(1e-12).ln()).ceil();
                if needed.is_finite() {
                    max_iters = max_iters.min(needed.max(16.0) as usize);
                }
            }
        }
    }
    let (mut h, _, _) = best?;
    // Two rounds of least-squares refinement on the consensus set.
    let mut result = None;
    for _ in 0..2 {
        let (inl, count, _) = score(&h, src, dst, thresh2);
        if count < k {
            break;
        }
        let (s2, d2): (Vec<Point>, Vec<Point>) = src
            .iter()
            .zip(dst)
            .zip(&inl)
            .filter(|(_, &i)| i)
            .map(|((a, b), _)| (*a, *b))
            .unzip();
        if let Some(r) = fit_model(model, &s2, &d2) {
            h = r;
        }
        let (inl, count, rms) = score(&h, src, dst, thresh2);
        result = Some(RansacResult {
            model: h,
            inliers: inl,
            inlier_count: count,
            rms_error: rms,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Mat3, b: &Mat3, tol: f64) -> bool {
        let (a, b) = (normalize(a), normalize(b));
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn invert_roundtrip() {
        let h = [1.1, 0.05, 10.0, -0.03, 0.95, 4.0, 1e-4, -2e-4, 1.0];
        let hi = invert(&h).unwrap();
        assert!(close(&mul(&h, &hi), &IDENTITY, 1e-9));
    }

    #[test]
    fn homography_exact_from_corners() {
        let src = [[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0]];
        let dst = [[10.0, 5.0], [120.0, 12.0], [110.0, 95.0], [3.0, 90.0]];
        let h = homography_from_corners(&src, &dst).unwrap();
        for i in 0..4 {
            let p = apply(&h, src[i]);
            assert!((p[0] - dst[i][0]).abs() < 1e-6 && (p[1] - dst[i][1]).abs() < 1e-6);
        }
    }

    #[test]
    fn ransac_rejects_outliers() {
        let truth = [0.98, -0.12, 30.0, 0.1, 1.02, -12.0, 2e-4, -1e-4, 1.0];
        let mut rng = XorShift::new(7);
        let mut src = vec![];
        let mut dst = vec![];
        for i in 0..200 {
            let p = [(rng.below(640)) as f64, (rng.below(480)) as f64];
            let mut q = apply(&truth, p);
            if i % 4 == 0 {
                q[0] += 40.0 + rng.below(50) as f64;
            }
            src.push(p);
            dst.push(q);
        }
        for model in [MotionModel::Perspective] {
            let r = ransac(model, &src, &dst, 1.0, 1).unwrap();
            assert_eq!(r.inlier_count, 150);
            assert!(close(&r.model, &truth, 1e-6));
        }
    }

    #[test]
    fn similarity_and_affine_fit() {
        let a = (0.3f64).cos() * 1.2;
        let b = (0.3f64).sin() * 1.2;
        let sim = [a, -b, 5.0, b, a, -7.0, 0.0, 0.0, 1.0];
        let src: Vec<Point> = (0..10)
            .map(|i| [i as f64 * 13.0, (i * i) as f64 * 3.0])
            .collect();
        let dst: Vec<Point> = src.iter().map(|&p| apply(&sim, p)).collect();
        assert!(close(
            &fit_model(MotionModel::Similarity, &src, &dst).unwrap(),
            &sim,
            1e-8
        ));
        assert!(close(
            &fit_model(MotionModel::Affine, &src, &dst).unwrap(),
            &sim,
            1e-8
        ));
    }
}

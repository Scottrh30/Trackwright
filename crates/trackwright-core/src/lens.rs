//! Lens distortion: a radial (Brown) model in image space, calibration from lines the
//! user marks along edges that should be straight, and undistort/redistort helpers.
//!
//! Coordinates are normalised by the image half-diagonal around the centre, so the
//! coefficients do not depend on resolution or focal length:
//! `p_distorted = c + R * u * (1 + k1 |u|^2 + k2 |u|^4)` with `u = (p_undistorted - c) / R`.

use crate::geometry::Point;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Lens {
    pub k1: f64,
    pub k2: f64,
}

impl Lens {
    pub fn is_identity(&self) -> bool {
        self.k1 == 0.0 && self.k2 == 0.0
    }

    fn frame(width: f64, height: f64) -> (Point, f64) {
        (
            [width / 2.0, height / 2.0],
            0.5 * (width * width + height * height).sqrt(),
        )
    }

    /// Undistorted (ideal pinhole) pixel -> where it appears in the real footage.
    pub fn distort(&self, p: Point, width: f64, height: f64) -> Point {
        if self.is_identity() {
            return p;
        }
        let (c, r) = Self::frame(width, height);
        let (ux, uy) = ((p[0] - c[0]) / r, (p[1] - c[1]) / r);
        let r2 = ux * ux + uy * uy;
        let f = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        [c[0] + r * ux * f, c[1] + r * uy * f]
    }

    /// Footage pixel -> ideal pinhole pixel (Newton iterations on the radius).
    pub fn undistort(&self, p: Point, width: f64, height: f64) -> Point {
        if self.is_identity() {
            return p;
        }
        let (c, r) = Self::frame(width, height);
        let (dx, dy) = ((p[0] - c[0]) / r, (p[1] - c[1]) / r);
        let rd = (dx * dx + dy * dy).sqrt();
        if rd < 1e-12 {
            return p;
        }
        // Solve ru * (1 + k1 ru^2 + k2 ru^4) = rd for ru.
        let mut ru = rd;
        for _ in 0..30 {
            let r2 = ru * ru;
            let f = ru * (1.0 + self.k1 * r2 + self.k2 * r2 * r2) - rd;
            let df = 1.0 + 3.0 * self.k1 * r2 + 5.0 * self.k2 * r2 * r2;
            if df.abs() < 1e-12 {
                break;
            }
            let step = f / df;
            ru -= step;
            if step.abs() < 1e-14 {
                break;
            }
        }
        let s = ru / rd;
        [c[0] + r * dx * s, c[1] + r * dy * s]
    }
}

/// Sum of squared distances of points from their best-fit line, and the count.
fn line_residuals(pts: &[Point], out: &mut Vec<f64>) {
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let my = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in pts {
        let (x, y) = (p[0] - mx, p[1] - my);
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    // Normal of the principal direction = eigenvector of the smaller eigenvalue.
    let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (nx, ny) = (-th.sin(), th.cos());
    for p in pts {
        out.push((p[0] - mx) * nx + (p[1] - my) * ny);
    }
}

fn residuals(lens: &Lens, lines: &[Vec<Point>], w: f64, h: f64) -> Vec<f64> {
    let mut r = Vec::new();
    for l in lines.iter().filter(|l| l.len() >= 3) {
        let und: Vec<Point> = l.iter().map(|&p| lens.undistort(p, w, h)).collect();
        line_residuals(&und, &mut r);
    }
    r
}

/// Result of calibrating from straight lines.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LensFit {
    pub lens: Lens,
    /// RMS distance (pixels) of the marked points from straight lines, before and after.
    pub rms_before: f64,
    pub rms_after: f64,
}

/// Find k1 (and k2 when `solve_k2`) that make every marked line straight.
/// Each line needs at least 3 points; long lines near the frame edges work best.
pub fn calibrate_from_lines(
    lines: &[Vec<Point>],
    width: f64,
    height: f64,
    solve_k2: bool,
) -> Option<LensFit> {
    let usable = lines.iter().filter(|l| l.len() >= 3).count();
    if usable == 0 {
        return None;
    }
    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len().max(1) as f64).sqrt();
    let mut lens = Lens::default();
    let before = rms(&residuals(&lens, lines, width, height));
    let np = if solve_k2 { 2 } else { 1 };
    let mut lambda = 1e-3;
    let mut cur = residuals(&lens, lines, width, height);
    let mut cost: f64 = cur.iter().map(|x| x * x).sum();
    for _ in 0..100 {
        // Numeric Jacobian.
        let eps = 1e-6;
        let mut jac: Vec<Vec<f64>> = Vec::new();
        for k in 0..np {
            let mut l2 = lens;
            if k == 0 {
                l2.k1 += eps;
            } else {
                l2.k2 += eps;
            }
            let r2 = residuals(&l2, lines, width, height);
            jac.push(r2.iter().zip(&cur).map(|(a, b)| (a - b) / eps).collect());
        }
        let mut jtj = vec![0.0; np * np];
        let mut jtr = vec![0.0; np];
        for a in 0..np {
            for b in 0..np {
                jtj[a * np + b] = jac[a].iter().zip(&jac[b]).map(|(x, y)| x * y).sum();
            }
            jtr[a] = -jac[a].iter().zip(&cur).map(|(x, y)| x * y).sum::<f64>();
        }
        let mut improved = false;
        for _ in 0..10 {
            let mut m = jtj.clone();
            for a in 0..np {
                m[a * np + a] *= 1.0 + lambda;
                m[a * np + a] += 1e-12;
            }
            let Some(d) = crate::linalg::solve_spd(&m, &jtr, np) else {
                break;
            };
            let mut trial = lens;
            trial.k1 += d[0];
            if np > 1 {
                trial.k2 += d[1];
            }
            let tr = residuals(&trial, lines, width, height);
            let tc: f64 = tr.iter().map(|x| x * x).sum();
            if tc < cost {
                lens = trial;
                cur = tr;
                let rel = (cost - tc) / cost.max(1e-30);
                cost = tc;
                lambda = (lambda * 0.3).max(1e-9);
                improved = true;
                if rel < 1e-10 {
                    return Some(LensFit {
                        lens,
                        rms_before: before,
                        rms_after: rms(&cur),
                    });
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    Some(LensFit {
        lens,
        rms_before: before,
        rms_after: rms(&cur),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distort_undistort_roundtrip() {
        let l = Lens {
            k1: -0.18,
            k2: 0.03,
        };
        for p in [
            [10.0, 10.0],
            [960.0, 540.0],
            [1900.0, 50.0],
            [300.0, 1000.0],
        ] {
            let d = l.distort(p, 1920.0, 1080.0);
            let u = l.undistort(d, 1920.0, 1080.0);
            assert!((u[0] - p[0]).abs() < 1e-8 && (u[1] - p[1]).abs() < 1e-8);
        }
    }

    #[test]
    fn recovers_barrel_distortion_from_lines() {
        let (w, h) = (1920.0, 1080.0);
        let truth = Lens { k1: -0.12, k2: 0.0 };
        // Straight lines in the ideal image, then distorted like the real footage.
        let mut lines = vec![];
        for &y in &[60.0, 300.0, 800.0, 1030.0] {
            lines.push(
                (0..12)
                    .map(|i| truth.distort([80.0 + i as f64 * 160.0, y], w, h))
                    .collect::<Vec<_>>(),
            );
        }
        for &x in &[70.0, 1850.0] {
            lines.push(
                (0..8)
                    .map(|i| truth.distort([x, 60.0 + i as f64 * 130.0], w, h))
                    .collect::<Vec<_>>(),
            );
        }
        let fit = calibrate_from_lines(&lines, w, h, false).unwrap();
        assert!((fit.lens.k1 - truth.k1).abs() < 1e-4, "{fit:?}");
        assert!(fit.rms_before > 5.0 && fit.rms_after < 0.01, "{fit:?}");
    }
}

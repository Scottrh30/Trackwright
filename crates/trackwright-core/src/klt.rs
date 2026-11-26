//! Pyramidal Lucas-Kanade point tracking (inverse-compositional, translational).

use crate::geometry::Point;
use crate::image::Pyramid;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KltParams {
    pub win_radius: i32,
    pub max_iterations: usize,
    /// Stop when the update is smaller than this (pixels at the current level).
    pub epsilon: f32,
    /// Reject patches whose mean absolute residual exceeds this (0..255 intensity).
    pub max_residual: f32,
}

impl Default for KltParams {
    fn default() -> Self {
        Self {
            win_radius: 7,
            max_iterations: 30,
            epsilon: 0.01,
            max_residual: 18.0,
        }
    }
}

/// Track point `p` from `prev` into `next`, starting the search at `guess`.
/// Returns the new position and its mean absolute residual.
pub fn track_point(
    prev: &Pyramid,
    next: &Pyramid,
    p: Point,
    guess: Point,
    params: &KltParams,
) -> Option<(Point, f32)> {
    let levels = prev.levels.len().min(next.levels.len());
    let r = params.win_radius;
    let side = (2 * r + 1) as usize;
    let mut tmpl = vec![0f32; side * side];
    let mut tgx = vec![0f32; side * side];
    let mut tgy = vec![0f32; side * side];

    let top = levels - 1;
    let scale_top = (1u32 << top) as f32;
    let mut d = [
        (guess[0] - p[0]) as f32 / scale_top,
        (guess[1] - p[1]) as f32 / scale_top,
    ];
    let mut residual = 0f32;

    for level in (0..levels).rev() {
        let s = (1u32 << level) as f32;
        let (img_p, img_n) = (&prev.levels[level], &next.levels[level]);
        let (gx, gy) = (&prev.grad_x[level], &prev.grad_y[level]);
        let px = p[0] as f32 / s;
        let py = p[1] as f32 / s;

        // Template patch and its gradient matrix.
        let (mut g11, mut g12, mut g22) = (0f32, 0f32, 0f32);
        let mut k = 0;
        for wy in -r..=r {
            for wx in -r..=r {
                let (sx, sy) = (px + wx as f32, py + wy as f32);
                tmpl[k] = img_p.sample(sx, sy);
                let (ix, iy) = (gx.sample(sx, sy), gy.sample(sx, sy));
                tgx[k] = ix;
                tgy[k] = iy;
                g11 += ix * ix;
                g12 += ix * iy;
                g22 += iy * iy;
                k += 1;
            }
        }
        let det = g11 * g22 - g12 * g12;
        let n_px = (side * side) as f32;
        // Minimum eigenvalue check: flat or edge-only patches are untrackable.
        let min_eig = 0.5 * (g11 + g22 - ((g11 - g22).powi(2) + 4.0 * g12 * g12).sqrt()) / n_px;
        if det.abs() < 1e-6 || min_eig < 1e-3 {
            return None;
        }
        let inv_det = 1.0 / det;

        for _ in 0..params.max_iterations {
            let (cx, cy) = (px + d[0], py + d[1]);
            if cx < -(r as f32)
                || cy < -(r as f32)
                || cx > img_n.width as f32 + r as f32
                || cy > img_n.height as f32 + r as f32
            {
                return None;
            }
            let (mut b1, mut b2) = (0f32, 0f32);
            let mut abs_sum = 0f32;
            let mut k = 0;
            for wy in -r..=r {
                for wx in -r..=r {
                    let e = tmpl[k] - img_n.sample(cx + wx as f32, cy + wy as f32);
                    b1 += e * tgx[k];
                    b2 += e * tgy[k];
                    abs_sum += e.abs();
                    k += 1;
                }
            }
            residual = abs_sum / n_px;
            let dx = inv_det * (g22 * b1 - g12 * b2);
            let dy = inv_det * (g11 * b2 - g12 * b1);
            d[0] += dx;
            d[1] += dy;
            if dx * dx + dy * dy < params.epsilon * params.epsilon {
                break;
            }
        }
        if level > 0 {
            d[0] *= 2.0;
            d[1] *= 2.0;
        }
    }

    let out = [p[0] + d[0] as f64, p[1] + d[1] as f64];
    let (w, h) = (next.width() as f64, next.height() as f64);
    if !(out[0] >= 0.0 && out[1] >= 0.0 && out[0] < w && out[1] < h)
        || residual > params.max_residual
    {
        return None;
    }
    Some((out, residual))
}

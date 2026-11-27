//! Pyramidal Lucas-Kanade point tracking (inverse-compositional, translational),
//! optionally compensating for brightness and contrast changes (gain and bias).

use crate::geometry::Point;
use crate::image::Pyramid;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KltParams {
    pub win_radius: i32,
    pub max_iterations: usize,
    /// Stop when the update is smaller than this (pixels at the current level).
    pub epsilon: f32,
    /// Reject patches whose mean absolute residual exceeds this (0..255 intensity,
    /// measured after brightness/contrast compensation when that is on).
    pub max_residual: f32,
    /// Model each patch as `gain * template + bias` so lighting changes, fades and
    /// flicker don't pull points off.
    pub illumination: bool,
    /// Finest pyramid level to track at; points are given and returned in that
    /// level's coordinates (0 = full resolution).
    pub base_level: usize,
}

impl Default for KltParams {
    fn default() -> Self {
        Self {
            win_radius: 7,
            max_iterations: 30,
            epsilon: 0.01,
            max_residual: 18.0,
            illumination: true,
            base_level: 0,
        }
    }
}

/// Track point `p` from `prev` into `next`, starting the search at `guess`.
/// `p` is in `prev`'s coordinates and `guess` in `next`'s, so the two images may be
/// different crops of the same frame. Returns the new position and its mean residual.
pub fn track_point(
    prev: &Pyramid,
    next: &Pyramid,
    p: Point,
    guess: Point,
    params: &KltParams,
) -> Option<(Point, f32)> {
    let levels = prev.levels.len().min(next.levels.len());
    let base = params.base_level.min(levels - 1);
    let r = params.win_radius;
    let side = (2 * r + 1) as usize;
    let n = side * side;
    let n_px = n as f32;
    let mut tmpl = vec![0f32; n];
    let mut tgx = vec![0f32; n];
    let mut tgy = vec![0f32; n];
    let mut cur = vec![0f32; n];

    let top = levels - 1;
    let scale_top = (1u32 << (top - base)) as f32;
    let mut d = [
        (guess[0] - p[0]) as f32 / scale_top,
        (guess[1] - p[1]) as f32 / scale_top,
    ];
    let mut residual = 0f32;

    for level in (base..levels).rev() {
        let s = (1u32 << (level - base)) as f32;
        let (img_p, img_n) = (&prev.levels[level], &next.levels[level]);
        let (gx, gy) = (&prev.grad_x[level], &prev.grad_y[level]);
        let px = p[0] as f32 / s;
        let py = p[1] as f32 / s;

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
        let min_eig = 0.5 * (g11 + g22 - ((g11 - g22).powi(2) + 4.0 * g12 * g12).sqrt()) / n_px;
        if det.abs() < 1e-6 || min_eig < 1e-3 {
            return None;
        }
        let inv_det = 1.0 / det;
        let t_mean = tmpl.iter().sum::<f32>() / n_px;
        let t_std = (tmpl.iter().map(|v| (v - t_mean).powi(2)).sum::<f32>() / n_px)
            .sqrt()
            .max(1e-3);

        for _ in 0..params.max_iterations {
            let (cx, cy) = (px + d[0], py + d[1]);
            if cx < -(r as f32)
                || cy < -(r as f32)
                || cx > img_n.width as f32 + r as f32
                || cy > img_n.height as f32 + r as f32
            {
                return None;
            }
            let mut k = 0;
            for wy in -r..=r {
                for wx in -r..=r {
                    cur[k] = img_n.sample(cx + wx as f32, cy + wy as f32);
                    k += 1;
                }
            }
            // Gain/bias so that gain * (J - mean_J) matches (T - mean_T).
            let (gain, j_mean) = if params.illumination {
                let m = cur.iter().sum::<f32>() / n_px;
                let sd = (cur.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n_px)
                    .sqrt()
                    .max(1e-3);
                // Contrast is only corrected when it has clearly changed: estimating
                // it on every patch adds noise when the lighting is steady.
                let g = (t_std / sd).clamp(0.2, 5.0);
                (if (g - 1.0).abs() < 0.08 { 1.0 } else { g }, m)
            } else {
                (1.0, t_mean)
            };
            let (mut b1, mut b2, mut abs_sum) = (0f32, 0f32, 0f32);
            for k in 0..n {
                let e = (tmpl[k] - t_mean) - gain * (cur[k] - j_mean);
                b1 += e * tgx[k];
                b2 += e * tgy[k];
                abs_sum += e.abs();
            }
            residual = abs_sum / n_px;
            // Template gradients describe T; J is scaled by the gain, so scale the step.
            let dx = inv_det * (g22 * b1 - g12 * b2) / gain;
            let dy = inv_det * (g11 * b2 - g12 * b1) / gain;
            d[0] += dx;
            d[1] += dy;
            if dx * dx + dy * dy < params.epsilon * params.epsilon {
                break;
            }
        }
        if level > base {
            d[0] *= 2.0;
            d[1] *= 2.0;
        }
    }

    let out = [p[0] + d[0] as f64, p[1] + d[1] as f64];
    let lvl = &next.levels[base];
    let (w, h) = (lvl.width as f64, lvl.height as f64);
    if !(out[0] >= 0.0 && out[1] >= 0.0 && out[0] < w && out[1] < h)
        || residual > params.max_residual
    {
        return None;
    }
    Some((out, residual))
}

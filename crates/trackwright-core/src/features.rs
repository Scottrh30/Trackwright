//! Shi-Tomasi "good features to track" restricted to a polygonal mask.

use crate::geometry::Point;
use crate::image::GrayImage;
use crate::shape::{bbox, point_in_polygon};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureParams {
    pub max_features: usize,
    /// Minimum response relative to the strongest corner (0..1).
    pub quality: f32,
    pub min_distance: f32,
    /// Structure tensor window radius.
    pub block_radius: usize,
    /// Keep features at least this far from the image edge.
    pub border: usize,
}

impl Default for FeatureParams {
    fn default() -> Self {
        Self {
            max_features: 400,
            quality: 0.01,
            min_distance: 7.0,
            block_radius: 2,
            border: 10,
        }
    }
}

/// Detect corners inside `mask` (image coordinates). `None` = whole image.
pub fn good_features(
    img: &GrayImage,
    gx: &GrayImage,
    gy: &GrayImage,
    mask: Option<&[Point]>,
    params: &FeatureParams,
) -> Vec<Point> {
    let (w, h) = (img.width, img.height);
    let r = params.block_radius;
    let border = params.border.max(r + 1);
    if w <= 2 * border || h <= 2 * border {
        return vec![];
    }
    let (mut x0, mut y0, mut x1, mut y1) = (border, border, w - border, h - border);
    if let Some(m) = mask {
        let Some(b) = bbox(m) else { return vec![] };
        x0 = x0.max(b[0].floor().max(0.0) as usize);
        y0 = y0.max(b[1].floor().max(0.0) as usize);
        x1 = x1.min(b[2].ceil().max(0.0) as usize + 1);
        y1 = y1.min(b[3].ceil().max(0.0) as usize + 1);
    }
    if x1 <= x0 + 2 || y1 <= y0 + 2 {
        return vec![];
    }

    // Integral images of the structure tensor over the region (+ window margin).
    let (rx0, ry0) = (x0 - r, y0 - r);
    let (rw, rh) = (x1 - x0 + 2 * r, y1 - y0 + 2 * r);
    let iw = rw + 1;
    let mut ixx = vec![0f64; iw * (rh + 1)];
    let mut ixy = vec![0f64; iw * (rh + 1)];
    let mut iyy = vec![0f64; iw * (rh + 1)];
    for y in 0..rh {
        let (mut sxx, mut sxy, mut syy) = (0f64, 0f64, 0f64);
        for x in 0..rw {
            let (ax, ay) = ((rx0 + x).min(w - 1), (ry0 + y).min(h - 1));
            let dx = gx.get(ax, ay) as f64;
            let dy = gy.get(ax, ay) as f64;
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
            let i = (y + 1) * iw + x + 1;
            ixx[i] = ixx[i - iw] + sxx;
            ixy[i] = ixy[i - iw] + sxy;
            iyy[i] = iyy[i - iw] + syy;
        }
    }
    let box_sum = |ii: &[f64], cx: usize, cy: usize| {
        // cx, cy are region-local centre coords (already offset by r).
        let (a, b) = (cx - r, cy - r);
        let (c, d) = (cx + r + 1, cy + r + 1);
        ii[d * iw + c] - ii[b * iw + c] - ii[d * iw + a] + ii[b * iw + a]
    };

    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut resp = vec![0f32; cw * ch];
    let mut max_resp = 0f32;
    for y in 0..ch {
        for x in 0..cw {
            let (cx, cy) = (x + r, y + r);
            let a = box_sum(&ixx, cx, cy);
            let b = box_sum(&ixy, cx, cy);
            let c = box_sum(&iyy, cx, cy);
            let half_tr = 0.5 * (a + c);
            let disc = (0.25 * (a - c) * (a - c) + b * b).sqrt();
            let e = (half_tr - disc) as f32;
            resp[y * cw + x] = e;
            max_resp = max_resp.max(e);
        }
    }
    if max_resp <= 1e-6 {
        return vec![];
    }
    let thresh = max_resp * params.quality;

    // 3x3 non-maximum suppression, then mask test.
    let mut cands: Vec<(f32, Point)> = Vec::new();
    for y in 1..ch.saturating_sub(1) {
        for x in 1..cw.saturating_sub(1) {
            let v = resp[y * cw + x];
            if v < thresh {
                continue;
            }
            let mut is_max = true;
            'n: for dy in [-1isize, 0, 1] {
                for dx in [-1isize, 0, 1] {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let o = resp[(y as isize + dy) as usize * cw + (x as isize + dx) as usize];
                    if o > v {
                        is_max = false;
                        break 'n;
                    }
                }
            }
            if !is_max {
                continue;
            }
            let p = [(x0 + x) as f64, (y0 + y) as f64];
            if let Some(m) = mask {
                if !point_in_polygon(m, p) {
                    continue;
                }
            }
            cands.push((v, p));
        }
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Greedy minimum-distance selection on a grid.
    let md = params.min_distance.max(1.0) as f64;
    let gw = (w as f64 / md).ceil() as usize + 1;
    let gh = (h as f64 / md).ceil() as usize + 1;
    let mut grid: Vec<Vec<Point>> = vec![Vec::new(); gw * gh];
    let mut out = Vec::new();
    for (_, p) in cands {
        let (gxi, gyi) = ((p[0] / md) as usize, (p[1] / md) as usize);
        let mut ok = true;
        'g: for yy in gyi.saturating_sub(1)..=(gyi + 1).min(gh - 1) {
            for xx in gxi.saturating_sub(1)..=(gxi + 1).min(gw - 1) {
                for q in &grid[yy * gw + xx] {
                    if (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) < md * md {
                        ok = false;
                        break 'g;
                    }
                }
            }
        }
        if ok {
            grid[gyi * gw + gxi].push(p);
            out.push(p);
            if out.len() >= params.max_features {
                break;
            }
        }
    }
    out
}

//! Mesh (warp) tracking: a grid of points over the surface that bends with it.
//!
//! The planar tracker moves the whole surface as one flat plane. Faces, mouths,
//! cheeks and cloth don't stay flat. The mesh tracker starts from the planar motion
//! and lets each grid point move on its own: every point is tracked locally (from the
//! previous frame, and checked against the reference frame to stop drift), then the
//! grid is smoothed so neighbours move together. Points that can't be tracked
//! (featureless skin, something in front) simply follow their neighbours.

use crate::geometry::{apply, Point};
use crate::image::Pyramid;
use crate::klt::{track_point, KltParams};

/// Largest grid, cells across or down.
pub const MAX_CELLS: usize = 32;

/// Point at grid coordinates `uv` (0..1 across, 0..1 down) on the quad `q`
/// (TL, TR, BR, BL), by blending its sides.
pub fn at_uv(q: &[Point; 4], uv: [f64; 2]) -> Point {
    let [s, t] = uv;
    let top = [
        q[0][0] + (q[1][0] - q[0][0]) * s,
        q[0][1] + (q[1][1] - q[0][1]) * s,
    ];
    let bot = [
        q[3][0] + (q[2][0] - q[3][0]) * s,
        q[3][1] + (q[2][1] - q[3][1]) * s,
    ];
    [top[0] + (bot[0] - top[0]) * t, top[1] + (bot[1] - top[1]) * t]
}

/// Grid coordinates of `p` on the quad `q` (the inverse of [`at_uv`]); may fall
/// outside 0..1 for points outside the quad.
pub fn to_uv(q: &[Point; 4], p: Point) -> [f64; 2] {
    let mut uv = [0.5, 0.5];
    for _ in 0..30 {
        let f = at_uv(q, uv);
        let (rx, ry) = (f[0] - p[0], f[1] - p[1]);
        if rx.abs() + ry.abs() < 1e-9 {
            break;
        }
        // Jacobian of at_uv.
        let [s, t] = uv;
        let dsx = (1.0 - t) * (q[1][0] - q[0][0]) + t * (q[2][0] - q[3][0]);
        let dsy = (1.0 - t) * (q[1][1] - q[0][1]) + t * (q[2][1] - q[3][1]);
        let dtx = (1.0 - s) * (q[3][0] - q[0][0]) + s * (q[2][0] - q[1][0]);
        let dty = (1.0 - s) * (q[3][1] - q[0][1]) + s * (q[2][1] - q[1][1]);
        let det = dsx * dty - dtx * dsy;
        if det.abs() < 1e-12 {
            break;
        }
        uv[0] -= (rx * dty - ry * dtx) / det;
        uv[1] -= (dsx * ry - dsy * rx) / det;
    }
    uv
}

/// Grid coordinates of a regular `cols` x `rows` cell grid, row by row.
pub fn grid_uv(cols: usize, rows: usize) -> Vec<[f64; 2]> {
    let (c, r) = (cols.max(1), rows.max(1));
    let mut v = Vec::with_capacity((c + 1) * (r + 1));
    for j in 0..=r {
        for i in 0..=c {
            v.push([i as f64 / c as f64, j as f64 / r as f64]);
        }
    }
    v
}

/// Vertices of a regular `cols` x `rows` cell grid over the quad `q` (TL, TR, BR,
/// BL), row by row.
pub fn grid(q: &[Point; 4], cols: usize, rows: usize) -> Vec<Point> {
    grid_uv(cols, rows).into_iter().map(|uv| at_uv(q, uv)).collect()
}

/// Where an even `cols` x `rows` grid over the surface `q` falls in an insert of
/// `size` (width, height) pinned to the surface in perspective: the insert's
/// coordinates for each mesh point, row by row.
pub fn insert_uv(q: &[Point; 4], cols: usize, rows: usize, size: [f64; 2]) -> Vec<Point> {
    let [w, h] = size;
    let rect = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    match crate::geometry::homography_from_corners(q, &rect) {
        Some(hs) => grid(q, cols, rows).into_iter().map(|p| apply(&hs, p)).collect(),
        None => grid_uv(cols, rows).into_iter().map(|p| [p[0] * w, p[1] * h]).collect(),
    }
}

/// Cells across and down of a grid with `count` vertices, `cols` cells across (0 if
/// unknown: then the grid is taken to be square).
pub fn dims(count: usize, cols: usize) -> Option<(usize, usize)> {
    if cols > 0 && count % (cols + 1) == 0 && count / (cols + 1) >= 2 {
        return Some((cols, count / (cols + 1) - 1));
    }
    let s = (count as f64).sqrt().round() as usize;
    (s >= 2 && s * s == count).then(|| (s - 1, s - 1))
}

/// Is vertex `k` one of the grid's four corners?
pub fn is_corner(k: usize, cols: usize, rows: usize) -> bool {
    let s = cols + 1;
    let last = (rows + 1) * s - 1;
    k == 0 || k == cols || k == last || k == last - cols
}

/// Weight with which vertex `j` follows a drag of vertex `k`: 1 at `k`, easing to 0
/// at `falloff` + 1 cells away (0 falloff: only `k`). Corners never follow.
pub fn falloff_weight(k: usize, j: usize, cols: usize, rows: usize, falloff: f64) -> f64 {
    if j == k {
        return 1.0;
    }
    if is_corner(j, cols, rows) {
        return 0.0;
    }
    let s = cols + 1;
    let d = ((k % s) as f64 - (j % s) as f64).hypot((k / s) as f64 - (j / s) as f64);
    let reach = falloff.max(0.0) + 1.0;
    if d >= reach {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * d / reach).cos())
    }
}

/// Grid neighbours (up, down, left, right) of vertex `k` in a `cols` x `rows` grid.
fn neighbours(k: usize, cols: usize, rows: usize) -> impl Iterator<Item = usize> {
    let s = cols + 1;
    let (i, j) = (k % s, k / s);
    [
        (i > 0).then(|| k - 1),
        (i < cols).then(|| k + 1),
        (j > 0).then(|| k - s),
        (j < rows).then(|| k + s),
    ]
    .into_iter()
    .flatten()
}

/// One frame of mesh tracking.
///
/// `prev_v`: vertices on the previous frame; `pred_v`: where the planar motion puts
/// them on this frame; `ref_v`: where they were on the reference frame.
/// `stiffness` (0..) is how strongly neighbours are held to move together.
#[allow(clippy::too_many_arguments)]
pub fn step(
    prev: &Pyramid,
    next: &Pyramid,
    reference: &Pyramid,
    prev_v: &[Point],
    pred_v: &[Point],
    ref_v: &[Point],
    cols: usize,
    rows: usize,
    cell_px: f64,
    stiffness: f64,
) -> Vec<Point> {
    let r = (cell_px / 3.0).clamp(6.0, 20.0) as i32;
    let params = KltParams {
        win_radius: r,
        illumination: true,
        max_residual: 20.0,
        ..Default::default()
    };
    // Where each vertex really went, if it could be told (and how sure).
    let measured: Vec<Option<Point>> =
        crate::par::map(&(0..prev_v.len()).collect::<Vec<_>>(), |&k| {
            let (q, _) = track_point(prev, next, prev_v[k], pred_v[k], &params)?;
            let (back, _) = track_point(next, prev, q, prev_v[k], &params)?;
            if (back[0] - prev_v[k][0]).hypot(back[1] - prev_v[k][1]) > 0.7 {
                return None;
            }
            // Against the reference too, which doesn't drift: use it when it agrees.
            if let Some((qr, _)) = track_point(reference, next, ref_v[k], q, &params) {
                if (qr[0] - q[0]).hypot(qr[1] - q[1]) < 1.5 {
                    return Some(qr);
                }
            }
            Some(q)
        });
    // Smooth: displacement from the planar prediction, held close to the measurement
    // and to the neighbours' displacements.
    let m = pred_v.len();
    let target: Vec<Option<Point>> = (0..m)
        .map(|k| measured[k].map(|q| [q[0] - pred_v[k][0], q[1] - pred_v[k][1]]))
        .collect();
    // Robust: a measurement far from what its measured neighbours say is an outlier.
    let target: Vec<Option<Point>> = (0..m)
        .map(|k| {
            let t = target[k]?;
            let near: Vec<Point> = neighbours(k, cols, rows).filter_map(|j| target[j]).collect();
            if near.len() >= 2 {
                let mx = near.iter().map(|p| p[0]).sum::<f64>() / near.len() as f64;
                let my = near.iter().map(|p| p[1]).sum::<f64>() / near.len() as f64;
                if (t[0] - mx).hypot(t[1] - my) > (0.35 * cell_px).max(3.0) {
                    return None;
                }
            }
            Some(t)
        })
        .collect();
    let lambda = stiffness.max(0.0);
    let mut u = vec![[0.0f64; 2]; m];
    for _ in 0..80 {
        for k in 0..m {
            let (w, t) = match target[k] {
                Some(t) => (1.0, t),
                None => (0.0, [0.0, 0.0]),
            };
            let mut sx = w * t[0];
            let mut sy = w * t[1];
            let mut deg = 0.0;
            for j in neighbours(k, cols, rows) {
                sx += lambda * u[j][0];
                sy += lambda * u[j][1];
                deg += lambda;
            }
            if w + deg > 0.0 {
                u[k] = [sx / (w + deg), sy / (w + deg)];
            }
        }
    }
    (0..m)
        .map(|k| [pred_v[k][0] + u[k][0], pred_v[k][1] + u[k][1]])
        .collect()
}

/// Map `pts` through `h` (a whole-mesh rigid move, for frames with no mesh solve).
pub fn carry(pts: &[Point], h: &crate::geometry::Mat3) -> Vec<Point> {
    pts.iter().map(|p| apply(h, *p)).collect()
}

/// Triangles of a `cols` x `rows` grid as vertex indices (two per cell).
pub fn triangles(cols: usize, rows: usize) -> Vec<[usize; 3]> {
    let s = cols + 1;
    let mut t = Vec::with_capacity(cols * rows * 2);
    for j in 0..rows {
        for i in 0..cols {
            let a = j * s + i;
            t.push([a, a + 1, a + s + 1]);
            t.push([a, a + s + 1, a + s]);
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_corners_and_size() {
        let q = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]];
        let g = grid(&q, 4, 4);
        assert_eq!(g.len(), 25);
        assert_eq!(g[0], [0.0, 0.0]);
        assert_eq!(g[4], [100.0, 0.0]);
        assert_eq!(g[24], [100.0, 50.0]);
        assert_eq!(dims(25, 0), Some((4, 4)));
        assert_eq!(triangles(4, 4).len(), 32);
        // Not square: 6 across, 2 down.
        let g = grid(&q, 6, 2);
        assert_eq!(g.len(), 21);
        assert_eq!(g[6], [100.0, 0.0]);
        assert_eq!(g[20], [100.0, 50.0]);
        assert_eq!(dims(21, 6), Some((6, 2)));
        assert_eq!(triangles(6, 2).len(), 24);
        assert!(is_corner(14, 6, 2) && is_corner(20, 6, 2) && !is_corner(7, 6, 2));
        // Up to 32 x 32.
        assert_eq!(grid(&q, 32, 32).len(), 33 * 33);
    }

    #[test]
    fn uv_round_trips_on_a_skewed_quad() {
        let q = [[10.0, 20.0], [130.0, 5.0], [150.0, 110.0], [0.0, 90.0]];
        for uv in [[0.0, 0.0], [0.3, 0.7], [1.0, 0.5], [0.9, 0.1]] {
            let back = to_uv(&q, at_uv(&q, uv));
            assert!((back[0] - uv[0]).abs() < 1e-9 && (back[1] - uv[1]).abs() < 1e-9);
        }
        assert_eq!(falloff_weight(8, 8, 4, 4, 0.0), 1.0);
        assert_eq!(falloff_weight(8, 9, 4, 4, 0.0), 0.0);
        assert!(falloff_weight(8, 9, 4, 4, 2.0) > 0.5);
        assert_eq!(falloff_weight(6, 0, 4, 4, 5.0), 0.0); // a corner never follows
    }
}

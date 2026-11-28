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

/// Vertices of a `n` x `n` cell grid over the quad `q` (TL, TR, BR, BL), row by row.
pub fn grid(q: &[Point; 4], n: usize) -> Vec<Point> {
    let n = n.max(1);
    let mut v = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        let t = j as f64 / n as f64;
        for i in 0..=n {
            let s = i as f64 / n as f64;
            let top = [
                q[0][0] + (q[1][0] - q[0][0]) * s,
                q[0][1] + (q[1][1] - q[0][1]) * s,
            ];
            let bot = [
                q[3][0] + (q[2][0] - q[3][0]) * s,
                q[3][1] + (q[2][1] - q[3][1]) * s,
            ];
            v.push([
                top[0] + (bot[0] - top[0]) * t,
                top[1] + (bot[1] - top[1]) * t,
            ]);
        }
    }
    v
}

/// Side of a grid with `count` vertices (count = (n + 1)^2), or `None`.
pub fn grid_side(count: usize) -> Option<usize> {
    let s = (count as f64).sqrt().round() as usize;
    (s >= 2 && s * s == count).then_some(s - 1)
}

/// Grid neighbours (up, down, left, right) of vertex `k` in an `n` x `n` cell grid.
fn neighbours(k: usize, n: usize) -> impl Iterator<Item = usize> {
    let s = n + 1;
    let (i, j) = (k % s, k / s);
    [
        (i > 0).then(|| k - 1),
        (i + 1 < s).then(|| k + 1),
        (j > 0).then(|| k - s),
        (j + 1 < s).then(|| k + s),
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
    n: usize,
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
            let near: Vec<Point> = neighbours(k, n).filter_map(|j| target[j]).collect();
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
            for j in neighbours(k, n) {
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

/// Triangles of the grid as vertex indices (two per cell).
pub fn triangles(n: usize) -> Vec<[usize; 3]> {
    let s = n + 1;
    let mut t = Vec::with_capacity(n * n * 2);
    for j in 0..n {
        for i in 0..n {
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
        let g = grid(&q, 4);
        assert_eq!(g.len(), 25);
        assert_eq!(g[0], [0.0, 0.0]);
        assert_eq!(g[4], [100.0, 0.0]);
        assert_eq!(g[24], [100.0, 50.0]);
        assert_eq!(grid_side(25), Some(4));
        assert_eq!(triangles(4).len(), 32);
    }
}

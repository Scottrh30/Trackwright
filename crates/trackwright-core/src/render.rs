//! Pixel rendering: perspective warps, insert compositing and stabilization.

use crate::geometry::{apply, homography_from_corners, invert, mul, Mat3, Point};
use crate::project::Layer;
use serde::{Deserialize, Serialize};

/// 8-bit RGBA image.
#[derive(Clone, Debug)]
pub struct Rgba {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Rgba {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height * 4],
        }
    }

    /// Bilinear sample with premultiplied blending; outside the image is transparent.
    #[inline]
    pub fn sample(&self, x: f64, y: f64) -> [f32; 4] {
        let x = x - 0.5;
        let y = y - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
        let (xi, yi) = (x0 as i64, y0 as i64);
        let px = |dx: i64, dy: i64| -> [f32; 4] {
            let (sx, sy) = (xi + dx, yi + dy);
            if sx < 0 || sy < 0 || sx >= self.width as i64 || sy >= self.height as i64 {
                return [0.0; 4];
            }
            let i = (sy as usize * self.width + sx as usize) * 4;
            let a = self.data[i + 3] as f32 / 255.0;
            [
                self.data[i] as f32 * a,
                self.data[i + 1] as f32 * a,
                self.data[i + 2] as f32 * a,
                a,
            ]
        };
        let (a, b, c, d) = (px(0, 0), px(1, 0), px(0, 1), px(1, 1));
        let mut out = [0f32; 4];
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bot = c[k] + (d[k] - c[k]) * fx;
            out[k] = top + (bot - top) * fy;
        }
        // Back to straight (non-premultiplied) colour.
        if out[3] > 1e-6 {
            for k in 0..3 {
                out[k] /= out[3];
            }
        }
        out
    }
}

/// Render an `out_w` x `out_h` image where each output pixel centre `p` shows `src`
/// at `out_to_src * p`. Uncovered pixels get `background`.
pub fn warp(
    src: &Rgba,
    out_to_src: &Mat3,
    out_w: usize,
    out_h: usize,
    background: [u8; 4],
) -> Rgba {
    let mut out = Rgba::new(out_w, out_h);
    crate::par::rows(&mut out.data, out_w * 4, |y, row| {
        for x in 0..out_w {
            let p = apply(out_to_src, [x as f64 + 0.5, y as f64 + 0.5]);
            let s = src.sample(p[0], p[1]);
            let a = s[3];
            for k in 0..3 {
                row[x * 4 + k] = (s[k] * a + background[k] as f32 * (1.0 - a))
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
            row[x * 4 + 3] = ((a * 255.0) + background[3] as f32 * (1.0 - a))
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    });
    out
}

/// Composite `insert` over `frame`, corner-pinned to `corners` (TL, TR, BR, BL).
pub fn composite_insert(frame: &mut Rgba, insert: &Rgba, corners: &[Point; 4], opacity: f64) {
    let (iw, ih) = (insert.width as f64, insert.height as f64);
    let rect = [[0.0, 0.0], [iw, 0.0], [iw, ih], [0.0, ih]];
    let Some(h) = homography_from_corners(&rect, corners) else {
        return;
    };
    let Some(hi) = invert(&h) else { return };
    let b = crate::shape::bbox(corners).unwrap();
    let (fw, fh) = (frame.width, frame.height);
    let y0 = (b[1].floor().max(0.0) as usize).min(fh);
    let y1 = ((b[3].ceil() + 1.0).max(0.0) as usize).min(fh);
    let x0 = (b[0].floor().max(0.0) as usize).min(fw);
    let x1 = ((b[2].ceil() + 1.0).max(0.0) as usize).min(fw);
    if y1 <= y0 || x1 <= x0 {
        return;
    }
    let op = opacity.clamp(0.0, 1.0) as f32;
    let band = &mut frame.data[y0 * fw * 4..y1 * fw * 4];
    crate::par::rows(band, fw * 4, |yy, row| {
        let y = y0 + yy;
        for x in x0..x1 {
            // Use the homography's projective map (not affine) so perspective is exact.
            let p = apply(&hi, [x as f64 + 0.5, y as f64 + 0.5]);
            let s = insert.sample(p[0], p[1]);
            let a = s[3] * op;
            if a <= 0.0 {
                continue;
            }
            for k in 0..3 {
                let d = row[x * 4 + k] as f32;
                row[x * 4 + k] = (s[k] * a + d * (1.0 - a)).round().clamp(0.0, 255.0) as u8;
            }
            let da = row[x * 4 + 3] as f32 / 255.0;
            row[x * 4 + 3] = ((a + da * (1.0 - a)) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    });
}

/// Composite `insert` over `frame` warped by a tracked mesh: `uv` are the grid's
/// vertices in insert pixels, `pts` the same vertices on this frame (row by row, an
/// `n` x `n` cell grid). Each grid triangle is mapped on its own, so the insert bends
/// with the surface.
pub fn composite_insert_mesh(
    frame: &mut Rgba,
    insert: &Rgba,
    n: usize,
    uv: &[Point],
    pts: &[Point],
    opacity: f64,
) {
    if uv.len() != pts.len() || pts.len() != (n + 1) * (n + 1) {
        return;
    }
    let op = opacity.clamp(0.0, 1.0) as f32;
    let (fw, fh) = (frame.width, frame.height);
    for t in crate::mesh::triangles(n) {
        let (a, b, c) = (pts[t[0]], pts[t[1]], pts[t[2]]);
        let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if det.abs() < 1e-9 {
            continue;
        }
        let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize;
        let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize;
        let x1 = (a[0].max(b[0]).max(c[0]).ceil() as usize + 1).min(fw);
        let y1 = (a[1].max(b[1]).max(c[1]).ceil() as usize + 1).min(fh);
        let (ua, ub, uc) = (uv[t[0]], uv[t[1]], uv[t[2]]);
        for y in y0..y1 {
            for x in x0..x1 {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                // Barycentric coordinates; a hair of slack so seams don't open up.
                let l1 = ((px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])) / det;
                let l2 = ((b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1])) / det;
                let l0 = 1.0 - l1 - l2;
                if l0 < -1e-6 || l1 < -1e-6 || l2 < -1e-6 {
                    continue;
                }
                let u = l0 * ua[0] + l1 * ub[0] + l2 * uc[0];
                let v = l0 * ua[1] + l1 * ub[1] + l2 * uc[1];
                let s = insert.sample(u, v);
                let al = s[3] * op;
                if al <= 0.0 {
                    continue;
                }
                let i = (y * fw + x) * 4;
                for k in 0..3 {
                    let d = frame.data[i + k] as f32;
                    frame.data[i + k] =
                        (s[k] * al + d * (1.0 - al)).round().clamp(0.0, 255.0) as u8;
                }
                let da = frame.data[i + 3] as f32 / 255.0;
                frame.data[i + 3] =
                    ((al + da * (1.0 - al)) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Copy pixels of `original` back into `frame` inside `poly` (with a 1 px soft edge),
/// so occluders stay in front of an insert.
pub fn restore_inside(frame: &mut Rgba, original: &Rgba, poly: &[Point]) {
    let Some(b) = crate::shape::bbox(poly) else {
        return;
    };
    let w = frame.width;
    let y0 = (b[1].floor() - 1.0).max(0.0) as usize;
    let y1 = ((b[3].ceil() + 2.0).max(0.0) as usize).min(frame.height);
    let x0 = (b[0].floor() - 1.0).max(0.0) as usize;
    let x1 = ((b[2].ceil() + 2.0).max(0.0) as usize).min(w);
    for y in y0..y1 {
        for x in x0..x1 {
            // 2x2 supersampled coverage for a clean edge.
            let mut cov = 0.0f32;
            for (dx, dy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                if crate::shape::point_in_polygon(poly, [x as f64 + dx, y as f64 + dy]) {
                    cov += 0.25;
                }
            }
            if cov == 0.0 {
                continue;
            }
            let i = (y * w + x) * 4;
            for k in 0..4 {
                let (a, o) = (frame.data[i + k] as f32, original.data[i + k] as f32);
                frame.data[i + k] = (o * cov + a * (1.0 - cov)).round() as u8;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum StabilizeMode {
    /// The surface stays exactly where it is on the reference frame.
    Lock,
    /// Keep the overall move but remove jitter: corners are smoothed over this many
    /// frames either side.
    Smooth { radius: usize },
}

/// Output-to-source transforms for stabilizing `frames` (which must be tracked).
/// `zoom` > 1 scales up around the centre to hide the moving edges.
pub fn stabilize_maps(
    layer: &Layer,
    frames: &[usize],
    mode: StabilizeMode,
    zoom: f64,
    width: usize,
    height: usize,
) -> Vec<(usize, Mat3)> {
    let (cx, cy) = (width as f64 / 2.0, height as f64 / 2.0);
    let z = zoom.max(0.1);
    // Output pixel -> un-zoomed output coordinates.
    let unzoom: Mat3 = [
        1.0 / z,
        0.0,
        cx - cx / z,
        0.0,
        1.0 / z,
        cy - cy / z,
        0.0,
        0.0,
        1.0,
    ];
    let tracked: Vec<usize> = frames
        .iter()
        .copied()
        .filter(|f| layer.tracks.contains_key(f))
        .collect();
    tracked
        .iter()
        .filter_map(|&f| {
            let m = match mode {
                StabilizeMode::Lock => layer.final_h_at(f)?,
                StabilizeMode::Smooth { radius } => {
                    let actual = layer.surface_at(f)?;
                    let sigma = (radius as f64 / 2.0).max(0.5);
                    let mut acc = [[0.0; 2]; 4];
                    let mut wsum = 0.0;
                    let lo = f.saturating_sub(radius);
                    for g in lo..=f + radius {
                        let Some(c) = layer.surface_at(g) else {
                            continue;
                        };
                        let d = g as f64 - f as f64;
                        let w = (-(d * d) / (2.0 * sigma * sigma)).exp();
                        for i in 0..4 {
                            acc[i][0] += c[i][0] * w;
                            acc[i][1] += c[i][1] * w;
                        }
                        wsum += w;
                    }
                    let smooth: [Point; 4] =
                        std::array::from_fn(|i| [acc[i][0] / wsum, acc[i][1] / wsum]);
                    homography_from_corners(&smooth, &actual)?
                }
            };
            Some((f, mul(&m, &unzoom)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::FrameTrack;
    use crate::shape::Shape;

    fn checker(w: usize, h: usize) -> Rgba {
        let mut img = Rgba::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let on = ((x / 8) + (y / 8)) % 2 == 0;
                let i = (y * w + x) * 4;
                img.data[i..i + 4].copy_from_slice(&if on {
                    [250, 250, 250, 255]
                } else {
                    [10, 10, 10, 255]
                });
            }
        }
        img
    }

    #[test]
    fn identity_warp_is_lossless() {
        let src = checker(64, 48);
        let out = warp(&src, &crate::geometry::IDENTITY, 64, 48, [0, 0, 0, 255]);
        assert_eq!(out.data, src.data);
    }

    #[test]
    fn lock_stabilize_undoes_motion() {
        // Frame 3 is the reference shifted by (+5, -2); locking maps it back.
        let mut l = Layer::new(1, 0, Shape::rect(10.0, 10.0, 30.0, 20.0));
        l.tracks.insert(
            3,
            FrameTrack {
                h_from_ref: [1.0, 0.0, 5.0, 0.0, 1.0, -2.0, 0.0, 0.0, 1.0],
                ..FrameTrack::reference()
            },
        );
        let maps = stabilize_maps(&l, &[0, 3], StabilizeMode::Lock, 1.0, 64, 48);
        assert_eq!(maps.len(), 2);
        let p = apply(&maps[1].1, [20.0, 20.0]);
        assert!((p[0] - 25.0).abs() < 1e-9 && (p[1] - 18.0).abs() < 1e-9);
        // Smooth mode on a steady move keeps the move (no correction in the middle).
        for f in 1..10 {
            l.tracks.insert(
                f,
                FrameTrack {
                    h_from_ref: [1.0, 0.0, f as f64, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    ..FrameTrack::reference()
                },
            );
        }
        let maps = stabilize_maps(&l, &[5], StabilizeMode::Smooth { radius: 3 }, 1.0, 64, 48);
        let p = apply(&maps[0].1, [20.0, 20.0]);
        assert!((p[0] - 20.0).abs() < 1e-6 && (p[1] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn insert_lands_on_corners() {
        let mut frame = Rgba::new(100, 80);
        for px in frame.data.chunks_mut(4) {
            px.copy_from_slice(&[0, 0, 0, 255]);
        }
        let mut ins = Rgba::new(10, 10);
        for px in ins.data.chunks_mut(4) {
            px.copy_from_slice(&[255, 0, 0, 255]);
        }
        composite_insert(
            &mut frame,
            &ins,
            &[[20.0, 20.0], [60.0, 20.0], [60.0, 50.0], [20.0, 50.0]],
            1.0,
        );
        let at = |x: usize, y: usize| frame.data[(y * 100 + x) * 4];
        assert_eq!(at(40, 35), 255);
        assert_eq!(at(10, 10), 0);
        assert_eq!(at(70, 35), 0);
    }

    #[test]
    fn mesh_insert_matches_corner_pin_when_flat_and_bends_when_not() {
        let mut ins = Rgba::new(40, 40);
        for (i, px) in ins.data.chunks_mut(4).enumerate() {
            let (x, y) = (i % 40, i / 40);
            px.copy_from_slice(&[(x * 6) as u8, (y * 6) as u8, 100, 255]);
        }
        let quad = [[20.0, 20.0], [70.0, 20.0], [70.0, 60.0], [20.0, 60.0]];
        let black = || {
            let mut f = Rgba::new(100, 80);
            for px in f.data.chunks_mut(4) {
                px.copy_from_slice(&[0, 0, 0, 255]);
            }
            f
        };
        let mut a = black();
        composite_insert(&mut a, &ins, &quad, 1.0);
        let mut b = black();
        let uv = crate::mesh::grid(&[[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]], 4);
        let pts = crate::mesh::grid(&quad, 4);
        composite_insert_mesh(&mut b, &ins, 4, &uv, &pts, 1.0);
        // Inside the quad (edge pixels are partly covered either way).
        let mut diff = 0;
        for y in 22..58 {
            for x in 22..68 {
                for k in 0..3 {
                    let i = (y * 100 + x) * 4 + k;
                    diff = diff.max((a.data[i] as i32 - b.data[i] as i32).abs());
                }
            }
        }
        assert!(diff <= 2, "flat mesh = corner pin (max diff {diff})");
        // Push the middle vertex down: the insert follows it.
        let mut bent = pts.clone();
        bent[12][1] += 10.0;
        let mut c = black();
        composite_insert_mesh(&mut c, &ins, 4, &uv, &bent, 1.0);
        let at = |f: &Rgba, x: usize, y: usize| f.data[(y * 100 + x) * 4 + 1];
        // Green encodes the insert's row: lower on the frame for the same row.
        assert!(at(&c, 45, 48) < at(&b, 45, 48), "bent down");
    }
}

// ---------------------------------------------------------------------------
// Mattes (rotoscoping)

/// Distance from `p` to the outline of `poly`.
fn outline_distance(poly: &[Point], p: Point) -> f64 {
    let n = poly.len();
    let mut best = f64::INFINITY;
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 {
            (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (qx, qy) = (a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
        best = best.min(qx * qx + qy * qy);
    }
    best.sqrt()
}

/// Alpha matte (0..255, one byte per pixel) of the union of `polys`, with a soft edge
/// `feather` pixels wide centred on the outline (at least 1 px for anti-aliasing).
pub fn matte(width: usize, height: usize, polys: &[Vec<Point>], feather: f64) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    let fw = feather.max(1.0);
    for poly in polys.iter().filter(|p| p.len() >= 3) {
        let b = crate::shape::bbox(poly).unwrap();
        let y0 = ((b[1] - fw).floor().max(0.0) as usize).min(height);
        let y1 = ((b[3] + fw).ceil().max(0.0) as usize + 1).min(height);
        let x0 = ((b[0] - fw).floor().max(0.0) as usize).min(width);
        let x1 = ((b[2] + fw).ceil().max(0.0) as usize + 1).min(width);
        if y1 <= y0 || x1 <= x0 {
            continue;
        }
        let band = &mut out[y0 * width..y1 * width];
        crate::par::rows(band, width, |yy, row| {
            let y = (y0 + yy) as f64 + 0.5;
            for x in x0..x1 {
                let p = [x as f64 + 0.5, y];
                let inside = crate::shape::point_in_polygon(poly, p);
                let d = outline_distance(poly, p);
                let a = if d >= fw / 2.0 {
                    if inside {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let sd = if inside { d } else { -d };
                    (0.5 + sd / fw).clamp(0.0, 1.0)
                };
                let v = (a * 255.0).round() as u8;
                row[x] = row[x].max(v);
            }
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Object removal

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RemoveParams {
    /// How many frames either side to borrow pixels from.
    pub search: usize,
    /// Grow the removal area by this many pixels (covers edges, shadows, blur).
    pub grow: f64,
    /// Match brightness/colour of borrowed pixels to the surrounding area.
    pub match_light: bool,
    /// Soft blend at the edge of the filled area, in pixels.
    pub feather: f64,
}

impl Default for RemoveParams {
    fn default() -> Self {
        Self {
            search: 20,
            grow: 6.0,
            match_light: true,
            feather: 3.0,
        }
    }
}

fn in_mask(poly: &[Point], p: Point, grow: f64) -> bool {
    crate::shape::point_in_polygon(poly, p) || (grow > 0.0 && outline_distance(poly, p) < grow)
}

/// Fill the removal area of frame `f` with the background seen in other frames.
/// `bg` tracks the surface behind the object; `mask(g)` is the object's outline at
/// frame g (None when it isn't there); `load(g)` returns frame g's pixels.
pub fn remove_object(
    f: usize,
    frame: &Rgba,
    bg: &Layer,
    frame_count: usize,
    mask: &dyn Fn(usize) -> Option<Vec<Point>>,
    load: &mut dyn FnMut(usize) -> Option<std::sync::Arc<Rgba>>,
    params: &RemoveParams,
) -> Rgba {
    let mut out = frame.clone();
    let (Some(poly), Some(hf)) = (mask(f), bg.final_h_at(f)) else {
        return out;
    };
    let Some(hf_inv) = invert(&hf) else {
        return out;
    };
    let (w, h) = (frame.width, frame.height);
    let b = crate::shape::bbox(&poly).unwrap();
    let pad = params.grow + params.feather + 1.0;
    let x0 = ((b[0] - pad).floor().max(0.0) as usize).min(w);
    let x1 = ((b[2] + pad).ceil().max(0.0) as usize + 1).min(w);
    let y0 = ((b[1] - pad).floor().max(0.0) as usize).min(h);
    let y1 = ((b[3] + pad).ceil().max(0.0) as usize + 1).min(h);
    // Pixels to fill, and a ring around them for lighting comparison.
    let mut todo: Vec<(usize, usize)> = Vec::new();
    let mut ring: Vec<(usize, usize)> = Vec::new();
    let ring_w = 14.0;
    let rx0 = x0.saturating_sub(ring_w as usize);
    let ry0 = y0.saturating_sub(ring_w as usize);
    let rx1 = (x1 + ring_w as usize).min(w);
    let ry1 = (y1 + ring_w as usize).min(h);
    for y in ry0..ry1 {
        for x in rx0..rx1 {
            let p = [x as f64 + 0.5, y as f64 + 0.5];
            if in_mask(&poly, p, params.grow) {
                todo.push((x, y));
            } else if params.match_light
                && in_mask(&poly, p, params.grow + ring_w)
                && (x + y) % 2 == 0
            {
                ring.push((x, y));
            }
        }
    }
    if todo.is_empty() {
        return out;
    }
    let mut filled: Vec<Option<[f32; 3]>> = vec![None; todo.len()];
    let mut remaining = todo.len();
    let mut order = Vec::new();
    for d in 1..=params.search {
        order.push(f as isize + d as isize);
        order.push(f as isize - d as isize);
    }
    for g in order {
        if remaining == 0 {
            break;
        }
        if g < 0 || g as usize >= frame_count {
            continue;
        }
        let g = g as usize;
        let Some(hg) = bg.final_h_at(g) else { continue };
        let map = crate::geometry::mul(&hg, &hf_inv);
        let gmask = mask(g);
        let Some(src) = load(g) else { continue };
        let usable = |q: Point| {
            q[0] >= 1.0
                && q[1] >= 1.0
                && q[0] < w as f64 - 1.0
                && q[1] < h as f64 - 1.0
                && gmask.as_ref().map_or(true, |m| !in_mask(m, q, params.grow))
        };
        // Lighting: compare the ring around the hole in f with the same area in g.
        let mut gain = [1.0f32; 3];
        if params.match_light {
            let (mut sf, mut sg, mut n) = ([0f64; 3], [0f64; 3], 0usize);
            for &(x, y) in &ring {
                let q = apply(&map, [x as f64 + 0.5, y as f64 + 0.5]);
                if !usable(q) {
                    continue;
                }
                let a = src.sample(q[0], q[1]);
                let i = (y * w + x) * 4;
                for c in 0..3 {
                    sf[c] += frame.data[i + c] as f64;
                    sg[c] += a[c] as f64;
                }
                n += 1;
            }
            if n > 20 {
                for c in 0..3 {
                    gain[c] = ((sf[c] + 1.0) / (sg[c] + 1.0)).clamp(0.5, 2.0) as f32;
                }
            }
        }
        for (k, &(x, y)) in todo.iter().enumerate() {
            if filled[k].is_some() {
                continue;
            }
            let q = apply(&map, [x as f64 + 0.5, y as f64 + 0.5]);
            if usable(q) {
                let a = src.sample(q[0], q[1]);
                filled[k] = Some([a[0] * gain[0], a[1] * gain[1], a[2] * gain[2]]);
                remaining -= 1;
            }
        }
    }
    // Anything never visible: diffuse in from the edges.
    let mut vals: Vec<[f32; 3]> = filled.iter().map(|v| v.unwrap_or([0.0; 3])).collect();
    if remaining > 0 {
        let idx: std::collections::HashMap<(usize, usize), usize> =
            todo.iter().enumerate().map(|(i, p)| (*p, i)).collect();
        let unknown: Vec<usize> = (0..todo.len()).filter(|&k| filled[k].is_none()).collect();
        for _ in 0..400 {
            let mut change = 0f32;
            for &k in &unknown {
                let (x, y) = todo[k];
                let mut acc = [0f32; 3];
                let mut n = 0.0;
                for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let v = match idx.get(&(nx as usize, ny as usize)) {
                        Some(&j) => vals[j],
                        None => {
                            let i = (ny as usize * w + nx as usize) * 4;
                            [
                                frame.data[i] as f32,
                                frame.data[i + 1] as f32,
                                frame.data[i + 2] as f32,
                            ]
                        }
                    };
                    for c in 0..3 {
                        acc[c] += v[c];
                    }
                    n += 1.0;
                }
                if n > 0.0 {
                    let nv = [acc[0] / n, acc[1] / n, acc[2] / n];
                    change = change.max((nv[0] - vals[k][0]).abs());
                    vals[k] = nv;
                }
            }
            if change < 0.05 {
                break;
            }
        }
    }
    // Write back with a soft edge.
    for (k, &(x, y)) in todo.iter().enumerate() {
        let p = [x as f64 + 0.5, y as f64 + 0.5];
        let a = if params.feather > 0.0 && !crate::shape::point_in_polygon(&poly, p) {
            // Outside the object outline but inside the grown area: fade towards the edge.
            let d = outline_distance(&poly, p);
            (((params.grow - d) / params.feather.max(1e-6)).clamp(0.0, 1.0)) as f32
        } else {
            1.0
        };
        let i = (y * w + x) * 4;
        for c in 0..3 {
            let o = out.data[i + c] as f32;
            out.data[i + c] = (vals[k][c] * a + o * (1.0 - a)).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// A single clean background image in the background layer's reference-frame geometry:
/// the per-pixel median of the surface over `frames`, ignoring the object (`mask`).
pub fn clean_plate(
    bg: &Layer,
    width: usize,
    height: usize,
    frames: &[usize],
    mask: &dyn Fn(usize) -> Option<Vec<Point>>,
    load: &mut dyn FnMut(usize) -> Option<std::sync::Arc<Rgba>>,
    grow: f64,
) -> Rgba {
    let max_samples = 15usize;
    let step = (frames.len() / max_samples).max(1);
    let chosen: Vec<usize> = frames
        .iter()
        .copied()
        .step_by(step)
        .take(max_samples)
        .collect();
    let k = chosen.len();
    let n = width * height;
    let mut samples = vec![0u8; n * k * 3];
    let mut valid = vec![false; n * k];
    for (si, &g) in chosen.iter().enumerate() {
        let (Some(hg), Some(src)) = (bg.final_h_at(g), load(g)) else {
            continue;
        };
        let m = mask(g);
        for y in 0..height {
            for x in 0..width {
                let q = apply(&hg, [x as f64 + 0.5, y as f64 + 0.5]);
                if q[0] < 1.0
                    || q[1] < 1.0
                    || q[0] >= src.width as f64 - 1.0
                    || q[1] >= src.height as f64 - 1.0
                {
                    continue;
                }
                if m.as_ref().is_some_and(|m| in_mask(m, q, grow)) {
                    continue;
                }
                let a = src.sample(q[0], q[1]);
                let pi = y * width + x;
                for c in 0..3 {
                    samples[(pi * k + si) * 3 + c] = a[c].round().clamp(0.0, 255.0) as u8;
                }
                valid[pi * k + si] = true;
            }
        }
    }
    let mut out = Rgba::new(width, height);
    let mut buf: Vec<u8> = Vec::with_capacity(k);
    for pi in 0..n {
        for c in 0..3 {
            buf.clear();
            for si in 0..k {
                if valid[pi * k + si] {
                    buf.push(samples[(pi * k + si) * 3 + c]);
                }
            }
            if buf.is_empty() {
                continue;
            }
            buf.sort_unstable();
            out.data[pi * 4 + c] = buf[buf.len() / 2];
            out.data[pi * 4 + 3] = 255;
        }
    }
    out
}

#[cfg(test)]
mod remove_tests {
    use super::*;
    use crate::project::FrameTrack;
    use crate::shape::Shape;
    use std::sync::Arc;

    fn textured(w: usize, h: usize, shift: f64) -> Rgba {
        let mut img = Rgba::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let xs = x as f64 + shift;
                let v = (((xs * 0.21).sin() * (y as f64 * 0.17).cos()) * 90.0 + 128.0) as u8;
                let i = (y * w + x) * 4;
                img.data[i..i + 4].copy_from_slice(&[v, v / 2 + 40, 255 - v, 255]);
            }
        }
        img
    }

    #[test]
    fn fills_object_from_other_frames() {
        // Background pans 4 px/frame; a black box sits still in the middle of every frame.
        let (w, h, n) = (120usize, 80usize, 12usize);
        let mut bg = Layer::new(1, 0, Shape::rect(0.0, 0.0, 120.0, 80.0));
        for f in 1..n {
            bg.tracks.insert(
                f,
                FrameTrack {
                    h_from_ref: [1.0, 0.0, -4.0 * f as f64, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    ..FrameTrack::reference()
                },
            );
        }
        let box_poly = vec![[50.0, 30.0], [70.0, 30.0], [70.0, 50.0], [50.0, 50.0]];
        let frames: Vec<Arc<Rgba>> = (0..n)
            .map(|f| {
                let mut img = textured(w, h, 4.0 * f as f64);
                for y in 30..50 {
                    for x in 50..70 {
                        let i = (y * w + x) * 4;
                        img.data[i..i + 3].copy_from_slice(&[0, 0, 0]);
                    }
                }
                Arc::new(img)
            })
            .collect();
        let mask = |_g: usize| Some(box_poly.clone());
        let mut load = |g: usize| frames.get(g).cloned();
        let p = RemoveParams {
            search: 10,
            grow: 2.0,
            match_light: true,
            feather: 0.0,
        };
        let out = remove_object(6, &frames[6], &bg, n, &mask, &mut load, &p);
        let truth = textured(w, h, 24.0);
        let mut worst = 0i32;
        for y in 32..48 {
            for x in 52..68 {
                let i = (y * w + x) * 4;
                worst = worst.max((out.data[i] as i32 - truth.data[i] as i32).abs());
            }
        }
        assert!(
            worst <= 6,
            "fill differs from the true background by {worst}"
        );
    }

    #[test]
    fn matte_is_soft_and_correct() {
        let poly = vec![[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]];
        let m = matte(40, 40, &[poly], 4.0);
        assert_eq!(m[20 * 40 + 20], 255);
        assert_eq!(m[2 * 40 + 2], 0);
        let edge = m[20 * 40 + 10];
        assert!(edge > 60 && edge < 200, "edge {edge}");
    }
}

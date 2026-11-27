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
}

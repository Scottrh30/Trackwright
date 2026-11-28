//! Finding the pupil inside a tracked eye.
//!
//! A pupil (with the iris around it) is the darkest round spot in the eye. Tracking
//! a tiny box on the pupil itself is fragile: the eye turns, the lids cut into it,
//! and it has little texture. Tracking the whole eye is steady, and inside it the
//! pupil is easy to find on every frame: the spot where a dark disk sits inside a
//! lighter ring. The centre is then refined to sub-pixel accuracy from the dark
//! pixels around it.

use crate::geometry::Point;
use crate::image::Pyramid;
use crate::shape::point_in_polygon;

/// A pupil found on one frame: centre and radius in frame pixels, and how clearly it
/// stood out (ring brightness minus disk brightness, 0..255).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pupil {
    pub centre: Point,
    pub radius: f64,
    pub contrast: f64,
}

/// Look for the pupil inside `eye` (the eye's outline in frame pixels). `None` when
/// nothing round and dark stands out (a closed eye).
pub fn find_pupil(frame: &Pyramid, eye: &[Point]) -> Option<Pupil> {
    let b = crate::shape::bbox(eye)?;
    let width = (b[2] - b[0]).max(b[3] - b[1]);
    if width < 8.0 {
        return None;
    }
    // Work at a level where the eye is about 120 px across.
    let mut lvl = 0;
    while lvl + 1 < frame.levels.len() && width / (1u32 << (lvl + 1)) as f64 >= 120.0 {
        lvl += 1;
    }
    let k = (1u32 << lvl) as f64;
    let img = &frame.levels[lvl];
    let (x0, y0) = (
        (b[0] / k).floor().max(0.0) as i64,
        (b[1] / k).floor().max(0.0) as i64,
    );
    let (x1, y1) = (
        ((b[2] / k).ceil() as i64).min(img.width as i64 - 1),
        ((b[3] / k).ceil() as i64).min(img.height as i64 - 1),
    );
    if x1 - x0 < 6 || y1 - y0 < 6 {
        return None;
    }
    let (w, h) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
    // Integral image of the crop for fast box means.
    let mut integ = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0.0;
        for x in 0..w {
            row += img.get(x0 as usize + x, y0 as usize + y) as f64;
            integ[(y + 1) * (w + 1) + x + 1] = integ[y * (w + 1) + x + 1] + row;
        }
    }
    let boxsum = |cx: i64, cy: i64, r: i64| -> Option<(f64, f64)> {
        let (ax, ay, bx, by) = (cx - r, cy - r, cx + r + 1, cy + r + 1);
        if ax < 0 || ay < 0 || bx > w as i64 || by > h as i64 {
            return None;
        }
        let at = |x: i64, y: i64| integ[y as usize * (w + 1) + x as usize];
        let s = at(bx, by) - at(ax, by) - at(bx, ay) + at(ax, ay);
        Some((s, ((bx - ax) * (by - ay)) as f64))
    };
    let poly: Vec<Point> = eye
        .iter()
        .map(|p| [p[0] / k - x0 as f64, p[1] / k - y0 as f64])
        .collect();
    let crop_w = width / k;
    // Pupil + iris is roughly a fifth to half of the eye's width across.
    let (rmin, rmax) = (
        (crop_w * 0.06).max(2.0) as i64,
        (crop_w * 0.25).max(3.0) as i64,
    );
    let inside: Vec<bool> = (0..w * h)
        .map(|i| point_in_polygon(&poly, [(i % w) as f64 + 0.5, (i / w) as f64 + 0.5]))
        .collect();
    let mut best: Option<(f64, i64, i64, i64)> = None;
    for r in rmin..=rmax {
        let ring = (r as f64 * 1.6).round() as i64;
        for cy in 0..h as i64 {
            for cx in 0..w as i64 {
                if !inside[cy as usize * w + cx as usize] {
                    continue;
                }
                let (Some((si, ni)), Some((so, no))) = (boxsum(cx, cy, r), boxsum(cx, cy, ring))
                else {
                    continue;
                };
                let inner = si / ni;
                let outer = (so - si) / (no - ni).max(1.0);
                // Clearly darker than around it, a bit more for bigger (steadier) disks.
                let score = (outer - inner) * (1.0 + 0.02 * r as f64);
                if best.is_none_or(|b| score > b.0) {
                    best = Some((score, cx, cy, r));
                }
            }
        }
    }
    let (_, cx, cy, r) = best?;
    let (si, ni) = boxsum(cx, cy, r)?;
    let ring = (r as f64 * 1.6).round() as i64;
    let (so, no) = boxsum(cx, cy, ring)?;
    let (inner, outer) = (si / ni, (so - si) / (no - ni).max(1.0));
    let contrast = outer - inner;
    if contrast < 12.0 {
        return None;
    }
    // Sub-pixel centre: weighted centroid of the pixels darker than half way.
    let mid = (inner + outer) * 0.5;
    let reach = ring as f64;
    let (mut sw, mut sx, mut sy) = (0.0, 0.0, 0.0);
    for y in (cy - ring).max(0)..=(cy + ring).min(h as i64 - 1) {
        for x in (cx - ring).max(0)..=(cx + ring).min(w as i64 - 1) {
            let d = ((x - cx) as f64).hypot((y - cy) as f64);
            if d > reach {
                continue;
            }
            let v = img.get((x0 + x) as usize, (y0 + y) as usize) as f64;
            let wgt = (mid - v).max(0.0);
            sw += wgt;
            sx += wgt * x as f64;
            sy += wgt * y as f64;
        }
    }
    let (px, py) = if sw > 0.0 {
        (sx / sw, sy / sw)
    } else {
        (cx as f64, cy as f64)
    };
    Some(Pupil {
        centre: [
            (px + x0 as f64 + 0.5) * k - 0.5,
            (py + y0 as f64 + 0.5) * k - 0.5,
        ],
        radius: r as f64 * k,
        contrast,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::GrayImage;

    #[test]
    fn finds_a_dark_disk_to_a_fraction_of_a_pixel() {
        let (w, h) = (400usize, 300usize);
        let (cx, cy, r) = (213.3f64, 147.8f64, 18.0f64);
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                // White of the eye with some texture, a dark disk with a soft rim.
                let t = ((x * 7 + y * 13) % 17) as f32;
                let edge = ((d - r) / 1.5).clamp(-1.0, 1.0) * 0.5 + 0.5;
                img.data[y * w + x] = 25.0 + (190.0 + t - 25.0) * edge as f32;
            }
        }
        let p = Pyramid::build(img, 4);
        let eye = vec![[120.0, 90.0], [320.0, 90.0], [320.0, 210.0], [120.0, 210.0]];
        let f = find_pupil(&p, &eye).unwrap();
        let err = (f.centre[0] - cx).hypot(f.centre[1] - cy);
        assert!(err < 0.5, "{f:?} err {err}");
        // A closed eye (flat skin) has no pupil.
        let flat = Pyramid::build(
            GrayImage {
                width: w,
                height: h,
                data: vec![150.0; w * h],
            },
            4,
        );
        assert!(find_pupil(&flat, &eye).is_none());
    }
}

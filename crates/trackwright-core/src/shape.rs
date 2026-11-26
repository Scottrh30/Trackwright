//! Spline shapes that define the region a planar layer tracks.

use crate::geometry::{apply, Mat3, Point};
use serde::{Deserialize, Serialize};

/// A closed Catmull-Rom spline through its control points (Mocha-style X-spline stand-in).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Shape {
    pub points: Vec<Point>,
    pub closed: bool,
    /// Round the outline through its points (spline) instead of straight edges.
    #[serde(default)]
    pub smooth: bool,
}

impl Shape {
    pub fn new(points: Vec<Point>) -> Self {
        Self {
            points,
            closed: true,
            smooth: false,
        }
    }

    pub fn rect(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self::new(vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    }

    pub fn is_valid(&self) -> bool {
        self.closed && self.points.len() >= 3
    }

    /// Polygonal approximation of the spline.
    pub fn flatten(&self, steps_per_segment: usize) -> Vec<Point> {
        let n = self.points.len();
        if n < 3 || steps_per_segment <= 1 || !self.smooth {
            return self.points.clone();
        }
        let p = |i: isize| self.points[i.rem_euclid(n as isize) as usize];
        let mut out = Vec::with_capacity(n * steps_per_segment);
        for i in 0..n as isize {
            let (p0, p1, p2, p3) = (p(i - 1), p(i), p(i + 1), p(i + 2));
            for s in 0..steps_per_segment {
                let t = s as f64 / steps_per_segment as f64;
                let (t2, t3) = (t * t, t * t * t);
                let c = |k: usize| {
                    0.5 * ((2.0 * p1[k])
                        + (-p0[k] + p2[k]) * t
                        + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * t2
                        + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * t3)
                };
                out.push([c(0), c(1)]);
            }
        }
        out
    }

    pub fn transformed(&self, h: &Mat3) -> Shape {
        Shape {
            points: self.points.iter().map(|&p| apply(h, p)).collect(),
            closed: self.closed,
            smooth: self.smooth,
        }
    }

    /// Nearest point on the outline to `p`: (index of the control segment it lies on,
    /// the point on the outline, distance). Inserting a control point at `segment + 1`
    /// puts it on that segment.
    pub fn nearest_edge(&self, p: Point) -> Option<(usize, Point, f64)> {
        let n = self.points.len();
        if n < 2 {
            return None;
        }
        let steps = if self.smooth { 10 } else { 1 };
        let poly = self.flatten(steps);
        let m = poly.len();
        let mut best: Option<(usize, Point, f64)> = None;
        for i in 0..m {
            let (a, b) = (poly[i], poly[(i + 1) % m]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let len2 = dx * dx + dy * dy;
            let t = if len2 > 0.0 {
                (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q = [a[0] + t * dx, a[1] + t * dy];
            let d = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt();
            if best.map_or(true, |b| d < b.2) {
                best = Some((i / steps, q, d));
            }
        }
        best
    }

    pub fn translated(&self, d: Point) -> Shape {
        Shape {
            points: self
                .points
                .iter()
                .map(|p| [p[0] + d[0], p[1] + d[1]])
                .collect(),
            ..self.clone()
        }
    }

    pub fn bbox(&self) -> Option<[f64; 4]> {
        bbox(&self.points)
    }
}

pub fn bbox(points: &[Point]) -> Option<[f64; 4]> {
    if points.is_empty() {
        return None;
    }
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in points {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    Some(b)
}

/// Even-odd point in polygon test.
pub fn point_in_polygon(poly: &[Point], p: Point) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0];
            if p[0] < x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Default surface for a shape: its bounding rectangle, ordered TL, TR, BR, BL.
pub fn default_surface(shape: &Shape) -> [Point; 4] {
    let b = shape.bbox().unwrap_or([0.0, 0.0, 100.0, 100.0]);
    [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_edge_segments() {
        let s = Shape::rect(0.0, 0.0, 10.0, 10.0);
        let (seg, q, d) = s.nearest_edge([5.0, -1.0]).unwrap();
        assert_eq!((seg, q, d), (0, [5.0, 0.0], 1.0));
        assert_eq!(s.nearest_edge([11.0, 5.0]).unwrap().0, 1);
        assert_eq!(s.nearest_edge([-1.0, 5.0]).unwrap().0, 3);
    }

    #[test]
    fn pip_square() {
        let sq = Shape::rect(0.0, 0.0, 10.0, 10.0).points;
        assert!(point_in_polygon(&sq, [5.0, 5.0]));
        assert!(!point_in_polygon(&sq, [15.0, 5.0]));
    }

    #[test]
    fn spline_passes_through_controls() {
        let mut s = Shape::rect(0.0, 0.0, 10.0, 10.0);
        assert_eq!(s.flatten(8).len(), 4);
        s.smooth = true;
        let f = s.flatten(8);
        assert_eq!(f.len(), 32);
        assert_eq!(f[0], [0.0, 0.0]);
        assert_eq!(f[8], [10.0, 0.0]);
    }
}

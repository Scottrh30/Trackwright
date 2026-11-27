//! Read Trackwright planar tracks and evaluate them at any frame or time.
//!
//! Built for Reelforge (and usable from Forma3D / Tidefire): load a track exported by
//! Trackwright and ask where the tracked surface is at any moment, as four corners, a
//! position/rotation/scale transform, or a homography that corner-pins a layer onto it.
//! No dependencies beyond `serde` / `serde_json`.
//!
//! ```no_run
//! use trackwright_import::CornerPinTrack;
//! let track = CornerPinTrack::load("shot_Screen.rftrack.json").unwrap();
//! // Corner-pin a 1920x1080 graphic onto the tracked screen at 2.5 seconds:
//! let h = track.pin_layer_at_time(2.5, 1920.0, 1080.0).unwrap();
//! let on_screen = trackwright_import::apply(&h, [960.0, 540.0]);
//! # let _ = on_screen;
//! ```
//!
//! © Scott Hambach. All rights reserved.

use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

pub type Point = [f64; 2];
/// Row-major 3x3 matrix mapping `[x, y, 1]` (column vector).
pub type Mat3 = [f64; 9];

/// Corners in order: top-left, top-right, bottom-right, bottom-left. Pixels, y down.
pub type Corners = [Point; 4];

#[derive(Debug)]
pub enum ImportError {
    Io(String),
    Json(String),
    Format(String),
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImportError::Io(e) => write!(f, "could not read track: {e}"),
            ImportError::Json(e) => write!(f, "track is not valid JSON: {e}"),
            ImportError::Format(e) => write!(f, "not a Trackwright track: {e}"),
        }
    }
}

impl std::error::Error for ImportError {}

/// Position (surface centre), rotation and scale relative to the reference frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform2D {
    pub position: Point,
    pub rotation_deg: f64,
    pub scale: Point,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrackFrame {
    pub corners: Corners,
    /// Reference-frame pixels -> this frame's pixels.
    pub homography: Option<Mat3>,
    pub transform: Option<Transform2D>,
    /// Number of good tracking points (0 when unknown).
    pub inliers: u32,
    /// Fit error in pixels (0 when unknown).
    pub rms_px: f64,
}

/// A Trackwright planar track.
#[derive(Clone, Debug, PartialEq)]
pub struct CornerPinTrack {
    /// "trackwright.reelforge.v1" or "trackwright.planar.v1".
    pub format: String,
    /// Path of the footage that was tracked.
    pub source_path: String,
    pub width: f64,
    pub height: f64,
    pub fps: f64,
    /// Frame number of the footage's first frame (e.g. 1001); frame numbers below use this numbering.
    pub first_frame: i64,
    pub layer_name: String,
    pub reference_frame: i64,
    pub reference_corners: Corners,
    pub frames: BTreeMap<i64, TrackFrame>,
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn point(v: &Value) -> Option<Point> {
    Some([num(v.get(0)?)?, num(v.get(1)?)?])
}

fn corners_obj(v: &Value) -> Option<Corners> {
    Some([
        point(v.get("tl")?)?,
        point(v.get("tr")?)?,
        point(v.get("br")?)?,
        point(v.get("bl")?)?,
    ])
}

fn corners_arr(v: &Value) -> Option<Corners> {
    let a = v.as_array()?;
    if a.len() != 4 {
        return None;
    }
    Some([point(&a[0])?, point(&a[1])?, point(&a[2])?, point(&a[3])?])
}

fn mat3(v: &Value) -> Option<Mat3> {
    let a = v.as_array()?;
    if a.len() != 9 {
        return None;
    }
    let mut m = [0.0; 9];
    for (i, x) in a.iter().enumerate() {
        m[i] = x.as_f64()?;
    }
    Some(m)
}

impl CornerPinTrack {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ImportError> {
        let text =
            std::fs::read_to_string(path.as_ref()).map_err(|e| ImportError::Io(e.to_string()))?;
        Self::from_json(&text)
    }

    /// Parse `.rftrack.json` (Reelforge) or `.f3dtrack.json` / `.tftrack.json` (planar) text.
    pub fn from_json(text: &str) -> Result<Self, ImportError> {
        let v: Value = serde_json::from_str(text).map_err(|e| ImportError::Json(e.to_string()))?;
        let bad = |what: &str| ImportError::Format(what.to_string());
        let format = v
            .get("format")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("missing \"format\""))?
            .to_string();
        let src = v.get("source").ok_or_else(|| bad("missing \"source\""))?;
        let width = src
            .get("width")
            .and_then(num)
            .ok_or_else(|| bad("missing source width"))?;
        let height = src
            .get("height")
            .and_then(num)
            .ok_or_else(|| bad("missing source height"))?;
        let fps = src.get("fps").and_then(num).unwrap_or(24.0);
        let first_frame = src.get("first_frame").and_then(Value::as_i64).unwrap_or(0);
        let source_path = src
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        let (layer_name, reference_frame, reference_corners) = match format.as_str() {
            "trackwright.reelforge.v1" => {
                let l = v.get("layer").ok_or_else(|| bad("missing layer"))?;
                (
                    l.get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    l.get("reference_frame")
                        .and_then(Value::as_i64)
                        .unwrap_or(first_frame),
                    l.get("reference_corners")
                        .and_then(corners_obj)
                        .ok_or_else(|| bad("missing reference corners"))?,
                )
            }
            "trackwright.planar.v1" => {
                let pl = v.get("plane").ok_or_else(|| bad("missing plane"))?;
                (
                    pl.get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    pl.get("reference_frame")
                        .and_then(Value::as_i64)
                        .unwrap_or(first_frame),
                    pl.get("corners_px")
                        .and_then(corners_arr)
                        .ok_or_else(|| bad("missing plane corners"))?,
                )
            }
            other => return Err(bad(&format!("unsupported format {other}"))),
        };

        let mut frames = BTreeMap::new();
        for fr in v
            .get("frames")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("missing frames"))?
        {
            let n = fr
                .get("frame")
                .and_then(Value::as_i64)
                .ok_or_else(|| bad("frame without a number"))?;
            let corners = fr
                .get("corners")
                .and_then(corners_obj)
                .or_else(|| fr.get("corners_px").and_then(corners_arr))
                .ok_or_else(|| bad("frame without corners"))?;
            let transform = fr.get("transform").and_then(|t| {
                Some(Transform2D {
                    position: point(t.get("position")?)?,
                    rotation_deg: num(t.get("rotation_deg")?)?,
                    scale: point(t.get("scale")?)?,
                })
            });
            let conf = fr.get("confidence");
            let inliers = conf
                .and_then(|c| c.get("inliers"))
                .or_else(|| fr.get("inliers"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            let rms_px = conf
                .and_then(|c| c.get("rms_px"))
                .or_else(|| fr.get("rms_px"))
                .and_then(num)
                .unwrap_or(0.0);
            frames.insert(
                n,
                TrackFrame {
                    corners,
                    homography: fr.get("homography").and_then(mat3),
                    transform,
                    inliers,
                    rms_px,
                },
            );
        }
        if frames.is_empty() {
            return Err(bad("track has no frames"));
        }
        Ok(Self {
            format,
            source_path,
            width,
            height,
            fps,
            first_frame,
            layer_name,
            reference_frame,
            reference_corners,
            frames,
        })
    }

    /// First and last tracked frame numbers.
    pub fn frame_range(&self) -> (i64, i64) {
        (
            *self.frames.keys().next().unwrap(),
            *self.frames.keys().next_back().unwrap(),
        )
    }

    /// Footage frame number (possibly fractional) at `seconds` from the start of the clip.
    pub fn frame_at_time(&self, seconds: f64) -> f64 {
        self.first_frame as f64 + seconds * self.fps
    }

    /// Corners at a (possibly fractional) frame number. Interpolates linearly between
    /// tracked frames and holds the first/last frame outside the tracked range.
    pub fn corners_at(&self, frame: f64) -> Corners {
        let below = self.frames.range(..=frame.floor() as i64).next_back();
        let above = self.frames.range(frame.ceil() as i64..).next();
        match (below, above) {
            (Some((&a, fa)), Some((&b, fb))) if b > a => {
                let t = (frame - a as f64) / (b - a) as f64;
                std::array::from_fn(|i| {
                    [
                        fa.corners[i][0] + (fb.corners[i][0] - fa.corners[i][0]) * t,
                        fa.corners[i][1] + (fb.corners[i][1] - fa.corners[i][1]) * t,
                    ]
                })
            }
            (Some((_, f)), _) | (None, Some((_, f))) => f.corners,
            (None, None) => self.reference_corners,
        }
    }

    pub fn corners_at_time(&self, seconds: f64) -> Corners {
        self.corners_at(self.frame_at_time(seconds))
    }

    /// Position (surface centre), rotation and scale relative to the reference frame.
    pub fn transform_at(&self, frame: f64) -> Transform2D {
        let c = self.corners_at(frame);
        let r = &self.reference_corners;
        let centre = [
            (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4.0,
            (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4.0,
        ];
        let ang = |a: Point, b: Point| (b[1] - a[1]).atan2(b[0] - a[0]);
        let mut rot = (ang(c[0], c[1]) - ang(r[0], r[1])).to_degrees();
        if rot > 180.0 {
            rot -= 360.0;
        } else if rot < -180.0 {
            rot += 360.0;
        }
        let dist = |a: Point, b: Point| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        let sx = dist(c[0], c[1]) / dist(r[0], r[1]).max(1e-9);
        let sy = dist(c[0], c[3]) / dist(r[0], r[3]).max(1e-9);
        Transform2D {
            position: centre,
            rotation_deg: rot,
            scale: [sx, sy],
        }
    }

    /// Homography that maps a `layer_w` x `layer_h` layer (pixels, origin top-left) onto
    /// the tracked surface at `frame`: the layer's corners land exactly on the corners.
    pub fn pin_layer_at(&self, frame: f64, layer_w: f64, layer_h: f64) -> Option<Mat3> {
        let rect = [
            [0.0, 0.0],
            [layer_w, 0.0],
            [layer_w, layer_h],
            [0.0, layer_h],
        ];
        homography(&rect, &self.corners_at(frame))
    }

    pub fn pin_layer_at_time(&self, seconds: f64, layer_w: f64, layer_h: f64) -> Option<Mat3> {
        self.pin_layer_at(self.frame_at_time(seconds), layer_w, layer_h)
    }

    /// The same track for footage shown at a different resolution (e.g. a 1080p proxy of
    /// a 4K plate, or a timeline of another size with the clip scaled to fit).
    pub fn rescaled(&self, width: f64, height: f64) -> Self {
        let (sx, sy) = (width / self.width, height / self.height);
        let s = |c: &Corners| c.map(|p| [p[0] * sx, p[1] * sy]);
        let scale_h = |h: &Mat3| {
            let m = [sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0];
            let mi = [1.0 / sx, 0.0, 0.0, 0.0, 1.0 / sy, 0.0, 0.0, 0.0, 1.0];
            mul(&mul(&m, h), &mi)
        };
        let mut out = self.clone();
        out.width = width;
        out.height = height;
        out.reference_corners = s(&self.reference_corners);
        for f in out.frames.values_mut() {
            f.corners = s(&f.corners);
            f.homography = f.homography.as_ref().map(scale_h);
            if let Some(t) = f.transform.as_mut() {
                t.position = [t.position[0] * sx, t.position[1] * sy];
            }
        }
        out
    }
}

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    r
}

pub fn apply(m: &Mat3, p: Point) -> Point {
    let w = m[6] * p[0] + m[7] * p[1] + m[8];
    let w = if w.abs() < 1e-12 { 1e-12 } else { w };
    [
        (m[0] * p[0] + m[1] * p[1] + m[2]) / w,
        (m[3] * p[0] + m[4] * p[1] + m[5]) / w,
    ]
}

pub fn invert(m: &Mat3) -> Option<Mat3> {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let co0 = e * i - f * h;
    let co1 = -(d * i - f * g);
    let co2 = d * h - e * g;
    let det = a * co0 + b * co1 + c * co2;
    if det.abs() < 1e-12 {
        return None;
    }
    let k = 1.0 / det;
    Some([
        co0 * k,
        -(b * i - c * h) * k,
        (b * f - c * e) * k,
        co1 * k,
        (a * i - c * g) * k,
        -(a * f - c * d) * k,
        co2 * k,
        -(a * h - b * g) * k,
        (a * e - b * d) * k,
    ])
}

/// Homography mapping four `src` points exactly onto four `dst` points.
pub fn homography(src: &Corners, dst: &Corners) -> Option<Mat3> {
    // 8x8 linear system with h33 = 1.
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (x, y) = (src[i][0], src[i][1]);
        let (u, v) = (dst[i][0], dst[i][1]);
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v];
    }
    for col in 0..8 {
        let piv = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        for row in 0..8 {
            if row != col {
                let f = a[row][col] / a[col][col];
                for k in col..9 {
                    a[row][k] -= f * a[col][k];
                }
            }
        }
    }
    let mut h = [0.0; 9];
    for i in 0..8 {
        h[i] = a[i][8] / a[i][i];
    }
    h[8] = 1.0;
    h.iter().all(|v| v.is_finite()).then_some(h)
}

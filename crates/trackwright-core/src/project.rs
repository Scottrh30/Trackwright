//! Project document: footage, layers, shapes, surfaces and per-frame track data.

use crate::geometry::{
    apply, homography_from_corners, invert, mul, normalize, Mat3, MotionModel, Point, IDENTITY,
};
use crate::shape::{default_surface, Shape};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PROJECT_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Footage {
    /// File or folder the user opened.
    pub source: String,
    /// One image file per frame (video is decoded to a frame cache).
    pub frames: Vec<String>,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    /// Frame number shown for index 0 (e.g. 1001 for VFX plates).
    pub first_frame: i64,
}

impl Footage {
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
    pub fn frame_number(&self, index: usize) -> i64 {
        self.first_frame + index as i64
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TrackSettings {
    pub model: MotionModel,
    pub max_features: usize,
    pub min_distance: f32,
    pub feature_quality: f32,
    pub search_radius: i32,
    pub pyramid_levels: usize,
    /// RANSAC inlier threshold in pixels.
    pub ransac_threshold: f64,
    /// Forward-backward consistency tolerance in pixels.
    pub fb_threshold: f64,
    pub min_inliers: usize,
    /// Re-solve each frame directly against the reference frame to cancel drift.
    pub refine_against_reference: bool,
    /// Largest correction (pixels) the reference refinement may apply.
    pub max_refine_correction: f64,
    /// Compensate brightness/contrast changes (fades, flicker, moving light).
    pub illumination: bool,
    /// When a frame fails, retry it with a wider search (helps with motion blur).
    pub retry_hard_frames: bool,
    /// Resolution for the fast frame-to-frame pass: 0 = full, 1 = half, 2 = quarter.
    /// `None` picks automatically (half for footage 2.5K and wider).
    pub detail_level: Option<usize>,
    /// Occlusion solver: when the surface is covered (a blink, something passing in
    /// front) keep it moving with what's around it, then find it again and join up.
    pub bridge_occlusions: bool,
    /// Longest stretch (frames) the occlusion solver may predict before giving up.
    pub max_bridge_frames: usize,
}

impl Default for TrackSettings {
    fn default() -> Self {
        Self {
            model: MotionModel::Perspective,
            max_features: 400,
            min_distance: 7.0,
            feature_quality: 0.01,
            search_radius: 7,
            pyramid_levels: 4,
            ransac_threshold: 1.5,
            fb_threshold: 1.0,
            min_inliers: 10,
            refine_against_reference: true,
            max_refine_correction: 12.0,
            illumination: true,
            retry_hard_frames: true,
            detail_level: None,
            bridge_occlusions: true,
            max_bridge_frames: 48,
        }
    }
}

/// Track result for one frame. `h_from_ref` maps reference-frame pixels to this frame.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameTrack {
    pub h_from_ref: Mat3,
    pub inliers: u32,
    pub rms_error: f64,
    /// Inlier feature positions in this frame (for overlay display).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub points: Vec<Point>,
    /// True for the reference frame or frames the user fixed by hand.
    #[serde(default)]
    pub keyframe: bool,
    /// Solved only on a second, wider attempt (often motion blur).
    #[serde(default)]
    pub retried: bool,
    /// Not seen: predicted by the occlusion solver while the surface was covered.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub predicted: bool,
}

impl FrameTrack {
    pub fn reference() -> Self {
        Self {
            h_from_ref: IDENTITY,
            inliers: 0,
            rms_error: 0.0,
            points: vec![],
            keyframe: true,
            retried: false,
            predicted: false,
        }
    }
}

/// Why a tracked frame looks unreliable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Problem {
    HighError,
    FewPoints,
    Retried,
    Predicted,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Layer {
    pub id: u32,
    pub name: String,
    pub color: String,
    pub visible: bool,
    /// Frame index at which `shape` and `surface` were drawn.
    pub ref_frame: usize,
    pub shape: Shape,
    /// Corner-pin surface at the reference frame: TL, TR, BR, BL.
    pub surface: [Point; 4],
    pub settings: TrackSettings,
    pub tracks: BTreeMap<usize, FrameTrack>,
    /// Follow another layer's motion instead of tracking on its own.
    #[serde(default)]
    pub link: Option<u32>,
    /// Image or clip pinned to the surface for previewing the track.
    #[serde(default)]
    pub insert: Option<Insert>,
    /// Layers whose shapes this layer's tracker ignores (occluders, reflections...).
    #[serde(default)]
    pub exclusions: Vec<u32>,
    /// Animated shape: keyframes of the shape in reference-frame coordinates.
    /// Empty = the shape is fixed to the plane (`shape`).
    #[serde(default)]
    pub shape_keys: BTreeMap<usize, Shape>,
    /// Adjust Track: per-frame corner offsets (frame pixels, TL TR BR BL) added to the
    /// tracked surface, interpolated between keys and held outside them.
    #[serde(default)]
    pub adjust_keys: BTreeMap<usize, [Point; 4]>,
    /// Soft edge for mattes and masks, in pixels.
    #[serde(default)]
    pub feather: f64,
}

/// Media corner-pinned onto a layer's surface (Mocha-style "insert").
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Insert {
    pub source: String,
    /// One file per insert frame (a still image has one).
    pub frames: Vec<String>,
    pub width: u32,
    pub height: u32,
    pub opacity: f64,
    /// Footage frame index at which insert frame 0 plays.
    pub start_frame: usize,
    pub looped: bool,
}

impl Insert {
    /// Insert frame shown at footage frame `frame`.
    pub fn frame_for(&self, frame: usize) -> Option<&str> {
        let n = self.frames.len() as i64;
        if n == 0 {
            return None;
        }
        let i = frame as i64 - self.start_frame as i64;
        let idx = if n == 1 {
            0
        } else if self.looped {
            i.rem_euclid(n)
        } else {
            i.clamp(0, n - 1)
        };
        self.frames.get(idx as usize).map(String::as_str)
    }
}

const LAYER_COLORS: [&str; 6] = [
    "#ffb020", "#3fd0ff", "#ff5fa2", "#7dff6a", "#b28bff", "#ff7a45",
];

impl Layer {
    pub fn new(id: u32, ref_frame: usize, shape: Shape) -> Self {
        let surface = default_surface(&shape);
        let mut tracks = BTreeMap::new();
        tracks.insert(ref_frame, FrameTrack::reference());
        Self {
            id,
            name: format!("Layer {id}"),
            color: Self::color_for(id),
            visible: true,
            ref_frame,
            shape,
            surface,
            settings: TrackSettings::default(),
            tracks,
            link: None,
            insert: None,
            exclusions: vec![],
            shape_keys: BTreeMap::new(),
            adjust_keys: BTreeMap::new(),
            feather: 0.0,
        }
    }

    pub fn color_for(id: u32) -> String {
        LAYER_COLORS[(id as usize).wrapping_sub(1) % LAYER_COLORS.len()].to_string()
    }

    pub fn h_at(&self, frame: usize) -> Option<Mat3> {
        self.tracks.get(&frame).map(|t| t.h_from_ref)
    }

    /// Shape in reference coordinates at `frame` (interpolates shape keyframes).
    pub fn shape_ref_at(&self, frame: usize) -> Shape {
        if self.shape_keys.is_empty() {
            return self.shape.clone();
        }
        let before = self.shape_keys.range(..=frame).next_back();
        let after = self.shape_keys.range(frame..).next();
        match (before, after) {
            (Some((&fa, a)), Some((&fb, b))) if fa != fb && a.points.len() == b.points.len() => {
                let t = (frame - fa) as f64 / (fb - fa) as f64;
                let mut s = a.clone();
                for (p, q) in s.points.iter_mut().zip(&b.points) {
                    *p = [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
                }
                s
            }
            (Some((_, a)), _) => a.clone(),
            (None, Some((_, b))) => b.clone(),
            _ => self.shape.clone(),
        }
    }

    pub fn shape_at(&self, frame: usize) -> Option<Shape> {
        self.h_at(frame)
            .map(|h| self.shape_ref_at(frame).transformed(&h))
    }

    /// Shape at `frame`, or (if that frame isn't tracked) where it sits at the nearest
    /// tracked frame. Untracked exclusion layers simply hold still this way.
    pub fn shape_at_or_nearest(&self, frame: usize) -> Option<Shape> {
        if let Some(s) = self.shape_at(frame) {
            return Some(s);
        }
        let before = self.tracks.range(..frame).next_back().map(|(f, _)| *f);
        let after = self.tracks.range(frame..).next().map(|(f, _)| *f);
        let nearest = match (before, after) {
            (Some(a), Some(b)) => Some(if frame - a <= b - frame { a } else { b }),
            (a, b) => a.or(b),
        }?;
        let h = self.h_at(nearest)?;
        Some(self.shape_ref_at(frame).transformed(&h))
    }

    /// Corner offsets from Adjust Track keys at `frame` (zero when there are none).
    pub fn adjust_at(&self, frame: usize) -> [Point; 4] {
        let before = self.adjust_keys.range(..=frame).next_back();
        let after = self.adjust_keys.range(frame..).next();
        match (before, after) {
            (Some((&fa, a)), Some((&fb, b))) if fa != fb => {
                let t = (frame - fa) as f64 / (fb - fa) as f64;
                std::array::from_fn(|i| {
                    [
                        a[i][0] + (b[i][0] - a[i][0]) * t,
                        a[i][1] + (b[i][1] - a[i][1]) * t,
                    ]
                })
            }
            (Some((_, a)), _) => *a,
            (None, Some((_, b))) => *b,
            _ => [[0.0; 2]; 4],
        }
    }

    /// Surface corners from tracking alone (no Adjust Track).
    pub fn tracked_surface_at(&self, frame: usize) -> Option<[Point; 4]> {
        self.h_at(frame).map(|h| self.surface.map(|p| apply(&h, p)))
    }

    /// Final surface corners at `frame`, including Adjust Track corrections.
    pub fn surface_at(&self, frame: usize) -> Option<[Point; 4]> {
        let c = self.tracked_surface_at(frame)?;
        if self.adjust_keys.is_empty() {
            return Some(c);
        }
        let o = self.adjust_at(frame);
        Some(std::array::from_fn(|i| {
            [c[i][0] + o[i][0], c[i][1] + o[i][1]]
        }))
    }

    /// Reference-to-frame transform including Adjust Track corrections.
    pub fn final_h_at(&self, frame: usize) -> Option<Mat3> {
        let h = self.h_at(frame)?;
        if self.adjust_keys.is_empty() {
            return Some(h);
        }
        let corners = self.surface_at(frame)?;
        Some(homography_from_corners(&self.surface, &corners).unwrap_or(h))
    }

    /// Move one surface corner at `frame`, keeping the reference surface consistent.
    pub fn set_surface_corner(&mut self, frame: usize, corner: usize, pos: Point) -> bool {
        let Some(h) = self.h_at(frame) else {
            return false;
        };
        let Some(hi) = invert(&h) else { return false };
        let o = self.adjust_at(frame)[corner % 4];
        self.surface[corner % 4] = apply(&hi, [pos[0] - o[0], pos[1] - o[1]]);
        true
    }

    /// Adjust Track: put one corner at `pos` on this frame by setting an adjust key.
    pub fn set_adjust_corner(&mut self, frame: usize, corner: usize, pos: Point) -> bool {
        let Some(t) = self.tracked_surface_at(frame) else {
            return false;
        };
        let mut off = self.adjust_at(frame);
        off[corner % 4] = [pos[0] - t[corner % 4][0], pos[1] - t[corner % 4][1]];
        self.adjust_keys.insert(frame, off);
        true
    }

    /// Adjust Track: lock the current corners here (a key with the current offsets).
    pub fn add_adjust_key(&mut self, frame: usize) -> bool {
        if !self.tracks.contains_key(&frame) {
            return false;
        }
        let off = self.adjust_at(frame);
        self.adjust_keys.insert(frame, off);
        true
    }

    /// Replace the shape as drawn at `frame`. With shape keyframes this sets a key.
    pub fn set_shape_at(&mut self, frame: usize, shape_at_frame: Shape) -> bool {
        let Some(h) = self.h_at(frame) else {
            return false;
        };
        let Some(hi) = invert(&h) else { return false };
        let shape = shape_at_frame.transformed(&hi);
        if self.shape_keys.is_empty() {
            self.shape = shape;
        } else {
            self.shape_keys.insert(frame, shape);
        }
        true
    }

    /// Start (or extend) shape animation with a key holding the current shape.
    pub fn add_shape_key(&mut self, frame: usize) -> bool {
        if !self.tracks.contains_key(&frame) {
            return false;
        }
        let s = self.shape_ref_at(frame);
        self.shape_keys.insert(frame, s);
        true
    }

    pub fn remove_shape_key(&mut self, frame: usize) -> bool {
        let Some(removed) = self.shape_keys.remove(&frame) else {
            return false;
        };
        if self.shape_keys.is_empty() {
            self.shape = removed;
        }
        true
    }

    /// Stop animating; keep the shape as it is at `frame`.
    pub fn clear_shape_keys(&mut self, frame: usize) {
        self.shape = self.shape_ref_at(frame);
        self.shape_keys.clear();
    }

    /// Insert a control point after `segment`, placed at `pos` (frame coordinates) on
    /// this frame and at the matching spot along the segment on every shape key.
    pub fn insert_shape_point(&mut self, frame: usize, segment: usize, pos: Point) -> bool {
        let Some(hi) = self.h_at(frame).and_then(|h| invert(&h)) else {
            return false;
        };
        let q = apply(&hi, pos);
        let cur = self.shape_ref_at(frame);
        let n = cur.points.len();
        if n < 2 {
            return false;
        }
        let (a, b) = (cur.points[segment % n], cur.points[(segment + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 {
            (((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let at = segment % n + 1;
        let lerp_into = |s: &mut Shape| {
            let m = s.points.len();
            if m == n {
                let (a, b) = (s.points[segment % m], s.points[(segment + 1) % m]);
                s.points
                    .insert(at, [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
            }
        };
        lerp_into(&mut self.shape);
        for k in self.shape_keys.values_mut() {
            lerp_into(k);
        }
        if let Some(k) = self.shape_keys.get_mut(&frame) {
            k.points[at] = q;
        } else if self.shape_keys.is_empty() {
            self.shape.points[at] = q;
        }
        true
    }

    /// Remove control point `index` from the shape and all its keys (min 3 points).
    pub fn remove_shape_point(&mut self, index: usize) -> bool {
        if self.shape.points.len() <= 3 || index >= self.shape.points.len() {
            return false;
        }
        self.shape.points.remove(index);
        for k in self.shape_keys.values_mut() {
            if index < k.points.len() {
                k.points.remove(index);
            }
        }
        true
    }

    /// Fewer good points than this is worth a look: normally twice the minimum, but
    /// for a small target that never has many (an eye, a pupil) half what it usually
    /// gets, so the warning still means something. Sampled, so it stays cheap.
    pub fn few_points_limit(&self) -> usize {
        let base = (self.settings.min_inliers * 2).max(16);
        let seen: Vec<u32> = self
            .tracks
            .values()
            .filter(|t| !t.predicted && !t.keyframe)
            .map(|t| t.inliers)
            .collect();
        if seen.len() < 8 {
            return base;
        }
        let stride = (seen.len() / 64).max(1);
        let mut sample: Vec<u32> = seen.iter().step_by(stride).copied().collect();
        sample.sort_unstable();
        let typical = sample[sample.len() / 2] as usize;
        base.min((typical / 2).max(self.settings.min_inliers + 2))
    }

    /// Frames that look unreliable, for the quality graph and "next problem" button.
    pub fn problem_at(&self, frame: usize) -> Option<Problem> {
        let t = self.tracks.get(&frame)?;
        if t.keyframe || frame == self.ref_frame {
            return None;
        }
        if t.predicted {
            Some(Problem::Predicted)
        } else if t.rms_error > 1.0 {
            Some(Problem::HighError)
        } else if (t.inliers as usize) < self.few_points_limit() {
            Some(Problem::FewPoints)
        } else if t.retried {
            Some(Problem::Retried)
        } else {
            None
        }
    }

    /// Make `frame` the new reference: all homographies are re-expressed relative to it.
    pub fn rebase(&mut self, frame: usize) -> bool {
        let Some(h) = self.h_at(frame) else {
            return false;
        };
        let Some(hi) = invert(&h) else { return false };
        self.shape = self.shape.transformed(&h);
        for k in self.shape_keys.values_mut() {
            *k = k.transformed(&h);
        }
        self.surface = self.surface.map(|p| apply(&h, p));
        for t in self.tracks.values_mut() {
            t.h_from_ref = mul(&t.h_from_ref, &hi);
            t.keyframe = false;
        }
        if let Some(t) = self.tracks.get_mut(&frame) {
            t.h_from_ref = IDENTITY;
            t.keyframe = true;
        }
        self.ref_frame = frame;
        true
    }

    /// Drop all tracking except the reference frame.
    pub fn clear_tracks(&mut self) {
        self.tracks.clear();
        self.tracks.insert(self.ref_frame, FrameTrack::reference());
    }

    /// Drop tracking strictly after (`forward`) or before a frame.
    pub fn clear_from(&mut self, frame: usize, forward: bool) {
        let rf = self.ref_frame;
        self.tracks
            .retain(|&f, _| f == rf || if forward { f <= frame } else { f >= frame });
    }

    pub fn tracked_range(&self) -> Option<(usize, usize)> {
        Some((
            *self.tracks.keys().next()?,
            *self.tracks.keys().next_back()?,
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Project {
    pub version: u32,
    pub footage: Option<Footage>,
    pub layers: Vec<Layer>,
    pub next_layer_id: u32,
    /// Auto point tracks and the camera solve.
    #[serde(default)]
    pub camera: crate::sfm::CameraData,
    /// Lens distortion of the footage (Lens panel).
    #[serde(default)]
    pub lens: crate::lens::Lens,
    /// Lines drawn along edges that should be straight, for lens calibration (frame, points).
    #[serde(default)]
    pub lens_lines: Vec<(usize, Vec<Point>)>,
}

impl Default for Project {
    fn default() -> Self {
        Self {
            version: PROJECT_VERSION,
            footage: None,
            layers: vec![],
            next_layer_id: 1,
            camera: Default::default(),
            lens: Default::default(),
            lens_lines: vec![],
        }
    }
}

impl Project {
    pub fn add_layer(&mut self, ref_frame: usize, shape: Shape) -> u32 {
        let id = self.next_layer_id;
        self.next_layer_id += 1;
        self.layers.push(Layer::new(id, ref_frame, shape));
        id
    }

    pub fn layer(&self, id: u32) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn layer_mut(&mut self, id: u32) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    /// Copy a layer (shape, surface, settings, track, insert) under a new id.
    pub fn duplicate_layer(&mut self, id: u32) -> Option<u32> {
        let mut copy = self.layer(id)?.clone();
        let new_id = self.next_layer_id;
        self.next_layer_id += 1;
        copy.id = new_id;
        copy.name = format!("{} copy", copy.name);
        copy.color = Layer::color_for(new_id);
        let pos = self
            .layers
            .iter()
            .position(|l| l.id == id)
            .map_or(self.layers.len(), |p| p + 1);
        self.layers.insert(pos, copy);
        Some(new_id)
    }

    /// Would linking `child` to `parent` create a cycle?
    pub fn link_creates_cycle(&self, child: u32, parent: u32) -> bool {
        let mut cur = Some(parent);
        let mut guard = 0;
        while let Some(id) = cur {
            if id == child || guard > self.layers.len() {
                return true;
            }
            guard += 1;
            cur = self.layer(id).and_then(|l| l.link);
        }
        false
    }

    /// Recompute the track of every linked layer from its parent's motion.
    pub fn resolve_links(&mut self) {
        for _ in 0..=self.layers.len() {
            let mut changed = false;
            for i in 0..self.layers.len() {
                let Some(pid) = self.layers[i].link else {
                    continue;
                };
                let (cid, cref) = (self.layers[i].id, self.layers[i].ref_frame);
                if pid == cid || self.link_creates_cycle(cid, pid) {
                    self.layers[i].link = None;
                    changed = true;
                    continue;
                }
                let Some(parent) = self.layer(pid) else {
                    self.layers[i].link = None;
                    changed = true;
                    continue;
                };
                let tracks = linked_tracks(parent, cref);
                if self.layers[i].tracks != tracks {
                    self.layers[i].tracks = tracks;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("project serialises")
    }

    pub fn from_json(s: &str) -> Result<Self, String> {
        let p: Project =
            serde_json::from_str(s).map_err(|e| format!("Not a Trackwright project: {e}"))?;
        if p.version > PROJECT_VERSION {
            return Err(format!(
                "Project was saved by a newer Trackwright (format v{})",
                p.version
            ));
        }
        Ok(p)
    }
}

/// Outlines (frame coordinates) of `layer`'s exclusion layers at `frame`.
pub fn exclusion_polys(layers: &[Layer], layer: &Layer, frame: usize) -> Vec<Vec<Point>> {
    layer
        .exclusions
        .iter()
        .filter(|&&id| id != layer.id)
        .filter_map(|id| layers.iter().find(|l| l.id == *id))
        .filter_map(|l| l.shape_at_or_nearest(frame))
        .map(|s| s.flatten(8))
        .collect()
}

/// Track data for a layer whose reference frame is `child_ref`, moving with `parent`.
fn linked_tracks(parent: &Layer, child_ref: usize) -> BTreeMap<usize, FrameTrack> {
    let mut m = BTreeMap::new();
    let Some(hr) = parent.final_h_at(child_ref).and_then(|h| invert(&h)) else {
        m.insert(child_ref, FrameTrack::reference());
        return m;
    };
    for (&f, t) in &parent.tracks {
        let hp = parent.final_h_at(f).unwrap_or(t.h_from_ref);
        let h = if f == child_ref {
            IDENTITY
        } else {
            normalize(&mul(&hp, &hr))
        };
        m.insert(
            f,
            FrameTrack {
                h_from_ref: h,
                inliers: t.inliers,
                rms_error: t.rms_error,
                points: vec![],
                keyframe: f == child_ref,
                retried: t.retried,
                predicted: t.predicted,
            },
        );
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::apply;

    fn moving_parent() -> Layer {
        let mut p = Layer::new(1, 0, Shape::rect(0.0, 0.0, 100.0, 100.0));
        for f in 1..10 {
            let t = f as f64;
            p.tracks.insert(
                f,
                FrameTrack {
                    h_from_ref: [1.0, 0.0, 5.0 * t, 0.0, 1.0, 2.0 * t, 0.0, 0.0, 1.0],
                    ..FrameTrack::reference()
                },
            );
        }
        p
    }

    #[test]
    fn linked_layer_follows_parent() {
        let mut pr = Project::default();
        pr.layers.push(moving_parent());
        pr.next_layer_id = 2;
        // Child drawn at frame 4 follows the parent relative to that frame.
        let mut child = Layer::new(2, 4, Shape::rect(20.0, 20.0, 10.0, 10.0));
        child.link = Some(1);
        pr.layers.push(child);
        pr.next_layer_id = 3;
        pr.resolve_links();
        let c = pr.layer(2).unwrap();
        assert_eq!(c.tracks.len(), 10);
        let p4 = apply(&c.h_at(4).unwrap(), [20.0, 20.0]);
        let p9 = apply(&c.h_at(9).unwrap(), [20.0, 20.0]);
        assert!(
            (p4[0] - 20.0).abs() < 1e-9
                && (p9[0] - 45.0).abs() < 1e-9
                && (p9[1] - 30.0).abs() < 1e-9
        );
        // Cycles are refused.
        assert!(pr.link_creates_cycle(1, 2));
    }

    #[test]
    fn shape_keys_interpolate_and_points_sync() {
        let mut l = moving_parent();
        assert!(l.add_shape_key(0));
        let mut s9 = l.shape_at(9).unwrap();
        s9.points[0][0] += 10.0; // widen at frame 9
        assert!(l.set_shape_at(9, s9));
        let x0 = |f: usize| l.shape_ref_at(f).points[0][0];
        assert!((x0(0) - 0.0).abs() < 1e-9 && (x0(9) - 10.0).abs() < 1e-9);
        assert!((x0(3) - 10.0 / 3.0).abs() < 1e-9);
        // A point added on frame 9 shows up on every key.
        let p = l.shape_at(9).unwrap();
        let mid = [
            (p.points[1][0] + p.points[2][0]) / 2.0,
            (p.points[1][1] + p.points[2][1]) / 2.0,
        ];
        assert!(l.insert_shape_point(9, 1, mid));
        assert!(l.shape_keys.values().all(|k| k.points.len() == 5) && l.shape.points.len() == 5);
        assert!(l.remove_shape_point(2));
        assert!(l.shape_keys.values().all(|k| k.points.len() == 4));
    }

    #[test]
    fn adjust_track_offsets() {
        let mut l = moving_parent();
        let t5 = l.tracked_surface_at(5).unwrap();
        assert!(l.add_adjust_key(2));
        assert!(l.set_adjust_corner(8, 0, [t5[0][0], t5[0][1]]));
        // Key at 2 has zero offset, key at 8 moves corner 0; frame 5 is halfway.
        let off8 = l.adjust_at(8)[0];
        let off5 = l.adjust_at(5)[0];
        assert!((off5[0] - off8[0] / 2.0).abs() < 1e-9);
        // Final transform reproduces the adjusted corners.
        let h = l.final_h_at(8).unwrap();
        let c = l.surface_at(8).unwrap();
        let m = apply(&h, l.surface[0]);
        assert!((m[0] - c[0][0]).abs() < 1e-6 && (m[1] - c[0][1]).abs() < 1e-6);
        // Moving the surface on an adjusted frame stays where it was put.
        assert!(l.set_surface_corner(8, 1, [500.0, 7.0]));
        let c = l.surface_at(8).unwrap();
        assert!((c[1][0] - 500.0).abs() < 1e-6 && (c[1][1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn duplicate_and_insert_frames() {
        let mut pr = Project::default();
        let id = pr.add_layer(0, Shape::rect(0.0, 0.0, 5.0, 5.0));
        let d = pr.duplicate_layer(id).unwrap();
        assert_eq!(pr.layers.len(), 2);
        assert_eq!(pr.layer(d).unwrap().name, "Layer 1 copy");
        let ins = Insert {
            source: String::new(),
            frames: vec!["a".into(), "b".into(), "c".into()],
            width: 1,
            height: 1,
            opacity: 1.0,
            start_frame: 5,
            looped: true,
        };
        assert_eq!(ins.frame_for(5), Some("a"));
        assert_eq!(ins.frame_for(9), Some("b"));
        assert_eq!(ins.frame_for(4), Some("c"));
    }
}

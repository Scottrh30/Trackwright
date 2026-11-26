//! Project document: footage, layers, shapes, surfaces and per-frame track data.

use crate::geometry::{apply, invert, mul, normalize, Mat3, MotionModel, Point, IDENTITY};
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
}

impl FrameTrack {
    pub fn reference() -> Self {
        Self {
            h_from_ref: IDENTITY,
            inliers: 0,
            rms_error: 0.0,
            points: vec![],
            keyframe: true,
        }
    }
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
        }
    }

    pub fn color_for(id: u32) -> String {
        LAYER_COLORS[(id as usize).wrapping_sub(1) % LAYER_COLORS.len()].to_string()
    }

    pub fn h_at(&self, frame: usize) -> Option<Mat3> {
        self.tracks.get(&frame).map(|t| t.h_from_ref)
    }

    pub fn shape_at(&self, frame: usize) -> Option<Shape> {
        self.h_at(frame).map(|h| self.shape.transformed(&h))
    }

    pub fn surface_at(&self, frame: usize) -> Option<[Point; 4]> {
        self.h_at(frame).map(|h| self.surface.map(|p| apply(&h, p)))
    }

    /// Move one surface corner at `frame`, keeping the reference surface consistent.
    pub fn set_surface_corner(&mut self, frame: usize, corner: usize, pos: Point) -> bool {
        let Some(h) = self.h_at(frame) else {
            return false;
        };
        let Some(hi) = invert(&h) else { return false };
        self.surface[corner % 4] = apply(&hi, pos);
        true
    }

    /// Replace the shape as drawn at `frame`, re-basing it into the reference frame.
    pub fn set_shape_at(&mut self, frame: usize, shape_at_frame: Shape) -> bool {
        let Some(h) = self.h_at(frame) else {
            return false;
        };
        let Some(hi) = invert(&h) else { return false };
        self.shape = shape_at_frame.transformed(&hi);
        true
    }

    /// Make `frame` the new reference: all homographies are re-expressed relative to it.
    pub fn rebase(&mut self, frame: usize) -> bool {
        let Some(h) = self.h_at(frame) else {
            return false;
        };
        let Some(hi) = invert(&h) else { return false };
        self.shape = self.shape.transformed(&h);
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
}

impl Default for Project {
    fn default() -> Self {
        Self {
            version: PROJECT_VERSION,
            footage: None,
            layers: vec![],
            next_layer_id: 1,
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

/// Track data for a layer whose reference frame is `child_ref`, moving with `parent`.
fn linked_tracks(parent: &Layer, child_ref: usize) -> BTreeMap<usize, FrameTrack> {
    let mut m = BTreeMap::new();
    let Some(hr) = parent.h_at(child_ref).and_then(|h| invert(&h)) else {
        m.insert(child_ref, FrameTrack::reference());
        return m;
    };
    for (&f, t) in &parent.tracks {
        let h = if f == child_ref {
            IDENTITY
        } else {
            normalize(&mul(&t.h_from_ref, &hr))
        };
        m.insert(
            f,
            FrameTrack {
                h_from_ref: h,
                inliers: t.inliers,
                rms_error: t.rms_error,
                points: vec![],
                keyframe: f == child_ref,
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

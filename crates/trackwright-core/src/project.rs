//! Project document: footage, layers, shapes, surfaces and per-frame track data.

use crate::geometry::{apply, invert, mul, Mat3, MotionModel, Point, IDENTITY};
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
            color: LAYER_COLORS[(id as usize).wrapping_sub(1) % LAYER_COLORS.len()].to_string(),
            visible: true,
            ref_frame,
            shape,
            surface,
            settings: TrackSettings::default(),
            tracks,
        }
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

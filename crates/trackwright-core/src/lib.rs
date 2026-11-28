//! Trackwright core: planar motion tracking engine.
//!
//! © Scott Hambach. All rights reserved.

pub mod ai;
pub mod alembic;
pub mod camera_export;
pub mod epipolar;
pub mod export;
pub mod fbx;
pub mod features;
pub mod geometry;
pub mod image;
pub mod klt;
pub mod learning;
pub mod lens;
pub mod linalg;
pub mod mesh;
mod par;
pub mod planar;
pub mod points;
pub mod project;
pub mod pupil;
pub mod render;
pub mod sfm;
pub mod shape;

pub use export::ExportFormat;
pub use geometry::{Mat3, MotionModel, Point};
pub use image::{GrayImage, Pyramid};
pub use learning::{LearningStore, Tuning};
pub use planar::{
    no_exclusions, retrack_section, track_layer, FrameSource, StopReason, TrackError,
};
pub use project::{
    exclusion_polys, Footage, FrameTrack, Layer, Problem, Project, RotationMode, TargetKind,
    TrackSettings,
};
pub use shape::Shape;

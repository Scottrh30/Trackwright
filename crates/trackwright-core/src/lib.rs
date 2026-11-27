//! Trackwright core: planar motion tracking engine.
//!
//! © Scott Hambach. All rights reserved.

pub mod export;
pub mod features;
pub mod geometry;
pub mod image;
pub mod klt;
mod par;
pub mod planar;
pub mod project;
pub mod shape;

pub use export::ExportFormat;
pub use geometry::{Mat3, MotionModel, Point};
pub use image::{GrayImage, Pyramid};
pub use planar::{
    no_exclusions, retrack_section, track_layer, FrameSource, StopReason, TrackError,
};
pub use project::{exclusion_polys, Footage, FrameTrack, Layer, Problem, Project, TrackSettings};
pub use shape::Shape;

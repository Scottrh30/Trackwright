//! AI point tracking (optional).
//!
//! A learned point tracker (AllTracker) follows points through blinks, hands, fast
//! moves and blur far better than any hand-made rule, at the cost of speed. The
//! engine doesn't depend on it: an app that has the model loaded registers it here,
//! and the occlusion solver uses it (when the layer asks) to predict where a
//! covered surface is.

use crate::geometry::Point;
use std::sync::{Arc, RwLock};

/// Tracks points through a short run of frames.
pub trait PointTracker: Send + Sync {
    /// Width and height of the frames it expects.
    fn frame_size(&self) -> (usize, usize);
    /// `frames`: grey frames (0..255, row by row, `frame_size`), the first is where
    /// `points` are. Returns, for every frame, each point's position and how sure it
    /// is that the point is visible (0..1).
    fn track(
        &self,
        frames: &[Vec<f32>],
        points: &[Point],
    ) -> Result<Vec<Vec<(Point, f32)>>, String>;
}

static TRACKER: RwLock<Option<Arc<dyn PointTracker>>> = RwLock::new(None);

/// Make a point tracker available to the engine (`None` removes it).
pub fn set_point_tracker(t: Option<Arc<dyn PointTracker>>) {
    *TRACKER.write().unwrap() = t;
}

/// The registered point tracker, if any.
pub fn point_tracker() -> Option<Arc<dyn PointTracker>> {
    TRACKER.read().unwrap().clone()
}

//! The planar tracker: follows a layer's shape region frame to frame and solves
//! a reference-to-frame transform for its surface, Mocha style.

use crate::features::{good_features, FeatureParams};
use crate::geometry::{apply, invert, mul, ransac, Mat3, Point, IDENTITY};
use crate::image::{GrayImage, Pyramid};
use crate::klt::{track_point, KltParams};
use crate::project::{FrameTrack, Layer, TrackSettings};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Supplies frame pyramids to the tracker (implemented by the app's frame cache).
pub trait FrameSource {
    fn frame_count(&self) -> usize;
    fn pyramid(&mut self, index: usize, levels: usize) -> Result<Arc<Pyramid>, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackError {
    NotTracked(usize),
    InvalidShape,
    NoFeatures(usize),
    Lost { frame: usize, inliers: usize },
    Source(String),
}

impl fmt::Display for TrackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrackError::NotTracked(fr) => {
                write!(f, "Frame {fr} has no track yet; start from a tracked frame")
            }
            TrackError::InvalidShape => {
                write!(f, "Draw a closed shape with at least 3 points first")
            }
            TrackError::NoFeatures(fr) => {
                write!(f, "Not enough texture inside the shape at frame {fr}")
            }
            TrackError::Lost { frame, inliers } => {
                write!(f, "Track lost at frame {frame} ({inliers} good points). Adjust the shape or surface and continue")
            }
            TrackError::Source(e) => write!(f, "Could not read footage: {e}"),
        }
    }
}

impl std::error::Error for TrackError {}

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    Finished,
    Cancelled,
}

fn feature_params(s: &TrackSettings) -> FeatureParams {
    FeatureParams {
        max_features: s.max_features,
        quality: s.feature_quality,
        min_distance: s.min_distance,
        ..Default::default()
    }
}

fn klt_params(s: &TrackSettings) -> KltParams {
    KltParams {
        win_radius: s.search_radius,
        ..Default::default()
    }
}

struct Correspondences {
    src: Vec<Point>,
    dst: Vec<Point>,
}

/// Track `pts` from `a` into `b`, with optional per-point guesses and a forward-backward check.
fn track_points(
    a: &Pyramid,
    b: &Pyramid,
    pts: &[Point],
    guess: &dyn Fn(Point) -> Point,
    s: &TrackSettings,
    fb: bool,
) -> Correspondences {
    let kp = klt_params(s);
    let mut src = Vec::with_capacity(pts.len());
    let mut dst = Vec::with_capacity(pts.len());
    for &p in pts {
        let Some((q, _)) = track_point(a, b, p, guess(p), &kp) else {
            continue;
        };
        if fb {
            let Some((back, _)) = track_point(b, a, q, p, &kp) else {
                continue;
            };
            if (back[0] - p[0]).powi(2) + (back[1] - p[1]).powi(2) > s.fb_threshold * s.fb_threshold
            {
                continue;
            }
        }
        src.push(p);
        dst.push(q);
    }
    Correspondences { src, dst }
}

fn max_corner_delta(surface: &[Point; 4], h1: &Mat3, h2: &Mat3) -> f64 {
    surface
        .iter()
        .map(|&p| {
            let (a, b) = (apply(h1, p), apply(h2, p));
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        })
        .fold(0.0, f64::max)
}

/// Render the reference image as it should appear in a frame under `h` (ref -> frame),
/// only around the shape. Pixels outside that area are left black.
fn warp_reference(
    reference: &GrayImage,
    h: &Mat3,
    poly: &[Point],
    w: usize,
    hgt: usize,
) -> Option<GrayImage> {
    let hi = invert(h)?;
    let b = crate::shape::bbox(poly)?;
    let margin = 24.0;
    let x0 = (b[0] - margin).floor().max(0.0) as usize;
    let y0 = (b[1] - margin).floor().max(0.0) as usize;
    let x1 = ((b[2] + margin).ceil().max(0.0) as usize).min(w);
    let y1 = ((b[3] + margin).ceil().max(0.0) as usize).min(hgt);
    if x1 <= x0 + 4 || y1 <= y0 + 4 {
        return None;
    }
    let mut out = GrayImage::new(w, hgt);
    for y in y0..y1 {
        for x in x0..x1 {
            let p = apply(&hi, [x as f64, y as f64]);
            out.data[y * w + x] = reference.sample(p[0] as f32, p[1] as f32);
        }
    }
    Some(out)
}

/// Track `layer` from tracked frame `start` towards `end` (either direction, inclusive).
/// `on_frame` is called after each solved frame. Existing data in the range is overwritten.
pub fn track_layer<S: FrameSource>(
    source: &mut S,
    layer: &mut Layer,
    start: usize,
    end: usize,
    cancel: &AtomicBool,
    mut on_frame: impl FnMut(usize, &FrameTrack),
) -> Result<StopReason, TrackError> {
    if !layer.shape.is_valid() {
        return Err(TrackError::InvalidShape);
    }
    let count = source.frame_count();
    let end = end.min(count.saturating_sub(1));
    let mut h_cur = layer.h_at(start).ok_or(TrackError::NotTracked(start))?;
    if start == end {
        return Ok(StopReason::Finished);
    }
    let s = layer.settings.clone();
    let fparams = feature_params(&s);
    let step: isize = if end > start { 1 } else { -1 };
    let levels = s.pyramid_levels;

    // Reference frame for drift-cancelling refinement.
    let ref_pyr = source
        .pyramid(layer.ref_frame, levels)
        .map_err(TrackError::Source)?;

    let mut velocity: Mat3 = IDENTITY;
    let mut cur = start;
    let seed_base = layer.id as u64 * 1_000_003;

    while cur != end {
        if cancel.load(Ordering::Relaxed) {
            return Ok(StopReason::Cancelled);
        }
        let next = (cur as isize + step) as usize;
        let prev_pyr = source.pyramid(cur, levels).map_err(TrackError::Source)?;
        let next_pyr = source.pyramid(next, levels).map_err(TrackError::Source)?;

        // 1. Frame-to-frame: features inside the shape as it sits in the current frame.
        let poly = layer.shape.transformed(&h_cur).flatten(8);
        let feats = good_features(
            &prev_pyr.levels[0],
            &prev_pyr.grad_x[0],
            &prev_pyr.grad_y[0],
            Some(&poly),
            &fparams,
        );
        if feats.len() < s.min_inliers {
            return Err(TrackError::NoFeatures(cur));
        }
        let vel = velocity;
        let c = track_points(
            &prev_pyr,
            &next_pyr,
            &feats,
            &move |p| apply(&vel, p),
            &s,
            true,
        );
        let fit = ransac(
            s.model,
            &c.src,
            &c.dst,
            s.ransac_threshold,
            seed_base + next as u64,
        );
        let fit = match fit {
            Some(f) if f.inlier_count >= s.min_inliers => f,
            other => {
                return Err(TrackError::Lost {
                    frame: next,
                    inliers: other.map_or(0, |f| f.inlier_count),
                })
            }
        };
        let mut h_next = mul(&fit.model, &h_cur);
        let mut inliers = fit.inlier_count;
        let mut rms = fit.rms_error;
        let mut shown: Vec<Point> = c
            .dst
            .iter()
            .zip(&fit.inliers)
            .filter(|(_, &i)| i)
            .map(|(p, _)| *p)
            .collect();

        // 2. Drift cancellation: warp the reference frame onto this frame with the
        //    current estimate and solve the small residual motion. Warping first keeps
        //    patches comparable even after large rotation, scale or perspective change.
        if s.refine_against_reference && next != layer.ref_frame {
            let shape_next = layer.shape.transformed(&h_next);
            let poly_next = shape_next.flatten(8);
            if let Some(warped) = warp_reference(
                &ref_pyr.levels[0],
                &h_next,
                &poly_next,
                next_pyr.width(),
                next_pyr.height(),
            ) {
                let wp = Pyramid::build(warped, levels);
                let wf = good_features(
                    &wp.levels[0],
                    &wp.grad_x[0],
                    &wp.grad_y[0],
                    Some(&poly_next),
                    &fparams,
                );
                if wf.len() >= s.min_inliers {
                    let rc = track_points(&wp, &next_pyr, &wf, &|p| p, &s, false);
                    if let Some(rf) = ransac(
                        s.model,
                        &rc.src,
                        &rc.dst,
                        s.ransac_threshold,
                        seed_base ^ (next as u64) << 20,
                    ) {
                        let candidate = mul(&rf.model, &h_next);
                        let enough = rf.inlier_count >= s.min_inliers.max(wf.len() * 3 / 10);
                        if enough
                            && max_corner_delta(&layer.surface, &candidate, &h_next)
                                <= s.max_refine_correction
                        {
                            h_next = candidate;
                            inliers = rf.inlier_count;
                            rms = rf.rms_error;
                            shown = rc
                                .dst
                                .iter()
                                .zip(&rf.inliers)
                                .filter(|(_, &i)| i)
                                .map(|(p, _)| *p)
                                .collect();
                        }
                    }
                }
            }
        }

        shown.truncate(250);
        let ft = FrameTrack {
            h_from_ref: h_next,
            inliers: inliers as u32,
            rms_error: rms,
            points: shown,
            keyframe: false,
        };
        velocity = fit.model;
        layer.tracks.insert(next, ft.clone());
        on_frame(next, &ft);
        h_cur = h_next;
        cur = next;
    }
    Ok(StopReason::Finished)
}

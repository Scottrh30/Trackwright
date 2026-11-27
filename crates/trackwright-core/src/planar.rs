//! The planar tracker: follows a layer's shape region frame to frame and solves
//! a reference-to-frame transform for its surface, Mocha style.

use crate::features::{good_features, FeatureParams};
use crate::geometry::{apply, homography_from_corners, invert, mul, ransac, Mat3, Point, IDENTITY};
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
            TrackError::NotTracked(fr) => write!(f, "Frame {fr} has no track yet; start from a tracked frame"),
            TrackError::InvalidShape => write!(f, "Draw a closed shape with at least 3 points first"),
            TrackError::NoFeatures(fr) => write!(
                f,
                "Not enough texture inside the shape at frame {fr}. Enlarge the shape or check exclusions"
            ),
            TrackError::Lost { frame, inliers } => write!(
                f,
                "Track lost at frame {frame} ({inliers} good points). Fix it with Adjust (A), add an exclusion, or reshape and continue"
            ),
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

/// Exclusion outlines (frame coordinates) for a frame.
pub type ExcludeFn<'a> = &'a dyn Fn(usize) -> Vec<Vec<Point>>;

/// No exclusions.
pub fn no_exclusions(_: usize) -> Vec<Vec<Point>> {
    vec![]
}

/// Parameters for one attempt; the retry pass loosens these.
#[derive(Clone)]
struct Pass {
    features: FeatureParams,
    klt: KltParams,
    ransac_threshold: f64,
    fb_threshold: f64,
    /// Pyramid level used for the frame-to-frame pass.
    level: usize,
}

fn auto_level(s: &TrackSettings, width: usize) -> usize {
    s.detail_level.unwrap_or(if width >= 5000 {
        2
    } else if width >= 2500 {
        1
    } else {
        0
    })
}

fn normal_pass(s: &TrackSettings, level: usize) -> Pass {
    let scale = (1u32 << level) as f32;
    Pass {
        features: FeatureParams {
            max_features: s.max_features,
            quality: s.feature_quality,
            min_distance: (s.min_distance / scale).max(3.0),
            exclude_margin: (s.search_radius as f32 + 1.0) / scale,
            ..Default::default()
        },
        klt: KltParams {
            win_radius: s.search_radius,
            illumination: s.illumination,
            base_level: level,
            ..Default::default()
        },
        ransac_threshold: s.ransac_threshold * scale as f64,
        fb_threshold: s.fb_threshold * scale as f64,
        level,
    }
}

/// Wider, more forgiving search at full resolution for blurry or difficult frames.
fn retry_pass(s: &TrackSettings) -> Pass {
    Pass {
        features: FeatureParams {
            max_features: s.max_features * 2,
            quality: (s.feature_quality * 0.5).max(0.002),
            min_distance: s.min_distance,
            exclude_margin: ((s.search_radius * 2).clamp(9, 25) + 1) as f32,
            ..Default::default()
        },
        klt: KltParams {
            win_radius: (s.search_radius * 2).clamp(9, 25),
            illumination: true,
            max_residual: 40.0,
            max_iterations: 40,
            ..Default::default()
        },
        ransac_threshold: s.ransac_threshold * 1.5,
        fb_threshold: s.fb_threshold * 2.5,
        level: 0,
    }
}

struct Correspondences {
    src: Vec<Point>,
    dst: Vec<Point>,
}

/// Track `pts` (in `a`'s coordinates at `klt.base_level`) into `b`.
/// `guess` maps a point to its expected position in `b`. Runs points in parallel.
fn track_points(
    a: &Pyramid,
    b: &Pyramid,
    pts: &[Point],
    guess: &(dyn Fn(Point) -> Point + Sync),
    klt: &KltParams,
    fb: Option<f64>,
) -> Correspondences {
    let results = crate::par::map(pts, |&p| {
        let g = guess(p);
        let (q, _) = track_point(a, b, p, g, klt)?;
        if let Some(tol) = fb {
            // Track back to where we came from (in `a`'s coordinates).
            let (back, _) = track_point(b, a, q, p, klt)?;
            if (back[0] - p[0]).powi(2) + (back[1] - p[1]).powi(2) > tol * tol {
                return None;
            }
        }
        Some((p, q))
    });
    let (src, dst) = results.into_iter().flatten().unzip();
    Correspondences { src, dst }
}

fn scale_pts(pts: &[Point], k: f64) -> Vec<Point> {
    pts.iter().map(|p| [p[0] * k, p[1] * k]).collect()
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

/// The reference image as it should appear in a frame under `h` (ref -> frame),
/// rendered only for the area around `poly`. Returns the crop and its top-left offset
/// in frame coordinates.
fn warp_reference(
    reference: &GrayImage,
    h: &Mat3,
    poly: &[Point],
    w: usize,
    hgt: usize,
) -> Option<(GrayImage, Point)> {
    let hi = invert(h)?;
    let b = crate::shape::bbox(poly)?;
    let margin = 32.0;
    let x0 = (b[0] - margin).floor().max(0.0) as usize;
    let y0 = (b[1] - margin).floor().max(0.0) as usize;
    let x1 = ((b[2] + margin).ceil().max(0.0) as usize).min(w);
    let y1 = ((b[3] + margin).ceil().max(0.0) as usize).min(hgt);
    if x1 <= x0 + 8 || y1 <= y0 + 8 {
        return None;
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut out = GrayImage::new(cw, ch);
    crate::par::rows(&mut out.data, cw, |y, row| {
        for (x, o) in row.iter_mut().enumerate() {
            let p = apply(&hi, [(x + x0) as f64, (y + y0) as f64]);
            *o = reference.sample(p[0] as f32, p[1] as f32);
        }
    });
    Some((out, [x0 as f64, y0 as f64]))
}

struct StepResult {
    h_next: Mat3,
    step: Mat3,
    inliers: usize,
    rms: f64,
    shown: Vec<Point>,
}

/// One frame-to-frame step, then drift refinement against the warped reference.
#[allow(clippy::too_many_arguments)]
fn solve_step(
    layer: &Layer,
    pass: &Pass,
    prev: &Pyramid,
    next: &Pyramid,
    ref_pyr: &Pyramid,
    cur: usize,
    next_i: usize,
    h_cur: &Mat3,
    velocity: &Mat3,
    exclude: ExcludeFn,
    seed: u64,
) -> Result<StepResult, TrackError> {
    let s = &layer.settings;
    let lvl = pass
        .level
        .min(prev.levels.len() - 1)
        .min(next.levels.len() - 1);
    let k = (1u32 << lvl) as f64;

    // 1. Frame to frame, at the chosen detail level.
    let poly = scale_pts(
        &layer.shape_ref_at(cur).transformed(h_cur).flatten(8),
        1.0 / k,
    );
    let excl: Vec<Vec<Point>> = exclude(cur).iter().map(|e| scale_pts(e, 1.0 / k)).collect();
    let feats = good_features(
        &prev.levels[lvl],
        &prev.grad_x[lvl],
        &prev.grad_y[lvl],
        Some(&poly),
        &excl,
        &pass.features,
    );
    if feats.len() < s.min_inliers {
        return Err(TrackError::NoFeatures(cur));
    }
    let vel = *velocity;
    let mut klt = pass.klt.clone();
    klt.base_level = lvl;
    let guess = move |p: Point| scale_pts(&[apply(&vel, [p[0] * k, p[1] * k])], 1.0 / k)[0];
    let c = track_points(
        prev,
        next,
        &feats,
        &guess,
        &klt,
        Some(pass.fb_threshold / k),
    );
    let (src, dst) = (scale_pts(&c.src, k), scale_pts(&c.dst, k));
    let fit = match ransac(s.model, &src, &dst, pass.ransac_threshold, seed) {
        Some(f) if f.inlier_count >= s.min_inliers => f,
        other => {
            return Err(TrackError::Lost {
                frame: next_i,
                inliers: other.map_or(0, |f| f.inlier_count),
            })
        }
    };
    let mut out = StepResult {
        h_next: mul(&fit.model, h_cur),
        step: fit.model,
        inliers: fit.inlier_count,
        rms: fit.rms_error / k,
        shown: dst
            .iter()
            .zip(&fit.inliers)
            .filter(|(_, &i)| i)
            .map(|(p, _)| *p)
            .collect(),
    };

    // 2. Drift cancellation at full resolution: warp the reference onto this frame
    //    with the current estimate and solve the small residual motion.
    if s.refine_against_reference && next_i != layer.ref_frame {
        let poly_next = layer
            .shape_ref_at(next_i)
            .transformed(&out.h_next)
            .flatten(8);
        if let Some((crop, o)) = warp_reference(
            &ref_pyr.levels[0],
            &out.h_next,
            &poly_next,
            next.width(),
            next.height(),
        ) {
            let shift = |pts: &[Point]| {
                pts.iter()
                    .map(|p| [p[0] - o[0], p[1] - o[1]])
                    .collect::<Vec<_>>()
            };
            // Ignore occluders in this frame, and in the reference (mapped into this frame).
            let mut excl: Vec<Vec<Point>> = exclude(next_i).iter().map(|e| shift(e)).collect();
            for e in exclude(layer.ref_frame) {
                let mapped: Vec<Point> = e.iter().map(|&p| apply(&out.h_next, p)).collect();
                excl.push(shift(&mapped));
            }
            let wp = Pyramid::build(crop, ref_pyr.levels.len());
            let mut fp = pass.features.clone();
            fp.min_distance = s.min_distance;
            fp.exclude_margin = (pass.klt.win_radius + 1) as f32;
            let wf = good_features(
                &wp.levels[0],
                &wp.grad_x[0],
                &wp.grad_y[0],
                Some(&shift(&poly_next)),
                &excl,
                &fp,
            );
            if wf.len() >= s.min_inliers {
                let mut klt0 = pass.klt.clone();
                klt0.base_level = 0;
                let rc = track_points(
                    &wp,
                    next,
                    &wf,
                    &move |p| [p[0] + o[0], p[1] + o[1]],
                    &klt0,
                    None,
                );
                let rsrc: Vec<Point> = rc.src.iter().map(|p| [p[0] + o[0], p[1] + o[1]]).collect();
                let thr = s.ransac_threshold.max(pass.ransac_threshold / k);
                if let Some(rf) = ransac(s.model, &rsrc, &rc.dst, thr, seed ^ 0xA5A5_0000) {
                    let candidate = mul(&rf.model, &out.h_next);
                    let enough = rf.inlier_count >= s.min_inliers.max(wf.len() * 3 / 10);
                    // Allow bigger corrections when the coarse pass was used.
                    let limit = s.max_refine_correction * k;
                    if enough && max_corner_delta(&layer.surface, &candidate, &out.h_next) <= limit
                    {
                        out.h_next = candidate;
                        out.inliers = rf.inlier_count;
                        out.rms = rf.rms_error;
                        out.shown = rc
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
    Ok(out)
}

/// Track `layer` from tracked frame `start` towards `end` (either direction, inclusive).
/// `exclude` gives occluder outlines per frame. `on_frame` is called after each solved
/// frame. Existing data in the range is overwritten.
pub fn track_layer<S: FrameSource>(
    source: &mut S,
    layer: &mut Layer,
    start: usize,
    end: usize,
    exclude: ExcludeFn,
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
    let step: isize = if end > start { 1 } else { -1 };

    let ref_pyr = source
        .pyramid(layer.ref_frame, s.pyramid_levels.max(1))
        .map_err(TrackError::Source)?;
    let level = auto_level(&s, ref_pyr.width());
    // Coarse passes need enough levels above them for large motion.
    let levels = s.pyramid_levels.max(1) + level;
    let ref_pyr = if ref_pyr.levels.len() < levels {
        source
            .pyramid(layer.ref_frame, levels)
            .map_err(TrackError::Source)?
    } else {
        ref_pyr
    };
    let normal = normal_pass(&s, level);
    let retry = retry_pass(&s);

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
        let seed = seed_base + next as u64;

        let mut retried = false;
        let result = match solve_step(
            layer, &normal, &prev_pyr, &next_pyr, &ref_pyr, cur, next, &h_cur, &velocity, exclude,
            seed,
        ) {
            Ok(r) => Ok(r),
            Err(e) if s.retry_hard_frames => {
                retried = true;
                // Also try without the motion prediction, which can be wrong after a jolt.
                solve_step(
                    layer,
                    &retry,
                    &prev_pyr,
                    &next_pyr,
                    &ref_pyr,
                    cur,
                    next,
                    &h_cur,
                    &velocity,
                    exclude,
                    seed ^ 1,
                )
                .or_else(|_| {
                    solve_step(
                        layer,
                        &retry,
                        &prev_pyr,
                        &next_pyr,
                        &ref_pyr,
                        cur,
                        next,
                        &h_cur,
                        &IDENTITY,
                        exclude,
                        seed ^ 2,
                    )
                })
                .map_err(|_| e)
            }
            Err(e) => Err(e),
        };
        let r = result?;

        let mut shown = r.shown;
        shown.truncate(250);
        let ft = FrameTrack {
            h_from_ref: r.h_next,
            inliers: r.inliers as u32,
            rms_error: r.rms,
            points: shown,
            keyframe: false,
            retried,
        };
        velocity = r.step;
        layer.tracks.insert(next, ft.clone());
        on_frame(next, &ft);
        h_cur = r.h_next;
        cur = next;
    }
    Ok(StopReason::Finished)
}

/// Re-track the section `a`..=`b` and blend it into the existing track so it joins
/// the old data at `b` seamlessly (the mismatch at `b` is spread across the section).
/// Frames outside the section are untouched.
pub fn retrack_section<S: FrameSource>(
    source: &mut S,
    layer: &mut Layer,
    a: usize,
    b: usize,
    exclude: ExcludeFn,
    cancel: &AtomicBool,
    on_frame: impl FnMut(usize, &FrameTrack),
) -> Result<StopReason, TrackError> {
    let old_b = layer.tracked_surface_at(b);
    let reason = track_layer(source, layer, a, b, exclude, cancel, on_frame)?;
    if reason != StopReason::Finished || a == b {
        return Ok(reason);
    }
    let (Some(old), Some(new)) = (old_b, layer.tracked_surface_at(b)) else {
        return Ok(reason);
    };
    let delta: [Point; 4] = std::array::from_fn(|i| [old[i][0] - new[i][0], old[i][1] - new[i][1]]);
    let span = a.abs_diff(b) as f64;
    let (lo, hi) = (a.min(b), a.max(b));
    for f in lo..=hi {
        let w = f.abs_diff(a) as f64 / span;
        if w == 0.0 {
            continue;
        }
        let Some(c) = layer.tracked_surface_at(f) else {
            continue;
        };
        let target: [Point; 4] =
            std::array::from_fn(|i| [c[i][0] + delta[i][0] * w, c[i][1] + delta[i][1] * w]);
        if let (Some(h), Some(t)) = (
            homography_from_corners(&layer.surface, &target),
            layer.tracks.get_mut(&f),
        ) {
            t.h_from_ref = h;
        }
    }
    Ok(reason)
}

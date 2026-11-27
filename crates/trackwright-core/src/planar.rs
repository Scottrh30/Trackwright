//! The planar tracker: follows a layer's shape region frame to frame and solves
//! a reference-to-frame transform for its surface, Mocha style.

use crate::features::{good_features, FeatureParams};
use crate::geometry::{
    apply, homography_from_corners, invert, mul, ransac, Mat3, MotionModel, Point, IDENTITY,
};
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
    Lost {
        frame: usize,
        inliers: usize,
    },
    /// Covered from `frame` on and not found again within `frames` frames.
    Occluded {
        frame: usize,
        frames: usize,
    },
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
            TrackError::Occluded { frame, frames } => write!(
                f,
                "Track lost at frame {frame} and not found again within {frames} frames. Track on from where it shows again (Adjust it there), or raise \"Bridge up to\" in Track Settings"
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
    // How much the surface usually looks like the reference (running average), to
    // notice it being covered even while points still track (an eyelid closing).
    let mut look: Option<f64> = None;
    let hidden_from = |look: Option<f64>, score: Option<f64>| match (look, score) {
        (Some(avg), Some(sc)) => avg >= 0.7 && sc < 0.45 && sc < avg - 0.3,
        _ => false,
    };

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
        // Covered? The step failed, or it "worked" but no longer looks like the surface.
        let lost = match &result {
            Err(TrackError::Lost { .. } | TrackError::NoFeatures(_)) => true,
            Ok(r) if s.bridge_occlusions => {
                let sc = appearance(
                    layer,
                    &ref_pyr.levels[0],
                    &next_pyr.levels[0],
                    &r.h_next,
                    next,
                    exclude,
                );
                if hidden_from(look, sc) {
                    true
                } else {
                    if let Some(sc) = sc {
                        look = Some(look.map_or(sc, |a| a * 0.8 + sc * 0.2));
                    }
                    false
                }
            }
            _ => false,
        };
        if lost && s.bridge_occlusions {
            match bridge(
                source,
                layer,
                &normal,
                &retry,
                &ref_pyr,
                levels,
                cur,
                end,
                step,
                exclude,
                cancel,
                &mut on_frame,
            )? {
                Bridged::Found { frame, velocity: v } => {
                    h_cur = layer.h_at(frame).unwrap_or(h_cur);
                    velocity = v;
                    cur = frame;
                    continue;
                }
                Bridged::ToEnd => return Ok(StopReason::Finished),
                Bridged::Cancelled => return Ok(StopReason::Cancelled),
                Bridged::GaveUp => {
                    return Err(TrackError::Occluded {
                        frame: next,
                        frames: s.max_bridge_frames,
                    })
                }
            }
        }
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
            predicted: false,
        };
        velocity = r.step;
        layer.tracks.insert(next, ft.clone());
        on_frame(next, &ft);
        h_cur = r.h_next;
        cur = next;
    }
    Ok(StopReason::Finished)
}

// ---------------------------------------------------------------------------
// Occlusion solver
//
// When the surface is covered (a blink over a pupil, a hand passing in front)
// the frame-to-frame step fails or starts following the cover. Then each hidden
// frame is predicted: moved with the area around the surface (skin, face,
// background) when that still tracks, otherwise carried on with the surface's
// own recent motion, slowing down. Every predicted frame also tries to find the
// surface again by tracking straight from the last frame it was seen, and accepts
// the result only if it lands near the prediction and looks like the reference
// again. Then the predicted stretch is corrected so it joins the found frame.

/// How much the frame looks like the reference under `h` inside the layer's shape:
/// normalised cross-correlation, 1 = identical (brightness and contrast don't matter).
/// `None` when too little of the shape is visible to judge.
fn appearance(
    layer: &Layer,
    reference: &GrayImage,
    frame: &GrayImage,
    h: &Mat3,
    index: usize,
    exclude: ExcludeFn,
) -> Option<f64> {
    use crate::shape::point_in_polygon;
    let hi = invert(h)?;
    let poly = layer.shape_ref_at(index).transformed(h).flatten(8);
    let b = crate::shape::bbox(&poly)?;
    let excl = exclude(index);
    let ref_excl = exclude(layer.ref_frame);
    let area = (b[2] - b[0]).max(1.0) * (b[3] - b[1]).max(1.0);
    let stride = (area / 3000.0).sqrt().max(1.0);
    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut y = b[1];
    while y <= b[3] {
        let mut x = b[0];
        while x <= b[2] {
            let p = [x, y];
            let inside_frame = x >= 1.0
                && y >= 1.0
                && x < frame.width as f64 - 2.0
                && y < frame.height as f64 - 2.0;
            if inside_frame
                && point_in_polygon(&poly, p)
                && !excl.iter().any(|e| point_in_polygon(e, p))
            {
                let r = apply(&hi, p);
                let inside_ref = r[0] >= 1.0
                    && r[1] >= 1.0
                    && r[0] < reference.width as f64 - 2.0
                    && r[1] < reference.height as f64 - 2.0;
                if inside_ref && !ref_excl.iter().any(|e| point_in_polygon(e, r)) {
                    let a = frame.sample(x as f32, y as f32) as f64;
                    let c = reference.sample(r[0] as f32, r[1] as f32) as f64;
                    n += 1.0;
                    sa += a;
                    sb += c;
                    saa += a * a;
                    sbb += c * c;
                    sab += a * c;
                }
            }
            x += stride;
        }
        y += stride;
    }
    if n < 60.0 {
        return None;
    }
    let va = saa - sa * sa / n;
    let vb = sbb - sb * sb / n;
    if va <= 1e-6 || vb <= 1e-6 {
        return Some(0.0);
    }
    Some((sab - sa * sb / n) / (va * vb).sqrt())
}

/// Motion of the area around the shape (a ring 2.5x its size, the shape itself
/// left out) from frame `a` to frame `b`, as a similarity transform.
#[allow(clippy::too_many_arguments)]
fn context_motion(
    layer: &Layer,
    a: &Pyramid,
    b: &Pyramid,
    ai: usize,
    h_a: &Mat3,
    guess: &Mat3,
    exclude: ExcludeFn,
    seed: u64,
) -> Option<Mat3> {
    let s = &layer.settings;
    let poly = layer.shape_ref_at(ai).transformed(h_a).flatten(8);
    let n = poly.len() as f64;
    let c = poly
        .iter()
        .fold([0.0, 0.0], |acc, p| [acc[0] + p[0] / n, acc[1] + p[1] / n]);
    let grow = |k: f64| -> Vec<Point> {
        poly.iter()
            .map(|p| [c[0] + (p[0] - c[0]) * k, c[1] + (p[1] - c[1]) * k])
            .collect()
    };
    let outer = grow(2.5);
    let mut excl = exclude(ai);
    excl.push(grow(1.15));
    let fp = FeatureParams {
        max_features: s.max_features,
        quality: (s.feature_quality * 0.5).max(0.002),
        min_distance: s.min_distance,
        exclude_margin: (s.search_radius + 1) as f32,
        ..Default::default()
    };
    let feats = good_features(
        &a.levels[0],
        &a.grad_x[0],
        &a.grad_y[0],
        Some(&outer),
        &excl,
        &fp,
    );
    if feats.len() < 12 {
        return None;
    }
    let klt = KltParams {
        win_radius: s.search_radius.max(7),
        illumination: true,
        base_level: 0,
        ..Default::default()
    };
    let g = *guess;
    let cr = track_points(
        a,
        b,
        &feats,
        &move |p| apply(&g, p),
        &klt,
        Some(s.fb_threshold * 1.5),
    );
    let fit = ransac(
        MotionModel::Similarity,
        &cr.src,
        &cr.dst,
        s.ransac_threshold * 1.5,
        seed,
    )?;
    (fit.inlier_count >= 10 && fit.inlier_count * 3 >= feats.len()).then_some(fit.model)
}

enum Bridged {
    /// Seen again at `frame`; `velocity` is the last frame-to-frame motion.
    Found {
        frame: usize,
        velocity: Mat3,
    },
    ToEnd,
    Cancelled,
    GaveUp,
}

/// Predict through a covered stretch after `anchor` (the last frame the surface
/// was seen), find it again, and correct the prediction to join up.
#[allow(clippy::too_many_arguments)]
fn bridge<S: FrameSource>(
    source: &mut S,
    layer: &mut Layer,
    normal: &Pass,
    retry: &Pass,
    ref_pyr: &Pyramid,
    levels: usize,
    anchor: usize,
    end: usize,
    step: isize,
    exclude: ExcludeFn,
    cancel: &AtomicBool,
    on_frame: &mut impl FnMut(usize, &FrameTrack),
) -> Result<Bridged, TrackError> {
    let s = layer.settings.clone();
    let h_anchor = layer.h_at(anchor).ok_or(TrackError::NotTracked(anchor))?;
    let corners = |h: &Mat3, l: &Layer| l.surface.map(|p| apply(h, p));
    // Recent motion of each corner (pixels per frame), from up to 8 seen frames.
    let mut hist: Vec<(f64, [Point; 4])> = Vec::new();
    let mut f = anchor as isize;
    while hist.len() < 8 && f >= 0 {
        match layer.tracks.get(&(f as usize)) {
            Some(t) if !t.predicted => hist.push((
                (f - anchor as isize) as f64 * step as f64,
                corners(&t.h_from_ref, layer),
            )),
            _ => break,
        }
        f -= step;
    }
    let vel: [Point; 4] = std::array::from_fn(|i| {
        let n = hist.len() as f64;
        if n < 2.0 {
            return [0.0, 0.0];
        }
        let mt = hist.iter().map(|h| h.0).sum::<f64>() / n;
        let mx = hist.iter().map(|h| h.1[i][0]).sum::<f64>() / n;
        let my = hist.iter().map(|h| h.1[i][1]).sum::<f64>() / n;
        let stt = hist
            .iter()
            .map(|h| (h.0 - mt).powi(2))
            .sum::<f64>()
            .max(1e-9);
        [
            hist.iter()
                .map(|h| (h.0 - mt) * (h.1[i][0] - mx))
                .sum::<f64>()
                / stt,
            hist.iter()
                .map(|h| (h.0 - mt) * (h.1[i][1] - my))
                .sum::<f64>()
                / stt,
        ]
    });
    let anchor_corners = corners(&h_anchor, layer);
    let size = {
        let d = |a: Point, b: Point| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        d(anchor_corners[0], anchor_corners[2]).max(d(anchor_corners[1], anchor_corners[3]))
    };
    let anchor_pyr = source.pyramid(anchor, levels).map_err(TrackError::Source)?;
    let seed_base = layer.id as u64 * 7_000_003;
    let mut ctx_vel = IDENTITY;
    let mut prev = anchor;
    let mut drift = [[0.0f64; 2]; 4]; // displacement from the anchor by own motion
    let mut damp_sum = 0.0;
    let mut k = 0usize;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(Bridged::Cancelled);
        }
        let g = (prev as isize + step) as usize;
        k += 1;
        let prev_pyr = source.pyramid(prev, levels).map_err(TrackError::Source)?;
        let g_pyr = source.pyramid(g, levels).map_err(TrackError::Source)?;
        let h_prev = layer.h_at(prev).unwrap_or(h_anchor);

        // 1. Prediction: the surroundings' motion, else the surface's own (slowing).
        damp_sum += 0.9f64.powi(k as i32);
        let own: [Point; 4] = std::array::from_fn(|i| {
            [
                anchor_corners[i][0] + vel[i][0] * damp_sum * step as f64,
                anchor_corners[i][1] + vel[i][1] * damp_sum * step as f64,
            ]
        });
        let h_pred = match context_motion(
            layer,
            &prev_pyr,
            &g_pyr,
            prev,
            &h_prev,
            &ctx_vel,
            exclude,
            seed_base + g as u64,
        ) {
            Some(m) => {
                ctx_vel = m;
                let h = mul(&m, &h_prev);
                // Keep the own-motion estimate in step with where the surroundings put it.
                let c = corners(&h, layer);
                drift = std::array::from_fn(|i| [c[i][0] - own[i][0], c[i][1] - own[i][1]]);
                h
            }
            None => {
                let c: [Point; 4] =
                    std::array::from_fn(|i| [own[i][0] + drift[i][0], own[i][1] + drift[i][1]]);
                homography_from_corners(&layer.surface, &c).unwrap_or(h_prev)
            }
        };
        let ft = FrameTrack {
            h_from_ref: h_pred,
            inliers: 0,
            rms_error: 0.0,
            points: vec![],
            keyframe: false,
            retried: false,
            predicted: true,
        };
        layer.tracks.insert(g, ft.clone());
        on_frame(g, &ft);

        // 2. Seen again? Track straight from the anchor with the prediction as the guess.
        let guess = invert(&h_anchor)
            .map(|hi| mul(&h_pred, &hi))
            .unwrap_or(IDENTITY);
        let found = [normal, retry].iter().enumerate().find_map(|(i, pass)| {
            let r = solve_step(
                layer,
                pass,
                &anchor_pyr,
                &g_pyr,
                ref_pyr,
                anchor,
                g,
                &h_anchor,
                &guess,
                exclude,
                seed_base ^ (g as u64 * 31 + i as u64),
            )
            .ok()?;
            let jump = max_corner_delta(&layer.surface, &r.h_next, &h_pred);
            let near = jump <= 0.35 * size + 3.0 * k as f64 + 8.0;
            let look = appearance(
                layer,
                &ref_pyr.levels[0],
                &g_pyr.levels[0],
                &r.h_next,
                g,
                exclude,
            )?;
            (near && look >= 0.6 && r.inliers >= s.min_inliers.max(12)).then_some(r)
        });
        if let Some(r) = found {
            // Spread the miss at the found frame back over the predicted frames.
            let miss: [Point; 4] = {
                let a = corners(&r.h_next, layer);
                let b = corners(&h_pred, layer);
                std::array::from_fn(|i| [a[i][0] - b[i][0], a[i][1] - b[i][1]])
            };
            for j in 1..k {
                let f = (anchor as isize + step * j as isize) as usize;
                let w = j as f64 / k as f64;
                let Some(h) = layer.h_at(f) else { continue };
                let c = corners(&h, layer);
                let t: [Point; 4] =
                    std::array::from_fn(|i| [c[i][0] + miss[i][0] * w, c[i][1] + miss[i][1] * w]);
                if let (Some(hn), Some(tr)) = (
                    homography_from_corners(&layer.surface, &t),
                    layer.tracks.get_mut(&f),
                ) {
                    tr.h_from_ref = hn;
                    let tr = tr.clone();
                    on_frame(f, &tr);
                }
            }
            let mut shown = r.shown;
            shown.truncate(250);
            let ft = FrameTrack {
                h_from_ref: r.h_next,
                inliers: r.inliers as u32,
                rms_error: r.rms,
                points: shown,
                keyframe: false,
                retried: false,
                predicted: false,
            };
            layer.tracks.insert(g, ft.clone());
            on_frame(g, &ft);
            let before = layer
                .h_at(prev)
                .and_then(|h| invert(&h))
                .unwrap_or(IDENTITY);
            return Ok(Bridged::Found {
                frame: g,
                velocity: mul(&r.h_next, &before),
            });
        }
        if g == end {
            return Ok(Bridged::ToEnd);
        }
        if k >= s.max_bridge_frames.max(1) {
            // Not found again: drop the guesses rather than leave a wandering track.
            for j in 1..=k {
                layer
                    .tracks
                    .remove(&((anchor as isize + step * j as isize) as usize));
            }
            return Ok(Bridged::GaveUp);
        }
        prev = g;
    }
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

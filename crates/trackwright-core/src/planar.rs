//! The planar tracker: follows a layer's shape region frame to frame and solves
//! a reference-to-frame transform for its surface, Mocha style.

use crate::features::{good_features, FeatureParams};
use crate::geometry::{
    apply, homography_from_corners, invert, mul, ransac, Mat3, MotionModel, Point, IDENTITY,
};
use crate::image::{GrayImage, Pyramid};
use crate::klt::{track_point, KltParams};
use crate::learning::Cues;
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
    // How much each frame usually looks like the one before it (running average). A
    // cover arrives suddenly (a lid coming down changes it a lot in one frame), where
    // a head turning or light changing drifts slowly, so this stays high for those.
    let mut steady: Option<f64> = None;
    // Recent frames' frame-to-frame scores, to start a prediction from the last clean frame.
    let mut recent: Vec<(usize, f64, Option<f64>)> = Vec::new();
    // Appearance memory: a few clean frames along the way (how it looked at different
    // angles and moments), to recognise it again after a cover.
    let mut memory: Vec<(usize, Mat3)> = vec![(start, h_cur)];
    // Scores compare fine detail: relative to how this surface usually scores. `look`
    // is the best recent score, fading 1% a frame, so a head turning over seconds is
    // fine but a lid closing over a few frames is not.
    let hidden_from = |look: Option<f64>,
                       steady: Option<f64>,
                       sc: Option<f64>,
                       cons: Option<f64>| {
        let gone = matches!((look, sc), (Some(peak), Some(sc)) if peak >= 0.3 && sc < s.tuning.gone * peak);
        let sudden = matches!((steady, cons), (Some(avg), Some(c)) if avg >= 0.5 && c < s.tuning.sudden * avg);
        gone || sudden
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
        let mut cues = None;
        let lost = match &result {
            Err(TrackError::Lost { .. } | TrackError::NoFeatures(_)) => true,
            Ok(r) if s.bridge_occlusions => {
                let sc = appearance(layer, &ref_pyr, &next_pyr, &r.h_next, next, exclude);
                let cons = appearance_vs(
                    layer, &prev_pyr, &h_cur, cur, &next_pyr, &r.h_next, next, exclude,
                );
                // What the calls below are based on, kept for learning from fixes.
                cues = Some(Cues {
                    sudden: match (steady, cons) {
                        (Some(a), Some(c)) if a >= 0.5 => Some((c / a) as f32),
                        _ => None,
                    },
                    gone: match (look, sc) {
                        (Some(p), Some(v)) if p >= 0.3 => Some((v / p) as f32),
                        _ => None,
                    },
                    ..Default::default()
                });
                if hidden_from(look, steady, sc, cons) {
                    true
                } else {
                    if let Some(sc) = sc {
                        look = Some(look.map_or(sc, |p| (p * 0.99).max(sc)));
                    }
                    if let Some(c) = cons {
                        let clean = steady.is_none_or(|a| c >= 0.9 * a);
                        if clean && memory.last().is_none_or(|m| m.0.abs_diff(next) >= 8) {
                            memory.push((next, r.h_next));
                            if memory.len() > 6 {
                                memory.remove(1); // keep the first (where the user started)
                            }
                        }
                        steady = Some(steady.map_or(c, |a| a * 0.8 + c * 0.2));
                        recent.push((next, c, sc));
                        if recent.len() > 8 {
                            recent.remove(0);
                        }
                    }
                    false
                }
            }
            _ => false,
        };
        if lost && s.bridge_occlusions {
            // The cover may have started a frame or two earlier (a lid starting to drop):
            // back up to the last frame that still looked normal and predict from there.
            if let Some(base) = steady {
                let peak = look.unwrap_or(0.0);
                let mut backed = 0;
                while backed < 8 {
                    match recent.last() {
                        Some(&(f, c, sc))
                            if f == cur
                                && f != start
                                && (c < 0.9 * base || sc.is_some_and(|v| v < 0.75 * peak))
                                && layer.tracks.get(&f).is_some_and(|t| !t.keyframe) =>
                        {
                            recent.pop();
                            layer.tracks.remove(&f);
                            cur = (cur as isize - step) as usize;
                            backed += 1;
                        }
                        _ => break,
                    }
                }
            }
            // The solve this frame would have used, for telling a real cover from a
            // false alarm when the user fixes the track later.
            let trigger = cues.take().map(|mut c| {
                c.rejected = result
                    .as_ref()
                    .ok()
                    .map(|r| layer.surface.map(|p| apply(&r.h_next, p)));
                c
            });
            let outcome = bridge(
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
                &memory,
                look.unwrap_or(0.0),
                &mut on_frame,
            )?;
            if matches!(outcome, Bridged::Found { .. } | Bridged::ToEnd) {
                if let Some(t) = layer.tracks.get_mut(&next) {
                    if t.cues.is_none() {
                        t.cues = trigger;
                    }
                }
            }
            match outcome {
                Bridged::Found { frame, velocity: v } => {
                    h_cur = layer.h_at(frame).unwrap_or(h_cur);
                    velocity = v;
                    cur = frame;
                    // Keep the usual look from before the cover; carry on from how it looks now.
                    let found = source.pyramid(frame, levels).map_err(TrackError::Source)?;
                    // It may look different now (an eye turned during a blink): that is the
                    // new normal to compare against.
                    look = appearance(layer, &ref_pyr, &found, &h_cur, frame, exclude);
                    recent.clear();
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
            cues,
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
/// Detail at `(x, y)` of pyramid level `lvl`: the pixel minus the local average (the
/// same spot two levels coarser). Comparing detail, not raw brightness, keeps soft
/// shading (a shadow across skin) from looking like the target.
fn detail(p: &Pyramid, lvl: usize, x: f32, y: f32) -> f32 {
    let lvl = lvl.min(p.levels.len() - 1);
    let coarse = (lvl + 2).min(p.levels.len() - 1);
    if coarse == lvl {
        return p.levels[lvl].sample(x, y);
    }
    let k = (1u32 << (coarse - lvl)) as f32;
    p.levels[lvl].sample(x, y) - p.levels[coarse].sample((x + 0.5) / k - 0.5, (y + 0.5) / k - 0.5)
}

fn appearance(
    layer: &Layer,
    reference_pyr: &Pyramid,
    frame_pyr: &Pyramid,
    h: &Mat3,
    index: usize,
    exclude: ExcludeFn,
) -> Option<f64> {
    appearance_vs(
        layer,
        reference_pyr,
        &IDENTITY,
        layer.ref_frame,
        frame_pyr,
        h,
        index,
        exclude,
    )
}

/// Like [`appearance`], but compared with another frame `src_index` (its pyramid and
/// its ref->frame transform `h_src`) instead of the reference frame.
#[allow(clippy::too_many_arguments)]
fn appearance_vs(
    layer: &Layer,
    reference_pyr: &Pyramid,
    h_src: &Mat3,
    src_index: usize,
    frame_pyr: &Pyramid,
    h: &Mat3,
    index: usize,
    exclude: ExcludeFn,
) -> Option<f64> {
    let reference = &reference_pyr.levels[0];
    let frame = &frame_pyr.levels[0];
    use crate::shape::point_in_polygon;
    // frame -> source frame
    let hi = mul(h_src, &invert(h)?);
    let poly = layer.shape_ref_at(index).transformed(h).flatten(8);
    let b = crate::shape::bbox(&poly)?;
    let excl = exclude(index);
    let ref_excl = exclude(src_index);
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
                    let a = detail(frame_pyr, 0, x as f32, y as f32) as f64;
                    let c = detail(reference_pyr, 0, r[0] as f32, r[1] as f32) as f64;
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

/// How much frame `index` (its pyramid `frame`) looks like the layer's reference
/// frame (`reference`) under `h`: 1 = identical, below ~0.5 = something else is
/// there. `None` when too little of the shape is in view. For tools and tests.
pub fn appearance_score(
    layer: &Layer,
    reference: &Pyramid,
    frame: &Pyramid,
    h: &Mat3,
    index: usize,
) -> Option<f64> {
    appearance(layer, reference, frame, h, index, &no_exclusions)
}

/// How much frame `index` (pyramid `frame`, transform `h`) looks like another frame
/// `src_index` (pyramid `src`, transform `h_src`) inside the layer's shape. For tools and tests.
#[allow(clippy::too_many_arguments)]
pub fn appearance_between(
    layer: &Layer,
    src: &Pyramid,
    h_src: &Mat3,
    src_index: usize,
    frame: &Pyramid,
    h: &Mat3,
    index: usize,
) -> Option<f64> {
    appearance_vs(
        layer,
        src,
        h_src,
        src_index,
        frame,
        h,
        index,
        &no_exclusions,
    )
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
    // Fast moves blur the surroundings and many points drop out or slip, but a clear
    // group moving together is still the right motion (a head jerking up mid-blink).
    (fit.inlier_count >= 12 && fit.inlier_count * 5 >= cr.src.len()).then_some(fit.model)
}

/// Search frame `b` for the shape's look in frame `a` (under `h_a`), shifted
/// anywhere within `radius` pixels of where `h_pred` puts it. The look is carried
/// to `b` the way the prediction moves it (turned and sized), compared as detail
/// (masked normalised cross-correlation), coarse level first, then refined at full
/// resolution. Returns the best shift (pixels, added to `h_pred`) and its score.
#[allow(clippy::too_many_arguments)]
fn search_shift(
    layer: &Layer,
    a: &Pyramid,
    _ai: usize,
    h_a: &Mat3,
    b: &Pyramid,
    bi: usize,
    h_pred: &Mat3,
    radius: f64,
) -> Option<(Point, f64)> {
    use crate::shape::point_in_polygon;
    // b -> a is h_a * h_pred^-1.
    let to_a = mul(h_a, &invert(h_pred)?);
    let poly_b = layer.shape_ref_at(bi).transformed(h_pred).flatten(8);
    let bb = crate::shape::bbox(&poly_b)?;
    let w = (bb[2] - bb[0]).max(bb[3] - bb[1]);
    let levels = a.levels.len().min(b.levels.len());
    // Level where the template is about 48 px across.
    let mut lvl = 0;
    while lvl + 1 < levels
        && w / (1u32 << (lvl + 1)) as f64 >= 24.0
        && w / (1u32 << lvl) as f64 > 48.0
    {
        lvl += 1;
    }
    let (aw, ah) = (a.levels[0].width as f64, a.levels[0].height as f64);

    // Template on level `l`'s pixel grid, centred on pixel (cx, cy): (i, j, value), with
    // `step` thinning it out.
    let template = |l: usize, cx: i64, cy: i64, step: i64| -> Vec<(i64, i64, f64)> {
        let k = (1u32 << l) as f64;
        let (i0, i1) = (
            (bb[0] / k).floor() as i64 - cx,
            (bb[2] / k).ceil() as i64 - cx,
        );
        let (j0, j1) = (
            (bb[1] / k).floor() as i64 - cy,
            (bb[3] / k).ceil() as i64 - cy,
        );
        let mut t = Vec::new();
        let mut j = j0;
        while j <= j1 {
            let mut i = i0;
            while i <= i1 {
                let p = [(cx + i) as f64 * k, (cy + j) as f64 * k];
                if point_in_polygon(&poly_b, p) {
                    let q = apply(&to_a, p);
                    if q[0] >= 1.0 && q[1] >= 1.0 && q[0] < aw - 2.0 && q[1] < ah - 2.0 {
                        t.push((
                            i,
                            j,
                            detail(a, l, (q[0] / k) as f32, (q[1] / k) as f32) as f64,
                        ));
                    }
                }
                i += step;
            }
            j += step;
        }
        t
    };
    // Detail of `b` on level `l` over a window, computed once and then looked up.
    struct Window {
        x0: i64,
        y0: i64,
        w: i64,
        h: i64,
        v: Vec<f32>,
    }
    let window = |l: usize, x0: i64, y0: i64, x1: i64, y1: i64| -> Window {
        let img = &b.levels[l];
        let (x0, y0) = (x0.max(1), y0.max(1));
        let (x1, y1) = (x1.min(img.width as i64 - 3), y1.min(img.height as i64 - 3));
        let (w, h) = ((x1 - x0 + 1).max(0), (y1 - y0 + 1).max(0));
        let mut v = vec![0.0f32; (w * h) as usize];
        crate::par::rows(&mut v, w.max(1) as usize, |y, row| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = detail(b, l, (x0 + x as i64) as f32, (y0 + y as i64) as f32);
            }
        });
        Window { x0, y0, w, h, v }
    };
    let score = |t: &[(i64, i64, f64)], win: &Window, cx: i64, cy: i64| -> Option<f64> {
        let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for &(i, j, v) in t {
            let (x, y) = (cx + i - win.x0, cy + j - win.y0);
            if x < 0 || y < 0 || x >= win.w || y >= win.h {
                continue;
            }
            let u = win.v[(y * win.w + x) as usize] as f64;
            n += 1.0;
            sa += v;
            sb += u;
            saa += v * v;
            sbb += u * u;
            sab += v * u;
        }
        if n < t.len() as f64 * 0.7 || n < 20.0 {
            return None;
        }
        let va = saa - sa * sa / n;
        let vb = sbb - sb * sb / n;
        (va > 1e-6 && vb > 1e-6).then(|| (sab - sa * sb / n) / (va * vb).sqrt())
    };
    let extent = |t: &[(i64, i64, f64)]| {
        t.iter().fold((0i64, 0i64, 0i64, 0i64), |e, &(i, j, _)| {
            (e.0.min(i), e.1.min(j), e.2.max(i), e.3.max(j))
        })
    };

    // Coarse pass.
    let k = (1u32 << lvl) as f64;
    let (cx, cy) = (
        ((bb[0] + bb[2]) / 2.0 / k).round() as i64,
        ((bb[1] + bb[3]) / 2.0 / k).round() as i64,
    );
    let t = template(lvl, cx, cy, 1);
    if t.len() < 20 {
        return None;
    }
    let r = (radius / k).ceil() as i64;
    let e = extent(&t);
    let win = window(lvl, cx + e.0 - r, cy + e.1 - r, cx + e.2 + r, cy + e.3 + r);
    let rows: Vec<i64> = (-r..=r).collect();
    let best = crate::par::map(&rows, |&dy| {
        let mut best: Option<(i64, i64, f64)> = None;
        for dx in -r..=r {
            if dx * dx + dy * dy > r * r {
                continue;
            }
            if let Some(sc) = score(&t, &win, cx + dx, cy + dy) {
                if best.is_none_or(|b| sc > b.2) {
                    best = Some((dx, dy, sc));
                }
            }
        }
        best
    })
    .into_iter()
    .flatten()
    .fold(None, |acc: Option<(i64, i64, f64)>, b| {
        if acc.is_none_or(|a| b.2 > a.2) {
            Some(b)
        } else {
            acc
        }
    });
    let (bx, by, coarse) = best?;
    if lvl == 0 {
        return Some((
            [
                (cx + bx) as f64 - (bb[0] + bb[2]) / 2.0,
                (cy + by) as f64 - (bb[1] + bb[3]) / 2.0,
            ],
            coarse,
        ));
    }
    // Refine at full resolution around the coarse answer (template thinned to ~3000 px).
    let (fx, fy) = (
        ((bb[0] + bb[2]) / 2.0).round() as i64,
        ((bb[1] + bb[3]) / 2.0).round() as i64,
    );
    let area = (bb[2] - bb[0]) * (bb[3] - bb[1]);
    let step = ((area / 3000.0).sqrt().floor() as i64).max(1);
    let t0 = template(0, fx, fy, step);
    let (sx, sy) = (fx + bx * k as i64, fy + by * k as i64);
    let rr = k as i64 + 1;
    let e0 = extent(&t0);
    let win0 = window(
        0,
        sx + e0.0 - rr,
        sy + e0.1 - rr,
        sx + e0.2 + rr,
        sy + e0.3 + rr,
    );
    let mut found: Option<(i64, i64, f64)> = None;
    for dy in -rr..=rr {
        for dx in -rr..=rr {
            if let Some(sc) = score(&t0, &win0, sx + dx, sy + dy) {
                if found.is_none_or(|f| sc > f.2) {
                    found = Some((sx + dx, sy + dy, sc));
                }
            }
        }
    }
    let (x, y, sc) = found?;
    Some((
        [
            x as f64 - (bb[0] + bb[2]) / 2.0,
            y as f64 - (bb[1] + bb[3]) / 2.0,
        ],
        sc,
    ))
}

fn translate(m: &Mat3, d: Point) -> Mat3 {
    mul(&[1.0, 0.0, d[0], 0.0, 1.0, d[1], 0.0, 0.0, 1.0], m)
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
    memory: &[(usize, Mat3)],
    peak: f64,
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
    // What it looked like: just before the cover, and the remembered clean frames.
    let mut looks: Vec<(usize, Mat3, Arc<Pyramid>)> = vec![(anchor, h_anchor, anchor_pyr.clone())];
    for &(f, h) in memory.iter().rev() {
        if f != anchor && looks.len() < 6 {
            looks.push((f, h, source.pyramid(f, levels).map_err(TrackError::Source)?));
        }
    }
    // How alike two clean views of it usually are (the two latest remembered frames),
    // for telling "back" from "still covered".
    let usual_level = match (looks.get(1), looks.get(2)) {
        (Some((f1, h1, p1)), Some((f2, h2, p2))) => {
            appearance_vs(layer, p2, h2, *f2, p1, h1, *f1, exclude).unwrap_or(peak)
        }
        _ => peak,
    };
    let seed_base = layer.id as u64 * 7_000_003;
    let mut ctx_vel = IDENTITY;
    let mut prev = anchor;
    let mut drift = [[0.0f64; 2]; 4]; // displacement from the anchor by own motion
    let mut damp_sum = 0.0;
    let mut k = 0usize;
    let mut last_shift: Option<Point> = None;
    // The least it has looked like itself while hidden (at the best spot found).
    let mut hidden_low = f64::INFINITY;
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
                // Only the kinds of motion the layer allows: a translation-only layer just
                // moves with the surroundings, it doesn't turn or grow with them.
                let m = if s.model == MotionModel::Translation {
                    let c = layer.shape_ref_at(prev).transformed(&h_prev).flatten(8);
                    let n = c.len().max(1) as f64;
                    let o = c
                        .iter()
                        .fold([0.0, 0.0], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n]);
                    let q = apply(&m, o);
                    [1.0, 0.0, q[0] - o[0], 0.0, 1.0, q[1] - o[1], 0.0, 0.0, 1.0]
                } else {
                    m
                };
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
            cues: None,
        };
        layer.tracks.insert(g, ft.clone());
        on_frame(g, &ft);

        // 2. Seen again? Where the prediction says, or (if it moved while hidden, like an
        //    eye turning during a blink) wherever its last look matches best nearby; then
        //    track straight from the anchor with that as the guess.
        let radius = 0.6 * size + 4.0 * k as f64 + 8.0;
        // How well it matches the last frame it was seen, and how much match a spot
        // needs: more the further it is from the prediction (the surroundings' motion
        // is good evidence; a look-alike off to one side is not).
        let like_anchor = |h: &Mat3| {
            looks
                .iter()
                .filter_map(|(f, hf, p)| appearance_vs(layer, p, hf, *f, &g_pyr, h, g, exclude))
                .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))))
        };
        let needed = |d: Point| 0.40 + 0.3 * d[0].hypot(d[1]) / size.max(1.0);
        // For a position found by the search alone (no point tracking to confirm it):
        // back for real: about as like itself (the reference or a clean remembered frame)
        // as two clean views usually are, or clearly more than the best spot was on an
        // earlier hidden frame (it may look a little different now: an eye that turned).
        // Never on the first hidden frame, and not on a lid that has stayed shut: a
        // closing lid matches a closed one, and the frame just before the cover, well
        // enough.
        let usual = |h: &Mat3| {
            looks
                .iter()
                .skip(1)
                .filter_map(|(f, hf, p)| appearance_vs(layer, p, hf, *f, &g_pyr, h, g, exclude))
                .chain(appearance(layer, ref_pyr, &g_pyr, h, g, exclude))
                .fold(0.0, f64::max)
        };
        let low_before = hidden_low;
        let like_usual = |h: &Mat3| {
            usual(h)
                >= (s.tuning.reacquire_level * usual_level)
                    .min(low_before + s.tuning.reacquire_margin)
        };
        // The best spot for any remembered look, by how far it clears what it needs.
        let searched = looks
            .iter()
            .filter_map(|(f, hf, p)| search_shift(layer, p, *f, hf, &g_pyr, g, &h_pred, radius))
            .filter(|(d, sc)| *sc >= needed(*d))
            .max_by(|a, b| (a.1 - needed(a.0)).total_cmp(&(b.1 - needed(b.0))));
        // Nothing nearby looks like it and the predicted spot doesn't either (an eye
        // fully shut): don't spend time trying to track points that aren't there.
        let hopeless = searched.is_none() && like_anchor(&h_pred).is_none_or(|v| v < 0.25);
        let mut guesses = if hopeless { vec![] } else { vec![h_pred] };
        if let Some((d, _)) = searched {
            if d[0].hypot(d[1]) > 2.0 {
                guesses.insert(0, translate(&h_pred, d));
            }
        }
        let found = guesses.iter().enumerate().find_map(|(gi, start)| {
            let guess = invert(&h_anchor)
                .map(|hi| mul(start, &hi))
                .unwrap_or(IDENTITY);
            [normal, retry].iter().enumerate().find_map(|(i, pass)| {
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
                    seed_base ^ (g as u64 * 31 + gi as u64 * 7 + i as u64),
                )
                .ok()?;
                let jump = max_corner_delta(&layer.surface, &r.h_next, start);
                let near = jump <= 0.25 * size + 2.0 * k as f64 + 6.0;
                let look = appearance(layer, &ref_pyr, &g_pyr, &r.h_next, g, exclude)?
                    .max(like_anchor(&r.h_next)?);
                (near && look >= 0.5 && r.inliers >= s.min_inliers).then_some(r)
            })
        });
        // It changed too much to match point by point (an eye that turned while closed)
        // but the search finds it clearly, at the same place on two frames running:
        // take the searched position.
        let steady = |d: Point| {
            last_shift
                .is_some_and(|p: Point| (p[0] - d[0]).hypot(p[1] - d[1]) <= (0.04 * size).max(3.0))
        };
        let found = found.or_else(|| {
            let (d, _) = searched?;
            let h = translate(&h_pred, d);
            (steady(d) && like_anchor(&h)? >= 0.4 && like_usual(&h)).then(|| StepResult {
                h_next: h,
                step: IDENTITY,
                inliers: 0,
                rms: 0.0,
                shown: vec![],
            })
        });
        last_shift = searched.map(|(d, _)| d);
        // How alike the best spot was on this frame, for judging later frames.
        let best_spot = searched.map_or(h_pred, |(d, _)| translate(&h_pred, d));
        hidden_low = hidden_low.min(usual(&best_spot));
        // Confirm on the next frame: still there and looking no worse (an eye opening or
        // open), not getting worse (an eye still closing).
        let found = found.filter(|r| {
            if g == end {
                return true;
            }
            let nxt = (g as isize + step) as usize;
            let (Ok(np), Some(now)) = (
                source.pyramid(nxt, levels),
                appearance(layer, &ref_pyr, &g_pyr, &r.h_next, g, exclude),
            ) else {
                return false;
            };
            let ahead = solve_step(
                layer,
                normal,
                &g_pyr,
                &np,
                ref_pyr,
                g,
                nxt,
                &r.h_next,
                &IDENTITY,
                exclude,
                seed_base ^ 0x5151 ^ g as u64,
            )
            .or_else(|_| {
                solve_step(
                    layer,
                    retry,
                    &g_pyr,
                    &np,
                    ref_pyr,
                    g,
                    nxt,
                    &r.h_next,
                    &IDENTITY,
                    exclude,
                    seed_base ^ 0x5152 ^ g as u64,
                )
            });
            match ahead {
                Ok(a) => appearance(layer, &ref_pyr, &np, &a.h_next, nxt, exclude)
                    .is_some_and(|next| next >= now - 0.05),
                // Too plain to track a step: judge it by the look where it is.
                Err(_) => appearance(layer, &ref_pyr, &np, &r.h_next, nxt, exclude)
                    .is_some_and(|next| next >= now - 0.05),
            }
        });
        if let Some(r) = found {
            // Found by the search alone: keep what it was judged on, for learning.
            let cues = (r.inliers == 0).then(|| {
                let u = usual(&r.h_next);
                Cues {
                    reacquire: Some([
                        (u / usual_level.max(1e-6)) as f32,
                        if low_before.is_finite() {
                            (u - low_before) as f32
                        } else {
                            -1.0
                        },
                    ]),
                    ..Default::default()
                }
            });
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
                cues,
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

//! Automatic point tracking for the camera solve: detect corners over the whole frame,
//! follow them with KLT, drop points that don't move consistently with the camera, and
//! top up as points leave the frame.

use crate::epipolar::ransac_fundamental;
use crate::features::{good_features, FeatureParams};
use crate::geometry::Point;
use crate::klt::{track_point, KltParams};
use crate::planar::{ExcludeFn, FrameSource, StopReason, TrackError};
use crate::sfm::PointTrack;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug)]
pub struct AutoTrackSettings {
    pub max_points: usize,
    /// Minimum spacing (pixels at this footage's resolution).
    pub min_distance: f64,
    pub win_radius: i32,
    pub fb_threshold: f64,
    /// Epipolar (fundamental matrix) inlier threshold in pixels.
    pub epipolar_px: f64,
    pub min_length: usize,
    pub pyramid_levels: usize,
}

impl AutoTrackSettings {
    pub fn for_width(width: usize, max_points: usize, min_distance_at_1080: f64) -> Self {
        let scale = width as f64 / 1920.0;
        Self {
            max_points,
            min_distance: (min_distance_at_1080 * scale).max(6.0),
            win_radius: if width >= 2500 { 11 } else { 9 },
            fb_threshold: 1.0 * scale.max(0.5),
            epipolar_px: 1.5 * scale.max(0.5),
            min_length: 5,
            pyramid_levels: if width >= 2500 { 5 } else { 4 },
        }
    }
}

/// Track points from `start` to `end` (inclusive, either direction).
pub fn auto_track<S: FrameSource>(
    source: &mut S,
    start: usize,
    end: usize,
    s: &AutoTrackSettings,
    exclude: ExcludeFn,
    cancel: &AtomicBool,
    mut progress: impl FnMut(usize, usize),
) -> Result<(Vec<PointTrack>, StopReason), TrackError> {
    let end = end.min(source.frame_count().saturating_sub(1));
    let step: isize = if end >= start { 1 } else { -1 };
    let total = start.abs_diff(end) + 1;
    let fparams = |max: usize| FeatureParams {
        max_features: max,
        quality: 0.005,
        min_distance: s.min_distance as f32,
        border: (s.win_radius + 4) as usize,
        ..Default::default()
    };
    let klt = KltParams {
        win_radius: s.win_radius,
        illumination: true,
        max_residual: 22.0,
        ..Default::default()
    };

    let mut tracks: Vec<PointTrack> = Vec::new();
    // (track index, current position)
    let mut active: Vec<(usize, Point)> = Vec::new();
    let mut next_id = 1u32;

    let mut add_features = |pyr: &crate::image::Pyramid,
                            frame: usize,
                            active: &mut Vec<(usize, Point)>,
                            tracks: &mut Vec<PointTrack>| {
        if active.len() >= s.max_points {
            return;
        }
        let excl = exclude(frame);
        let cands = good_features(
            &pyr.levels[0],
            &pyr.grad_x[0],
            &pyr.grad_y[0],
            None,
            &excl,
            &fparams(s.max_points * 2),
        );
        // Keep away from existing points.
        let cell = s.min_distance.max(1.0);
        let gw = (pyr.width() as f64 / cell).ceil() as usize + 1;
        let gh = (pyr.height() as f64 / cell).ceil() as usize + 1;
        let mut grid = vec![false; gw * gh];
        let mark = |grid: &mut Vec<bool>, p: Point| {
            let (gx, gy) = ((p[0] / cell) as usize, (p[1] / cell) as usize);
            for y in gy.saturating_sub(1)..=(gy + 1).min(gh - 1) {
                for x in gx.saturating_sub(1)..=(gx + 1).min(gw - 1) {
                    grid[y * gw + x] = true;
                }
            }
        };
        for (_, p) in active.iter() {
            mark(&mut grid, *p);
        }
        for p in cands {
            if active.len() >= s.max_points {
                break;
            }
            let (gx, gy) = ((p[0] / cell) as usize, (p[1] / cell) as usize);
            if grid[gy.min(gh - 1) * gw + gx.min(gw - 1)] {
                continue;
            }
            mark(&mut grid, p);
            let mut pts = BTreeMap::new();
            pts.insert(frame, p);
            tracks.push(PointTrack {
                id: next_id,
                points: pts,
            });
            next_id += 1;
            active.push((tracks.len() - 1, p));
        }
    };

    let first = source
        .pyramid(start, s.pyramid_levels)
        .map_err(TrackError::Source)?;
    add_features(&first, start, &mut active, &mut tracks);
    progress(1, total);
    let mut cur = start;
    let mut reason = StopReason::Finished;
    while cur != end {
        if cancel.load(Ordering::Relaxed) {
            reason = StopReason::Cancelled;
            break;
        }
        let next = (cur as isize + step) as usize;
        let a = source
            .pyramid(cur, s.pyramid_levels)
            .map_err(TrackError::Source)?;
        let b = source
            .pyramid(next, s.pyramid_levels)
            .map_err(TrackError::Source)?;
        let results = crate::par::map(&active, |&(ti, p)| {
            let (q, _) = track_point(&a, &b, p, p, &klt)?;
            let (back, _) = track_point(&b, &a, q, p, &klt)?;
            ((back[0] - p[0]).powi(2) + (back[1] - p[1]).powi(2) <= s.fb_threshold * s.fb_threshold)
                .then_some((ti, p, q))
        });
        let moved: Vec<(usize, Point, Point)> = results.into_iter().flatten().collect();
        // Points on moving objects disagree with the camera's epipolar geometry.
        let keep: Vec<bool> = if moved.len() >= 16 {
            let p1: Vec<Point> = moved.iter().map(|m| m.1).collect();
            let p2: Vec<Point> = moved.iter().map(|m| m.2).collect();
            ransac_fundamental(&p1, &p2, s.epipolar_px, next as u64)
                .map(|(_, m)| m)
                .unwrap_or_else(|| vec![true; moved.len()])
        } else {
            vec![true; moved.len()]
        };
        let excl = exclude(next);
        active.clear();
        for ((ti, _, q), k) in moved.into_iter().zip(keep) {
            if !k || excl.iter().any(|e| crate::shape::point_in_polygon(e, q)) {
                continue;
            }
            tracks[ti].points.insert(next, q);
            active.push((ti, q));
        }
        add_features(&b, next, &mut active, &mut tracks);
        cur = next;
        progress(cur.abs_diff(start) + 1, total);
    }
    tracks.retain(|t| t.points.len() >= s.min_length);
    Ok((tracks, reason))
}

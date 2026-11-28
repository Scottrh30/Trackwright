//! Learning from the user's manual tracking.
//!
//! The occlusion solver makes a few judgement calls on every frame: has the surface
//! just been covered (a lid dropping, a hand passing)? Has it slowly disappeared? Is
//! the spot the search found really the surface coming back? Each call compares a
//! score against a threshold. The defaults suit most footage; the right values
//! depend on what is being tracked (a pupil blinks, a talking mouth changes shape all
//! the time, a phone screen never does).
//!
//! While tracking, the solver writes down the scores it based each call on
//! ([`Cues`] on each frame). When the user fixes the track by hand (Adjust Track
//! keys), the fixed corners say what the right answer was. [`harvest`] turns the
//! frames the user worked on into examples: "this frame was fine", "this frame should
//! have been flagged as covered", "this re-find was wrong". [`Profile::fit`] picks the
//! thresholds that would have made the fewest mistakes on those examples, held near
//! the defaults until there is enough evidence. Profiles are kept per [`TargetKind`].
//!
//! Only frames the user touched are used (inside the span of their Adjust keys), so
//! the tracker never learns from its own unchecked output.

use crate::geometry::{apply, Point};
use crate::project::{Layer, TargetKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The solver's adjustable judgement thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tuning {
    /// "Suddenly covered" when the frame-to-frame look drops below this fraction of
    /// its usual level.
    pub sudden: f64,
    /// "Gone" when the look against the reference drops below this fraction of its
    /// recent best.
    pub gone: f64,
    /// A position found by the search alone is accepted when it looks at least this
    /// fraction as alike as two clean views usually are...
    pub reacquire_level: f64,
    /// ...or at least this much more alike than while it was hidden.
    pub reacquire_margin: f64,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            sudden: 0.7,
            gone: 0.45,
            reacquire_level: 0.85,
            reacquire_margin: 0.12,
        }
    }
}

/// The scores behind the solver's calls on one frame (written while tracking).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cues {
    /// Frame-to-frame look as a fraction of its usual level (`None` until settled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sudden: Option<f32>,
    /// Look against the reference as a fraction of its recent best.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gone: Option<f32>,
    /// Where the normal solve put the corners on a frame the solver then called
    /// covered (so a fix can tell a real cover from a false alarm).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected: Option<[Point; 4]>,
    /// Found again by the search alone: (look / usual level, look - look while hidden).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reacquire: Option<[f32; 2]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectExample {
    pub sudden: Option<f32>,
    pub gone: Option<f32>,
    /// True: it was covered or slipping and should have been flagged.
    pub covered: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReacquireExample {
    pub level: f32,
    pub margin: f32,
    /// True: the re-found position was right.
    pub correct: bool,
}

/// Examples from one layer (one harvest).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Harvest {
    pub detect: Vec<DetectExample>,
    pub reacquire: Vec<ReacquireExample>,
}

impl Harvest {
    pub fn len(&self) -> usize {
        self.detect.len() + self.reacquire.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Examples from the frames the user worked on by hand.
pub fn harvest(layer: &Layer) -> Harvest {
    let mut out = Harvest::default();
    let (Some((&first, _)), Some((&last, _))) = (
        layer.adjust_keys.first_key_value(),
        layer.adjust_keys.last_key_value(),
    ) else {
        return out;
    };
    let dist = |a: &[Point; 4], b: &[Point; 4]| {
        (0..4)
            .map(|i| (a[i][0] - b[i][0]).hypot(a[i][1] - b[i][1]))
            .fold(0.0, f64::max)
    };
    // Error of the automatic track against the user's corners.
    let err = |f: usize| -> Option<(f64, f64, [Point; 4])> {
        let t = layer.tracks.get(&f)?;
        let auto = layer.surface.map(|p| apply(&t.h_from_ref, p));
        let truth = layer.surface_at(f)?;
        let size = dist(
            &[truth[0], truth[1], truth[0], truth[1]],
            &[truth[2], truth[3], truth[2], truth[3]],
        );
        let tol = (0.03 * size).max(2.0);
        Some((dist(&auto, &truth), tol, truth))
    };
    let mut prev_err: Option<f64> = None;
    for f in first..=last {
        let Some(t) = layer.tracks.get(&f) else {
            prev_err = None;
            continue;
        };
        let Some((e, tol, truth)) = err(f) else {
            prev_err = None;
            continue;
        };
        if t.keyframe || f == layer.ref_frame {
            prev_err = Some(e);
            continue;
        }
        if let Some(c) = &t.cues {
            if let Some(rej) = &c.rejected {
                // The solver called this frame covered; was the normal solve right?
                out.detect.push(DetectExample {
                    sudden: c.sudden,
                    gone: c.gone,
                    covered: dist(rej, &truth) > tol,
                });
            } else if !t.predicted && (c.sudden.is_some() || c.gone.is_some()) {
                if e <= tol {
                    out.detect.push(DetectExample {
                        sudden: c.sudden,
                        gone: c.gone,
                        covered: false,
                    });
                } else if prev_err.is_none_or(|p| p <= tol || e - p > tol * 0.5) {
                    // Where it went wrong (not every frame of a slip that carried on).
                    out.detect.push(DetectExample {
                        sudden: c.sudden,
                        gone: c.gone,
                        covered: true,
                    });
                }
            }
            if let Some([level, margin]) = c.reacquire {
                out.reacquire.push(ReacquireExample {
                    level,
                    margin,
                    correct: e <= tol,
                });
            }
        }
        prev_err = Some(e);
    }
    out
}

/// What has been learned for one kind of target.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    /// Examples by where they came from (project + layer), so learning from the same
    /// layer again replaces its examples instead of counting them twice.
    pub sources: BTreeMap<String, Harvest>,
    pub tuning: Tuning,
}

/// Most examples kept per profile; the oldest sources go first.
const MAX_EXAMPLES: usize = 20_000;

impl Profile {
    pub fn example_count(&self) -> usize {
        self.sources.values().map(Harvest::len).sum()
    }

    pub fn add(&mut self, source: &str, h: Harvest) {
        self.sources.remove(source);
        if !h.is_empty() {
            self.sources.insert(source.to_string(), h);
        }
        while self.example_count() > MAX_EXAMPLES && self.sources.len() > 1 {
            let k = self.sources.keys().next().cloned().unwrap();
            self.sources.remove(&k);
        }
        self.tuning = self.fit();
    }

    /// Thresholds that would have made the fewest mistakes on the examples. Missing a
    /// cover or accepting a wrong re-find costs twice a false alarm (a guessed frame is
    /// cheap to accept, a slipped track is not). Moving away from the default costs
    /// as much as a few mistakes, so a handful of examples only nudges it.
    pub fn fit(&self) -> Tuning {
        let d = Tuning::default();
        let detect: Vec<&DetectExample> = self.sources.values().flat_map(|h| &h.detect).collect();
        let reacq: Vec<&ReacquireExample> =
            self.sources.values().flat_map(|h| &h.reacquire).collect();
        let prior = 3.0;
        let grid = |lo: f64, hi: f64, step: f64| {
            let n = ((hi - lo) / step).round() as usize;
            (0..=n).map(move |i| lo + step * i as f64)
        };

        let (mut sudden, mut gone) = (d.sudden, d.gone);
        if !detect.is_empty() {
            let mut best = f64::INFINITY;
            for s in grid(0.35, 0.9, 0.01) {
                for g in grid(0.25, 0.65, 0.01) {
                    let mut cost = prior * ((s - d.sudden).abs() + (g - d.gone).abs()) / 0.1;
                    for e in &detect {
                        let flagged = e.sudden.is_some_and(|v| (v as f64) < s)
                            || e.gone.is_some_and(|v| (v as f64) < g);
                        cost += match (e.covered, flagged) {
                            (true, false) => 2.0,
                            (false, true) => 1.0,
                            _ => 0.0,
                        };
                    }
                    if cost < best - 1e-9 {
                        best = cost;
                        (sudden, gone) = (s, g);
                    }
                }
            }
        }

        let (mut level, mut margin) = (d.reacquire_level, d.reacquire_margin);
        if !reacq.is_empty() {
            let mut best = f64::INFINITY;
            for l in grid(0.7, 1.0, 0.01) {
                for m in grid(0.04, 0.4, 0.01) {
                    let mut cost = prior
                        * ((l - d.reacquire_level).abs() / 0.1
                            + (m - d.reacquire_margin).abs() / 0.1);
                    for e in &reacq {
                        let accepted = e.level as f64 >= l || e.margin as f64 >= m;
                        cost += match (e.correct, accepted) {
                            (false, true) => 2.0,
                            (true, false) => 1.0,
                            _ => 0.0,
                        };
                    }
                    if cost < best - 1e-9 {
                        best = cost;
                        (level, margin) = (l, m);
                    }
                }
            }
        }
        Tuning {
            sudden,
            gone,
            reacquire_level: level,
            reacquire_margin: margin,
        }
    }
}

/// Everything learned, by target kind. Saved as one file in the app's settings folder.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LearningStore {
    #[serde(default)]
    pub profiles: BTreeMap<TargetKind, Profile>,
}

impl LearningStore {
    /// Learn from one layer's hand-fixed frames. Returns how many examples it gave.
    pub fn learn(&mut self, source: &str, layer: &Layer) -> usize {
        let h = harvest(layer);
        let n = h.len();
        let kind = layer.settings.target;
        let had = self.profiles.contains_key(&kind);
        if n == 0 && !had {
            return 0;
        }
        self.profiles.entry(kind).or_default().add(source, h);
        if self
            .profiles
            .get(&kind)
            .is_some_and(|p| p.sources.is_empty())
        {
            self.profiles.remove(&kind);
        }
        n
    }

    /// Keep a source's examples for good under a new name, so learning from the same
    /// layer again (after a re-track has absorbed its fixes) adds to them instead of
    /// replacing them. Returns whether there was anything to keep.
    pub fn freeze(&mut self, source: &str) -> bool {
        let mut done = false;
        for p in self.profiles.values_mut() {
            if let Some(h) = p.sources.remove(source) {
                let n = p
                    .sources
                    .keys()
                    .filter(|k| k.starts_with(&format!("{source}#run")))
                    .count();
                p.sources.insert(format!("{source}#run{}", n + 1), h);
                done = true;
            }
        }
        done
    }

    pub fn tuning_for(&self, kind: TargetKind) -> Tuning {
        self.profiles
            .get(&kind)
            .map(|p| p.tuning)
            .unwrap_or_default()
    }

    /// Forget one kind of target, or everything.
    pub fn reset(&mut self, kind: Option<TargetKind>) {
        match kind {
            Some(k) => {
                self.profiles.remove(&k);
            }
            None => self.profiles.clear(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(&serde_json::json!({
            "trackwright_learning": 1,
            "profiles": self.profiles,
        }))
        .unwrap_or_default()
    }

    pub fn from_json(s: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("Not a learning file: {e}"))?;
        if v.get("trackwright_learning").is_none() {
            return Err("Not a Trackwright learning file".into());
        }
        let profiles = serde_json::from_value(v["profiles"].clone())
            .map_err(|e| format!("Learning file is damaged: {e}"))?;
        let mut s = Self { profiles };
        // Recompute in case the fitting has changed since the file was written.
        for p in s.profiles.values_mut() {
            p.tuning = p.fit();
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::IDENTITY;
    use crate::project::FrameTrack;
    use crate::shape::Shape;

    fn ex(sudden: f32, covered: bool) -> DetectExample {
        DetectExample {
            sudden: Some(sudden),
            gone: Some(0.9),
            covered,
        }
    }

    #[test]
    fn no_examples_means_defaults() {
        assert_eq!(Profile::default().fit(), Tuning::default());
    }

    #[test]
    fn a_few_examples_only_nudge() {
        let mut p = Profile::default();
        p.add(
            "a",
            Harvest {
                detect: vec![ex(0.66, false), ex(0.4, true)],
                reacquire: vec![],
            },
        );
        // One false alarm at 0.66 isn't worth a big move.
        assert!((p.tuning.sudden - 0.7).abs() <= 0.06, "{:?}", p.tuning);
    }

    #[test]
    fn many_false_alarms_lower_the_cover_threshold() {
        // A talking mouth: its look drops to 0.6 of usual all the time while fine;
        // real covers drop it to 0.3.
        let mut detect = vec![];
        for i in 0..60 {
            detect.push(ex(0.55 + (i % 10) as f32 * 0.02, false));
        }
        for i in 0..10 {
            detect.push(ex(0.25 + i as f32 * 0.01, true));
        }
        let mut p = Profile::default();
        p.add(
            "m",
            Harvest {
                detect,
                reacquire: vec![],
            },
        );
        assert!(p.tuning.sudden < 0.56, "{:?}", p.tuning);
        assert!(
            p.tuning.sudden > 0.34,
            "still catches real covers: {:?}",
            p.tuning
        );
    }

    #[test]
    fn missed_covers_raise_it() {
        let mut detect = vec![];
        for i in 0..40 {
            detect.push(ex(0.9 + (i % 5) as f32 * 0.02, false));
        }
        for i in 0..20 {
            detect.push(ex(0.72 + (i % 5) as f32 * 0.02, true));
        }
        let mut p = Profile::default();
        p.add(
            "e",
            Harvest {
                detect,
                reacquire: vec![],
            },
        );
        assert!(p.tuning.sudden > 0.8, "{:?}", p.tuning);
    }

    #[test]
    fn wrong_refinds_make_it_stricter() {
        let mut reacquire = vec![];
        for _ in 0..15 {
            reacquire.push(ReacquireExample {
                level: 0.6,
                margin: 0.14,
                correct: false,
            });
        }
        for _ in 0..15 {
            reacquire.push(ReacquireExample {
                level: 0.95,
                margin: 0.3,
                correct: true,
            });
        }
        let mut p = Profile::default();
        p.add(
            "e",
            Harvest {
                detect: vec![],
                reacquire,
            },
        );
        assert!(p.tuning.reacquire_margin > 0.14, "{:?}", p.tuning);
        assert!(p.tuning.reacquire_level <= 0.95, "{:?}", p.tuning);
    }

    #[test]
    fn same_source_replaces_and_reset_forgets() {
        let mut s = LearningStore::default();
        let mut l = Layer::new(1, 0, Shape::rect(0.0, 0.0, 100.0, 100.0));
        l.settings.target = TargetKind::Eye;
        // Auto track: identity on frames 0..10, user moved frame 5 by 20 px.
        for f in 1..10 {
            l.tracks.insert(
                f,
                FrameTrack {
                    cues: Some(Cues {
                        sudden: Some(0.95),
                        gone: Some(0.95),
                        ..Default::default()
                    }),
                    ..FrameTrack::reference()
                },
            );
            l.tracks.get_mut(&f).unwrap().keyframe = false;
        }
        assert!(l.add_adjust_key(2));
        assert!(l.set_adjust_corner(5, 0, [20.0, 20.0]));
        assert!(l.add_adjust_key(8));
        let n = s.learn("p#1", &l);
        assert!(n > 0);
        let again = s.learn("p#1", &l);
        assert_eq!(again, n);
        assert_eq!(s.profiles[&TargetKind::Eye].example_count(), n);
        let covered = s.profiles[&TargetKind::Eye].sources["p#1"]
            .detect
            .iter()
            .filter(|e| e.covered)
            .count();
        assert!(covered >= 1, "the moved frame counts as a miss");
        let back = LearningStore::from_json(&s.to_json()).unwrap();
        assert_eq!(back.profiles[&TargetKind::Eye].example_count(), n);
        s.reset(Some(TargetKind::Eye));
        assert!(s.profiles.is_empty());
        assert_eq!(s.tuning_for(TargetKind::Eye), Tuning::default());
        let _ = IDENTITY;
    }
}

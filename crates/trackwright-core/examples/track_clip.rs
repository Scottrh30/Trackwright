//! Track a layer through a folder of real frames and print what happened on
//! each frame, for testing the tracker on footage outside the app.
//!
//!   cargo run --release --example track_clip --features parallel -- \
//!       FRAMES_DIR START END x0,y0,x1,y1[,x2,y2,...] [out.json] [key=value ...]
//!
//! Settings: bridge=0|1, model=perspective|affine|similarity|translation,
//! max_bridge=N, min_inliers=N, max_features=N.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use trackwright_core::geometry::{apply, MotionModel, Point};
use trackwright_core::*;

struct Folder {
    frames: Vec<std::path::PathBuf>,
    cache: HashMap<usize, Arc<Pyramid>>,
    order: VecDeque<usize>,
}

impl FrameSource for Folder {
    fn frame_count(&self) -> usize {
        self.frames.len()
    }
    fn pyramid(&mut self, index: usize, levels: usize) -> Result<Arc<Pyramid>, String> {
        if let Some(p) = self.cache.get(&index) {
            if p.levels.len() >= levels {
                return Ok(p.clone());
            }
        }
        let path = self.frames.get(index).ok_or("out of range")?;
        let img = ::image::open(path).map_err(|e| e.to_string())?.to_luma8();
        let (w, h) = img.dimensions();
        let p = Arc::new(Pyramid::build(
            GrayImage::from_luma8(w as usize, h as usize, img.as_raw()),
            levels,
        ));
        self.cache.insert(index, p.clone());
        self.order.push_back(index);
        while self.order.len() > 6 {
            if let Some(o) = self.order.pop_front() {
                self.cache.remove(&o);
            }
        }
        Ok(p)
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!(
            "usage: track_clip FRAMES_DIR START END x0,y0,x1,y1,... [out.json] [key=value ...]"
        );
        std::process::exit(2);
    }
    let mut frames: Vec<_> = std::fs::read_dir(&a[1])
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jpg" || x == "png"))
        .collect();
    frames.sort();
    let (start, end): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let nums: Vec<f64> = a[4].split(',').map(|v| v.parse().unwrap()).collect();
    let pts: Vec<Point> = nums.chunks(2).map(|c| [c[0], c[1]]).collect();
    let shape = if pts.len() == 2 {
        Shape::rect(
            pts[0][0],
            pts[0][1],
            pts[1][0] - pts[0][0],
            pts[1][1] - pts[0][1],
        )
    } else {
        Shape {
            points: pts,
            closed: true,
            smooth: false,
        }
    };
    let mut layer = Layer::new(1, start, shape);
    let out = a.get(5).filter(|s| !s.contains('=')).cloned();
    for kv in a.iter().skip(5).filter(|s| s.contains('=')) {
        let (k, v) = kv.split_once('=').unwrap();
        let s = &mut layer.settings;
        match k {
            "bridge" => s.bridge_occlusions = v == "1",
            "max_bridge" => s.max_bridge_frames = v.parse().unwrap(),
            "min_inliers" => s.min_inliers = v.parse().unwrap(),
            "max_features" => s.max_features = v.parse().unwrap(),
            "model" => {
                s.model = match v {
                    "affine" => MotionModel::Affine,
                    "similarity" => MotionModel::Similarity,
                    "translation" => MotionModel::Translation,
                    _ => MotionModel::Perspective,
                }
            }
            _ => panic!("unknown setting {k}"),
        }
    }
    let mut src = Folder {
        frames,
        cache: HashMap::new(),
        order: VecDeque::new(),
    };
    let cancel = AtomicBool::new(false);
    let t0 = std::time::Instant::now();
    let surf = layer.surface;
    let r = track_layer(
        &mut src,
        &mut layer,
        start,
        end,
        &no_exclusions,
        &cancel,
        |f, t| {
            let c = layer_centre(t, &surf);
            eprintln!(
                "{f:4} {} inl {:4} rms {:.2} centre ~({:.0},{:.0})",
                if t.predicted { "PRED" } else { "seen" },
                t.inliers,
                t.rms_error,
                c[0],
                c[1]
            );
        },
    );
    let secs = t0.elapsed().as_secs_f64();
    match &r {
        Ok(s) => println!("result: {s:?} in {secs:.1}s"),
        Err(e) => println!("result: ERROR {e} (after {secs:.1}s)"),
    }
    let pred: Vec<usize> = layer
        .tracks
        .iter()
        .filter(|(_, t)| t.predicted)
        .map(|(f, _)| *f)
        .collect();
    println!(
        "tracked {} frames, predicted {:?}",
        layer.tracks.len(),
        pred
    );
    if let Some(path) = out {
        let refp = src.pyramid(layer.ref_frame, 4).unwrap();
        let rows: Vec<serde_json::Value> = layer
            .tracks
            .iter()
            .map(|(f, t)| {
                let c = layer.surface.map(|p| apply(&t.h_from_ref, p));
                let fp = src.pyramid(*f, 4).unwrap();
                let look = trackwright_core::planar::appearance_score(&layer, &refp, &fp, &t.h_from_ref, *f);
                // Against the frame before (same direction as tracked).
                let pf = if start <= end { f.checked_sub(1) } else { Some(f + 1) };
                let look_prev = pf.and_then(|pf| {
                    let pt = layer.tracks.get(&pf)?;
                    let pp = src.pyramid(pf, 4).ok()?;
                    trackwright_core::planar::appearance_between(&layer, &pp, &pt.h_from_ref, pf, &fp, &t.h_from_ref, *f)
                });
                serde_json::json!({ "frame": f, "corners": c, "predicted": t.predicted, "inliers": t.inliers, "rms": t.rms_error, "look": look, "look_prev": look_prev,
                    "shape": layer.shape.transformed(&t.h_from_ref).points })
            })
            .collect();
        std::fs::write(path, serde_json::to_string(&rows).unwrap()).unwrap();
    }
}

fn layer_centre(t: &FrameTrack, surf: &[Point; 4]) -> Point {
    // Where the middle of the surface is in this frame.
    let c = [
        surf.iter().map(|p| p[0]).sum::<f64>() / 4.0,
        surf.iter().map(|p| p[1]).sum::<f64>() / 4.0,
    ];
    apply(&t.h_from_ref, c)
}

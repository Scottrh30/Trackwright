//! AllTracker (Harley et al., ICCV 2025; MIT licence) run through ONNX Runtime.
//!
//! The model comes as two ONNX files exported from the authors' weights (see
//! `tools/ai-model`): an image encoder and a 16-frame tracking window. This crate
//! runs them the same way the authors' `forward_sliding` does: windows of 16 frames,
//! each new window overlapping the last by half and starting from its estimates,
//! all tracked from the first frame. ONNX Runtime itself is a separate library
//! (`onnxruntime.dll`) loaded when the tracker is created.

use std::path::Path;
use std::sync::{Mutex, OnceLock};
use trackwright_core::ai::PointTracker;
use trackwright_core::geometry::Point;

use ort::session::Session;
use ort::value::Tensor;

/// Frame size the exported model works at.
pub const WIDTH: usize = 384;
pub const HEIGHT: usize = 256;
/// Frames per tracking window.
const S: usize = 16;
const C: usize = 256;
const H8: usize = HEIGHT / 8;
const W8: usize = WIDTH / 8;

static RUNTIME: OnceLock<Result<ort::environment::Environment, String>> = OnceLock::new();

/// Load ONNX Runtime from `library` (only the first call loads it; it can't be
/// swapped while the app runs).
fn runtime(library: &Path) -> Result<&'static ort::environment::Environment, String> {
    RUNTIME
        .get_or_init(|| {
            ort::init_from(library)
                .map_err(|e| format!("Could not load ONNX Runtime ({}): {e}", library.display()))?
                .with_name("trackwright")
                .build()
                .map_err(|e| format!("Could not start ONNX Runtime: {e}"))
        })
        .as_ref()
        .map_err(|e| e.clone())
}

pub struct AllTracker {
    encoder: Mutex<Session>,
    window: Mutex<Session>,
}

impl AllTracker {
    /// `library`: onnxruntime.dll / libonnxruntime.so; `encoder`, `window`: the model.
    pub fn load(
        library: &Path,
        encoder: &Path,
        window: &Path,
        threads: usize,
    ) -> Result<Self, String> {
        let env = runtime(library)?;
        let open = |p: &Path| -> Result<Session, String> {
            Session::builder(env)
                .map_err(|e| e.to_string())?
                .with_intra_threads(threads.max(1))
                .map_err(|e| e.to_string())?
                .commit_from_file(p)
                .map_err(|e| format!("Could not load the AI model {}: {e}", p.display()))
        };
        Ok(Self {
            encoder: Mutex::new(open(encoder)?),
            window: Mutex::new(open(window)?),
        })
    }

    /// Feature maps for `n` normalised frames (n x 3 x H x W).
    fn encode(&self, images: Vec<f32>, n: usize) -> Result<Vec<f32>, String> {
        let t = Tensor::from_array(([n, 3, HEIGHT, WIDTH], images)).map_err(|e| e.to_string())?;
        let mut s = self.encoder.lock().unwrap();
        let out = s
            .run(ort::inputs!["images" => t])
            .map_err(|e| e.to_string())?;
        let (_, d) = out["fmaps"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        Ok(d.to_vec())
    }

    /// Dense tracks from frame 0: per frame, flow (x then y planes) and visibility.
    fn slide(&self, frames: &[Vec<f32>]) -> Result<Vec<(Vec<f32>, Vec<f32>)>, String> {
        let t = frames.len();
        // ImageNet normalisation of a grey frame repeated in all three channels.
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let norm = |f: &Vec<f32>, out: &mut Vec<f32>| {
            for c in 0..3 {
                out.extend(f.iter().map(|v| (v / 255.0 - mean[c]) / std[c]));
            }
        };
        let stride = S / 2;
        let mut starts = vec![];
        let mut s0 = 0;
        while s0 + S < t {
            starts.push(s0);
            s0 += stride;
        }
        starts.push(s0);
        let tp = s0 + S;
        let frame = |i: usize| &frames[i.min(t - 1)];
        let encode_range = |a: usize, b: usize| -> Result<Vec<f32>, String> {
            let mut buf = Vec::with_capacity((b - a) * 3 * HEIGHT * WIDTH);
            for i in a..b {
                norm(frame(i), &mut buf);
            }
            self.encode(buf, b - a)
        };
        let anchor = encode_range(0, 1)?;
        let per = C * H8 * W8;
        let plane8 = 2 * H8 * W8;
        let mut out: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![None; tp];
        let mut fmaps: Vec<f32> = vec![];
        let mut flows8 = vec![0f32; S * plane8];
        let mut vis8 = vec![0f32; S * plane8];
        for (ii, &ind) in starts.iter().enumerate() {
            if ii == 0 {
                fmaps = encode_range(ind, ind + S)?;
            } else {
                // Keep the second half of the last window, start the new half from
                // its last estimate.
                let keep = |v: &Vec<f32>, n: usize| {
                    let mut o = v[stride * n..(stride + S / 2) * n].to_vec();
                    let last = &v[(stride + S / 2 - 1) * n..(stride + S / 2) * n];
                    for _ in 0..S / 2 {
                        o.extend_from_slice(last);
                    }
                    o
                };
                flows8 = keep(&flows8, plane8);
                vis8 = keep(&vis8, plane8);
                let mut f = fmaps[stride * per..(stride + S / 2) * per].to_vec();
                f.extend(encode_range(ind + S / 2, ind + S)?);
                fmaps = f;
            }
            let mut w = self.window.lock().unwrap();
            let inputs = ort::inputs![
                "fmap_anchor" => Tensor::from_array(([1, C, H8, W8], anchor.clone())).map_err(|e| e.to_string())?,
                "fmaps" => Tensor::from_array(([1, S, C, H8, W8], fmaps.clone())).map_err(|e| e.to_string())?,
                "flows8" => Tensor::from_array(([S, 2, H8, W8], flows8.clone())).map_err(|e| e.to_string())?,
                "visconfs8" => Tensor::from_array(([S, 2, H8, W8], vis8.clone())).map_err(|e| e.to_string())?,
            ];
            let res = w.run(inputs).map_err(|e| e.to_string())?;
            let get = |name: &str| -> Result<Vec<f32>, String> {
                Ok(res[name]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| e.to_string())?
                    .1
                    .to_vec())
            };
            let flows = get("flows")?;
            let vis = get("visconfs")?;
            flows8 = get("flows8_out")?;
            vis8 = get("visconfs8_out")?;
            drop(res);
            drop(w);
            let plane = 2 * HEIGHT * WIDTH;
            for k in 0..S {
                let f = ind + k;
                if out[f].is_none() {
                    out[f] = Some((
                        flows[k * plane..(k + 1) * plane].to_vec(),
                        vis[k * plane..(k + 1) * plane].to_vec(),
                    ));
                }
            }
        }
        Ok(out
            .into_iter()
            .take(t)
            .map(|o| o.unwrap_or_default())
            .collect())
    }
}

fn bilinear(plane: &[f32], x: f64, y: f64) -> f32 {
    let x = x.clamp(0.0, (WIDTH - 1) as f64);
    let y = y.clamp(0.0, (HEIGHT - 1) as f64);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(WIDTH - 1), (y0 + 1).min(HEIGHT - 1));
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
    let at = |xx: usize, yy: usize| plane[yy * WIDTH + xx];
    let a = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
    let b = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
    a * (1.0 - fy) + b * fy
}

impl PointTracker for AllTracker {
    fn frame_size(&self) -> (usize, usize) {
        (WIDTH, HEIGHT)
    }

    fn track(
        &self,
        frames: &[Vec<f32>],
        points: &[Point],
    ) -> Result<Vec<Vec<(Point, f32)>>, String> {
        if frames.is_empty() {
            return Ok(vec![]);
        }
        if frames.iter().any(|f| f.len() != WIDTH * HEIGHT) {
            return Err(format!("AI tracker frames must be {WIDTH}x{HEIGHT}"));
        }
        let dense = self.slide(frames)?;
        let n = WIDTH * HEIGHT;
        Ok(dense
            .iter()
            .map(|(flow, vis)| {
                points
                    .iter()
                    .map(|p| {
                        if flow.is_empty() {
                            return (*p, 0.0);
                        }
                        // Pixel centres: the flow at pixel (x, y) is for point (x, y).
                        let dx = bilinear(&flow[..n], p[0], p[1]) as f64;
                        let dy = bilinear(&flow[n..], p[0], p[1]) as f64;
                        // Channel 1 of the visibility/confidence output is visibility.
                        let v = bilinear(&vis[n..], p[0], p[1]);
                        ([p[0] + dx, p[1] + dy], v)
                    })
                    .collect()
            })
            .collect())
    }
}

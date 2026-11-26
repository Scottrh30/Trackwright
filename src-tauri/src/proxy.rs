//! Low-resolution proxy frames for smooth playback and scrubbing.

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Proxies are only built when the footage is bigger than this on its long side.
pub const PROXY_MAX_DIM: u32 = 1280;

#[derive(Clone, Debug, Default, Serialize)]
pub struct ProxyInfo {
    /// Path prefix; frame `i` is at `{prefix}{i:06}.jpg`. `None` = no proxies needed.
    pub prefix: Option<String>,
    /// Frames 0..ready are available.
    pub ready: usize,
    pub total: usize,
}

pub struct ProxyBuilder {
    generation: Arc<AtomicU64>,
    pub info: Arc<Mutex<ProxyInfo>>,
}

impl ProxyBuilder {
    pub fn new() -> Self {
        Self {
            generation: Arc::new(AtomicU64::new(0)),
            info: Arc::new(Mutex::new(ProxyInfo::default())),
        }
    }

    /// Start (or reuse) proxies for `frames`. `on_progress` is called from a worker thread.
    pub fn start(
        &self,
        frames: Vec<String>,
        width: u32,
        height: u32,
        cache_root: PathBuf,
        on_progress: impl Fn(ProxyInfo) + Send + 'static,
    ) -> Option<PathBuf> {
        let gen = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let total = frames.len();
        if width.max(height) <= PROXY_MAX_DIM || total == 0 {
            *self.info.lock().unwrap() = ProxyInfo {
                prefix: None,
                ready: total,
                total,
            };
            on_progress(self.info.lock().unwrap().clone());
            return None;
        }
        let mut hash: u64 = 1469598103934665603;
        for f in [frames.first(), frames.last()].into_iter().flatten() {
            for b in f.bytes() {
                hash = (hash ^ b as u64).wrapping_mul(1099511628211);
            }
        }
        hash ^= total as u64;
        if let Some(m) = frames
            .first()
            .and_then(|f| std::fs::metadata(f).ok())
            .and_then(|m| m.modified().ok())
        {
            if let Ok(d) = m.duration_since(std::time::UNIX_EPOCH) {
                hash = (hash ^ d.as_secs()).wrapping_mul(1099511628211);
            }
        }
        let dir = cache_root.join(format!("{hash:016x}"));
        let _ = std::fs::create_dir_all(&dir);
        let prefix = dir.join("p_").to_string_lossy().to_string();
        *self.info.lock().unwrap() = ProxyInfo {
            prefix: Some(prefix.clone()),
            ready: 0,
            total,
        };

        let generation = self.generation.clone();
        let info = self.info.clone();
        let scale = PROXY_MAX_DIM as f64 / width.max(height) as f64;
        let (pw, ph) = (
            ((width as f64 * scale).round() as u32).max(1),
            ((height as f64 * scale).round() as u32).max(1),
        );
        std::thread::spawn(move || {
            for (i, src) in frames.iter().enumerate() {
                if generation.load(Ordering::SeqCst) != gen {
                    return;
                }
                let out = PathBuf::from(format!("{prefix}{i:06}.jpg"));
                if !out.exists() {
                    let _ = make_proxy(Path::new(src), &out, pw, ph);
                }
                let snapshot = {
                    let mut inf = info.lock().unwrap();
                    inf.ready = i + 1;
                    inf.clone()
                };
                if i % 8 == 7 || i + 1 == frames.len() {
                    on_progress(snapshot);
                }
            }
        });
        Some(dir)
    }

    pub fn cancel(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        *self.info.lock().unwrap() = ProxyInfo::default();
    }
}

fn make_proxy(src: &Path, out: &Path, w: u32, h: u32) -> Result<(), String> {
    let img = image::open(src).map_err(|e| e.to_string())?.to_rgb8();
    let small = image::imageops::thumbnail(&img, w, h);
    let tmp = out.with_extension("tmp");
    let file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    let mut enc =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 85);
    enc.encode_image(&small).map_err(|e| e.to_string())?;
    drop(enc);
    std::fs::rename(&tmp, out).map_err(|e| e.to_string())
}

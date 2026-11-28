//! The optional AI tracker: downloading the model, loading it, and its status.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};

const RELEASE: &str = "https://github.com/Scottrh30/Trackwright/releases/download/ai-models";
const FILES: [&str; 2] = ["at_encoder.onnx", "at_window.onnx"];
/// Shown before the download.
pub const DOWNLOAD_MB: u32 = 66;

#[derive(Clone, Serialize, Default)]
pub struct AiStatus {
    /// ONNX Runtime library found next to the app.
    pub runtime: bool,
    /// Model files downloaded.
    pub downloaded: bool,
    /// Loaded and in use.
    pub ready: bool,
    pub downloading: bool,
    pub message: String,
    pub size_mb: u32,
}

#[derive(Default)]
pub struct AiState {
    pub status: Mutex<AiStatus>,
}

fn model_dir(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join("ai"))
}

/// The ONNX Runtime library: next to trackwright.exe (zip and installer), in the
/// app's resource folder, or named by TW_ONNXRUNTIME (development).
pub fn runtime_path(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("TW_ONNXRUNTIME") {
        return Some(PathBuf::from(p)).filter(|p| p.exists());
    }
    let name = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    let mut dirs = vec![];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs.push(d.to_path_buf());
        }
    }
    if let Ok(r) = app.path().resource_dir() {
        dirs.push(r);
    }
    dirs.into_iter().map(|d| d.join(name)).find(|p| p.exists())
}

fn downloaded(app: &AppHandle) -> bool {
    model_dir(app).is_some_and(|d| FILES.iter().all(|f| d.join(f).exists()))
}

/// Refresh the status (and load the model if it's there and not loaded yet).
pub fn refresh(app: &AppHandle) -> AiStatus {
    let state = app.state::<AiState>();
    let mut st = state.status.lock().unwrap().clone();
    st.size_mb = DOWNLOAD_MB;
    st.runtime = runtime_path(app).is_some();
    st.downloaded = downloaded(app);
    st.ready = trackwright_core::ai::point_tracker().is_some();
    if !st.ready && st.downloaded && st.runtime && !st.downloading {
        match load(app) {
            Ok(()) => {
                st.ready = true;
                st.message = "AI tracker ready".into();
            }
            Err(e) => st.message = e,
        }
    } else if !st.runtime && !st.ready {
        st.message = "The AI tracker needs onnxruntime.dll next to trackwright.exe (it comes with the download zip and the installer)".into();
    } else if !st.downloaded && !st.downloading {
        st.message = format!("Not downloaded yet ({} MB)", DOWNLOAD_MB);
    }
    *state.status.lock().unwrap() = st.clone();
    st
}

fn load(app: &AppHandle) -> Result<(), String> {
    let rt = runtime_path(app).ok_or("ONNX Runtime not found")?;
    let dir = model_dir(app).ok_or("No app data folder")?;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let t =
        trackwright_ai::AllTracker::load(&rt, &dir.join(FILES[0]), &dir.join(FILES[1]), threads)?;
    trackwright_core::ai::set_point_tracker(Some(std::sync::Arc::new(t)));
    Ok(())
}

#[derive(Clone, Serialize)]
struct Progress {
    done_mb: f64,
    total_mb: f64,
}

fn fetch(url: &str) -> Result<ureq::Response, String> {
    ureq::get(url)
        .call()
        .map_err(|e| format!("Download failed ({url}): {e}"))
}

/// Download the model files (checked against the release's SHA256SUMS), then load.
pub fn download(app: &AppHandle) -> Result<AiStatus, String> {
    {
        let state = app.state::<AiState>();
        let mut st = state.status.lock().unwrap();
        if st.downloading {
            return Err("Already downloading".into());
        }
        st.downloading = true;
    }
    let result = (|| -> Result<(), String> {
        let dir = model_dir(app).ok_or("No app data folder")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut sums = String::new();
        fetch(&format!("{RELEASE}/SHA256SUMS.txt"))?
            .into_reader()
            .take(64 * 1024)
            .read_to_string(&mut sums)
            .map_err(|e| e.to_string())?;
        let want = |name: &str| {
            sums.lines()
                .find(|l| l.trim_end().ends_with(name))
                .and_then(|l| l.split_whitespace().next())
                .map(str::to_lowercase)
        };
        let mut done = 0u64;
        let total = DOWNLOAD_MB as f64;
        for name in FILES {
            let expect =
                want(name).ok_or(format!("{name} is not in the release's checksum list"))?;
            let tmp = dir.join(format!("{name}.part"));
            let mut out = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            let mut r = fetch(&format!("{RELEASE}/{name}"))?.into_reader();
            let mut hasher = Sha256::new();
            let mut buf = vec![0u8; 1 << 20];
            let mut last = std::time::Instant::now();
            loop {
                let n = r
                    .read(&mut buf)
                    .map_err(|e| format!("Download interrupted: {e}"))?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
                std::io::Write::write_all(&mut out, &buf[..n]).map_err(|e| e.to_string())?;
                done += n as u64;
                if last.elapsed().as_millis() > 200 {
                    last = std::time::Instant::now();
                    let _ = app.emit(
                        "ai-progress",
                        Progress {
                            done_mb: done as f64 / 1e6,
                            total_mb: total,
                        },
                    );
                }
            }
            drop(out);
            let got = format!("{:x}", hasher.finalize());
            if got != expect {
                let _ = std::fs::remove_file(&tmp);
                return Err(format!(
                    "{name} didn't download correctly (checksum mismatch); try again"
                ));
            }
            std::fs::rename(&tmp, dir.join(name)).map_err(|e| e.to_string())?;
        }
        Ok(())
    })();
    app.state::<AiState>().status.lock().unwrap().downloading = false;
    result?;
    let st = refresh(app);
    if !st.ready {
        return Err(st.message);
    }
    Ok(st)
}

/// Forget the model: unload it and delete the files.
pub fn remove(app: &AppHandle) -> AiStatus {
    trackwright_core::ai::set_point_tracker(None);
    if let Some(d) = model_dir(app) {
        for f in FILES {
            let _ = std::fs::remove_file(d.join(f));
        }
    }
    let st = refresh(app);
    let state = app.state::<AiState>();
    let mut s = state.status.lock().unwrap();
    s.message = format!("Removed. Download again any time ({} MB)", DOWNLOAD_MB);
    let mut out = st;
    out.message = s.message.clone();
    out
}

//! Trackwright desktop app (Tauri backend).
//!
//! © Scott Hambach. All rights reserved. "Trackwright" is a trademark of Scott Hambach.

mod footage;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;
use trackwright_core::export::{self, ExportFormat};
use trackwright_core::{
    track_layer, FrameSource, FrameTrack, GrayImage, Layer, Project, Pyramid, Shape, StopReason,
};

// ---------------------------------------------------------------------------
// Frame cache

struct FrameCache {
    frames: Vec<String>,
    cache: HashMap<usize, Arc<Pyramid>>,
    order: VecDeque<usize>,
    capacity: usize,
}

impl FrameCache {
    fn new() -> Self {
        Self {
            frames: vec![],
            cache: HashMap::new(),
            order: VecDeque::new(),
            capacity: 8,
        }
    }

    fn reset(&mut self, frames: Vec<String>) {
        self.frames = frames;
        self.cache.clear();
        self.order.clear();
    }
}

impl FrameSource for FrameCache {
    fn frame_count(&self) -> usize {
        self.frames.len()
    }

    fn pyramid(&mut self, index: usize, levels: usize) -> Result<Arc<Pyramid>, String> {
        if let Some(p) = self.cache.get(&index) {
            return Ok(p.clone());
        }
        let path = self
            .frames
            .get(index)
            .ok_or_else(|| format!("frame {index} is out of range"))?;
        let img = image::open(path)
            .map_err(|e| format!("{path}: {e}"))?
            .to_luma8();
        let (w, h) = img.dimensions();
        let gray = GrayImage::from_luma8(w as usize, h as usize, img.as_raw());
        let pyr = Arc::new(Pyramid::build(gray, levels));
        self.cache.insert(index, pyr.clone());
        self.order.push_back(index);
        while self.order.len() > self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.cache.remove(&old);
            }
        }
        Ok(pyr)
    }
}

// ---------------------------------------------------------------------------
// Settings

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub reelforge_dir: String,
    pub forma3d_dir: String,
    pub tidefire_dir: String,
    pub default_format: ExportFormat,
}

impl Default for AppSettings {
    fn default() -> Self {
        let docs = dirs::document_dir()
            .or_else(dirs::home_dir)
            .unwrap_or_else(|| PathBuf::from("."));
        let d = |app: &str| {
            docs.join(app)
                .join("Imports")
                .join("Tracks")
                .to_string_lossy()
                .to_string()
        };
        Self {
            reelforge_dir: d("Reelforge"),
            forma3d_dir: d("Forma3D"),
            tidefire_dir: d("Tidefire"),
            default_format: ExportFormat::Reelforge,
        }
    }
}

// ---------------------------------------------------------------------------
// State

struct AppState {
    project: Mutex<Project>,
    project_path: Mutex<Option<PathBuf>>,
    cache: Arc<Mutex<FrameCache>>,
    cancel: Arc<AtomicBool>,
    tracking: Arc<AtomicBool>,
    settings: Mutex<AppSettings>,
}

type CmdResult<T> = Result<T, String>;

fn settings_path(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|d| d.join("settings.json"))
}

fn load_settings(app: &AppHandle) -> AppSettings {
    settings_path(app)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn ensure_not_tracking(state: &AppState) -> CmdResult<()> {
    if state.tracking.load(Ordering::SeqCst) {
        Err("Stop tracking first".into())
    } else {
        Ok(())
    }
}

/// Let the webview display frames from this folder through the asset protocol.
fn allow_frames(app: &AppHandle, frames: &[String]) {
    let scope = app.asset_protocol_scope();
    let mut dirs: Vec<PathBuf> = frames
        .iter()
        .filter_map(|f| Path::new(f).parent().map(|p| p.to_path_buf()))
        .collect();
    dirs.sort();
    dirs.dedup();
    for d in dirs {
        let _ = scope.allow_directory(&d, false);
    }
}

fn install_footage(app: &AppHandle, state: &AppState, f: trackwright_core::Footage) -> Project {
    allow_frames(app, &f.frames);
    state.cache.lock().unwrap().reset(f.frames.clone());
    let mut p = state.project.lock().unwrap();
    let count = f.frames.len();
    // Keep layers when re-linking footage of the same length; otherwise start fresh.
    if p.footage.as_ref().map(|o| o.frames.len()) != Some(count) {
        p.layers.clear();
    }
    p.footage = Some(f);
    p.clone()
}

// ---------------------------------------------------------------------------
// Commands: project & footage

#[tauri::command]
fn get_project(state: State<'_, AppState>) -> Project {
    state.project.lock().unwrap().clone()
}

#[tauri::command]
fn new_project(state: State<'_, AppState>) -> CmdResult<Project> {
    ensure_not_tracking(&state)?;
    *state.project.lock().unwrap() = Project::default();
    *state.project_path.lock().unwrap() = None;
    state.cache.lock().unwrap().reset(vec![]);
    Ok(Project::default())
}

/// `kind`: "file" (video / image / sequence frame) or "folder".
#[tauri::command]
async fn open_footage(
    app: AppHandle,
    state: State<'_, AppState>,
    kind: String,
    path: Option<String>,
) -> CmdResult<Project> {
    ensure_not_tracking(&state)?;
    let path = match path {
        Some(p) => PathBuf::from(p),
        None => {
            let picked = if kind == "folder" {
                app.dialog()
                    .file()
                    .set_title("Open image sequence folder")
                    .blocking_pick_folder()
            } else {
                app.dialog()
                    .file()
                    .set_title("Open footage")
                    .add_filter(
                        "Footage",
                        &[
                            "mp4", "mov", "mkv", "avi", "webm", "m4v", "mxf", "png", "jpg", "jpeg",
                            "tif", "tiff", "bmp", "webp",
                        ],
                    )
                    .blocking_pick_file()
            };
            match picked.and_then(|p| p.into_path().ok()) {
                Some(p) => p,
                None => return Ok(state.project.lock().unwrap().clone()),
            }
        }
    };
    let cache_root = app
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?
        .join("frames");
    let f = tauri::async_runtime::spawn_blocking(move || footage::open_any(&path, &cache_root))
        .await
        .map_err(|e| e.to_string())??;
    Ok(install_footage(&app, &state, f))
}

#[tauri::command]
fn set_footage_timing(
    state: State<'_, AppState>,
    fps: f64,
    first_frame: i64,
) -> CmdResult<Project> {
    let mut p = state.project.lock().unwrap();
    let f = p.footage.as_mut().ok_or("Open footage first")?;
    if fps > 0.0 && fps < 1000.0 {
        f.fps = fps;
    }
    f.first_frame = first_frame;
    Ok(p.clone())
}

#[tauri::command]
async fn save_project(
    app: AppHandle,
    state: State<'_, AppState>,
    save_as: bool,
) -> CmdResult<Option<String>> {
    let existing = state.project_path.lock().unwrap().clone();
    let path = match (existing, save_as) {
        (Some(p), false) => p,
        _ => {
            let picked = app
                .dialog()
                .file()
                .set_title("Save Trackwright project")
                .add_filter("Trackwright project", &["twproj"])
                .set_file_name("untitled.twproj")
                .blocking_save_file();
            match picked.and_then(|p| p.into_path().ok()) {
                Some(mut p) => {
                    if p.extension().is_none() {
                        p.set_extension("twproj");
                    }
                    p
                }
                None => return Ok(None),
            }
        }
    };
    let json = state.project.lock().unwrap().to_json();
    std::fs::write(&path, json).map_err(|e| format!("Could not save: {e}"))?;
    *state.project_path.lock().unwrap() = Some(path.clone());
    Ok(Some(path.to_string_lossy().to_string()))
}

#[tauri::command]
async fn open_project(app: AppHandle, state: State<'_, AppState>) -> CmdResult<Project> {
    ensure_not_tracking(&state)?;
    let picked = app
        .dialog()
        .file()
        .set_title("Open Trackwright project")
        .add_filter("Trackwright project", &["twproj"])
        .blocking_pick_file();
    let Some(path) = picked.and_then(|p| p.into_path().ok()) else {
        return Ok(state.project.lock().unwrap().clone());
    };
    let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let project = Project::from_json(&text)?;
    if let Some(f) = &project.footage {
        if let Some(missing) = f.frames.iter().find(|p| !Path::new(p).exists()) {
            return Err(format!("Footage is missing ({missing}). Re-open the original footage, then load the project again"));
        }
        allow_frames(&app, &f.frames);
        state.cache.lock().unwrap().reset(f.frames.clone());
    }
    *state.project.lock().unwrap() = project.clone();
    *state.project_path.lock().unwrap() = Some(path);
    Ok(project)
}

// ---------------------------------------------------------------------------
// Commands: layers

#[tauri::command]
fn add_layer(
    state: State<'_, AppState>,
    frame: usize,
    points: Vec<[f64; 2]>,
) -> CmdResult<Project> {
    ensure_not_tracking(&state)?;
    let mut p = state.project.lock().unwrap();
    if p.footage.is_none() {
        return Err("Open footage first".into());
    }
    let shape = Shape::new(points);
    if !shape.is_valid() {
        return Err("A shape needs at least 3 points".into());
    }
    p.add_layer(frame, shape);
    Ok(p.clone())
}

/// Replace a layer with an edited copy from the UI (shape, surface, settings, name...).
#[tauri::command]
fn update_layer(state: State<'_, AppState>, layer: Layer) -> CmdResult<Project> {
    ensure_not_tracking(&state)?;
    let mut p = state.project.lock().unwrap();
    let slot = p.layer_mut(layer.id).ok_or("Layer not found")?;
    *slot = layer;
    Ok(p.clone())
}

#[tauri::command]
fn delete_layer(state: State<'_, AppState>, id: u32) -> CmdResult<Project> {
    ensure_not_tracking(&state)?;
    let mut p = state.project.lock().unwrap();
    p.layers.retain(|l| l.id != id);
    Ok(p.clone())
}

/// Make `frame` the layer's reference frame (shape and surface are re-based onto it).
#[tauri::command]
fn set_reference_frame(state: State<'_, AppState>, id: u32, frame: usize) -> CmdResult<Project> {
    ensure_not_tracking(&state)?;
    let mut p = state.project.lock().unwrap();
    let l = p.layer_mut(id).ok_or("Layer not found")?;
    if !l.rebase(frame) {
        return Err("That frame is not tracked yet".into());
    }
    Ok(p.clone())
}

#[tauri::command]
fn clear_track(
    state: State<'_, AppState>,
    id: u32,
    frame: Option<usize>,
    forward: bool,
) -> CmdResult<Project> {
    ensure_not_tracking(&state)?;
    let mut p = state.project.lock().unwrap();
    let l = p.layer_mut(id).ok_or("Layer not found")?;
    match frame {
        Some(f) => l.clear_from(f, forward),
        None => l.clear_tracks(),
    }
    Ok(p.clone())
}

// ---------------------------------------------------------------------------
// Commands: tracking

#[derive(Clone, Serialize)]
struct TrackFrameEvent {
    layer_id: u32,
    frame: usize,
    track: FrameTrack,
}

#[derive(Clone, Serialize)]
struct TrackDoneEvent {
    layer_id: u32,
    ok: bool,
    cancelled: bool,
    message: String,
    last_frame: Option<usize>,
}

/// Start tracking in the background. `direction` is 1 or -1; `steps` limits the
/// number of frames (1 = step a single frame, None = to the end of the clip).
#[tauri::command]
fn start_tracking(
    app: AppHandle,
    state: State<'_, AppState>,
    id: u32,
    from: usize,
    direction: i32,
    steps: Option<usize>,
) -> CmdResult<()> {
    if state.tracking.swap(true, Ordering::SeqCst) {
        return Err("Already tracking".into());
    }
    let setup = (|| -> CmdResult<(Layer, usize)> {
        let p = state.project.lock().unwrap();
        let f = p.footage.as_ref().ok_or("Open footage first")?;
        let layer = p.layer(id).ok_or("Layer not found")?.clone();
        let last = f.frames.len().saturating_sub(1);
        let end = match (direction >= 0, steps) {
            (true, Some(n)) => (from + n).min(last),
            (true, None) => last,
            (false, Some(n)) => from.saturating_sub(n),
            (false, None) => 0,
        };
        Ok((layer, end))
    })();
    let (mut layer, end) = match setup {
        Ok(v) => v,
        Err(e) => {
            state.tracking.store(false, Ordering::SeqCst);
            return Err(e);
        }
    };
    state.cancel.store(false, Ordering::SeqCst);
    let cache = state.cache.clone();
    let cancel = state.cancel.clone();
    let tracking = state.tracking.clone();

    std::thread::spawn(move || {
        let mut last_frame = None;
        let result = {
            let mut src = cache.lock().unwrap();
            track_layer(&mut *src, &mut layer, from, end, &cancel, |frame, track| {
                last_frame = Some(frame);
                let _ = app.emit(
                    "track-frame",
                    TrackFrameEvent {
                        layer_id: id,
                        frame,
                        track: track.clone(),
                    },
                );
            })
        };
        // Write results back whatever happened, so partial tracks are kept.
        if let Some(state) = app.try_state::<AppState>() {
            if let Some(l) = state.project.lock().unwrap().layer_mut(id) {
                l.tracks = layer.tracks.clone();
            }
        }
        tracking.store(false, Ordering::SeqCst);
        let done = match result {
            Ok(StopReason::Finished) => TrackDoneEvent {
                layer_id: id,
                ok: true,
                cancelled: false,
                message: "Tracking finished".into(),
                last_frame,
            },
            Ok(StopReason::Cancelled) => TrackDoneEvent {
                layer_id: id,
                ok: true,
                cancelled: true,
                message: "Tracking stopped".into(),
                last_frame,
            },
            Err(e) => TrackDoneEvent {
                layer_id: id,
                ok: false,
                cancelled: false,
                message: e.to_string(),
                last_frame,
            },
        };
        let _ = app.emit("track-done", done);
    });
    Ok(())
}

#[tauri::command]
fn stop_tracking(state: State<'_, AppState>) {
    state.cancel.store(true, Ordering::SeqCst);
}

// ---------------------------------------------------------------------------
// Commands: export & settings

#[derive(Serialize)]
struct FormatInfo {
    id: ExportFormat,
    label: &'static str,
}

#[tauri::command]
fn export_formats() -> Vec<FormatInfo> {
    ExportFormat::ALL
        .iter()
        .map(|&f| FormatInfo {
            id: f,
            label: f.label(),
        })
        .collect()
}

#[tauri::command]
fn get_settings(state: State<'_, AppState>) -> AppSettings {
    state.settings.lock().unwrap().clone()
}

#[tauri::command]
fn set_settings(
    app: AppHandle,
    state: State<'_, AppState>,
    settings: AppSettings,
) -> CmdResult<AppSettings> {
    if let Some(p) = settings_path(&app) {
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        std::fs::write(&p, serde_json::to_string_pretty(&settings).unwrap())
            .map_err(|e| e.to_string())?;
    }
    *state.settings.lock().unwrap() = settings.clone();
    Ok(settings)
}

#[tauri::command]
async fn pick_folder(app: AppHandle, title: String) -> CmdResult<Option<String>> {
    Ok(app
        .dialog()
        .file()
        .set_title(title)
        .blocking_pick_folder()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.to_string_lossy().to_string()))
}

/// `destination`: "desktop", "reelforge", "forma3d", "tidefire" or "choose".
#[tauri::command]
async fn export_layer(
    app: AppHandle,
    state: State<'_, AppState>,
    id: u32,
    format: ExportFormat,
    destination: String,
) -> CmdResult<Option<String>> {
    let (text, name) = {
        let p = state.project.lock().unwrap();
        let f = p.footage.as_ref().ok_or("Open footage first")?;
        let l = p.layer(id).ok_or("Layer not found")?;
        if l.tracks.len() < 2 {
            return Err("Track the layer before exporting".into());
        }
        (
            export::export(format, f, l),
            export::file_name(f, l, format),
        )
    };
    let settings = state.settings.lock().unwrap().clone();
    let dir = match destination.as_str() {
        "desktop" => dirs::desktop_dir()
            .or_else(dirs::home_dir)
            .ok_or("Could not find your Desktop folder")?,
        "reelforge" => PathBuf::from(settings.reelforge_dir),
        "forma3d" => PathBuf::from(settings.forma3d_dir),
        "tidefire" => PathBuf::from(settings.tidefire_dir),
        "choose" => {
            let picked = app
                .dialog()
                .file()
                .set_title("Export track")
                .set_file_name(&name)
                .blocking_save_file();
            let Some(path) = picked.and_then(|p| p.into_path().ok()) else {
                return Ok(None);
            };
            std::fs::write(&path, text).map_err(|e| format!("Export failed: {e}"))?;
            return Ok(Some(path.to_string_lossy().to_string()));
        }
        other => return Err(format!("Unknown destination {other}")),
    };
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Could not create {}: {e}", dir.display()))?;
    let path = dir.join(name);
    std::fs::write(&path, text).map_err(|e| format!("Export failed: {e}"))?;
    Ok(Some(path.to_string_lossy().to_string()))
}

/// Open footage outside a command (drag & drop, command line) and notify the UI.
fn open_and_emit(app: AppHandle, path: PathBuf) {
    let Some(state) = app.try_state::<AppState>() else {
        return;
    };
    if state.tracking.load(Ordering::SeqCst) {
        let _ = app.emit("app-error", "Stop tracking before opening new footage");
        return;
    }
    let path = if path.extension().map_or(false, |e| e == "twproj") {
        let _ = app.emit("app-error", "Use Open Project… to load .twproj files");
        return;
    } else {
        path
    };
    let cache_root = match app.path().app_cache_dir() {
        Ok(d) => d.join("frames"),
        Err(e) => {
            let _ = app.emit("app-error", e.to_string());
            return;
        }
    };
    let _ = app.emit("app-status", "Opening footage…");
    match footage::open_any(&path, &cache_root) {
        Ok(f) => {
            let p = install_footage(&app, &state, f);
            let _ = app.emit("project-changed", p);
        }
        Err(e) => {
            let _ = app.emit("app-error", e);
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let settings = load_settings(app.handle());
            app.manage(AppState {
                project: Mutex::new(Project::default()),
                project_path: Mutex::new(None),
                cache: Arc::new(Mutex::new(FrameCache::new())),
                cancel: Arc::new(AtomicBool::new(false)),
                tracking: Arc::new(AtomicBool::new(false)),
                settings: Mutex::new(settings),
            });
            // `trackwright <footage>` opens it straight away.
            if let Some(arg) = std::env::args().skip(1).find(|a| !a.starts_with('-')) {
                let handle = app.handle().clone();
                std::thread::spawn(move || open_and_emit(handle, PathBuf::from(arg)));
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::DragDrop(tauri::DragDropEvent::Drop { paths, .. }) = event {
                if let Some(p) = paths.first().cloned() {
                    let handle = window.app_handle().clone();
                    std::thread::spawn(move || open_and_emit(handle, p));
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_project,
            new_project,
            open_footage,
            set_footage_timing,
            save_project,
            open_project,
            add_layer,
            update_layer,
            delete_layer,
            set_reference_frame,
            clear_track,
            start_tracking,
            stop_tracking,
            export_formats,
            get_settings,
            set_settings,
            pick_folder,
            export_layer,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Trackwright");
}

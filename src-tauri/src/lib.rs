//! Trackwright desktop app (Tauri backend).
//!
//! © Scott Hambach. All rights reserved. "Trackwright" is a trademark of Scott Hambach.

mod footage;
mod proxy;

use proxy::{ProxyBuilder, ProxyInfo};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;
use trackwright_core::camera_export::{self, CameraFormat};
use trackwright_core::export::{self, ExportFormat};
use trackwright_core::lens::Lens;
use trackwright_core::linalg;
use trackwright_core::points::{auto_track, AutoTrackSettings};
use trackwright_core::project::Insert;
use trackwright_core::render;
use trackwright_core::sfm::{self, SolveOptions};
use trackwright_core::{
    exclusion_polys, retrack_section, track_layer, FrameSource, FrameTrack, GrayImage, Layer,
    Project, Pyramid, Shape, StopReason,
};

const MAX_UNDO: usize = 60;
const MAX_RECENT: usize = 10;
const AUTOSAVE_EVERY: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Frame cache

struct FrameCache {
    frames: Vec<String>,
    /// index -> (levels requested, pyramid)
    cache: HashMap<usize, (usize, Arc<Pyramid>)>,
    order: VecDeque<usize>,
    capacity: usize,
    /// The frame the tracker will probably ask for next, being decoded in the background.
    prefetch: Option<(
        usize,
        usize,
        std::thread::JoinHandle<Result<Pyramid, String>>,
    )>,
    last: Option<usize>,
}

fn load_pyramid(path: &str, levels: usize) -> Result<Pyramid, String> {
    let img = image::open(path)
        .map_err(|e| format!("{path}: {e}"))?
        .to_luma8();
    let (w, h) = img.dimensions();
    let gray = GrayImage::from_luma8(w as usize, h as usize, img.as_raw());
    Ok(Pyramid::build(gray, levels))
}

impl FrameCache {
    fn new() -> Self {
        Self {
            frames: vec![],
            cache: HashMap::new(),
            order: VecDeque::new(),
            capacity: 8,
            prefetch: None,
            last: None,
        }
    }

    fn reset(&mut self, frames: Vec<String>) {
        self.frames = frames;
        self.cache.clear();
        self.order.clear();
        self.prefetch = None;
        self.last = None;
    }

    fn insert(&mut self, index: usize, levels: usize, pyr: Arc<Pyramid>) {
        self.cache.insert(index, (levels, pyr));
        self.order.retain(|&i| i != index);
        self.order.push_back(index);
        while self.order.len() > self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.cache.remove(&old);
            }
        }
    }
}

impl FrameSource for FrameCache {
    fn frame_count(&self) -> usize {
        self.frames.len()
    }

    fn pyramid(&mut self, index: usize, levels: usize) -> Result<Arc<Pyramid>, String> {
        let hit = self
            .cache
            .get(&index)
            .filter(|(l, _)| *l >= levels)
            .map(|(_, p)| p.clone());
        let pyr = match hit {
            Some(p) => p,
            None => {
                let prefetched = match self.prefetch.take() {
                    Some((i, l, handle)) if i == index && l >= levels => {
                        handle.join().ok().and_then(|r| r.ok())
                    }
                    other => {
                        self.prefetch = other;
                        None
                    }
                };
                let path = self
                    .frames
                    .get(index)
                    .ok_or_else(|| format!("frame {index} is out of range"))?;
                let p = Arc::new(match prefetched {
                    Some(p) => p,
                    None => load_pyramid(path, levels)?,
                });
                self.insert(index, levels, p.clone());
                p
            }
        };
        // Decode the next frame in the direction of travel while this one is tracked.
        if let Some(last) = self.last {
            if last != index {
                let next = if index > last {
                    index.checked_add(1)
                } else {
                    index.checked_sub(1)
                };
                if let Some(n) = next.filter(|&n| n < self.frames.len()) {
                    let busy = self.prefetch.as_ref().is_some_and(|(i, _, _)| *i == n);
                    let cached = self.cache.get(&n).is_some_and(|(l, _)| *l >= levels);
                    if !busy && !cached {
                        let path = self.frames[n].clone();
                        self.prefetch = Some((
                            n,
                            levels,
                            std::thread::spawn(move || load_pyramid(&path, levels)),
                        ));
                    }
                }
            }
        }
        self.last = Some(index);
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
    pub recent_projects: Vec<String>,
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
            recent_projects: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// State

#[derive(Default)]
struct History {
    undo: Vec<Project>,
    redo: Vec<Project>,
}

struct AppState {
    project: Mutex<Project>,
    history: Mutex<History>,
    dirty: AtomicBool,
    /// Bumped on every change; lets autosave skip unchanged projects.
    edit_version: AtomicU64,
    project_path: Mutex<Option<PathBuf>>,
    cache: Arc<Mutex<FrameCache>>,
    cancel: Arc<AtomicBool>,
    tracking: Arc<AtomicBool>,
    settings: Mutex<AppSettings>,
    proxies: ProxyBuilder,
    allow_close: AtomicBool,
    rendering: Arc<AtomicBool>,
    render_cancel: Arc<AtomicBool>,
}

/// Everything the UI needs after a change.
#[derive(Clone, Serialize)]
struct AppView {
    project: Project,
    can_undo: bool,
    can_redo: bool,
    dirty: bool,
    path: Option<String>,
    recent: Vec<String>,
}

type CmdResult<T> = Result<T, String>;

fn view(state: &AppState) -> AppView {
    let project = state.project.lock().unwrap().clone();
    let (can_undo, can_redo) = {
        let h = state.history.lock().unwrap();
        (!h.undo.is_empty(), !h.redo.is_empty())
    };
    AppView {
        project,
        can_undo,
        can_redo,
        dirty: state.dirty.load(Ordering::SeqCst),
        path: state
            .project_path
            .lock()
            .unwrap()
            .as_ref()
            .map(|p| p.to_string_lossy().to_string()),
        recent: state.settings.lock().unwrap().recent_projects.clone(),
    }
}

fn mark_changed(state: &AppState) {
    state.dirty.store(true, Ordering::SeqCst);
    state.edit_version.fetch_add(1, Ordering::SeqCst);
}

fn push_undo(state: &AppState, before: Project) {
    let mut h = state.history.lock().unwrap();
    h.undo.push(before);
    if h.undo.len() > MAX_UNDO {
        h.undo.remove(0);
    }
    h.redo.clear();
}

fn ensure_not_tracking(state: &AppState) -> CmdResult<()> {
    if state.tracking.load(Ordering::SeqCst) {
        Err("Stop tracking first".into())
    } else {
        Ok(())
    }
}

/// Apply an undoable edit. On error the project is left untouched.
fn mutate<T>(state: &AppState, f: impl FnOnce(&mut Project) -> CmdResult<T>) -> CmdResult<T> {
    ensure_not_tracking(state)?;
    let mut p = state.project.lock().unwrap();
    let before = p.clone();
    match f(&mut p) {
        Ok(v) => {
            p.resolve_links();
            if *p != before {
                push_undo(state, before);
                mark_changed(state);
            }
            Ok(v)
        }
        Err(e) => {
            *p = before;
            Err(e)
        }
    }
}

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

fn store_settings(app: &AppHandle, settings: &AppSettings) -> CmdResult<()> {
    if let Some(p) = settings_path(app) {
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        std::fs::write(&p, serde_json::to_string_pretty(settings).unwrap())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn add_recent(app: &AppHandle, state: &AppState, path: &Path) {
    let p = path.to_string_lossy().to_string();
    let mut s = state.settings.lock().unwrap();
    s.recent_projects.retain(|r| r != &p);
    s.recent_projects.insert(0, p);
    s.recent_projects.truncate(MAX_RECENT);
    let _ = store_settings(app, &s);
}

fn remove_recent(app: &AppHandle, state: &AppState, path: &str) {
    let mut s = state.settings.lock().unwrap();
    s.recent_projects.retain(|r| r != path);
    let _ = store_settings(app, &s);
}

// ---------------------------------------------------------------------------
// Autosave

fn autosave_paths(app: &AppHandle) -> Option<(PathBuf, PathBuf)> {
    let d = app.path().app_data_dir().ok()?.join("autosave");
    Some((d.join("autosave.twproj"), d.join("autosave.json")))
}

#[derive(Serialize, Deserialize, Clone)]
struct RecoveryInfo {
    original_path: Option<String>,
    saved_at: u64,
}

fn delete_autosave(app: &AppHandle) {
    if let Some((a, b)) = autosave_paths(app) {
        let _ = std::fs::remove_file(a);
        let _ = std::fs::remove_file(b);
    }
}

fn write_autosave(app: &AppHandle, state: &AppState) -> Result<(), String> {
    let (proj, meta) = autosave_paths(app).ok_or("no data dir")?;
    if let Some(d) = proj.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let json = state.project.lock().unwrap().to_json();
    let tmp = proj.with_extension("tmp");
    std::fs::write(&tmp, json).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &proj).map_err(|e| e.to_string())?;
    let info = RecoveryInfo {
        original_path: state
            .project_path
            .lock()
            .unwrap()
            .as_ref()
            .map(|p| p.to_string_lossy().to_string()),
        saved_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
    };
    std::fs::write(meta, serde_json::to_string(&info).unwrap()).map_err(|e| e.to_string())
}

fn autosave_loop(app: AppHandle) {
    let mut last_saved = u64::MAX;
    loop {
        std::thread::sleep(AUTOSAVE_EVERY);
        let Some(state) = app.try_state::<AppState>() else {
            continue;
        };
        let v = state.edit_version.load(Ordering::SeqCst);
        if !state.dirty.load(Ordering::SeqCst)
            || state.tracking.load(Ordering::SeqCst)
            || v == last_saved
        {
            continue;
        }
        if write_autosave(&app, &state).is_ok() {
            last_saved = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Footage, assets, proxies

/// Let the webview display files from these folders through the asset protocol.
fn allow_dirs<'a>(app: &AppHandle, files: impl IntoIterator<Item = &'a String>) {
    let scope = app.asset_protocol_scope();
    let mut dirs: Vec<PathBuf> = files
        .into_iter()
        .filter_map(|f| Path::new(f).parent().map(|p| p.to_path_buf()))
        .collect();
    dirs.sort();
    dirs.dedup();
    for d in dirs {
        let _ = scope.allow_directory(&d, false);
    }
}

fn allow_project_assets(app: &AppHandle, p: &Project) {
    if let Some(f) = &p.footage {
        allow_dirs(app, &f.frames);
    }
    for l in &p.layers {
        if let Some(i) = &l.insert {
            allow_dirs(app, &i.frames);
        }
    }
}

fn start_proxies(app: &AppHandle, state: &AppState) {
    let footage = state.project.lock().unwrap().footage.clone();
    let Some(f) = footage else {
        state.proxies.cancel();
        return;
    };
    let Ok(root) = app.path().app_cache_dir().map(|d| d.join("proxies")) else {
        return;
    };
    let handle = app.clone();
    if let Some(dir) = state
        .proxies
        .start(f.frames.clone(), f.width, f.height, root, move |info| {
            let _ = handle.emit("proxy-progress", info);
        })
    {
        let _ = app.asset_protocol_scope().allow_directory(&dir, false);
    }
}

/// Install newly opened footage. Changing footage resets undo history.
fn install_footage(app: &AppHandle, state: &AppState, f: trackwright_core::Footage) -> AppView {
    allow_dirs(app, &f.frames);
    state.cache.lock().unwrap().reset(f.frames.clone());
    {
        let mut p = state.project.lock().unwrap();
        if p.footage.as_ref().map(|o| o.frames.len()) != Some(f.frames.len()) {
            p.layers.clear();
        }
        p.footage = Some(f);
    }
    *state.history.lock().unwrap() = History::default();
    mark_changed(state);
    start_proxies(app, state);
    view(state)
}

fn load_project_file(app: &AppHandle, path: &Path) -> CmdResult<Project> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("Could not open {}: {e}", path.display()))?;
    let mut project = Project::from_json(&text)?;
    if let Some(f) = &project.footage {
        if let Some(missing) = f.frames.iter().find(|p| !Path::new(p).exists()) {
            return Err(format!("Footage is missing ({missing}). Re-open the original footage, then load the project again"));
        }
    }
    project.resolve_links();
    allow_project_assets(app, &project);
    Ok(project)
}

fn install_project(
    app: &AppHandle,
    state: &AppState,
    project: Project,
    path: Option<PathBuf>,
    dirty: bool,
) {
    state.cache.lock().unwrap().reset(
        project
            .footage
            .as_ref()
            .map(|f| f.frames.clone())
            .unwrap_or_default(),
    );
    *state.project.lock().unwrap() = project;
    *state.history.lock().unwrap() = History::default();
    *state.project_path.lock().unwrap() = path;
    state.dirty.store(dirty, Ordering::SeqCst);
    state.edit_version.fetch_add(1, Ordering::SeqCst);
    start_proxies(app, state);
}

// ---------------------------------------------------------------------------
// Commands: project & history

#[tauri::command]
fn get_view(state: State<'_, AppState>) -> AppView {
    view(&state)
}

#[tauri::command]
fn new_project(app: AppHandle, state: State<'_, AppState>) -> CmdResult<AppView> {
    ensure_not_tracking(&state)?;
    install_project(&app, &state, Project::default(), None, false);
    delete_autosave(&app);
    Ok(view(&state))
}

#[tauri::command]
fn undo(state: State<'_, AppState>) -> CmdResult<AppView> {
    ensure_not_tracking(&state)?;
    {
        let mut p = state.project.lock().unwrap();
        let mut h = state.history.lock().unwrap();
        let prev = h.undo.pop().ok_or("Nothing to undo")?;
        let cur = std::mem::replace(&mut *p, prev);
        h.redo.push(cur);
    }
    mark_changed(&state);
    Ok(view(&state))
}

#[tauri::command]
fn redo(state: State<'_, AppState>) -> CmdResult<AppView> {
    ensure_not_tracking(&state)?;
    {
        let mut p = state.project.lock().unwrap();
        let mut h = state.history.lock().unwrap();
        let next = h.redo.pop().ok_or("Nothing to redo")?;
        let cur = std::mem::replace(&mut *p, next);
        h.undo.push(cur);
    }
    mark_changed(&state);
    Ok(view(&state))
}

fn pick_media(app: &AppHandle, kind: &str, title: &str) -> Option<PathBuf> {
    let picked = if kind == "folder" {
        app.dialog()
            .file()
            .set_title(format!("{title}: image sequence folder"))
            .blocking_pick_folder()
    } else {
        app.dialog()
            .file()
            .set_title(title)
            .add_filter(
                "Images and video",
                &[
                    "mp4", "mov", "mkv", "avi", "webm", "m4v", "mxf", "png", "jpg", "jpeg", "tif",
                    "tiff", "bmp", "webp",
                ],
            )
            .blocking_pick_file()
    };
    picked.and_then(|p| p.into_path().ok())
}

/// `kind`: "file" (video / image / sequence frame) or "folder".
#[tauri::command]
async fn open_footage(
    app: AppHandle,
    state: State<'_, AppState>,
    kind: String,
    path: Option<String>,
) -> CmdResult<AppView> {
    ensure_not_tracking(&state)?;
    let path = match path {
        Some(p) => PathBuf::from(p),
        None => match pick_media(&app, &kind, "Open footage") {
            Some(p) => p,
            None => return Ok(view(&state)),
        },
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
) -> CmdResult<AppView> {
    mutate(&state, |p| {
        let f = p.footage.as_mut().ok_or("Open footage first")?;
        if fps > 0.0 && fps < 1000.0 {
            f.fps = fps;
        }
        f.first_frame = first_frame;
        Ok(())
    })?;
    Ok(view(&state))
}

/// Returns `None` when the user cancels the save dialog.
#[tauri::command]
async fn save_project(
    app: AppHandle,
    state: State<'_, AppState>,
    save_as: bool,
) -> CmdResult<Option<AppView>> {
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
    let tmp = path.with_extension("twproj.tmp");
    std::fs::write(&tmp, json).map_err(|e| format!("Could not save: {e}"))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("Could not save: {e}"))?;
    *state.project_path.lock().unwrap() = Some(path.clone());
    state.dirty.store(false, Ordering::SeqCst);
    delete_autosave(&app);
    add_recent(&app, &state, &path);
    Ok(Some(view(&state)))
}

#[tauri::command]
async fn open_project(
    app: AppHandle,
    state: State<'_, AppState>,
    path: Option<String>,
) -> CmdResult<AppView> {
    ensure_not_tracking(&state)?;
    let path = match path {
        Some(p) => PathBuf::from(p),
        None => {
            let picked = app
                .dialog()
                .file()
                .set_title("Open Trackwright project")
                .add_filter("Trackwright project", &["twproj"])
                .blocking_pick_file();
            match picked.and_then(|p| p.into_path().ok()) {
                Some(p) => p,
                None => return Ok(view(&state)),
            }
        }
    };
    if !path.exists() {
        remove_recent(&app, &state, &path.to_string_lossy());
        return Err(format!("{} no longer exists", path.display()));
    }
    let project = load_project_file(&app, &path)?;
    install_project(&app, &state, project, Some(path.clone()), false);
    add_recent(&app, &state, &path);
    Ok(view(&state))
}

#[tauri::command]
fn recovery_info(app: AppHandle) -> Option<RecoveryInfo> {
    let (proj, meta) = autosave_paths(&app)?;
    if !proj.exists() {
        return None;
    }
    std::fs::read_to_string(meta)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .or(Some(RecoveryInfo {
            original_path: None,
            saved_at: 0,
        }))
}

#[tauri::command]
fn recover_autosave(app: AppHandle, state: State<'_, AppState>) -> CmdResult<AppView> {
    ensure_not_tracking(&state)?;
    let info = recovery_info(app.clone()).ok_or("No autosave found")?;
    let (proj, _) = autosave_paths(&app).ok_or("No autosave found")?;
    let project = load_project_file(&app, &proj)?;
    install_project(
        &app,
        &state,
        project,
        info.original_path.map(PathBuf::from),
        true,
    );
    Ok(view(&state))
}

#[tauri::command]
fn discard_autosave(app: AppHandle) {
    delete_autosave(&app);
}

/// Close the app without further prompts (after the UI has asked about saving).
#[tauri::command]
fn quit_app(app: AppHandle, state: State<'_, AppState>) {
    state.allow_close.store(true, Ordering::SeqCst);
    state.cancel.store(true, Ordering::SeqCst);
    delete_autosave(&app);
    app.exit(0);
}

// ---------------------------------------------------------------------------
// Commands: layers

#[tauri::command]
fn add_layer(
    state: State<'_, AppState>,
    frame: usize,
    points: Vec<[f64; 2]>,
) -> CmdResult<AppView> {
    mutate(&state, |p| {
        if p.footage.is_none() {
            return Err("Open footage first".into());
        }
        let shape = Shape::new(points);
        if !shape.is_valid() {
            return Err("A shape needs at least 3 points".into());
        }
        p.add_layer(frame, shape);
        Ok(())
    })?;
    Ok(view(&state))
}

/// Replace a layer with an edited copy from the UI (shape, surface, settings, name...).
#[tauri::command]
fn update_layer(state: State<'_, AppState>, layer: Layer) -> CmdResult<AppView> {
    mutate(&state, |p| {
        let slot = p.layer_mut(layer.id).ok_or("Layer not found")?;
        *slot = layer;
        Ok(())
    })?;
    Ok(view(&state))
}

#[tauri::command]
fn delete_layer(state: State<'_, AppState>, id: u32) -> CmdResult<AppView> {
    mutate(&state, |p| {
        p.layers.retain(|l| l.id != id);
        for l in &mut p.layers {
            if l.link == Some(id) {
                // Keep the motion it had, but stop following the deleted layer.
                l.link = None;
            }
        }
        Ok(())
    })?;
    Ok(view(&state))
}

#[tauri::command]
fn duplicate_layer(state: State<'_, AppState>, id: u32) -> CmdResult<AppView> {
    mutate(&state, |p| {
        p.duplicate_layer(id)
            .map(|_| ())
            .ok_or_else(|| "Layer not found".to_string())
    })?;
    Ok(view(&state))
}

/// Make `id` follow `parent`'s motion (or track independently with `None`).
#[tauri::command]
fn set_link(state: State<'_, AppState>, id: u32, parent: Option<u32>) -> CmdResult<AppView> {
    mutate(&state, |p| {
        if let Some(pid) = parent {
            if pid == id || p.link_creates_cycle(id, pid) {
                return Err("That link would make layers follow each other in a loop".into());
            }
            let parent_layer = p.layer(pid).ok_or("Layer not found")?;
            let child_ref = p.layer(id).ok_or("Layer not found")?.ref_frame;
            if !parent_layer.tracks.contains_key(&child_ref) {
                return Err(format!(
                    "{} isn't tracked at this layer's reference frame yet",
                    parent_layer.name
                ));
            }
        }
        let l = p.layer_mut(id).ok_or("Layer not found")?;
        l.link = parent;
        if parent.is_none() {
            l.clear_tracks();
        }
        Ok(())
    })?;
    Ok(view(&state))
}

/// Make `frame` the layer's reference frame (shape and surface are re-based onto it).
#[tauri::command]
fn set_reference_frame(state: State<'_, AppState>, id: u32, frame: usize) -> CmdResult<AppView> {
    mutate(&state, |p| {
        let l = p.layer_mut(id).ok_or("Layer not found")?;
        if !l.rebase(frame) {
            return Err("That frame is not tracked yet".into());
        }
        Ok(())
    })?;
    Ok(view(&state))
}

#[tauri::command]
fn clear_track(
    state: State<'_, AppState>,
    id: u32,
    frame: Option<usize>,
    forward: bool,
) -> CmdResult<AppView> {
    mutate(&state, |p| {
        let l = p.layer_mut(id).ok_or("Layer not found")?;
        if l.link.is_some() {
            return Err("This layer follows another layer. Unlink it to clear its track".into());
        }
        match frame {
            Some(f) => l.clear_from(f, forward),
            None => l.clear_tracks(),
        }
        Ok(())
    })?;
    Ok(view(&state))
}

/// Pin an image, clip or sequence onto a layer's surface. `start` is the footage frame
/// where the clip begins. Returns `None` if the user cancels.
#[tauri::command]
async fn choose_insert(
    app: AppHandle,
    state: State<'_, AppState>,
    id: u32,
    kind: String,
    start: usize,
) -> CmdResult<Option<AppView>> {
    ensure_not_tracking(&state)?;
    let Some(path) = pick_media(&app, &kind, "Choose insert image or clip") else {
        return Ok(None);
    };
    let cache_root = app
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?
        .join("frames");
    let f = tauri::async_runtime::spawn_blocking(move || footage::open_any(&path, &cache_root))
        .await
        .map_err(|e| e.to_string())??;
    allow_dirs(&app, &f.frames);
    mutate(&state, |p| {
        let l = p.layer_mut(id).ok_or("Layer not found")?;
        let multi = f.frames.len() > 1;
        l.insert = Some(Insert {
            source: f.source,
            frames: f.frames,
            width: f.width,
            height: f.height,
            opacity: l.insert.as_ref().map_or(1.0, |i| i.opacity),
            start_frame: if multi { start } else { 0 },
            looped: false,
        });
        Ok(())
    })?;
    Ok(Some(view(&state)))
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
/// number of frames (1 = step a single frame); `until` stops at that frame (the
/// In/Out point). With `blend`, the section from..until is re-tracked and blended
/// into the existing track at `until`. A whole tracking run is a single undo step.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
fn start_tracking(
    app: AppHandle,
    state: State<'_, AppState>,
    id: u32,
    from: usize,
    direction: i32,
    steps: Option<usize>,
    until: Option<usize>,
    blend: Option<bool>,
) -> CmdResult<()> {
    let blend = blend.unwrap_or(false);
    if state.tracking.swap(true, Ordering::SeqCst) {
        return Err("Already tracking".into());
    }
    let setup = (|| -> CmdResult<(Layer, usize, Project)> {
        let p = state.project.lock().unwrap();
        let f = p.footage.as_ref().ok_or("Open footage first")?;
        let layer = p.layer(id).ok_or("Layer not found")?.clone();
        if let Some(pid) = layer.link {
            let name = p.layer(pid).map(|l| l.name.clone()).unwrap_or_default();
            return Err(format!(
                "This layer follows {name}. Set Motion source to Own track to track it"
            ));
        }
        let last = f.frames.len().saturating_sub(1);
        let mut end = match (direction >= 0, steps) {
            (true, Some(n)) => (from + n).min(last),
            (true, None) => last,
            (false, Some(n)) => from.saturating_sub(n),
            (false, None) => 0,
        };
        if let Some(u) = until {
            let u = u.min(last);
            if (direction >= 0 && u > from && u < end) || (direction < 0 && u < from && u > end) {
                end = u;
            }
        }
        if blend && end == from {
            return Err("Set In and Out points around the section to re-track".into());
        }
        Ok((layer, end, p.clone()))
    })();
    let (mut layer, end, before) = match setup {
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
    // Occluder outlines come from a snapshot of the other layers.
    let all_layers = before.layers.clone();
    let me = layer.clone();

    std::thread::spawn(move || {
        let mut last_frame = None;
        let exclude = move |f: usize| exclusion_polys(&all_layers, &me, f);
        let result = {
            let mut src = cache.lock().unwrap();
            let emit = |frame: usize, track: &FrameTrack| {
                last_frame = Some(frame);
                let _ = app.emit(
                    "track-frame",
                    TrackFrameEvent {
                        layer_id: id,
                        frame,
                        track: track.clone(),
                    },
                );
            };
            if blend {
                retrack_section(&mut *src, &mut layer, from, end, &exclude, &cancel, emit)
            } else {
                track_layer(&mut *src, &mut layer, from, end, &exclude, &cancel, emit)
            }
        };
        // Write results back whatever happened, so partial tracks are kept.
        if let Some(state) = app.try_state::<AppState>() {
            let mut p = state.project.lock().unwrap();
            if let Some(l) = p.layer_mut(id) {
                l.tracks = layer.tracks.clone();
            }
            p.resolve_links();
            if *p != before {
                push_undo(&state, before);
                mark_changed(&state);
            }
        }
        tracking.store(false, Ordering::SeqCst);
        let done = match result {
            Ok(StopReason::Finished) => TrackDoneEvent {
                layer_id: id,
                ok: true,
                cancelled: false,
                message: if blend {
                    "Section re-tracked and blended into the existing track".into()
                } else {
                    "Tracking finished".into()
                },
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
// Commands: proxies, export & settings

#[tauri::command]
fn get_proxy_info(state: State<'_, AppState>) -> ProxyInfo {
    state.proxies.info.lock().unwrap().clone()
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
    let mut s = settings;
    // The recent list is managed here, not by the settings panel.
    s.recent_projects = state.settings.lock().unwrap().recent_projects.clone();
    store_settings(&app, &s)?;
    *state.settings.lock().unwrap() = s.clone();
    Ok(s)
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

// ---------------------------------------------------------------------------
// Rendering: stabilized clips and insert composites

#[derive(Deserialize)]
struct RenderRequest {
    /// Layer the render is about (not needed for "undistort").
    id: Option<u32>,
    /// "insert", "stabilize_lock", "stabilize_smooth", "remove", "clean_plate",
    /// "matte", "cutout" or "undistort"
    mode: String,
    smooth_frames: Option<usize>,
    /// Percent, 100 = no zoom.
    zoom: Option<f64>,
    from: Option<usize>,
    to: Option<usize>,
    /// "desktop" or "choose"
    destination: String,
    make_video: bool,
    /// Put the layer's exclusion areas (occluders) back over the insert.
    holdout: Option<bool>,
    /// Remove / clean plate: the layer tracking the surface behind the object.
    bg_layer: Option<u32>,
    search: Option<usize>,
    grow: Option<f64>,
    match_light: Option<bool>,
    /// Matte / cutout: combine every visible layer instead of just the selected one.
    all_layers: Option<bool>,
}

#[derive(Clone, Serialize)]
struct RenderProgress {
    done: usize,
    total: usize,
}

#[derive(Clone, Serialize)]
struct RenderDone {
    ok: bool,
    message: String,
    path: Option<String>,
}

fn load_rgba(path: &str) -> Result<render::Rgba, String> {
    let img = image::open(path)
        .map_err(|e| format!("{path}: {e}"))?
        .to_rgba8();
    let (w, h) = img.dimensions();
    Ok(render::Rgba {
        width: w as usize,
        height: h as usize,
        data: img.into_raw(),
    })
}

fn save_png(img: render::Rgba, path: &Path) -> Result<(), String> {
    image::RgbaImage::from_raw(img.width as u32, img.height as u32, img.data)
        .ok_or("bad image buffer")?
        .save(path)
        .map_err(|e| format!("Could not write {}: {e}", path.display()))
}

fn save_gray(w: usize, h: usize, data: Vec<u8>, path: &Path) -> Result<(), String> {
    image::GrayImage::from_raw(w as u32, h as u32, data)
        .ok_or("bad image buffer")?
        .save(path)
        .map_err(|e| format!("Could not write {}: {e}", path.display()))
}

fn unique_dir(base: PathBuf) -> PathBuf {
    if !base.exists() {
        return base;
    }
    (2..)
        .map(|n| PathBuf::from(format!("{}_{n}", base.display())))
        .find(|p| !p.exists())
        .unwrap()
}

/// Small LRU of decoded frames for renders that read several frames per output frame.
struct RgbaCache {
    frames: Vec<String>,
    map: HashMap<usize, Arc<render::Rgba>>,
    order: VecDeque<usize>,
    cap: usize,
}

impl RgbaCache {
    fn get(&mut self, i: usize) -> Option<Arc<render::Rgba>> {
        if let Some(v) = self.map.get(&i) {
            return Some(v.clone());
        }
        let img = Arc::new(load_rgba(self.frames.get(i)?).ok()?);
        self.map.insert(i, img.clone());
        self.order.push_back(i);
        while self.order.len() > self.cap {
            if let Some(o) = self.order.pop_front() {
                self.map.remove(&o);
            }
        }
        Some(img)
    }
}

/// Pick the output folder (Desktop or a dialog). `None` when the user cancels.
fn render_parent(app: &AppHandle, destination: &str) -> CmdResult<Option<PathBuf>> {
    Ok(match destination {
        "desktop" => Some(
            dirs::desktop_dir()
                .or_else(dirs::home_dir)
                .ok_or("Could not find your Desktop folder")?,
        ),
        _ => app
            .dialog()
            .file()
            .set_title("Choose where to render")
            .blocking_pick_folder()
            .and_then(|p| p.into_path().ok()),
    })
}

fn footage_stem(footage: &trackwright_core::Footage) -> String {
    Path::new(&footage.source)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "footage".into())
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Render as a PNG sequence (and optionally an MP4), or a single clean plate, in the background.
#[tauri::command]
async fn render_layer(
    app: AppHandle,
    state: State<'_, AppState>,
    request: RenderRequest,
) -> CmdResult<()> {
    ensure_not_tracking(&state)?;
    if state.rendering.load(Ordering::SeqCst) {
        return Err("A render is already running".into());
    }
    let project = state.project.lock().unwrap().clone();
    let footage = project.footage.clone().ok_or("Open footage first")?;
    let mode = request.mode.clone();
    let layer = match request.id {
        Some(id) => Some(project.layer(id).ok_or("Layer not found")?.clone()),
        None => None,
    };
    let all_layers_req = request.all_layers.unwrap_or(false);
    let need_layer =
        |l: &Option<Layer>| l.clone().ok_or_else(|| "Select a layer first".to_string());
    let last = footage.frames.len().saturating_sub(1);
    let bg = match mode.as_str() {
        "remove" | "clean_plate" => {
            let id = request
                .bg_layer
                .ok_or("Choose the background layer (the surface behind the object)")?;
            let b = project
                .layer(id)
                .ok_or("Background layer not found")?
                .clone();
            if b.tracks.len() < 2 {
                return Err(format!("Track the background layer ({}) first", b.name));
            }
            Some(b)
        }
        _ => None,
    };

    // Frames to render.
    let (lo, hi) = match mode.as_str() {
        "undistort" => (0, last),
        "remove" | "clean_plate" => {
            let b = bg.as_ref().unwrap();
            b.tracked_range()
                .ok_or("Track the background layer first")?
        }
        "matte" | "cutout" if request.all_layers.unwrap_or(false) => (0, last),
        _ => {
            let l = need_layer(&layer)?;
            if mode == "insert" && l.insert.is_none() {
                return Err(
                    "Pick an insert image or clip for this layer first (Insert Preview)".into(),
                );
            }
            if l.tracks.len() < 2 && !mode.starts_with("matte") && mode != "cutout" {
                return Err("Track the layer first".into());
            }
            l.tracked_range().ok_or("Track the layer first")?
        }
    };
    let from = request.from.unwrap_or(lo).max(lo);
    let to = request.to.unwrap_or(hi).min(hi).min(last);
    if to < from {
        return Err("Nothing to render in that range".into());
    }
    let frames: Vec<usize> = (from..=to).collect();
    if mode == "remove" || ((mode == "matte" || mode == "cutout") && !all_layers_req) {
        need_layer(&layer)?;
    }
    if mode == "undistort" && project.lens.is_identity() {
        return Err("Set the lens distortion first (Lens panel: Calibrate, or Solve Camera with lens distortion)".into());
    }

    let Some(parent) = render_parent(&app, &request.destination)? else {
        return Ok(());
    };
    let layer_name = match (&layer, mode.as_str()) {
        (_, "undistort") => "lens".to_string(),
        (_, "matte" | "cutout") if all_layers_req => "all_layers".to_string(),
        (Some(l), _) => l.name.clone(),
        (None, _) => "lens".to_string(),
    };
    let clean = |s: &str| {
        s.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>()
    };
    let suffix = match mode.as_str() {
        "insert" => "insert",
        "stabilize_lock" => "locked",
        "stabilize_smooth" => "stabilized",
        "remove" => "removed",
        "clean_plate" => "cleanplate",
        "matte" => "matte",
        "cutout" => "cutout",
        "undistort" => "undistorted",
        other => return Err(format!("Unknown render mode {other}")),
    };
    let base = format!("{}_{}_{suffix}", footage_stem(&footage), clean(&layer_name));
    let out_dir = unique_dir(parent.join(&base));
    if mode != "clean_plate" {
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| format!("Could not create {}: {e}", out_dir.display()))?;
    }

    let maps = match (&layer, mode.as_str()) {
        (Some(l), "stabilize_lock" | "stabilize_smooth") => {
            let sm = if mode == "stabilize_lock" {
                render::StabilizeMode::Lock
            } else {
                render::StabilizeMode::Smooth {
                    radius: request.smooth_frames.unwrap_or(15).clamp(1, 200),
                }
            };
            render::stabilize_maps(
                l,
                &frames,
                sm,
                request.zoom.unwrap_or(100.0) / 100.0,
                footage.width as usize,
                footage.height as usize,
            )
        }
        _ => vec![],
    };
    let holdout = request.holdout.unwrap_or(true);
    let remove_params = render::RemoveParams {
        search: request.search.unwrap_or(20).clamp(1, 120),
        grow: request.grow.unwrap_or(6.0).clamp(0.0, 200.0),
        match_light: request.match_light.unwrap_or(true),
        feather: layer.as_ref().map_or(3.0, |l| l.feather.max(2.0)),
    };
    let all_layers = request.all_layers.unwrap_or(false);
    let lens = project.lens;

    state.rendering.store(true, Ordering::SeqCst);
    state.render_cancel.store(false, Ordering::SeqCst);
    let rendering = state.rendering.clone();
    let cancel = state.render_cancel.clone();
    let make_video = request.make_video && mode != "clean_plate";
    std::thread::spawn(move || {
        let total = frames.len();
        let (w, h) = (footage.width as usize, footage.height as usize);
        let mut cache = RgbaCache {
            frames: footage.frames.clone(),
            map: HashMap::new(),
            order: VecDeque::new(),
            cap: remove_params.search * 2 + 3,
        };
        let mut insert_cache: Option<(String, render::Rgba)> = None;
        let mut last_map = None;
        // Outline of the selected layer (or all visible layers) at a frame.
        let shape_polys = |f: usize| -> Vec<Vec<trackwright_core::Point>> {
            if all_layers {
                project
                    .layers
                    .iter()
                    .filter(|l| l.visible)
                    .filter_map(|l| l.shape_at(f))
                    .map(|s| s.flatten(10))
                    .collect()
            } else {
                layer
                    .as_ref()
                    .and_then(|l| l.shape_at(f))
                    .map(|s| vec![s.flatten(10)])
                    .unwrap_or_default()
            }
        };
        let result = (|| -> Result<Option<PathBuf>, String> {
            if mode == "clean_plate" {
                let b = bg.as_ref().unwrap();
                let obj = layer.clone();
                let mask = |g: usize| {
                    obj.as_ref()
                        .and_then(|l| l.shape_at(g))
                        .map(|s| s.flatten(10))
                };
                let mut load = |g: usize| cache.get(g);
                let _ = app.emit("render-progress", RenderProgress { done: 0, total: 1 });
                let plate =
                    render::clean_plate(b, w, h, &frames, &mask, &mut load, remove_params.grow);
                let path = unique_dir(parent.join(&base)).with_extension("png");
                save_png(plate, &path)?;
                let _ = app.emit("render-progress", RenderProgress { done: 1, total: 1 });
                return Ok(Some(path));
            }
            for (k, &f) in frames.iter().enumerate() {
                if cancel.load(Ordering::SeqCst) {
                    return Ok(None);
                }
                let out_path = out_dir.join(format!("frame_{:06}.png", footage.frame_number(f)));
                match mode.as_str() {
                    "insert" => {
                        let l = layer.as_ref().unwrap();
                        let src = load_rgba(&footage.frames[f])?;
                        let original = if holdout && !l.exclusions.is_empty() {
                            Some(src.clone())
                        } else {
                            None
                        };
                        let mut frame = src;
                        if let (Some(ins), Some(corners)) = (l.insert.as_ref(), l.surface_at(f)) {
                            if let Some(path) = ins.frame_for(f) {
                                if insert_cache.as_ref().map(|(p, _)| p.as_str()) != Some(path) {
                                    insert_cache = Some((path.to_string(), load_rgba(path)?));
                                }
                                render::composite_insert(
                                    &mut frame,
                                    &insert_cache.as_ref().unwrap().1,
                                    &corners,
                                    ins.opacity,
                                );
                            }
                        }
                        if let Some(orig) = original {
                            for poly in exclusion_polys(&project.layers, l, f) {
                                render::restore_inside(&mut frame, &orig, &poly);
                            }
                        }
                        save_png(frame, &out_path)?;
                    }
                    "stabilize_lock" | "stabilize_smooth" => {
                        if let Some((_, m)) = maps.iter().find(|(g, _)| *g == f) {
                            last_map = Some(*m);
                        }
                        let m = last_map
                            .or_else(|| maps.first().map(|(_, m)| *m))
                            .ok_or("No tracked frames to stabilize")?;
                        let src = load_rgba(&footage.frames[f])?;
                        save_png(
                            render::warp(&src, &m, src.width, src.height, [0, 0, 0, 255]),
                            &out_path,
                        )?;
                    }
                    "remove" => {
                        let obj = layer.clone().unwrap();
                        let b = bg.as_ref().unwrap();
                        let mask = |g: usize| obj.shape_at(g).map(|s| s.flatten(10));
                        let src = cache.get(f).ok_or("Could not read a frame")?;
                        let mut load = |g: usize| cache.get(g);
                        let out = render::remove_object(
                            f,
                            &src,
                            b,
                            footage.frames.len(),
                            &mask,
                            &mut load,
                            &remove_params,
                        );
                        save_png(out, &out_path)?;
                    }
                    "matte" => {
                        let feather = layer.as_ref().map_or(0.0, |l| l.feather);
                        save_gray(
                            w,
                            h,
                            render::matte(w, h, &shape_polys(f), feather),
                            &out_path,
                        )?;
                    }
                    "cutout" => {
                        let feather = layer.as_ref().map_or(0.0, |l| l.feather);
                        let a = render::matte(w, h, &shape_polys(f), feather);
                        let mut src = load_rgba(&footage.frames[f])?;
                        for (i, v) in a.iter().enumerate() {
                            src.data[i * 4 + 3] = *v;
                        }
                        save_png(src, &out_path)?;
                    }
                    "undistort" => {
                        let src = load_rgba(&footage.frames[f])?;
                        let mut out = render::Rgba::new(w, h);
                        let (wf, hf) = (w as f64, h as f64);
                        for y in 0..h {
                            for x in 0..w {
                                let q = lens.distort([x as f64 + 0.5, y as f64 + 0.5], wf, hf);
                                let s = src.sample(q[0], q[1]);
                                let i = (y * w + x) * 4;
                                for c in 0..3 {
                                    out.data[i + c] = s[c].round().clamp(0.0, 255.0) as u8;
                                }
                                out.data[i + 3] = 255;
                            }
                        }
                        save_png(out, &out_path)?;
                    }
                    _ => {}
                }
                let _ = app.emit("render-progress", RenderProgress { done: k + 1, total });
            }
            Ok(Some(out_dir.clone()))
        })();
        let done = match result {
            Ok(None) => RenderDone {
                ok: true,
                message: format!("Render stopped. Frames so far are in {}", out_dir.display()),
                path: Some(out_dir.to_string_lossy().to_string()),
            },
            Err(e) => RenderDone {
                ok: false,
                message: e,
                path: None,
            },
            Ok(Some(path)) if mode == "clean_plate" => RenderDone {
                ok: true,
                message: format!("Clean plate saved to {}", path.display()),
                path: Some(path.to_string_lossy().to_string()),
            },
            Ok(Some(_)) if make_video => {
                let mp4 = out_dir.with_extension("mp4");
                let first = footage.frame_number(frames[0]);
                let fps = if footage.fps > 0.0 { footage.fps } else { 24.0 };
                let status = footage::command("ffmpeg")
                    .args([
                        "-v",
                        "error",
                        "-y",
                        "-framerate",
                        &format!("{fps}"),
                        "-start_number",
                        &first.to_string(),
                        "-i",
                    ])
                    .arg(out_dir.join("frame_%06d.png"))
                    .args([
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    ])
                    .arg(&mp4)
                    .output();
                match status {
                    Ok(o) if o.status.success() => RenderDone { ok: true, message: format!("Rendered {total} frames and {}", mp4.display()), path: Some(mp4.to_string_lossy().to_string()) },
                    Ok(o) => RenderDone {
                        ok: false,
                        message: format!("Frames rendered to {}, but the MP4 failed: {}", out_dir.display(), String::from_utf8_lossy(&o.stderr).trim()),
                        path: Some(out_dir.to_string_lossy().to_string()),
                    },
                    Err(_) => RenderDone {
                        ok: false,
                        message: format!("Frames rendered to {}. Run get-ffmpeg.bat (or install ffmpeg) to also get an MP4", out_dir.display()),
                        path: Some(out_dir.to_string_lossy().to_string()),
                    },
                }
            }
            Ok(Some(_)) => RenderDone {
                ok: true,
                message: format!("Rendered {total} frames to {}", out_dir.display()),
                path: Some(out_dir.to_string_lossy().to_string()),
            },
        };
        rendering.store(false, Ordering::SeqCst);
        let _ = app.emit("render-done", done);
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Camera solve and lens

#[derive(Clone, Serialize)]
struct CameraProgress {
    stage: String,
    fraction: f64,
}

#[derive(Clone, Serialize)]
struct CameraDone {
    ok: bool,
    message: String,
}

fn camera_range(
    project: &Project,
    from: Option<usize>,
    to: Option<usize>,
) -> CmdResult<(usize, usize)> {
    let f = project.footage.as_ref().ok_or("Open footage first")?;
    let last = f.frames.len().saturating_sub(1);
    let a = from.unwrap_or(0).min(last);
    let b = to.unwrap_or(last).min(last);
    if b < a + 2 {
        return Err("The range needs at least 3 frames".into());
    }
    Ok((a, b))
}

/// Outlines of the camera solve's ignored layers at a frame.
fn camera_excludes(
    layers: &[Layer],
    ids: &[u32],
    frame: usize,
) -> Vec<Vec<trackwright_core::Point>> {
    ids.iter()
        .filter_map(|id| layers.iter().find(|l| l.id == *id))
        .filter_map(|l| l.shape_at_or_nearest(frame))
        .map(|s| s.flatten(8))
        .collect()
}

/// Step 1 of a camera solve: track points over the whole frame (In/Out range).
#[tauri::command]
fn camera_track_points(
    app: AppHandle,
    state: State<'_, AppState>,
    from: Option<usize>,
    to: Option<usize>,
) -> CmdResult<()> {
    if state.tracking.swap(true, Ordering::SeqCst) {
        return Err("Already tracking".into());
    }
    let before = state.project.lock().unwrap().clone();
    let (a, b) = match camera_range(&before, from, to) {
        Ok(r) => r,
        Err(e) => {
            state.tracking.store(false, Ordering::SeqCst);
            return Err(e);
        }
    };
    let width = before.footage.as_ref().map_or(1920, |f| f.width as usize);
    let settings = AutoTrackSettings::for_width(
        width,
        before.camera.settings.max_points.clamp(50, 3000),
        before.camera.settings.min_distance,
    );
    state.cancel.store(false, Ordering::SeqCst);
    let (cache, cancel, tracking) = (
        state.cache.clone(),
        state.cancel.clone(),
        state.tracking.clone(),
    );
    let layers = before.layers.clone();
    let ids = before.camera.settings.exclude_layers.clone();
    std::thread::spawn(move || {
        let exclude = move |f: usize| camera_excludes(&layers, &ids, f);
        let result = {
            let mut src = cache.lock().unwrap();
            auto_track(
                &mut *src,
                a,
                b,
                &settings,
                &exclude,
                &cancel,
                |done, total| {
                    let _ = app.emit(
                        "camera-progress",
                        CameraProgress {
                            stage: format!("Tracking points {done} / {total}"),
                            fraction: done as f64 / total as f64,
                        },
                    );
                },
            )
        };
        let done = match result {
            Ok((tracks, reason)) => {
                let n = tracks.len();
                if let Some(state) = app.try_state::<AppState>() {
                    let mut p = state.project.lock().unwrap();
                    p.camera.tracks = tracks;
                    p.camera.solve = None;
                    if *p != before {
                        push_undo(&state, before);
                        mark_changed(&state);
                    }
                }
                let msg = if reason == StopReason::Cancelled {
                    format!("Stopped. Kept {n} point tracks")
                } else {
                    format!("Tracked {n} points. Next: Solve Camera")
                };
                CameraDone {
                    ok: true,
                    message: msg,
                }
            }
            Err(e) => CameraDone {
                ok: false,
                message: e.to_string(),
            },
        };
        tracking.store(false, Ordering::SeqCst);
        let _ = app.emit("camera-done", done);
    });
    Ok(())
}

/// Step 2: solve camera, focal length (and lens distortion) from the point tracks.
#[tauri::command]
fn camera_solve(
    app: AppHandle,
    state: State<'_, AppState>,
    from: Option<usize>,
    to: Option<usize>,
) -> CmdResult<()> {
    if state.tracking.swap(true, Ordering::SeqCst) {
        return Err("Already busy".into());
    }
    let before = state.project.lock().unwrap().clone();
    let setup = (|| -> CmdResult<(Vec<usize>, SolveOptions)> {
        if before.camera.tracks.is_empty() {
            return Err("Track points first (step 1)".into());
        }
        let f = before.footage.as_ref().ok_or("Open footage first")?;
        let (a, b) = camera_range(&before, from, to)?;
        let s = &before.camera.settings;
        Ok((
            (a..=b).collect(),
            SolveOptions {
                width: f.width as f64,
                height: f.height as f64,
                focal: s.focal,
                sensor_mm: s.sensor_mm,
                mode: s.mode,
                solve_distortion: s.solve_distortion,
                lens: before.lens,
            },
        ))
    })();
    let (frames, opts) = match setup {
        Ok(v) => v,
        Err(e) => {
            state.tracking.store(false, Ordering::SeqCst);
            return Err(e);
        }
    };
    state.cancel.store(false, Ordering::SeqCst);
    let (cancel, tracking) = (state.cancel.clone(), state.tracking.clone());
    std::thread::spawn(move || {
        let tracks = before.camera.tracks.clone();
        let result = sfm::solve_camera(&tracks, &frames, &opts, &cancel, |stage, fraction| {
            let _ = app.emit(
                "camera-progress",
                CameraProgress {
                    stage: stage.to_string(),
                    fraction,
                },
            );
        });
        let done = match result {
            Ok(solve) => {
                let sensor = before.camera.settings.sensor_mm;
                let msg = format!(
                    "{} camera solved: {} frames, focal {:.1} mm, error {:.2} px, {} points{}",
                    if solve.tripod { "Tripod" } else { "Moving" },
                    solve.poses.len(),
                    solve.focal_mm(sensor),
                    solve.rms_px,
                    solve.points.len(),
                    if opts.solve_distortion {
                        format!(", lens k1 {:.4}", solve.lens.k1)
                    } else {
                        String::new()
                    }
                );
                if let Some(state) = app.try_state::<AppState>() {
                    let mut p = state.project.lock().unwrap();
                    if opts.solve_distortion {
                        p.lens = solve.lens;
                    }
                    p.camera.solve = Some(solve);
                    push_undo(&state, before);
                    mark_changed(&state);
                }
                CameraDone {
                    ok: true,
                    message: msg,
                }
            }
            Err(e) => CameraDone {
                ok: false,
                message: e.to_string(),
            },
        };
        tracking.store(false, Ordering::SeqCst);
        let _ = app.emit("camera-done", done);
    });
    Ok(())
}

#[tauri::command]
fn set_camera_settings(
    state: State<'_, AppState>,
    settings: sfm::CameraSettings,
) -> CmdResult<AppView> {
    mutate(&state, |p| {
        p.camera.settings = settings;
        Ok(())
    })?;
    Ok(view(&state))
}

#[tauri::command]
fn clear_camera(state: State<'_, AppState>) -> CmdResult<AppView> {
    mutate(&state, |p| {
        p.camera.tracks.clear();
        p.camera.solve = None;
        Ok(())
    })?;
    Ok(view(&state))
}

/// Make a planar layer's surface the ground: origin at its centre, X along its top edge,
/// and (optionally) scale the scene so the surface is `width` units wide.
#[tauri::command]
fn camera_orient(
    state: State<'_, AppState>,
    layer_id: u32,
    frame: usize,
    width: Option<f64>,
) -> CmdResult<AppView> {
    mutate(&state, |p| {
        let solve = p.camera.solve.clone().ok_or("Solve the camera first")?;
        let layer = p.layer(layer_id).ok_or("Layer not found")?.clone();
        if !solve.poses.contains_key(&frame) {
            return Err("The camera isn't solved on this frame".into());
        }
        let shape = layer
            .shape_at(frame)
            .ok_or("That layer isn't tracked on this frame")?;
        let poly = shape.flatten(8);
        // Solved points whose track lies inside the layer's shape on this frame.
        let ids: HashMap<u32, usize> = p
            .camera
            .tracks
            .iter()
            .enumerate()
            .map(|(i, t)| (t.id, i))
            .collect();
        let mut pts = Vec::new();
        for sp in solve.points.iter().filter(|sp| sp.error < 2.0) {
            if let Some(&ti) = ids.get(&sp.id) {
                if let Some(px) = p.camera.tracks[ti].points.get(&frame) {
                    if trackwright_core::shape::point_in_polygon(&poly, *px) {
                        pts.push(sp.pos);
                    }
                }
            }
        }
        if pts.len() < 4 {
            return Err(format!("Only {} solved points lie on {} at this frame; pick a larger textured surface or another frame", pts.len(), layer.name));
        }
        let first =
            sfm::orient_to_plane(&solve, &pts, frame, None, None).ok_or("Could not fit a plane")?;
        // Plane in solve coordinates: origin and normal from the first fit.
        let origin = first.from_user([0.0, 0.0, 0.0]);
        let normal = linalg::sub(first.from_user([0.0, 1.0, 0.0]), origin);
        let corners = layer
            .surface_at(frame)
            .ok_or("That layer has no surface on this frame")?;
        let hit = |c: trackwright_core::Point| sfm::ray_plane(&solve, frame, c, origin, normal);
        let (Some(tl), Some(tr), Some(br), Some(bl)) = (
            hit(corners[0]),
            hit(corners[1]),
            hit(corners[2]),
            hit(corners[3]),
        ) else {
            return Err(
                "The surface corners don't hit the fitted plane; adjust the surface".into(),
            );
        };
        let hint = linalg::sub(tr, tl);
        let measured = linalg::norm(hint);
        let scale_ref = width.filter(|w| *w > 0.0).map(|w| (measured, w));
        let mut world = sfm::orient_to_plane(&solve, &pts, frame, Some(hint), scale_ref)
            .ok_or("Could not orient the scene")?;
        if scale_ref.is_none() {
            world.scale = 1.0 / measured.max(1e-9);
        }
        // Origin at the centre of the surface.
        let centre = linalg::scale(linalg::add(linalg::add(tl, tr), linalg::add(br, bl)), 0.25);
        let c_user = world.to_user(centre);
        world.translation = linalg::sub(world.translation, c_user);
        p.camera.solve.as_mut().unwrap().world = world;
        Ok(())
    })?;
    Ok(view(&state))
}

#[tauri::command]
fn camera_reset_orientation(state: State<'_, AppState>) -> CmdResult<AppView> {
    mutate(&state, |p| {
        let s = p.camera.solve.as_mut().ok_or("Solve the camera first")?;
        s.world = sfm::World::default();
        Ok(())
    })?;
    Ok(view(&state))
}

#[tauri::command]
async fn export_camera(
    app: AppHandle,
    state: State<'_, AppState>,
    format: CameraFormat,
    destination: String,
) -> CmdResult<Option<String>> {
    let (text, name) = {
        let p = state.project.lock().unwrap();
        let f = p.footage.as_ref().ok_or("Open footage first")?;
        let s = p.camera.solve.as_ref().ok_or("Solve the camera first")?;
        let name = format!("{}_camera", footage_stem(f));
        (
            camera_export::export_camera(format, f, s, p.camera.settings.sensor_mm, &name),
            format!("{name}.{}", format.extension()),
        )
    };
    let settings = state.settings.lock().unwrap().clone();
    let dir = match destination.as_str() {
        "desktop" => dirs::desktop_dir()
            .or_else(dirs::home_dir)
            .ok_or("Could not find your Desktop folder")?,
        "forma3d" => PathBuf::from(settings.forma3d_dir),
        "tidefire" => PathBuf::from(settings.tidefire_dir),
        "reelforge" => PathBuf::from(settings.reelforge_dir),
        _ => {
            let picked = app
                .dialog()
                .file()
                .set_title("Export camera")
                .set_file_name(&name)
                .blocking_save_file();
            let Some(path) = picked.and_then(|p| p.into_path().ok()) else {
                return Ok(None);
            };
            std::fs::write(&path, text).map_err(|e| format!("Export failed: {e}"))?;
            return Ok(Some(path.to_string_lossy().to_string()));
        }
    };
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Could not create {}: {e}", dir.display()))?;
    let path = dir.join(name);
    std::fs::write(&path, text).map_err(|e| format!("Export failed: {e}"))?;
    Ok(Some(path.to_string_lossy().to_string()))
}

#[tauri::command]
fn set_lens(state: State<'_, AppState>, k1: f64, k2: f64) -> CmdResult<AppView> {
    mutate(&state, |p| {
        p.lens = Lens {
            k1: k1.clamp(-1.0, 1.0),
            k2: k2.clamp(-1.0, 1.0),
        };
        Ok(())
    })?;
    Ok(view(&state))
}

#[tauri::command]
fn set_lens_lines(
    state: State<'_, AppState>,
    lines: Vec<(usize, Vec<[f64; 2]>)>,
) -> CmdResult<AppView> {
    mutate(&state, |p| {
        p.lens_lines = lines;
        Ok(())
    })?;
    Ok(view(&state))
}

#[derive(Serialize)]
struct LensResult {
    view: AppView,
    rms_before: f64,
    rms_after: f64,
}

/// Solve k1 (and k2) so the lines drawn along straight edges become straight.
#[tauri::command]
fn calibrate_lens(state: State<'_, AppState>, solve_k2: bool) -> CmdResult<LensResult> {
    let fit = mutate(&state, |p| {
        let f = p.footage.as_ref().ok_or("Open footage first")?;
        let lines: Vec<Vec<[f64; 2]>> = p.lens_lines.iter().map(|(_, l)| l.clone()).collect();
        if lines.iter().filter(|l| l.len() >= 3).count() < 2 {
            return Err("Draw at least two lines (3+ points each) along edges that should be straight, ideally near the frame edges".into());
        }
        let fit = trackwright_core::lens::calibrate_from_lines(
            &lines,
            f.width as f64,
            f.height as f64,
            solve_k2,
        )
        .ok_or("Calibration failed")?;
        p.lens = fit.lens;
        Ok(fit)
    })?;
    Ok(LensResult {
        view: view(&state),
        rms_before: fit.rms_before,
        rms_after: fit.rms_after,
    })
}

#[tauri::command]
fn stop_render(state: State<'_, AppState>) {
    state.render_cancel.store(true, Ordering::SeqCst);
}

/// Open footage or a project outside a command (drag & drop, command line) and notify the UI.
fn open_and_emit(app: AppHandle, path: PathBuf) {
    let Some(state) = app.try_state::<AppState>() else {
        return;
    };
    if state.tracking.load(Ordering::SeqCst) {
        let _ = app.emit("app-error", "Stop tracking before opening something new");
        return;
    }
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("twproj"))
    {
        if state.dirty.load(Ordering::SeqCst) {
            let _ = app.emit(
                "app-error",
                "Save or start a new project before dropping in another project",
            );
            return;
        }
        match load_project_file(&app, &path) {
            Ok(p) => {
                install_project(&app, &state, p, Some(path.clone()), false);
                add_recent(&app, &state, &path);
                let _ = app.emit("view-changed", view(&state));
            }
            Err(e) => {
                let _ = app.emit("app-error", e);
            }
        }
        return;
    }
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
            let v = install_footage(&app, &state, f);
            let _ = app.emit("view-changed", v);
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
                history: Mutex::new(History::default()),
                dirty: AtomicBool::new(false),
                edit_version: AtomicU64::new(0),
                project_path: Mutex::new(None),
                cache: Arc::new(Mutex::new(FrameCache::new())),
                cancel: Arc::new(AtomicBool::new(false)),
                tracking: Arc::new(AtomicBool::new(false)),
                settings: Mutex::new(settings),
                proxies: ProxyBuilder::new(),
                allow_close: AtomicBool::new(false),
                rendering: Arc::new(AtomicBool::new(false)),
                render_cancel: Arc::new(AtomicBool::new(false)),
            });
            let handle = app.handle().clone();
            std::thread::spawn(move || autosave_loop(handle));
            // `trackwright <footage or .twproj>` opens it straight away.
            if let Some(arg) = std::env::args().skip(1).find(|a| !a.starts_with('-')) {
                let handle = app.handle().clone();
                std::thread::spawn(move || open_and_emit(handle, PathBuf::from(arg)));
            }
            Ok(())
        })
        .on_window_event(|window, event| match event {
            tauri::WindowEvent::DragDrop(tauri::DragDropEvent::Drop { paths, .. }) => {
                if let Some(p) = paths.first().cloned() {
                    let handle = window.app_handle().clone();
                    std::thread::spawn(move || open_and_emit(handle, p));
                }
            }
            tauri::WindowEvent::CloseRequested { api, .. } => {
                if let Some(state) = window.app_handle().try_state::<AppState>() {
                    if state.dirty.load(Ordering::SeqCst)
                        && !state.allow_close.load(Ordering::SeqCst)
                    {
                        api.prevent_close();
                        let _ = window.emit("confirm-close", ());
                    }
                }
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![
            get_view,
            new_project,
            undo,
            redo,
            open_footage,
            set_footage_timing,
            save_project,
            open_project,
            recovery_info,
            recover_autosave,
            discard_autosave,
            quit_app,
            add_layer,
            update_layer,
            delete_layer,
            duplicate_layer,
            set_link,
            set_reference_frame,
            clear_track,
            choose_insert,
            start_tracking,
            stop_tracking,
            get_proxy_info,
            get_settings,
            set_settings,
            pick_folder,
            export_layer,
            render_layer,
            stop_render,
            camera_track_points,
            camera_solve,
            set_camera_settings,
            clear_camera,
            camera_orient,
            camera_reset_orientation,
            export_camera,
            set_lens,
            set_lens_lines,
            calibrate_lens,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Trackwright");
}

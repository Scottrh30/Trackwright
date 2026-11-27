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
use trackwright_core::export::{self, ExportFormat};
use trackwright_core::project::Insert;
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running Trackwright");
}

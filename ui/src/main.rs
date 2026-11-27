//! Trackwright UI (Leptos, client-side rendered inside Tauri).
//!
//! © Scott Hambach. All rights reserved.

mod draw;
mod tauri;

use draw::{draw_timeline, draw_viewport, image_for, Overlays, Scene, Tool, View};
use leptos::ev;
use leptos::html;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;
use trackwright_core::camera_export::CameraFormat;
use trackwright_core::geometry::{MotionModel, Point};
use trackwright_core::sfm::{CameraSettings, FocalMode, SolveMode};
use trackwright_core::shape::point_in_polygon;
use trackwright_core::{ExportFormat, FrameTrack, Layer, Project, Shape};
use wasm_bindgen::JsCast;

// ---------------------------------------------------------------------------
// Types mirrored from the backend

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct AppSettings {
    reelforge_dir: String,
    forma3d_dir: String,
    tidefire_dir: String,
    default_format: ExportFormat,
    #[serde(default)]
    recent_projects: Vec<String>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            reelforge_dir: String::new(),
            forma3d_dir: String::new(),
            tidefire_dir: String::new(),
            default_format: ExportFormat::Reelforge,
            recent_projects: vec![],
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct AppView {
    project: Project,
    can_undo: bool,
    can_redo: bool,
    dirty: bool,
    path: Option<String>,
    recent: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
struct ProxyInfo {
    prefix: Option<String>,
    ready: usize,
    total: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct RecoveryInfo {
    original_path: Option<String>,
    saved_at: u64,
}

#[derive(Clone, Deserialize)]
struct TrackFrameEvent {
    layer_id: u32,
    frame: usize,
    track: FrameTrack,
}

#[derive(Clone, Deserialize)]
struct CameraProgressEvent {
    stage: String,
    #[allow(dead_code)]
    fraction: f64,
}

#[derive(Clone, Deserialize)]
struct CameraDoneEvent {
    ok: bool,
    message: String,
}

#[derive(Clone, Deserialize)]
struct LensResult {
    view: AppView,
    rms_before: f64,
    rms_after: f64,
}

#[derive(Clone, Deserialize)]
struct RenderProgress {
    done: usize,
    total: usize,
}

#[derive(Clone, Deserialize)]
struct RenderDone {
    ok: bool,
    message: String,
}

#[derive(Clone, Deserialize)]
struct TrackDoneEvent {
    ok: bool,
    message: String,
}

// ---------------------------------------------------------------------------
// UI state

#[derive(Clone, PartialEq)]
enum Drag {
    None,
    Pan { start: Point, pan: Point },
    ShapePoint(usize),
    SurfaceCorner(usize),
    MoveShape { start: Point, orig: Shape },
    MoveSurface { start: Point, orig: [Point; 4] },
    AdjustCorner(usize),
    MoveAdjust { start: Point, orig: [Point; 4] },
}

/// Something that would throw away unsaved work.
#[derive(Clone, Debug, PartialEq)]
enum Pending {
    Close,
    New,
    OpenProject(Option<String>),
}

#[derive(Clone, Debug, PartialEq)]
enum Modal {
    None,
    About,
    Unsaved(Pending),
    Recovery(RecoveryInfo),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewerRes {
    Auto,
    Proxy,
    Full,
}

/// All shared UI state. Everything inside is a cheap, `Copy` handle.
#[derive(Clone, Copy)]
struct Ctx {
    project: RwSignal<Project>,
    can_undo: RwSignal<bool>,
    can_redo: RwSignal<bool>,
    dirty: RwSignal<bool>,
    path: RwSignal<Option<String>>,
    recent: RwSignal<Vec<String>>,
    recent_open: RwSignal<bool>,
    frame: RwSignal<usize>,
    selected: RwSignal<Option<u32>>,
    sel_point: RwSignal<Option<usize>>,
    renaming: RwSignal<Option<u32>>,
    tool: RwSignal<Tool>,
    draft: RwSignal<Vec<Point>>,
    hover: RwSignal<Option<Point>>,
    view: RwSignal<View>,
    redraw: RwSignal<u32>,
    tracking: RwSignal<bool>,
    playing: RwSignal<bool>,
    scrubbing: RwSignal<bool>,
    status: RwSignal<String>,
    status_err: RwSignal<bool>,
    show_surface: RwSignal<bool>,
    show_grid: RwSignal<bool>,
    show_points: RwSignal<bool>,
    show_shapes: RwSignal<bool>,
    show_inserts: RwSignal<bool>,
    viewer_res: RwSignal<ViewerRes>,
    proxy: RwSignal<ProxyInfo>,
    settings: RwSignal<AppSettings>,
    export_format: RwSignal<ExportFormat>,
    modal: RwSignal<Modal>,
    in_point: RwSignal<Option<usize>>,
    out_point: RwSignal<Option<usize>>,
    render_mode: RwSignal<String>,
    smooth_frames: RwSignal<usize>,
    zoom_pct: RwSignal<f64>,
    make_video: RwSignal<bool>,
    rendering: RwSignal<bool>,
    render_progress: RwSignal<(usize, usize)>,
    render_bg: RwSignal<Option<u32>>,
    render_search: RwSignal<usize>,
    render_grow: RwSignal<f64>,
    render_match: RwSignal<bool>,
    render_all_layers: RwSignal<bool>,
    show_tracks: RwSignal<bool>,
    show_cloud: RwSignal<bool>,
    show_ground: RwSignal<bool>,
    show_lens_grid: RwSignal<bool>,
    camera_busy: RwSignal<bool>,
    camera_progress: RwSignal<(String, f64)>,
    camera_format: RwSignal<CameraFormat>,
    orient_layer: RwSignal<Option<u32>>,
    orient_width: RwSignal<f64>,
    lens_k2: RwSignal<bool>,
    drag: StoredValue<Drag>,
    play_handle: StoredValue<Option<IntervalHandle>>,
    viewport: NodeRef<html::Canvas>,
    timeline: NodeRef<html::Canvas>,
}

fn dist2(a: Point, b: Point) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn enum_id<T: Serialize>(v: T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default()
}

fn file_name(p: &str) -> String {
    p.rsplit(['/', '\\']).next().unwrap_or(p).to_string()
}

impl Ctx {
    fn new() -> Self {
        Self {
            project: RwSignal::new(Project::default()),
            can_undo: RwSignal::new(false),
            can_redo: RwSignal::new(false),
            dirty: RwSignal::new(false),
            path: RwSignal::new(None),
            recent: RwSignal::new(vec![]),
            recent_open: RwSignal::new(false),
            frame: RwSignal::new(0),
            selected: RwSignal::new(None),
            sel_point: RwSignal::new(None),
            renaming: RwSignal::new(None),
            tool: RwSignal::new(Tool::Pan),
            draft: RwSignal::new(vec![]),
            hover: RwSignal::new(None),
            view: RwSignal::new(View::default()),
            redraw: RwSignal::new(0),
            tracking: RwSignal::new(false),
            playing: RwSignal::new(false),
            scrubbing: RwSignal::new(false),
            status: RwSignal::new("Ready".into()),
            status_err: RwSignal::new(false),
            show_surface: RwSignal::new(true),
            show_grid: RwSignal::new(false),
            show_points: RwSignal::new(true),
            show_shapes: RwSignal::new(true),
            show_inserts: RwSignal::new(true),
            viewer_res: RwSignal::new(ViewerRes::Auto),
            proxy: RwSignal::new(ProxyInfo::default()),
            settings: RwSignal::new(AppSettings::default()),
            export_format: RwSignal::new(ExportFormat::Reelforge),
            modal: RwSignal::new(Modal::None),
            in_point: RwSignal::new(None),
            out_point: RwSignal::new(None),
            render_mode: RwSignal::new("insert".into()),
            smooth_frames: RwSignal::new(15),
            zoom_pct: RwSignal::new(100.0),
            make_video: RwSignal::new(true),
            rendering: RwSignal::new(false),
            render_progress: RwSignal::new((0, 0)),
            render_bg: RwSignal::new(None),
            render_search: RwSignal::new(20),
            render_grow: RwSignal::new(6.0),
            render_match: RwSignal::new(true),
            render_all_layers: RwSignal::new(false),
            show_tracks: RwSignal::new(true),
            show_cloud: RwSignal::new(true),
            show_ground: RwSignal::new(true),
            show_lens_grid: RwSignal::new(false),
            camera_busy: RwSignal::new(false),
            camera_progress: RwSignal::new((String::new(), 0.0)),
            camera_format: RwSignal::new(CameraFormat::Forma3d),
            orient_layer: RwSignal::new(None),
            orient_width: RwSignal::new(0.0),
            lens_k2: RwSignal::new(false),
            drag: StoredValue::new(Drag::None),
            play_handle: StoredValue::new(None),
            viewport: NodeRef::new(),
            timeline: NodeRef::new(),
        }
    }

    fn say(self, msg: impl Into<String>, err: bool) {
        self.status.set(msg.into());
        self.status_err.set(err);
    }

    fn bump(self) {
        self.redraw.update(|n| *n = n.wrapping_add(1));
    }

    fn frame_count(self) -> usize {
        self.project
            .with(|p| p.footage.as_ref().map_or(0, |f| f.frames.len()))
    }

    fn frame_count_untracked(self) -> usize {
        self.project
            .with_untracked(|p| p.footage.as_ref().map_or(0, |f| f.frames.len()))
    }

    fn apply_view(self, v: AppView) {
        let n = v.project.footage.as_ref().map_or(0, |f| f.frames.len());
        let ids: Vec<u32> = v.project.layers.iter().map(|l| l.id).collect();
        self.project.set(v.project);
        self.can_undo.set(v.can_undo);
        self.can_redo.set(v.can_redo);
        self.dirty.set(v.dirty);
        self.path.set(v.path);
        self.recent.set(v.recent);
        if self.frame.get_untracked() >= n {
            self.frame.set(n.saturating_sub(1));
        }
        match self.selected.get_untracked() {
            Some(s) if ids.contains(&s) => {}
            _ => self.selected.set(ids.last().copied()),
        }
    }

    /// Run a backend command that returns an `AppView`.
    fn call(self, cmd: &'static str, args: serde_json::Value, ok_msg: Option<&'static str>) {
        spawn_local(async move {
            match tauri::call::<AppView>(cmd, args).await {
                Ok(v) => {
                    self.apply_view(v);
                    if let Some(m) = ok_msg {
                        self.say(m, false);
                    }
                }
                Err(e) => self.say(e, true),
            }
        });
    }

    /// Change the selected layer locally and push it to the backend (one undo step).
    fn edit_layer(self, f: impl FnOnce(&mut Layer) + 'static) {
        let Some(id) = self.selected.get_untracked() else {
            return;
        };
        let mut edited = None;
        self.project.update(|p| {
            if let Some(l) = p.layer_mut(id) {
                f(l);
                edited = Some(l.clone());
            }
        });
        if let Some(l) = edited {
            self.call("update_layer", json!({ "layer": l }), None);
        }
    }

    fn undo(self) {
        if !self.can_undo.get_untracked() {
            return;
        }
        self.call("undo", json!({}), Some("Undone"));
    }

    fn redo(self) {
        if !self.can_redo.get_untracked() {
            return;
        }
        self.call("redo", json!({}), Some("Redone"));
    }

    fn fit_view(self) {
        let Some(c) = self.viewport.get_untracked() else {
            return;
        };
        self.project.with_untracked(|p| {
            if let Some(f) = &p.footage {
                self.view.set(View::fit(
                    f.width as f64,
                    f.height as f64,
                    c.client_width() as f64,
                    c.client_height() as f64,
                ));
            }
        });
    }

    fn open_footage(self, kind: &'static str) {
        self.say("Opening footage… (video is decoded to frames first)", false);
        self.call(
            "open_footage",
            json!({ "kind": kind, "path": null }),
            Some("Footage loaded. Choose Draw Shape (D) and outline a flat surface"),
        );
    }

    /// Run `action` now, or ask about unsaved changes first.
    fn guard(self, action: Pending) {
        if self.dirty.get_untracked()
            && self
                .project
                .with_untracked(|p| p.footage.is_some() || !p.layers.is_empty())
        {
            self.modal.set(Modal::Unsaved(action));
        } else {
            self.perform(action);
        }
    }

    fn perform(self, action: Pending) {
        self.modal.set(Modal::None);
        match action {
            Pending::Close => spawn_local(async move {
                let _ = tauri::call::<()>("quit_app", json!({})).await;
            }),
            Pending::New => {
                self.selected.set(None);
                self.call("new_project", json!({}), Some("New project"));
            }
            Pending::OpenProject(path) => {
                self.selected.set(None);
                self.call(
                    "open_project",
                    json!({ "path": path }),
                    Some("Project opened"),
                );
            }
        }
    }

    /// Save; `then(true)` runs after a successful save.
    fn save_then(self, save_as: bool, then: impl FnOnce(bool) + 'static) {
        spawn_local(async move {
            match tauri::call::<Option<AppView>>("save_project", json!({ "saveAs": save_as })).await
            {
                Ok(Some(v)) => {
                    let p = v.path.clone().unwrap_or_default();
                    self.apply_view(v);
                    self.say(format!("Saved {p}"), false);
                    then(true);
                }
                Ok(None) => then(false),
                Err(e) => {
                    self.say(e, true);
                    then(false);
                }
            }
        });
    }

    fn save(self, save_as: bool) {
        self.save_then(save_as, |_| {});
    }

    fn start_track(self, direction: i32, steps: Option<usize>) {
        if self.tracking.get_untracked() {
            return;
        }
        let Some(id) = self.selected.get_untracked() else {
            self.say("Select or draw a layer to track", true);
            return;
        };
        let fr = self.frame.get_untracked();
        let (tracked, linked) = self.project.with_untracked(|p| {
            p.layer(id)
                .map(|l| (l.tracks.contains_key(&fr), l.link.is_some()))
                .unwrap_or((false, false))
        });
        if linked {
            self.say("This layer follows another layer's motion. Set Motion source to “Own track” to track it", true);
            return;
        }
        if !tracked {
            self.say("This frame isn't tracked for the layer. Move to a tracked frame (coloured on the timeline)", true);
            return;
        }
        self.tracking.set(true);
        self.say("Tracking…", false);
        // Stop at the Out point going forward, the In point going backward.
        let until = if direction >= 0 {
            self.out_point.get_untracked()
        } else {
            self.in_point.get_untracked()
        };
        spawn_local(async move {
            let args = json!({ "id": id, "from": fr, "direction": direction, "steps": steps, "until": until, "blend": false });
            if let Err(e) = tauri::call::<()>("start_tracking", args).await {
                self.tracking.set(false);
                self.say(e, true);
            }
        });
    }

    /// Re-track In..Out and blend it into the existing track at Out.
    fn retrack_section(self) {
        if self.tracking.get_untracked() {
            return;
        }
        let Some(id) = self.selected.get_untracked() else {
            self.say("Select a layer first", true);
            return;
        };
        let (Some(a), Some(b)) = (
            self.in_point.get_untracked(),
            self.out_point.get_untracked(),
        ) else {
            self.say("Set an In point (I) at a good frame and an Out point (O) where the track is good again", true);
            return;
        };
        let ok = self.project.with_untracked(|p| {
            p.layer(id)
                .map(|l| l.tracks.contains_key(&a) && l.link.is_none())
                .unwrap_or(false)
        });
        if !ok {
            self.say(
                "The In frame must already be tracked (and the layer must use its own track)",
                true,
            );
            return;
        }
        if b <= a {
            self.say("The Out point must be after the In point", true);
            return;
        }
        self.tracking.set(true);
        self.say("Re-tracking section…", false);
        spawn_local(async move {
            let args = json!({ "id": id, "from": a, "direction": 1, "steps": null, "until": b, "blend": true });
            if let Err(e) = tauri::call::<()>("start_tracking", args).await {
                self.tracking.set(false);
                self.say(e, true);
            }
        });
    }

    /// Jump to the next frame flagged on the quality graph (wraps around).
    fn next_problem(self) {
        let Some(id) = self.selected.get_untracked() else {
            return;
        };
        let fr = self.frame.get_untracked();
        let found = self.project.with_untracked(|p| {
            let l = p.layer(id)?;
            let probs: Vec<(usize, trackwright_core::Problem)> = l
                .tracks
                .keys()
                .filter_map(|&f| l.problem_at(f).map(|pr| (f, pr)))
                .collect();
            probs
                .iter()
                .find(|(f, _)| *f > fr)
                .or(probs.first())
                .copied()
        });
        match found {
            Some((f, pr)) => {
                self.frame.set(f);
                let why = match pr {
                    trackwright_core::Problem::HighError => "high error",
                    trackwright_core::Problem::FewPoints => "few good points",
                    trackwright_core::Problem::Retried => "needed a second, wider search (blur?)",
                };
                self.say(
                    format!(
                        "Check this frame: {why}. Fix with Adjust (A), or set In/Out and Re-track"
                    ),
                    false,
                );
            }
            None => self.say("No problem frames on this layer", false),
        }
    }

    fn problem_count(self) -> usize {
        let Some(id) = self.selected.get() else {
            return 0;
        };
        self.project.with(|p| {
            p.layer(id)
                .map(|l| {
                    l.tracks
                        .keys()
                        .filter(|&&f| l.problem_at(f).is_some())
                        .count()
                })
                .unwrap_or(0)
        })
    }

    fn set_in_out(self, is_in: bool) {
        let f = self.frame.get_untracked();
        if is_in {
            if self.out_point.get_untracked().is_some_and(|o| o <= f) {
                self.out_point.set(None);
            }
            self.in_point.set(Some(f));
        } else {
            if self.in_point.get_untracked().is_some_and(|i| i >= f) {
                self.in_point.set(None);
            }
            self.out_point.set(Some(f));
        }
    }

    fn stop_track(self) {
        spawn_local(async move {
            let _ = tauri::call::<()>("stop_tracking", json!({})).await;
        });
    }

    fn finish_draft(self) {
        if self.tool.get_untracked() == Tool::LensLine {
            let pts = self.draft.get_untracked();
            if pts.len() < 3 {
                self.say(
                    "A lens line needs at least 3 points along a straight edge",
                    true,
                );
                return;
            }
            self.draft.set(vec![]);
            let fr = self.frame.get_untracked();
            let mut lines = self.project.with_untracked(|p| p.lens_lines.clone());
            lines.push((fr, pts));
            let n = lines.len();
            self.call("set_lens_lines", json!({ "lines": lines }), None);
            self.say(format!("{n} lens line(s). Draw more along straight edges near the frame edges, then Calibrate"), false);
            return;
        }
        let pts = self.draft.get_untracked();
        if pts.len() < 3 {
            self.say("A shape needs at least 3 points", true);
            return;
        }
        self.draft.set(vec![]);
        let fr = self.frame.get_untracked();
        spawn_local(async move {
            match tauri::call::<AppView>("add_layer", json!({ "frame": fr, "points": pts })).await {
                Ok(v) => {
                    let id = v.project.layers.last().map(|l| l.id);
                    self.apply_view(v);
                    self.selected.set(id);
                    self.tool.set(Tool::Surface);
                    self.say("Layer created. Adjust the blue surface corners if needed, then Track Forward (Shift+→ steps)", false);
                }
                Err(e) => self.say(e, true),
            }
        });
    }

    fn step_frame(self, d: i64) {
        let n = self.frame_count_untracked() as i64;
        if n == 0 {
            return;
        }
        self.frame
            .update(|f| *f = (*f as i64 + d).clamp(0, n - 1) as usize);
    }

    fn toggle_play(self) {
        if self.playing.get_untracked() {
            self.playing.set(false);
            if let Some(h) = self.play_handle.get_value() {
                h.clear();
            }
            self.play_handle.set_value(None);
            return;
        }
        let fps = self
            .project
            .with_untracked(|p| p.footage.as_ref().map_or(24.0, |f| f.fps))
            .clamp(1.0, 120.0);
        self.playing.set(true);
        let h = set_interval_with_handle(
            move || {
                let n = self.frame_count_untracked();
                if n > 0 {
                    self.frame.update(|f| *f = (*f + 1) % n);
                }
            },
            Duration::from_millis((1000.0 / fps) as u64),
        );
        self.play_handle.set_value(h.ok());
    }

    fn export(self, destination: &'static str) {
        let Some(id) = self.selected.get_untracked() else {
            self.say("Select a tracked layer to export", true);
            return;
        };
        let fmt = self.export_format.get_untracked();
        spawn_local(async move {
            match tauri::call::<Option<String>>(
                "export_layer",
                json!({ "id": id, "format": fmt, "destination": destination }),
            )
            .await
            {
                Ok(Some(p)) => self.say(format!("Exported to {p}"), false),
                Ok(None) => {}
                Err(e) => self.say(e, true),
            }
        });
    }

    fn change_dir(self, which: &'static str) {
        spawn_local(async move {
            let title = format!("Choose the {which} import folder");
            if let Ok(Some(dir)) =
                tauri::call::<Option<String>>("pick_folder", json!({ "title": title })).await
            {
                let mut s = self.settings.get_untracked();
                match which {
                    "Reelforge" => s.reelforge_dir = dir,
                    "Forma3D" => s.forma3d_dir = dir,
                    _ => s.tidefire_dir = dir,
                }
                match tauri::call::<AppSettings>("set_settings", json!({ "settings": s })).await {
                    Ok(s) => self.settings.set(s),
                    Err(e) => self.say(e, true),
                }
            }
        });
    }

    fn duplicate_selected(self) {
        let Some(id) = self.selected.get_untracked() else {
            return;
        };
        spawn_local(async move {
            match tauri::call::<AppView>("duplicate_layer", json!({ "id": id })).await {
                Ok(v) => {
                    let new_id = v.project.next_layer_id.saturating_sub(1);
                    self.apply_view(v);
                    self.selected.set(Some(new_id));
                    self.say("Layer duplicated", false);
                }
                Err(e) => self.say(e, true),
            }
        });
    }

    fn delete_selected_point(self) {
        let Some(i) = self.sel_point.get_untracked() else {
            return;
        };
        let n = self.selected.get_untracked().and_then(|id| {
            self.project
                .with_untracked(|p| p.layer(id).map(|l| l.shape.points.len()))
        });
        if n.unwrap_or(0) <= 3 {
            self.say("A shape needs at least 3 points", true);
            return;
        }
        self.sel_point.set(None);
        self.edit_layer(move |l| {
            l.remove_shape_point(i);
        });
    }

    fn choose_insert(self, kind: &'static str) {
        let Some(id) = self.selected.get_untracked() else {
            return;
        };
        let start = self.frame.get_untracked();
        self.say("Loading insert…", false);
        spawn_local(async move {
            match tauri::call::<Option<AppView>>(
                "choose_insert",
                json!({ "id": id, "kind": kind, "start": start }),
            )
            .await
            {
                Ok(Some(v)) => {
                    self.apply_view(v);
                    self.show_inserts.set(true);
                    self.say(
                        "Insert pinned to the surface. Scrub or play to check the track",
                        false,
                    );
                }
                Ok(None) => self.say("Ready", false),
                Err(e) => self.say(e, true),
            }
        });
    }

    fn use_proxy(self) -> bool {
        match self.viewer_res.get() {
            ViewerRes::Full => false,
            ViewerRes::Proxy => true,
            ViewerRes::Auto => self.playing.get() || self.scrubbing.get() || self.tracking.get(),
        }
    }

    /// File to display for `frame`: a proxy when wanted and available, else the original.
    fn display_path(self, p: &Project, frame: usize, proxy: bool) -> Option<String> {
        let f = p.footage.as_ref()?;
        let original = f.frames.get(frame)?;
        if proxy {
            let info = self.proxy.get_untracked();
            if let Some(prefix) = info.prefix.as_ref() {
                if info.total == f.frames.len() && frame < info.ready {
                    return Some(format!("{prefix}{frame:06}.jpg"));
                }
            }
        }
        Some(original.clone())
    }
}

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(App);
}

// ---------------------------------------------------------------------------
// App

#[component]
fn App() -> impl IntoView {
    let ctx = Ctx::new();
    provide_context(ctx);

    // ---- backend events & startup ----
    tauri::listen::<TrackFrameEvent>("track-frame", move |e| {
        ctx.project.update(|p| {
            if let Some(l) = p.layer_mut(e.layer_id) {
                l.tracks.insert(e.frame, e.track.clone());
            }
        });
        ctx.frame.set(e.frame);
        ctx.say(
            format!(
                "Tracking… frame {}  ({} points, {:.2}px error)",
                e.frame, e.track.inliers, e.track.rms_error
            ),
            false,
        );
    });
    tauri::listen::<TrackDoneEvent>("track-done", move |e| {
        ctx.tracking.set(false);
        ctx.say(e.message, !e.ok);
        spawn_local(async move {
            if let Ok(v) = tauri::call::<AppView>("get_view", json!({})).await {
                ctx.apply_view(v);
            }
        });
    });
    tauri::listen::<AppView>("view-changed", move |v| {
        ctx.apply_view(v);
        ctx.say(
            "Opened. Choose Draw Shape (D) and outline a flat surface",
            false,
        );
    });
    tauri::listen::<ProxyInfo>("proxy-progress", move |p| ctx.proxy.set(p));
    tauri::listen::<RenderProgress>("render-progress", move |p| {
        ctx.render_progress.set((p.done, p.total));
        ctx.say(format!("Rendering… {} / {}", p.done, p.total), false);
    });
    tauri::listen::<CameraProgressEvent>("camera-progress", move |p| {
        ctx.camera_progress.set((p.stage.clone(), p.fraction));
        ctx.say(p.stage, false);
    });
    tauri::listen::<CameraDoneEvent>("camera-done", move |d| {
        ctx.camera_busy.set(false);
        ctx.say(d.message, !d.ok);
        spawn_local(async move {
            if let Ok(v) = tauri::call::<AppView>("get_view", json!({})).await {
                ctx.apply_view(v);
            }
        });
    });
    tauri::listen::<RenderDone>("render-done", move |d| {
        ctx.rendering.set(false);
        ctx.say(d.message, !d.ok);
    });
    tauri::listen::<String>("app-error", move |m| ctx.say(m, true));
    tauri::listen::<String>("app-status", move |m| ctx.say(m, false));
    tauri::listen::<serde_json::Value>("confirm-close", move |_| ctx.guard(Pending::Close));

    spawn_local(async move {
        if !tauri::in_tauri() {
            ctx.say(
                "Preview mode: start with `cargo tauri dev` to load footage",
                true,
            );
            return;
        }
        if let Ok(v) = tauri::call::<AppView>("get_view", json!({})).await {
            ctx.apply_view(v);
        }
        if let Ok(s) = tauri::call::<AppSettings>("get_settings", json!({})).await {
            ctx.export_format.set(s.default_format);
            ctx.settings.set(s);
        }
        if let Ok(Some(info)) =
            tauri::call::<Option<RecoveryInfo>>("recovery_info", json!({})).await
        {
            ctx.modal.set(Modal::Recovery(info));
        }
    });

    // Refit and refresh proxy status whenever new footage is loaded.
    Effect::new(move |prev: Option<String>| {
        let src = ctx.project.with(|p| {
            p.footage
                .as_ref()
                .map(|f| f.source.clone())
                .unwrap_or_default()
        });
        if prev.as_deref() != Some(src.as_str()) && !src.is_empty() {
            ctx.fit_view();
            spawn_local(async move {
                if let Ok(p) = tauri::call::<ProxyInfo>("get_proxy_info", json!({})).await {
                    ctx.proxy.set(p);
                }
            });
        }
        src
    });

    // ---- rendering ----
    Effect::new(move |_| {
        ctx.redraw.track();
        let Some(canvas) = ctx.viewport.get() else {
            return;
        };
        let fr = ctx.frame.get();
        let draft_pts = ctx.draft.get();
        let use_proxy = ctx.use_proxy();
        ctx.proxy.track();
        let ov = Overlays {
            surface: ctx.show_surface.get(),
            grid: ctx.show_grid.get(),
            points: ctx.show_points.get(),
            shapes: ctx.show_shapes.get(),
            inserts: ctx.show_inserts.get(),
            tracks: ctx.show_tracks.get(),
            cloud: ctx.show_cloud.get(),
            ground: ctx.show_ground.get(),
            lens_grid: ctx.show_lens_grid.get(),
        };
        let (sel, sel_point, tool, hover, view) = (
            ctx.selected.get(),
            ctx.sel_point.get(),
            ctx.tool.get(),
            ctx.hover.get(),
            ctx.view.get(),
        );
        let playing = ctx.playing.get();
        ctx.project.with(|p| {
            let bump = move || ctx.bump();
            let img = ctx
                .display_path(p, fr, use_proxy)
                .and_then(|path| image_for(&path, bump));
            // Preload upcoming frames for smooth scrubbing and playback.
            let ahead = if playing { 8 } else { 3 };
            for k in 1..=ahead {
                if let Some(path) = ctx.display_path(p, fr + k, use_proxy) {
                    let _ = image_for(&path, || {});
                }
            }
            if fr > 0 {
                if let Some(path) = ctx.display_path(p, fr - 1, use_proxy) {
                    let _ = image_for(&path, || {});
                }
            }
            let mut inserts = vec![];
            if ov.inserts {
                for l in p.layers.iter().filter(|l| l.visible) {
                    if let Some(path) = l.insert.as_ref().and_then(|i| i.frame_for(fr)) {
                        if let Some(img) = image_for(path, bump) {
                            inserts.push((l.id, img));
                        }
                        if let Some(next) = l.insert.as_ref().and_then(|i| i.frame_for(fr + 1)) {
                            let _ = image_for(next, || {});
                        }
                    }
                }
            }
            let scene = Scene {
                project: p,
                frame: fr,
                selected: sel,
                sel_point,
                tool,
                draft: &draft_pts,
                hover,
                view,
                overlays: ov,
                inserts,
            };
            draw_viewport(&canvas, &scene, img);
        });
    });
    Effect::new(move |_| {
        ctx.redraw.track();
        let Some(canvas) = ctx.timeline.get() else {
            return;
        };
        let (fr, sel) = (ctx.frame.get(), ctx.selected.get());
        let (ip, op) = (ctx.in_point.get(), ctx.out_point.get());
        ctx.project
            .with(|p| draw_timeline(&canvas, p, fr, sel, ip, op));
    });
    let _resize = window_event_listener(ev::resize, move |_| ctx.bump());
    let _up = window_event_listener(ev::mouseup, move |_| {
        if ctx.scrubbing.get_untracked() {
            ctx.scrubbing.set(false);
        }
    });

    // ---- keyboard ----
    let _keys = window_event_listener(ev::keydown, move |e: web_sys::KeyboardEvent| {
        let in_field = e
            .target()
            .and_then(|t| t.dyn_into::<web_sys::Element>().ok())
            .map(|el| matches!(el.tag_name().as_str(), "INPUT" | "SELECT" | "TEXTAREA"))
            .unwrap_or(false);
        if in_field {
            return;
        }
        let key = e.key().to_lowercase();
        let cmd = e.ctrl_key() || e.meta_key();
        if cmd {
            let handled = match key.as_str() {
                "z" if e.shift_key() => {
                    ctx.redo();
                    true
                }
                "z" => {
                    ctx.undo();
                    true
                }
                "y" => {
                    ctx.redo();
                    true
                }
                "s" => {
                    ctx.save(e.shift_key());
                    true
                }
                "o" => {
                    ctx.guard(Pending::OpenProject(None));
                    true
                }
                "n" => {
                    ctx.guard(Pending::New);
                    true
                }
                "d" => {
                    ctx.duplicate_selected();
                    true
                }
                _ => false,
            };
            if handled {
                e.prevent_default();
            }
            return;
        }
        match key.as_str() {
            "arrowright" if e.shift_key() => ctx.start_track(1, Some(1)),
            "arrowleft" if e.shift_key() => ctx.start_track(-1, Some(1)),
            "arrowright" => ctx.step_frame(1),
            "arrowleft" => ctx.step_frame(-1),
            "home" => ctx.frame.set(0),
            "end" => ctx.frame.set(ctx.frame_count_untracked().saturating_sub(1)),
            " " => {
                e.prevent_default();
                ctx.toggle_play();
            }
            "escape" => {
                if ctx.tracking.get_untracked() {
                    ctx.stop_track();
                }
                ctx.draft.set(vec![]);
                ctx.sel_point.set(None);
                ctx.recent_open.set(false);
            }
            "enter" if !ctx.draft.get_untracked().is_empty() => ctx.finish_draft(),
            "delete" | "backspace" if ctx.tool.get_untracked() == Tool::EditShape => {
                ctx.delete_selected_point()
            }
            "d" => ctx.tool.set(Tool::Draw),
            "a" => ctx.tool.set(Tool::Adjust),
            "l" => ctx.tool.set(Tool::LensLine),
            "i" => ctx.set_in_out(true),
            "o" => ctx.set_in_out(false),
            "x" => {
                ctx.in_point.set(None);
                ctx.out_point.set(None);
            }
            "n" => ctx.next_problem(),
            "e" => ctx.tool.set(Tool::EditShape),
            "s" => ctx.tool.set(Tool::Surface),
            "h" => ctx.tool.set(Tool::Pan),
            "f" => ctx.fit_view(),
            _ => {}
        }
    });

    view! {
        <div class="app">
            <TopBar/>
            <LeftPanel/>
            <Viewport/>
            <RightPanel/>
            <Bottom/>
            <Modals/>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Top bar

#[component]
fn TopBar() -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let tool_btn = move |t: Tool, label: &'static str, key: &'static str| {
        view! {
            <button class="tool" class:active=move || ctx.tool.get() == t title=format!("{label} ({key})")
                on:click=move |_| { ctx.tool.set(t); if !matches!(t, Tool::Draw | Tool::LensLine) { ctx.draft.set(vec![]); ctx.hover.set(None); } }>
                {label}<kbd>{key}</kbd>
            </button>
        }
    };
    let title = move || {
        let name = ctx
            .path
            .get()
            .map(|p| file_name(&p))
            .unwrap_or_else(|| "Untitled".into());
        format!("{name}{}", if ctx.dirty.get() { " •" } else { "" })
    };

    view! {
        <header class="topbar">
            <div class="brand"><span class="logo">"◆"</span>"Trackwright"<span class="doc" title=move || ctx.path.get().unwrap_or_default()>{title}</span></div>
            <div class="group">
                <button title="New project (Ctrl+N)" on:click=move |_| ctx.guard(Pending::New)>"New"</button>
                <button on:click=move |_| ctx.open_footage("file")>"Open Footage…"</button>
                <button on:click=move |_| ctx.open_footage("folder")>"Sequence…"</button>
                <button title="Open project (Ctrl+O)" on:click=move |_| ctx.guard(Pending::OpenProject(None))>"Open Project…"</button>
                <div class="menu-wrap">
                    <button disabled=move || ctx.recent.get().is_empty() on:click=move |_| ctx.recent_open.update(|v| *v = !*v)>"Recent ▾"</button>
                    <Show when=move || ctx.recent_open.get()>
                        <div class="menu-backdrop" on:click=move |_| ctx.recent_open.set(false)></div>
                        <ul class="menu">
                            {move || ctx.recent.get().into_iter().map(|p| {
                                let name = file_name(&p);
                                let full = p.clone();
                                view! {
                                    <li title=full.clone() on:click=move |_| { ctx.recent_open.set(false); ctx.guard(Pending::OpenProject(Some(full.clone()))); }>
                                        <span>{name}</span><small>{p.clone()}</small>
                                    </li>
                                }
                            }).collect_view()}
                        </ul>
                    </Show>
                </div>
                <button title="Save (Ctrl+S)" on:click=move |_| ctx.save(false)>"Save"</button>
                <button title="Save As (Ctrl+Shift+S)" on:click=move |_| ctx.save(true)>"Save As…"</button>
            </div>
            <div class="group">
                <button title="Undo (Ctrl+Z)" disabled=move || !ctx.can_undo.get() || ctx.tracking.get() on:click=move |_| ctx.undo()>"↶ Undo"</button>
                <button title="Redo (Ctrl+Shift+Z / Ctrl+Y)" disabled=move || !ctx.can_redo.get() || ctx.tracking.get() on:click=move |_| ctx.redo()>"↷ Redo"</button>
            </div>
            <div class="group tools">
                {tool_btn(Tool::Pan, "Pan", "H")}
                {tool_btn(Tool::Draw, "Draw", "D")}
                {tool_btn(Tool::EditShape, "Edit", "E")}
                {tool_btn(Tool::Surface, "Surface", "S")}
                {tool_btn(Tool::Adjust, "Adjust", "A")}
                {tool_btn(Tool::LensLine, "Lens", "L")}
            </div>
            <div class="spacer"></div>
            <button on:click=move |_| ctx.modal.set(Modal::About)>"About"</button>
        </header>
    }
}

// ---------------------------------------------------------------------------
// Left panel: layers, layer, insert, track settings

#[component]
fn LeftPanel() -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let sel_layer = move || {
        ctx.selected
            .get()
            .and_then(|id| ctx.project.with(|p| p.layer(id).cloned()))
    };

    let commit_rename = move |id: u32, name: String| {
        ctx.renaming.set(None);
        let name = name.trim().to_string();
        if name.is_empty() {
            return;
        }
        ctx.selected.set(Some(id));
        ctx.edit_layer(move |l| l.name = name);
    };

    view! {
        <aside class="panel left">
            <section>
                <h3>"Layers"
                    <span class="h-actions">
                        <button class="small" title="Duplicate selected layer (Ctrl+D)" disabled=move || ctx.selected.get().is_none() on:click=move |_| ctx.duplicate_selected()>"Duplicate"</button>
                        <button class="small" on:click=move |_| { ctx.tool.set(Tool::Draw); ctx.say("Click points around a flat surface; click the first point (or Enter) to close", false); }>"+ New"</button>
                    </span>
                </h3>
                <ul class="layers">
                    {move || ctx.project.with(|p| p.layers.iter().map(|l| {
                        let id = l.id;
                        let name = l.name.clone();
                        let color = l.color.clone();
                        let vis = l.visible;
                        let tracked = l.tracks.len();
                        let parent = l.link.and_then(|pid| p.layer(pid).map(|pl| pl.name.clone()));
                        let has_insert = l.insert.is_some();
                        let editing = ctx.renaming.get() == Some(id);
                        view! {
                            <li class:selected=move || ctx.selected.get() == Some(id) on:click=move |_| { if ctx.selected.get_untracked() != Some(id) { ctx.sel_point.set(None); } ctx.selected.set(Some(id)) }>
                                <span class="swatch" style=format!("background:{color}")></span>
                                {if editing {
                                    let start = name.clone();
                                    view! {
                                        <input id="rename-input" class="rename" type="text" prop:value=start
                                            on:click=|e| e.stop_propagation()
                                            on:keydown=move |e: web_sys::KeyboardEvent| {
                                                e.stop_propagation();
                                                if e.key() == "Enter" { commit_rename(id, event_target_value(&e)); }
                                                if e.key() == "Escape" { ctx.renaming.set(None); }
                                            }
                                            on:blur=move |e| { if ctx.renaming.get_untracked() == Some(id) { commit_rename(id, event_target_value(&e)); } } />
                                    }.into_any()
                                } else {
                                    view! {
                                        <span class="lname" title="Double-click to rename" on:dblclick=move |e| {
                                            e.stop_propagation();
                                            ctx.renaming.set(Some(id));
                                            request_animation_frame(|| {
                                                if let Some(el) = document().get_element_by_id("rename-input") {
                                                    if let Ok(input) = el.dyn_into::<web_sys::HtmlInputElement>() {
                                                        let _ = input.focus();
                                                        input.select();
                                                    }
                                                }
                                            });
                                        }>
                                            {name.clone()}
                                            {parent.map(|pn| view! { <small class="link">{format!("↳ {pn}")}</small> })}
                                        </span>
                                    }.into_any()
                                }}
                                {has_insert.then(|| view! { <span class="badge" title="Has an insert">"▣"</span> })}
                                <span class="muted">{format!("{tracked}f")}</span>
                                <button class="icon" title="Show / hide" on:click=move |ev| {
                                    ev.stop_propagation();
                                    ctx.selected.set(Some(id));
                                    ctx.edit_layer(|l| l.visible = !l.visible);
                                }>{if vis { "👁" } else { "–" }}</button>
                                <button class="icon" title="Delete layer" on:click=move |ev| {
                                    ev.stop_propagation();
                                    ctx.call("delete_layer", json!({ "id": id }), Some("Layer deleted (Ctrl+Z to undo)"));
                                }>"✕"</button>
                            </li>
                        }
                    }).collect_view())}
                </ul>
            </section>

            {move || sel_layer().map(|l| view! {
                <LayerSection layer=l.clone()/>
                <KeysSection layer=l.clone()/>
                <ExclusionsSection layer=l.clone()/>
                <InsertSection layer=l.clone()/>
                <TrackSettingsSection layer=l/>
            })}
        </aside>
    }
}

#[component]
fn LayerSection(layer: Layer) -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let l = layer;
    let id = l.id;
    let name = l.name.clone();
    let ref_label = ctx.project.with_untracked(|p| {
        p.footage
            .as_ref()
            .map_or(0, |f| f.frame_number(l.ref_frame))
    });
    let others: Vec<(u32, String)> = ctx.project.with_untracked(|p| {
        p.layers
            .iter()
            .filter(|o| o.id != id)
            .map(|o| (o.id, o.name.clone()))
            .collect()
    });
    let link = l.link;
    let linked = link.is_some();
    view! {
        <section>
            <h3>"Layer"</h3>
            <label class="field">"Name"
                <input type="text" prop:value=name on:change=move |ev| { let v = event_target_value(&ev); ctx.edit_layer(move |l| l.name = v); } />
            </label>
            <label class="field">"Motion source"
                <select on:change=move |ev| {
                    let v = event_target_value(&ev);
                    let parent: Option<u32> = v.parse().ok();
                    ctx.call("set_link", json!({ "id": id, "parent": parent }), Some(if parent.is_some() { "Layer now follows the chosen layer" } else { "Layer tracks on its own again" }));
                }>
                    <option value="" selected=!linked>"Own track"</option>
                    {others.into_iter().map(|(oid, oname)| view! {
                        <option value=oid.to_string() selected=link == Some(oid)>{format!("Follow {oname}")}</option>
                    }).collect_view()}
                </select>
            </label>
            <p class="muted">{format!("Reference frame {ref_label}")}</p>
            <div class="row">
                <button on:click=move |_| ctx.call("set_reference_frame", json!({ "id": id, "frame": ctx.frame.get_untracked() }), Some("Reference frame set"))>"Set Reference Here"</button>
            </div>
            <div class="row">
                <button disabled=linked on:click=move |_| ctx.call("clear_track", json!({ "id": id, "frame": ctx.frame.get_untracked(), "forward": false }), Some("Cleared track before this frame"))>"Clear ◀"</button>
                <button disabled=linked on:click=move |_| ctx.call("clear_track", json!({ "id": id, "frame": null, "forward": true }), Some("Track cleared"))>"Clear All"</button>
                <button disabled=linked on:click=move |_| ctx.call("clear_track", json!({ "id": id, "frame": ctx.frame.get_untracked(), "forward": true }), Some("Cleared track after this frame"))>"Clear ▶"</button>
            </div>
            <label class="toggle">
                <input type="checkbox" prop:checked=l.shape.smooth on:change=move |_| ctx.edit_layer(|l| l.shape.smooth = !l.shape.smooth) />
                "Smooth (spline) outline"
            </label>
            <label class="field">"Mask feather (px)"
                <input type="number" min="0" max="200" step="1" prop:value=format!("{:.0}", l.feather) on:change=move |ev| {
                    if let Ok(v) = event_target_value(&ev).parse::<f64>() { ctx.edit_layer(move |l| l.feather = v.clamp(0.0, 200.0)); }
                } />
            </label>
            <button class="wide" on:click=move |_| ctx.edit_layer(|l| l.surface = trackwright_core::shape::default_surface(&l.shape))>"Reset Surface to Shape Bounds"</button>
            <p class="hint">"Edit Shape (E): drag points, click an edge to add a point, drag inside to move the shape, Delete removes the selected point. Surface (S): drag corners or drag inside to move it."</p>
        </section>
    }
}

/// Camera solve: auto point tracking, solve, orientation and export.
#[component]
fn CameraSection() -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let settings = move || ctx.project.with(|p| p.camera.settings.clone());
    let edit = move |f: Box<dyn FnOnce(&mut CameraSettings)>| {
        let mut s = ctx.project.with_untracked(|p| p.camera.settings.clone());
        f(&mut s);
        ctx.call("set_camera_settings", json!({ "settings": s }), None);
    };
    let run = move |cmd: &'static str| {
        if ctx.tracking.get_untracked() || ctx.camera_busy.get_untracked() {
            return;
        }
        ctx.camera_busy.set(true);
        ctx.camera_progress.set((String::new(), 0.0));
        let (from, to) = (ctx.in_point.get_untracked(), ctx.out_point.get_untracked());
        spawn_local(async move {
            if let Err(e) = tauri::call::<()>(cmd, json!({ "from": from, "to": to })).await {
                ctx.camera_busy.set(false);
                ctx.say(e, true);
            }
        });
    };
    let export = move |destination: &'static str| {
        let fmt = ctx.camera_format.get_untracked();
        spawn_local(async move {
            match tauri::call::<Option<String>>(
                "export_camera",
                json!({ "format": fmt, "destination": destination }),
            )
            .await
            {
                Ok(Some(p)) => ctx.say(format!("Camera exported to {p}"), false),
                Ok(None) => {}
                Err(e) => ctx.say(e, true),
            }
        });
    };
    let summary = move || {
        ctx.project.with(|p| {
            let n = p.camera.tracks.len();
            match &p.camera.solve {
                None if n == 0 => "No point tracks yet.".to_string(),
                None => format!("{n} point tracks. Not solved yet."),
                Some(s) => format!(
                    "{}: {} frames · focal {:.1} mm · error {:.2} px · {} points{}{}",
                    if s.tripod { "Tripod" } else { "Moving camera" },
                    s.poses.len(),
                    s.focal_mm(p.camera.settings.sensor_mm),
                    s.rms_px,
                    s.points.len(),
                    if s.lens.is_identity() {
                        String::new()
                    } else {
                        format!(" · k1 {:.4}", s.lens.k1)
                    },
                    if s.world != trackwright_core::sfm::World::default() {
                        " · oriented"
                    } else {
                        ""
                    }
                ),
            }
        })
    };
    let toggle = move |sig: RwSignal<bool>, label: &'static str| {
        view! { <label class="toggle"><input type="checkbox" prop:checked=move || sig.get() on:change=move |_| sig.update(|v| *v = !*v) />{label}</label> }
    };
    view! {
        <section>
            <h3>"Camera Solve"</h3>
            <p class="hint">"Uses the In/Out range, or the whole clip. Tick moving objects and people under Ignore."</p>
            <div class="row">
                <label class="field">"Focal length"
                    <select on:change=move |ev| {
                        let auto = event_target_value(&ev) == "auto";
                        edit(Box::new(move |s| s.focal = if auto { FocalMode::Auto } else { FocalMode::Known { mm: match s.focal { FocalMode::Known { mm } => mm, _ => 35.0 } } }));
                    }>
                        <option value="auto" selected=move || matches!(settings().focal, FocalMode::Auto)>"Solve it"</option>
                        <option value="known" selected=move || matches!(settings().focal, FocalMode::Known { .. })>"I know it"</option>
                    </select>
                </label>
                <Show when=move || matches!(settings().focal, FocalMode::Known { .. })>
                    <label class="field">"mm"
                        <input type="number" step="0.1" prop:value=move || match settings().focal { FocalMode::Known { mm } => format!("{mm}"), _ => String::new() }
                            on:change=move |ev| { if let Ok(v) = event_target_value(&ev).parse::<f64>() { edit(Box::new(move |s| s.focal = FocalMode::Known { mm: v.clamp(1.0, 2000.0) })) } } />
                    </label>
                </Show>
                <label class="field">"Sensor width mm"
                    <input type="number" step="0.1" prop:value=move || format!("{}", settings().sensor_mm)
                        on:change=move |ev| { if let Ok(v) = event_target_value(&ev).parse::<f64>() { edit(Box::new(move |s| s.sensor_mm = v.clamp(1.0, 100.0))) } } />
                </label>
            </div>
            <div class="row">
                <label class="field">"Camera"
                    <select on:change=move |ev| {
                        let m = match event_target_value(&ev).as_str() { "free" => SolveMode::Free, "tripod" => SolveMode::Tripod, _ => SolveMode::Auto };
                        edit(Box::new(move |s| s.mode = m));
                    }>
                        <option value="auto" selected=move || settings().mode == SolveMode::Auto>"Detect"</option>
                        <option value="free" selected=move || settings().mode == SolveMode::Free>"Moving"</option>
                        <option value="tripod" selected=move || settings().mode == SolveMode::Tripod>"Tripod (pan/tilt)"</option>
                    </select>
                </label>
                <label class="field">"Points"
                    <input type="number" step="50" prop:value=move || settings().max_points.to_string()
                        on:change=move |ev| { if let Ok(v) = event_target_value(&ev).parse::<usize>() { edit(Box::new(move |s| s.max_points = v.clamp(50, 3000))) } } />
                </label>
            </div>
            <label class="toggle">
                <input type="checkbox" prop:checked=move || settings().solve_distortion on:change=move |_| edit(Box::new(|s| s.solve_distortion = !s.solve_distortion)) />
                "Solve lens distortion"
            </label>
            {move || {
                let layers: Vec<(u32, String)> = ctx.project.with(|p| p.layers.iter().map(|l| (l.id, l.name.clone())).collect());
                (!layers.is_empty()).then(|| view! {
                    <p class="hint">"Ignore:"</p>
                    {layers.into_iter().map(|(id, name)| view! {
                        <label class="toggle">
                            <input type="checkbox" prop:checked=move || settings().exclude_layers.contains(&id) on:change=move |_| edit(Box::new(move |s| {
                                if let Some(i) = s.exclude_layers.iter().position(|e| *e == id) { s.exclude_layers.remove(i); } else { s.exclude_layers.push(id); }
                            })) />
                            {name}
                        </label>
                    }).collect_view()}
                })
            }}
            <Show when=move || !ctx.camera_busy.get() fallback=move || view! {
                <div class="progress"><div class="bar" style=move || format!("width:{:.0}%", ctx.camera_progress.get().1 * 100.0)></div></div>
                <button class="wide stop" on:click=move |_| ctx.stop_track()>"Stop"</button>
            }>
                <div class="export-grid">
                    <button on:click=move |_| run("camera_track_points")>"1 · Track Points"</button>
                    <button class="primary-ish" disabled=move || ctx.project.with(|p| p.camera.tracks.is_empty()) on:click=move |_| run("camera_solve")>"2 · Solve Camera"</button>
                </div>
            </Show>
            <p class="muted">{summary}</p>
            <Show when=move || ctx.project.with(|p| p.camera.solve.is_some())>
                <div class="row">
                    {toggle(ctx.show_tracks, "Tracks")}
                    {toggle(ctx.show_cloud, "3D points")}
                    {toggle(ctx.show_ground, "Ground")}
                </div>
                <h3 class="sub">"Orient scene"</h3>
                <div class="row">
                    <label class="field">"Ground layer"
                        <select on:change=move |ev| ctx.orient_layer.set(event_target_value(&ev).parse().ok())>
                            <option value="">"Choose…"</option>
                            {move || ctx.project.with(|p| p.layers.iter().map(|l| {
                                let id = l.id;
                                view! { <option value=id.to_string() selected=move || ctx.orient_layer.get() == Some(id)>{l.name.clone()}</option> }
                            }).collect_view())}
                        </select>
                    </label>
                    <label class="field">"Real width"
                        <input type="number" step="0.01" placeholder="m" prop:value=move || { let w = ctx.orient_width.get(); if w > 0.0 { format!("{w}") } else { String::new() } }
                            on:change=move |ev| ctx.orient_width.set(event_target_value(&ev).parse().unwrap_or(0.0)) />
                    </label>
                </div>
                <div class="row">
                    <button on:click=move |_| {
                        let Some(id) = ctx.orient_layer.get_untracked() else { ctx.say("Choose the layer that lies on the ground (floor, table, road...)", true); return; };
                        let w = ctx.orient_width.get_untracked();
                        ctx.call("camera_orient", json!({ "layerId": id, "frame": ctx.frame.get_untracked(), "width": if w > 0.0 { Some(w) } else { None } }), Some("Scene oriented: the layer's surface is now the ground"));
                    }>"Set Ground"</button>
                    <button on:click=move |_| ctx.call("camera_reset_orientation", json!({}), Some("Orientation reset"))>"Reset"</button>
                </div>
                <h3 class="sub">"Export camera"</h3>
                <select on:change=move |ev| {
                    if let Ok(f) = serde_json::from_value::<CameraFormat>(json!(event_target_value(&ev))) { ctx.camera_format.set(f); }
                }>
                    {CameraFormat::ALL.iter().map(|f| {
                        let fmt = *f;
                        view! { <option value=enum_id(fmt) selected=move || ctx.camera_format.get() == fmt>{fmt.label()}</option> }
                    }).collect_view()}
                </select>
                <div class="export-grid" style="margin-top:6px">
                    <button class="primary" on:click=move |_| export("desktop")>"Save to Desktop"</button>
                    <button on:click=move |_| export("forma3d")>"Send to Forma3D"</button>
                    <button on:click=move |_| export("tidefire")>"Send to Tidefire"</button>
                    <button on:click=move |_| export("choose")>"Save As…"</button>
                </div>
            </Show>
            <Show when=move || ctx.project.with(|p| !p.camera.tracks.is_empty())>
                <button class="wide" on:click=move |_| ctx.call("clear_camera", json!({}), Some("Camera tracks cleared"))>"Clear Camera Tracks"</button>
            </Show>
        </section>
    }
}

/// Lens distortion: values, line calibration, and an overlay grid.
#[component]
fn LensSection() -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let lens = move || ctx.project.with(|p| p.lens);
    let set = move |k1: f64, k2: f64| ctx.call("set_lens", json!({ "k1": k1, "k2": k2 }), None);
    view! {
        <section>
            <h3>"Lens Distortion"</h3>
            <div class="row">
                <label class="field">"k1"
                    <input type="number" step="0.001" prop:value=move || format!("{:.5}", lens().k1)
                        on:change=move |ev| { if let Ok(v) = event_target_value(&ev).parse::<f64>() { set(v, lens().k2) } } />
                </label>
                <label class="field">"k2"
                    <input type="number" step="0.001" prop:value=move || format!("{:.5}", lens().k2)
                        on:change=move |ev| { if let Ok(v) = event_target_value(&ev).parse::<f64>() { set(lens().k1, v) } } />
                </label>
            </div>
            <p class="muted">{move || format!("{} line(s) drawn", ctx.project.with(|p| p.lens_lines.len()))}</p>
            <div class="row">
                <button on:click=move |_| { ctx.tool.set(Tool::LensLine); ctx.say("Click points along an edge that should be straight (3+), Enter to finish. Long lines near the frame edges work best", false); }>"Draw Lines (L)"</button>
                <button disabled=move || ctx.project.with(|p| p.lens_lines.is_empty()) on:click=move |_| ctx.call("set_lens_lines", json!({ "lines": [] }), Some("Lens lines cleared"))>"Clear"</button>
            </div>
            <label class="toggle">
                <input type="checkbox" prop:checked=move || ctx.lens_k2.get() on:change=move |_| ctx.lens_k2.update(|v| *v = !*v) />
                "Also solve k2 (strong fisheye-like bending)"
            </label>
            <button class="wide" on:click=move |_| {
                let k2 = ctx.lens_k2.get_untracked();
                spawn_local(async move {
                    match tauri::call::<LensResult>("calibrate_lens", json!({ "solveK2": k2 })).await {
                        Ok(r) => {
                            ctx.apply_view(r.view);
                            ctx.show_lens_grid.set(true);
                            ctx.say(format!("Lens calibrated: lines were {:.2} px off straight, now {:.2} px", r.rms_before, r.rms_after), false);
                        }
                        Err(e) => ctx.say(e, true),
                    }
                });
            }>"Calibrate from Lines"</button>
            <label class="toggle">
                <input type="checkbox" prop:checked=move || ctx.show_lens_grid.get() on:change=move |_| ctx.show_lens_grid.update(|v| *v = !*v) />
                "Show lens grid"
            </label>
            <p class="hint">"Render › Undistort footage writes straightened frames. The camera solve uses these values (or solves them)."</p>
        </section>
    }
}

/// Render a stabilized clip or the insert composited into the footage.
#[component]
fn RenderSection() -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let start = move |destination: &'static str| {
        let mode = ctx.render_mode.get_untracked();
        let id = ctx.selected.get_untracked();
        if id.is_none() && mode != "undistort" && !ctx.render_all_layers.get_untracked() {
            ctx.say("Select a layer to render", true);
            return;
        }
        let request = json!({
            "id": id,
            "bg_layer": ctx.render_bg.get_untracked(),
            "search": ctx.render_search.get_untracked(),
            "grow": ctx.render_grow.get_untracked(),
            "match_light": ctx.render_match.get_untracked(),
            "all_layers": ctx.render_all_layers.get_untracked(),
            "mode": ctx.render_mode.get_untracked(),
            "smooth_frames": ctx.smooth_frames.get_untracked(),
            "zoom": ctx.zoom_pct.get_untracked(),
            "from": ctx.in_point.get_untracked(),
            "to": ctx.out_point.get_untracked(),
            "destination": destination,
            "make_video": ctx.make_video.get_untracked(),
            "holdout": true,
        });
        ctx.rendering.set(true);
        ctx.render_progress.set((0, 0));
        ctx.say("Starting render…", false);
        spawn_local(async move {
            if let Err(e) = tauri::call::<()>("render_layer", json!({ "request": request })).await {
                ctx.rendering.set(false);
                ctx.say(e, true);
            }
        });
    };
    let mode_opt = move |id: &'static str, label: &'static str| {
        view! { <option value=id selected=move || ctx.render_mode.get() == id>{label}</option> }
    };
    view! {
        <section>
            <h3>"Render"</h3>
            <label class="field">"Output"
                <select on:change=move |ev| ctx.render_mode.set(event_target_value(&ev))>
                    {mode_opt("insert", "Insert composited into the footage")}
                    {mode_opt("stabilize_smooth", "Stabilize: smooth the motion")}
                    {mode_opt("stabilize_lock", "Stabilize: lock to the surface")}
                    {mode_opt("remove", "Remove object (fill from other frames)")}
                    {mode_opt("clean_plate", "Clean plate (single image)")}
                    {mode_opt("matte", "Matte (black & white mask)")}
                    {mode_opt("cutout", "Cut-out (footage with alpha)")}
                    {mode_opt("undistort", "Undistort footage (lens)")}
                </select>
            </label>
            <Show when=move || matches!(ctx.render_mode.get().as_str(), "remove" | "clean_plate")>
                <label class="field">"Background layer (tracks the surface behind the object)"
                    <select on:change=move |ev| ctx.render_bg.set(event_target_value(&ev).parse().ok())>
                        <option value="" selected=move || ctx.render_bg.get().is_none()>"Choose…"</option>
                        {move || ctx.project.with(|p| p.layers.iter().map(|l| {
                            let id = l.id;
                            view! { <option value=id.to_string() selected=move || ctx.render_bg.get() == Some(id)>{l.name.clone()}</option> }
                        }).collect_view())}
                    </select>
                </label>
                <div class="row">
                    <label class="field">"Search ± frames"
                        <input type="number" min="1" max="120" prop:value=move || ctx.render_search.get().to_string()
                            on:change=move |ev| { if let Ok(v) = event_target_value(&ev).parse::<usize>() { ctx.render_search.set(v.clamp(1, 120)) } } />
                    </label>
                    <label class="field">"Grow mask (px)"
                        <input type="number" min="0" max="200" prop:value=move || format!("{:.0}", ctx.render_grow.get())
                            on:change=move |ev| { if let Ok(v) = event_target_value(&ev).parse::<f64>() { ctx.render_grow.set(v.clamp(0.0, 200.0)) } } />
                    </label>
                </div>
                <Show when=move || ctx.render_mode.get() == "remove">
                    <label class="toggle">
                        <input type="checkbox" prop:checked=move || ctx.render_match.get() on:change=move |_| ctx.render_match.update(|v| *v = !*v) />
                        "Match lighting"
                    </label>
                </Show>
                <p class="hint">"The selected layer is the object to remove (track it, link it, or keyframe its shape). Areas never seen in other frames are filled smoothly."</p>
            </Show>
            <Show when=move || matches!(ctx.render_mode.get().as_str(), "matte" | "cutout")>
                <label class="toggle">
                    <input type="checkbox" prop:checked=move || ctx.render_all_layers.get() on:change=move |_| ctx.render_all_layers.update(|v| *v = !*v) />
                    "All visible layers"
                </label>
                <p class="hint">"Edge softness comes from each layer's Mask feather."</p>
            </Show>
            <Show when=move || ctx.render_mode.get().starts_with("stabilize")>
                <div class="row">
                    <Show when=move || ctx.render_mode.get() == "stabilize_smooth">
                        <label class="field">"Smoothing (frames)"
                            <input type="number" min="1" max="200" step="1" prop:value=move || ctx.smooth_frames.get().to_string()
                                on:change=move |ev| { if let Ok(v) = event_target_value(&ev).parse::<usize>() { ctx.smooth_frames.set(v.clamp(1, 200)) } } />
                        </label>
                    </Show>
                    <label class="field">"Zoom %"
                        <input type="number" min="50" max="300" step="1" prop:value=move || format!("{:.0}", ctx.zoom_pct.get())
                            on:change=move |ev| { if let Ok(v) = event_target_value(&ev).parse::<f64>() { ctx.zoom_pct.set(v.clamp(50.0, 300.0)) } } />
                    </label>
                </div>
            </Show>
            <label class="toggle">
                <input type="checkbox" prop:checked=move || ctx.make_video.get() on:change=move |_| ctx.make_video.update(|v| *v = !*v) />
                "Also make an MP4 (needs ffmpeg)"
            </label>
            <Show when=move || ctx.render_mode.get() == "insert">
                <p class="hint">"Areas covered by this layer's exclusions stay in front of the insert."</p>
            </Show>
            <p class="hint">{move || {
                let ranged = ctx.in_point.get().is_some() || ctx.out_point.get().is_some();
                match ctx.render_mode.get().as_str() {
                    "clean_plate" => "Writes one PNG: the background with the object taken out, built from every tracked frame.".to_string(),
                    "undistort" => format!("Writes the {} with the lens bending removed, as a PNG sequence.", if ranged { "In/Out range" } else { "whole clip" }),
                    "matte" => "Writes white-on-black PNG masks (a luma matte for Reelforge; Export Track › Reelforge mask gives the editable shape).".to_string(),
                    "cutout" => "Writes PNGs that are transparent outside the shape.".to_string(),
                    _ if ranged => "Renders the In/Out range (within the tracked frames) as a PNG sequence.".to_string(),
                    _ => "Renders the layer's whole tracked range as a PNG sequence.".to_string(),
                }
            }}</p>
            <Show when=move || !ctx.rendering.get() fallback=move || view! {
                <div class="progress"><div class="bar" style=move || {
                    let (d, t) = ctx.render_progress.get();
                    format!("width:{}%", if t == 0 { 0 } else { d * 100 / t })
                }></div></div>
                <button class="wide stop" on:click=move |_| spawn_local(async move { let _ = tauri::call::<()>("stop_render", json!({})).await; })>"Stop Render"</button>
            }>
                <div class="export-grid">
                    <button class="primary" on:click=move |_| start("desktop")>"Render to Desktop"</button>
                    <button on:click=move |_| start("choose")>"Render To…"</button>
                </div>
            </Show>
        </section>
    }
}

/// Shape keyframes and Adjust Track keys for the selected layer.
#[component]
fn KeysSection(layer: Layer) -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let n_shape = layer.shape_keys.len();
    let n_adjust = layer.adjust_keys.len();
    let linked = layer.link.is_some();
    let here = move || ctx.frame.get();
    let shape_key_here = {
        let keys: Vec<usize> = layer.shape_keys.keys().copied().collect();
        move || keys.contains(&here())
    };
    let adjust_key_here = {
        let keys: Vec<usize> = layer.adjust_keys.keys().copied().collect();
        move || keys.contains(&here())
    };
    let shape_key_here2 = shape_key_here.clone();
    let adjust_key_here2 = adjust_key_here.clone();
    view! {
        <section>
            <h3>"Keyframes"</h3>
            <p class="muted">{if n_shape == 0 { "Shape: fixed to the plane".to_string() } else { format!("Shape: animated, {n_shape} key(s)") }}</p>
            <div class="row">
                <button title="Add a shape key here; later edits on other frames add keys automatically" on:click=move |_| {
                    let f = ctx.frame.get_untracked();
                    ctx.edit_layer(move |l| { l.add_shape_key(f); });
                    ctx.tool.set(Tool::EditShape);
                    ctx.say("Shape key added. Edit the shape on other frames to animate it (E)", false);
                }>"◆ Shape Key"</button>
                <button disabled=move || !shape_key_here() on:click=move |_| { let f = ctx.frame.get_untracked(); ctx.edit_layer(move |l| { l.remove_shape_key(f); }); }>"Delete"</button>
                <button disabled=n_shape == 0 title="Stop animating the shape (keeps the shape on this frame)" on:click=move |_| { let f = ctx.frame.get_untracked(); ctx.edit_layer(move |l| l.clear_shape_keys(f)); }>"Clear"</button>
            </div>
            <p class="muted">{if n_adjust == 0 { "Adjust Track: no corrections".to_string() } else { format!("Adjust Track: {n_adjust} key(s)") }}</p>
            <div class="row">
                <button disabled=linked title="Lock the corners on this frame (add a key with the current correction)" on:click=move |_| {
                    let f = ctx.frame.get_untracked();
                    ctx.edit_layer(move |l| { l.add_adjust_key(f); });
                    ctx.tool.set(Tool::Adjust);
                }>"◆ Lock Here"</button>
                <button disabled=move || !adjust_key_here() on:click=move |_| { let f = ctx.frame.get_untracked(); ctx.edit_layer(move |l| { l.adjust_keys.remove(&f); }); }>"Delete"</button>
                <button disabled=n_adjust == 0 on:click=move |_| ctx.edit_layer(|l| l.adjust_keys.clear())>"Clear"</button>
            </div>
            <p class="hint">{move || if adjust_key_here2() || shape_key_here2() { "There is a key on this frame." } else { "" }}
                " Adjust (A): drag corners onto the right spot on a bad frame. Use ◆ Lock Here on good frames either side so the fix only covers the bad stretch."</p>
        </section>
    }
}

/// Pick layers whose areas this layer should ignore.
#[component]
fn ExclusionsSection(layer: Layer) -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let id = layer.id;
    let others: Vec<(u32, String, String, bool)> = ctx.project.with_untracked(|p| {
        p.layers
            .iter()
            .filter(|o| o.id != id)
            .map(|o| {
                (
                    o.id,
                    o.name.clone(),
                    o.color.clone(),
                    layer.exclusions.contains(&o.id),
                )
            })
            .collect()
    });
    let empty = others.is_empty();
    view! {
        <section>
            <h3>"Exclusions"</h3>
            {if empty {
                view! { <p class="hint">"Draw another layer over anything that moves in front of this surface (a hand, a person, a reflection), then tick it here so the tracker ignores that area. Track or keyframe that layer so it follows the occluder."</p> }.into_any()
            } else {
                view! {
                    <p class="hint">"Ignore the areas covered by:"</p>
                    {others.into_iter().map(|(oid, name, color, on)| view! {
                        <label class="toggle">
                            <input type="checkbox" prop:checked=on on:change=move |_| ctx.edit_layer(move |l| {
                                if let Some(pos) = l.exclusions.iter().position(|&e| e == oid) { l.exclusions.remove(pos); } else { l.exclusions.push(oid); }
                            }) />
                            <span class="swatch" style=format!("background:{color}")></span>{name}
                        </label>
                    }).collect_view()}
                }.into_any()
            }}
        </section>
    }
}

#[component]
fn InsertSection(layer: Layer) -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let body = match layer.insert.clone() {
        None => view! {
            <div class="row">
                <button on:click=move |_| ctx.choose_insert("file")>"Image / Clip…"</button>
                <button on:click=move |_| ctx.choose_insert("folder")>"Sequence…"</button>
            </div>
            <p class="hint">"Pin a picture or video onto the surface to see how well the track holds."</p>
        }
        .into_any(),
        Some(ins) => {
            let first = ctx.project.with_untracked(|p| p.footage.as_ref().map_or(0, |f| f.first_frame));
            let multi = ins.frames.len() > 1;
            let opacity = ins.opacity;
            let start_label = first + ins.start_frame as i64;
            let looped = ins.looped;
            view! {
                <p class="fname" title=ins.source.clone()>{file_name(&ins.source)}</p>
                <p class="muted">{format!("{} × {}{}", ins.width, ins.height, if multi { format!(" · {} frames", ins.frames.len()) } else { String::new() })}</p>
                <label class="field">{move || format!("Opacity {:.0}%", opacity * 100.0)}
                    <input type="range" min="0" max="1" step="0.05" prop:value=opacity.to_string() on:change=move |ev| {
                        if let Ok(v) = event_target_value(&ev).parse::<f64>() {
                            ctx.edit_layer(move |l| if let Some(i) = l.insert.as_mut() { i.opacity = v.clamp(0.0, 1.0) });
                        }
                    } />
                </label>
                {multi.then(|| view! {
                    <div class="row">
                        <label class="field">"Starts at frame"
                            <input type="number" step="1" prop:value=start_label.to_string() on:change=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse::<i64>() {
                                    let idx = (v - first).max(0) as usize;
                                    ctx.edit_layer(move |l| if let Some(i) = l.insert.as_mut() { i.start_frame = idx });
                                }
                            } />
                        </label>
                    </div>
                    <label class="toggle">
                        <input type="checkbox" prop:checked=looped on:change=move |_| ctx.edit_layer(|l| if let Some(i) = l.insert.as_mut() { i.looped = !i.looped }) />
                        "Loop clip"
                    </label>
                })}
                <div class="row">
                    <button on:click=move |_| ctx.choose_insert("file")>"Replace…"</button>
                    <button on:click=move |_| ctx.edit_layer(|l| l.insert = None)>"Remove"</button>
                </div>
            }
            .into_any()
        }
    };
    view! { <section><h3>"Insert Preview"</h3>{body}</section> }
}

#[component]
fn TrackSettingsSection(layer: Layer) -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let s = layer.settings.clone();
    let model = enum_id(s.model);
    let num = move |label: &'static str,
                    value: String,
                    step: &'static str,
                    apply: fn(&mut Layer, f64)| {
        view! {
            <label class="field">{label}
                <input type="number" step=step prop:value=value on:change=move |ev| {
                    if let Ok(v) = event_target_value(&ev).parse::<f64>() {
                        ctx.edit_layer(move |l| apply(l, v));
                    }
                } />
            </label>
        }
    };
    view! {
        <section class:disabled=layer.link.is_some()>
            <h3>"Track Settings"</h3>
            <label class="field">"Motion"
                <select on:change=move |ev| {
                    let v = event_target_value(&ev);
                    if let Ok(m) = serde_json::from_value::<MotionModel>(json!(v)) {
                        ctx.edit_layer(move |l| l.settings.model = m);
                    }
                }>
                    {MotionModel::ALL.iter().map(|m| {
                        let id = enum_id(*m);
                        let sel = id == model;
                        view! { <option value=id.clone() selected=sel>{m.label()}</option> }
                    }).collect_view()}
                </select>
            </label>
            {num("Max points", s.max_features.to_string(), "10", |l, v| l.settings.max_features = v.clamp(20.0, 3000.0) as usize)}
            {num("Min point spacing (px)", s.min_distance.to_string(), "1", |l, v| l.settings.min_distance = v.clamp(2.0, 50.0) as f32)}
            {num("Search window radius (px)", s.search_radius.to_string(), "1", |l, v| l.settings.search_radius = v.clamp(3.0, 31.0) as i32)}
            {num("Pyramid levels", s.pyramid_levels.to_string(), "1", |l, v| l.settings.pyramid_levels = v.clamp(1.0, 7.0) as usize)}
            {num("Inlier threshold (px)", s.ransac_threshold.to_string(), "0.1", |l, v| l.settings.ransac_threshold = v.clamp(0.2, 10.0))}
            {num("Min good points", s.min_inliers.to_string(), "1", |l, v| l.settings.min_inliers = v.clamp(4.0, 500.0) as usize)}
            <label class="toggle">
                <input type="checkbox" prop:checked=s.refine_against_reference on:change=move |_| ctx.edit_layer(|l| l.settings.refine_against_reference = !l.settings.refine_against_reference) />
                "Anti-drift (re-solve against reference)"
            </label>
            <label class="toggle">
                <input type="checkbox" prop:checked=s.illumination on:change=move |_| ctx.edit_layer(|l| l.settings.illumination = !l.settings.illumination) />
                "Lighting compensation (fades, flicker)"
            </label>
            <label class="toggle">
                <input type="checkbox" prop:checked=s.retry_hard_frames on:change=move |_| ctx.edit_layer(|l| l.settings.retry_hard_frames = !l.settings.retry_hard_frames) />
                "Retry hard frames (motion blur)"
            </label>
            <label class="field">"Speed"
                <select on:change=move |ev| {
                    let v = event_target_value(&ev);
                    let lvl: Option<usize> = v.parse().ok();
                    ctx.edit_layer(move |l| l.settings.detail_level = lvl);
                }>
                    <option value="auto" selected=s.detail_level.is_none()>"Auto (fast first pass on 2.5K+)"</option>
                    <option value="0" selected=s.detail_level == Some(0)>"Full resolution"</option>
                    <option value="1" selected=s.detail_level == Some(1)>"Fast: half-res first pass"</option>
                    <option value="2" selected=s.detail_level == Some(2)>"Fastest: quarter-res first pass"</option>
                </select>
            </label>
        </section>
    }
}

// ---------------------------------------------------------------------------
// Viewport

#[component]
fn Viewport() -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let mouse_pos = |e: &web_sys::MouseEvent| [e.offset_x() as f64, e.offset_y() as f64];

    let on_down = move |e: web_sys::MouseEvent| {
        let s = mouse_pos(&e);
        let v = ctx.view.get_untracked();
        let ip = v.to_image(s);
        let tool = ctx.tool.get_untracked();
        if e.button() == 2 && !ctx.draft.get_untracked().is_empty() {
            ctx.draft.update(|d| {
                d.pop();
            });
            return;
        }
        if e.button() == 1 || e.button() == 2 || tool == Tool::Pan || e.alt_key() {
            ctx.drag.set_value(Drag::Pan {
                start: s,
                pan: v.pan,
            });
            return;
        }
        if e.button() != 0 || ctx.project.with_untracked(|p| p.footage.is_none()) {
            return;
        }
        let fr = ctx.frame.get_untracked();
        let hit_r2 = 9.0f64.powi(2);
        match tool {
            Tool::LensLine => ctx.draft.update(|d| d.push(ip)),
            Tool::Draw => {
                let d = ctx.draft.get_untracked();
                if d.len() >= 3 && dist2(v.to_screen(d[0]), s) < hit_r2 * 1.5 {
                    ctx.finish_draft();
                } else {
                    ctx.draft.update(|d| d.push(ip));
                }
            }
            Tool::EditShape | Tool::Surface | Tool::Adjust => {
                if ctx.tracking.get_untracked() {
                    return;
                }
                let Some(id) = ctx.selected.get_untracked() else {
                    ctx.drag.set_value(Drag::Pan {
                        start: s,
                        pan: v.pan,
                    });
                    return;
                };
                let drag = if tool == Tool::EditShape {
                    let shape = ctx
                        .project
                        .with_untracked(|p| p.layer(id).and_then(|l| l.shape_at(fr)));
                    match shape {
                        None => None,
                        Some(sh) => {
                            if let Some(i) = sh
                                .points
                                .iter()
                                .position(|&q| dist2(v.to_screen(q), s) < hit_r2)
                            {
                                ctx.sel_point.set(Some(i));
                                Some(Drag::ShapePoint(i))
                            } else if let Some((seg, q, d)) =
                                sh.nearest_edge(ip).filter(|(_, _, d)| d * v.zoom < 7.0)
                            {
                                // Click on an edge: add a point there and start dragging it.
                                let _ = d;
                                let at = seg + 1;
                                ctx.project.update(|p| {
                                    if let Some(l) = p.layer_mut(id) {
                                        l.insert_shape_point(fr, seg, q);
                                    }
                                });
                                ctx.sel_point.set(Some(at));
                                Some(Drag::ShapePoint(at))
                            } else if point_in_polygon(&sh.flatten(10), ip) {
                                ctx.sel_point.set(None);
                                Some(Drag::MoveShape {
                                    start: ip,
                                    orig: sh,
                                })
                            } else {
                                ctx.sel_point.set(None);
                                None
                            }
                        }
                    }
                } else {
                    let corners = ctx
                        .project
                        .with_untracked(|p| p.layer(id).and_then(|l| l.surface_at(fr)));
                    let adjust = tool == Tool::Adjust;
                    corners.and_then(|c| {
                        if let Some(i) = c
                            .iter()
                            .position(|&q| dist2(v.to_screen(q), s) < hit_r2 * 1.4)
                        {
                            Some(if adjust {
                                Drag::AdjustCorner(i)
                            } else {
                                Drag::SurfaceCorner(i)
                            })
                        } else if point_in_polygon(&c, ip) {
                            Some(if adjust {
                                Drag::MoveAdjust { start: ip, orig: c }
                            } else {
                                Drag::MoveSurface { start: ip, orig: c }
                            })
                        } else {
                            None
                        }
                    })
                };
                ctx.drag.set_value(drag.unwrap_or(Drag::Pan {
                    start: s,
                    pan: v.pan,
                }));
            }
            Tool::Pan => {}
        }
    };

    let on_move = move |e: web_sys::MouseEvent| {
        let s = mouse_pos(&e);
        let v = ctx.view.get_untracked();
        let ip = v.to_image(s);
        if matches!(ctx.tool.get_untracked(), Tool::Draw | Tool::LensLine) {
            ctx.hover.set(Some(ip));
        }
        let fr = ctx.frame.get_untracked();
        let Some(id) = ctx.selected.get_untracked() else {
            if let Drag::Pan { start, pan } = ctx.drag.get_value() {
                ctx.view
                    .update(|v| v.pan = [pan[0] + s[0] - start[0], pan[1] + s[1] - start[1]]);
            }
            return;
        };
        match ctx.drag.get_value() {
            Drag::None => {}
            Drag::Pan { start, pan } => ctx
                .view
                .update(|v| v.pan = [pan[0] + s[0] - start[0], pan[1] + s[1] - start[1]]),
            Drag::ShapePoint(i) => ctx.project.update(|p| {
                if let Some(l) = p.layer_mut(id) {
                    if let Some(mut sh) = l.shape_at(fr) {
                        if i < sh.points.len() {
                            sh.points[i] = ip;
                            l.set_shape_at(fr, sh);
                        }
                    }
                }
            }),
            Drag::SurfaceCorner(i) => ctx.project.update(|p| {
                if let Some(l) = p.layer_mut(id) {
                    l.set_surface_corner(fr, i, ip);
                }
            }),
            Drag::MoveShape { start, orig } => ctx.project.update(|p| {
                if let Some(l) = p.layer_mut(id) {
                    l.set_shape_at(fr, orig.translated([ip[0] - start[0], ip[1] - start[1]]));
                }
            }),
            Drag::MoveSurface { start, orig } => ctx.project.update(|p| {
                if let Some(l) = p.layer_mut(id) {
                    let d = [ip[0] - start[0], ip[1] - start[1]];
                    for (i, c) in orig.iter().enumerate() {
                        l.set_surface_corner(fr, i, [c[0] + d[0], c[1] + d[1]]);
                    }
                }
            }),
            Drag::AdjustCorner(i) => ctx.project.update(|p| {
                if let Some(l) = p.layer_mut(id) {
                    l.set_adjust_corner(fr, i, ip);
                }
            }),
            Drag::MoveAdjust { start, orig } => ctx.project.update(|p| {
                if let Some(l) = p.layer_mut(id) {
                    let d = [ip[0] - start[0], ip[1] - start[1]];
                    for (i, c) in orig.iter().enumerate() {
                        l.set_adjust_corner(fr, i, [c[0] + d[0], c[1] + d[1]]);
                    }
                }
            }),
        }
    };

    let on_up = move |_e: web_sys::MouseEvent| {
        let d = ctx.drag.get_value();
        ctx.drag.set_value(Drag::None);
        if matches!(
            d,
            Drag::ShapePoint(_)
                | Drag::SurfaceCorner(_)
                | Drag::MoveShape { .. }
                | Drag::MoveSurface { .. }
                | Drag::AdjustCorner(_)
                | Drag::MoveAdjust { .. }
        ) {
            // Push the edited layer as one undo step.
            ctx.edit_layer(|_| {});
        }
    };

    let on_wheel = move |e: web_sys::WheelEvent| {
        e.prevent_default();
        let s = [e.offset_x() as f64, e.offset_y() as f64];
        ctx.view.update(|v| {
            let before = v.to_image(s);
            let factor = (-e.delta_y() * 0.0015).exp();
            v.zoom = (v.zoom * factor).clamp(0.05, 32.0);
            let after = v.to_screen(before);
            v.pan[0] += s[0] - after[0];
            v.pan[1] += s[1] - after[1];
        });
    };

    let on_dbl = move |_e: web_sys::MouseEvent| {
        if matches!(ctx.tool.get_untracked(), Tool::Draw | Tool::LensLine)
            && ctx.draft.with_untracked(|d| d.len() >= 3)
        {
            ctx.draft.update(|d| {
                d.pop(); // the double-click's second point
            });
            ctx.finish_draft();
        }
    };

    let toggle = move |sig: RwSignal<bool>, label: &'static str| {
        view! {
            <label class="toggle"><input type="checkbox" prop:checked=move || sig.get() on:change=move |_| sig.update(|v| *v = !*v) />{label}</label>
        }
    };
    view! {
        <main class="viewport">
            <div class="view-toolbar">
                {toggle(ctx.show_shapes, "Shapes")}
                {toggle(ctx.show_surface, "Surface")}
                {toggle(ctx.show_inserts, "Insert")}
                {toggle(ctx.show_grid, "Grid")}
                {toggle(ctx.show_points, "Points")}
                {move || ctx.project.with(|p| !p.camera.tracks.is_empty()).then(|| view! {
                    {toggle(ctx.show_tracks, "Cam tracks")}
                    {toggle(ctx.show_cloud, "3D pts")}
                    {toggle(ctx.show_ground, "Ground")}
                })}
                {toggle(ctx.show_lens_grid, "Lens")}
                <button class="small" on:click=move |_| ctx.fit_view() title="Fit (F)">"Fit"</button>
            </div>
            <canvas node_ref=ctx.viewport class="view"
                class:draw=move || matches!(ctx.tool.get(), Tool::Draw | Tool::LensLine)
                class:edit=move || matches!(ctx.tool.get(), Tool::EditShape | Tool::Surface | Tool::Adjust)
                on:mousedown=on_down on:mousemove=on_move on:mouseup=on_up
                on:mouseleave=move |e| { ctx.hover.set(None); on_up(e) }
                on:wheel=on_wheel on:dblclick=on_dbl
                on:contextmenu=|e| e.prevent_default()></canvas>
        </main>
    }
}

// ---------------------------------------------------------------------------
// Right panel

#[component]
fn RightPanel() -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let res_opt = move |r: ViewerRes, label: &'static str| {
        view! { <button class="seg" class:active=move || ctx.viewer_res.get() == r on:click=move |_| ctx.viewer_res.set(r)>{label}</button> }
    };
    view! {
        <aside class="panel right">
            <section>
                <h3>"Footage"</h3>
                {move || ctx.project.with(|p| match &p.footage {
                    None => view! { <p class="muted">"No footage loaded. Drag a video, image or folder onto the window."</p> }.into_any(),
                    Some(f) => {
                        let (fps, first) = (f.fps, f.first_frame);
                        view! {
                            <p class="fname" title=f.source.clone()>{file_name(&f.source)}</p>
                            <p class="muted">{format!("{} × {} · {} frames", f.width, f.height, f.frames.len())}</p>
                            <div class="row">
                                <label class="field">"FPS"
                                    <input type="number" step="0.001" prop:value=format!("{fps:.3}") on:change=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<f64>() {
                                            let first = ctx.project.with_untracked(|p| p.footage.as_ref().map_or(0, |f| f.first_frame));
                                            ctx.call("set_footage_timing", json!({ "fps": v, "firstFrame": first }), None);
                                        }
                                    } />
                                </label>
                                <label class="field">"First frame"
                                    <input type="number" step="1" prop:value=first.to_string() on:change=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<i64>() {
                                            let fps = ctx.project.with_untracked(|p| p.footage.as_ref().map_or(24.0, |f| f.fps));
                                            ctx.call("set_footage_timing", json!({ "fps": fps, "firstFrame": v }), None);
                                        }
                                    } />
                                </label>
                            </div>
                        }.into_any()
                    }
                })}
                <label class="field">"Viewer resolution"</label>
                <div class="segmented">
                    {res_opt(ViewerRes::Auto, "Auto")}
                    {res_opt(ViewerRes::Proxy, "Proxy")}
                    {res_opt(ViewerRes::Full, "Full")}
                </div>
                <p class="hint">{move || {
                    let p = ctx.proxy.get();
                    if ctx.project.with(|pr| pr.footage.is_none()) {
                        String::new()
                    } else if p.prefix.is_none() {
                        "Footage is small enough to play at full resolution.".to_string()
                    } else if p.ready < p.total {
                        format!("Building playback proxies… {} / {}", p.ready, p.total)
                    } else {
                        "Proxies ready. Auto uses them while playing, scrubbing and tracking.".to_string()
                    }
                }}</p>
            </section>

            <section>
                <h3>"Export Track"</h3>
                <label class="field">"Format"
                    <select on:change=move |ev| {
                        let v = event_target_value(&ev);
                        if let Ok(f) = serde_json::from_value::<ExportFormat>(json!(v)) { ctx.export_format.set(f); }
                    }>
                        {move || ExportFormat::ALL.iter().map(|f| {
                            let id = enum_id(*f);
                            let sel = *f == ctx.export_format.get();
                            view! { <option value=id selected=sel>{f.label()}</option> }
                        }).collect_view()}
                    </select>
                </label>
                <div class="export-grid">
                    <button class="primary" on:click=move |_| ctx.export("desktop")>"Save to Desktop"</button>
                    <button on:click=move |_| ctx.export("reelforge")>"Send to Reelforge"</button>
                    <button on:click=move |_| ctx.export("forma3d")>"Send to Forma3D"</button>
                    <button on:click=move |_| ctx.export("tidefire")>"Send to Tidefire"</button>
                    <button on:click=move |_| ctx.export("choose")>"Save As…"</button>
                </div>
            </section>

            <RenderSection/>
            <CameraSection/>
            <LensSection/>

            <section>
                <h3>"App Folders"</h3>
                {move || {
                    let s = ctx.settings.get();
                    [("Reelforge", s.reelforge_dir), ("Forma3D", s.forma3d_dir), ("Tidefire", s.tidefire_dir)].into_iter().map(|(app, dir)| view! {
                        <div class="dir">
                            <div class="dir-head"><strong>{app}</strong><button class="small" on:click=move |_| ctx.change_dir(app)>"Change…"</button></div>
                            <code title=dir.clone()>{dir.clone()}</code>
                        </div>
                    }).collect_view()
                }}
            </section>

            <section>
                <h3>"Shortcuts"</h3>
                <dl class="keys">
                    <dt>"Ctrl+Z / Ctrl+Y"</dt><dd>"Undo / redo"</dd>
                    <dt>"Ctrl+S / Ctrl+O"</dt><dd>"Save / open project"</dd>
                    <dt>"Ctrl+D"</dt><dd>"Duplicate layer"</dd>
                    <dt>"← / →"</dt><dd>"Previous / next frame"</dd>
                    <dt>"Shift + ← / →"</dt><dd>"Track one frame"</dd>
                    <dt>"Space"</dt><dd>"Play / pause"</dd>
                    <dt>"Enter / dbl-click"</dt><dd>"Close shape"</dd>
                    <dt>"Delete"</dt><dd>"Remove selected shape point"</dd>
                    <dt>"Right-click"</dt><dd>"Undo last point while drawing"</dd>
                    <dt>"A"</dt><dd>"Adjust Track: drag corners to fix a frame"</dd>
                    <dt>"I / O / X"</dt><dd>"Set In / Out, clear"</dd>
                    <dt>"N"</dt><dd>"Next problem frame"</dd>
                    <dt>"Esc"</dt><dd>"Stop tracking / cancel"</dd>
                    <dt>"Wheel / middle-drag"</dt><dd>"Zoom / pan"</dd>
                </dl>
            </section>
        </aside>
    }
}

// ---------------------------------------------------------------------------
// Transport + timeline

#[component]
fn Bottom() -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    let frame_label = move || {
        let f = ctx.frame.get();
        ctx.project
            .with(|p| p.footage.as_ref().map(|ft| ft.frame_number(f)).unwrap_or(0))
    };
    let on_timeline = move |e: web_sys::MouseEvent| {
        let down = e.type_() == "mousedown";
        if !down && e.buttons() & 1 == 0 {
            return;
        }
        if down {
            ctx.scrubbing.set(true);
        }
        let Some(c) = ctx.timeline.get_untracked() else {
            return;
        };
        let n = ctx.frame_count_untracked();
        if n == 0 {
            return;
        }
        let f = (e.offset_x() as f64 / c.client_width().max(1) as f64 * n as f64).floor() as i64;
        ctx.frame.set(f.clamp(0, n as i64 - 1) as usize);
    };
    view! {
        <footer class="bottom">
            <div class="transport">
                <button title="First frame (Home)" on:click=move |_| ctx.frame.set(0)>"⏮"</button>
                <button class="track" title="Track backward" disabled=move || ctx.tracking.get() on:click=move |_| ctx.start_track(-1, None)>"◀◀ Track"</button>
                <button class="track" title="Track one frame backward (Shift+←)" disabled=move || ctx.tracking.get() on:click=move |_| ctx.start_track(-1, Some(1))>"◀|"</button>
                <button class="stop" title="Stop (Esc)" disabled=move || !ctx.tracking.get() on:click=move |_| ctx.stop_track()>"■"</button>
                <button class="track" title="Track one frame forward (Shift+→)" disabled=move || ctx.tracking.get() on:click=move |_| ctx.start_track(1, Some(1))>"|▶"</button>
                <button class="track" title="Track forward" disabled=move || ctx.tracking.get() on:click=move |_| ctx.start_track(1, None)>"Track ▶▶"</button>
                <button title="Last frame (End)" on:click=move |_| ctx.frame.set(ctx.frame_count_untracked().saturating_sub(1))>"⏭"</button>
                <span class="sep"></span>
                <button class="inout" class:set=move || ctx.in_point.get().is_some() title="Set In point (I)" on:click=move |_| ctx.set_in_out(true)>"In"</button>
                <button class="inout" class:set=move || ctx.out_point.get().is_some() title="Set Out point (O)" on:click=move |_| ctx.set_in_out(false)>"Out"</button>
                <button class="small" title="Clear In/Out (X)" disabled=move || ctx.in_point.get().is_none() && ctx.out_point.get().is_none() on:click=move |_| { ctx.in_point.set(None); ctx.out_point.set(None); }>"✕"</button>
                <button class="track" title="Re-track In→Out and blend it into the existing track" disabled=move || ctx.tracking.get() || ctx.in_point.get().is_none() || ctx.out_point.get().is_none() on:click=move |_| ctx.retrack_section()>"Re-track In→Out"</button>
                <button class="warn" title="Jump to the next frame flagged on the quality graph (N)" disabled=move || ctx.problem_count() == 0 on:click=move |_| ctx.next_problem()>
                    {move || { let n = ctx.problem_count(); if n == 0 { "✓ No problems".to_string() } else { format!("⚠ Next problem ({n})") } }}
                </button>
                <span class="sep"></span>
                <button title="Previous frame (←)" on:click=move |_| ctx.step_frame(-1)>"‹"</button>
                <button title="Play / pause (Space)" on:click=move |_| ctx.toggle_play()>{move || if ctx.playing.get() { "❚❚" } else { "▶" }}</button>
                <button title="Next frame (→)" on:click=move |_| ctx.step_frame(1)>"›"</button>
                <span class="frame">{move || format!("Frame {}", frame_label())}</span>
                <span class="muted">{move || format!("{} / {}", ctx.frame.get() + 1, ctx.frame_count())}</span>
                <span class="sep"></span>
                <span class="status" class:error=move || ctx.status_err.get()>{move || ctx.status.get()}</span>
            </div>
            <div class="timeline-wrap"><canvas node_ref=ctx.timeline class="timeline" on:mousedown=on_timeline on:mousemove=on_timeline></canvas></div>
        </footer>
    }
}

// ---------------------------------------------------------------------------
// Modals

#[component]
fn Modals() -> impl IntoView {
    let ctx = expect_context::<Ctx>();
    move || {
        match ctx.modal.get() {
        Modal::None => ().into_any(),
        Modal::About => view! {
            <div class="modal-bg" on:click=move |_| ctx.modal.set(Modal::None)>
                <div class="modal" on:click=|e| e.stop_propagation()>
                    <h2>"Trackwright " {env!("CARGO_PKG_VERSION")}</h2>
                    <p>"Planar motion tracking. Camera tracking is coming next."</p>
                    <p class="legal">"© 2026 Scott Hambach. All rights reserved."<br/>"Trackwright is a trademark of Scott Hambach."</p>
                    <h3>"Open-source components"</h3>
                    <ul class="licenses">
                        <li>"Tauri (MIT / Apache-2.0)"</li>
                        <li>"Leptos (MIT)"</li>
                        <li>"image-rs (MIT / Apache-2.0)"</li>
                        <li>"serde, serde_json (MIT / Apache-2.0)"</li>
                        <li>"wasm-bindgen, js-sys, web-sys (MIT / Apache-2.0)"</li>
                        <li>"dirs (MIT / Apache-2.0)"</li>
                        <li>"FFmpeg, if installed, is used as a separate program for video decoding (LGPL/GPL)"</li>
                    </ul>
                    <div class="modal-actions"><button class="primary" on:click=move |_| ctx.modal.set(Modal::None)>"Close"</button></div>
                </div>
            </div>
        }
        .into_any(),
        Modal::Unsaved(action) => {
            let what = match &action {
                Pending::Close => "before closing",
                Pending::New => "before starting a new project",
                Pending::OpenProject(_) => "before opening another project",
            };
            let (a1, a2) = (action.clone(), action.clone());
            view! {
                <div class="modal-bg">
                    <div class="modal">
                        <h2>"Save changes?"</h2>
                        <p>{format!("This project has unsaved changes. Save them {what}?")}</p>
                        <div class="modal-actions">
                            <button on:click=move |_| ctx.modal.set(Modal::None)>"Cancel"</button>
                            <button on:click=move |_| {
                                let a = a1.clone();
                                spawn_local(async move {
                                    let _ = tauri::call::<()>("discard_autosave", json!({})).await;
                                    ctx.perform(a);
                                });
                            }>"Don't Save"</button>
                            <button class="primary" on:click=move |_| {
                                let a = a2.clone();
                                ctx.save_then(false, move |ok| if ok { ctx.perform(a) });
                            }>"Save"</button>
                        </div>
                    </div>
                </div>
            }
            .into_any()
        }
        Modal::Recovery(info) => {
            let when = if info.saved_at > 0 {
                let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(info.saved_at as f64 * 1000.0));
                String::from(d.to_locale_string("default", &wasm_bindgen::JsValue::UNDEFINED))
            } else {
                "earlier".into()
            };
            let name = info.original_path.as_deref().map(file_name).unwrap_or_else(|| "an unsaved project".into());
            view! {
                <div class="modal-bg">
                    <div class="modal">
                        <h2>"Recover unsaved work?"</h2>
                        <p>{format!("Trackwright closed without saving {name}. An autosave from {when} is available.")}</p>
                        <div class="modal-actions">
                            <button on:click=move |_| {
                                ctx.modal.set(Modal::None);
                                spawn_local(async move { let _ = tauri::call::<()>("discard_autosave", json!({})).await; });
                            }>"Discard"</button>
                            <button class="primary" on:click=move |_| {
                                ctx.modal.set(Modal::None);
                                ctx.call("recover_autosave", json!({}), Some("Recovered from autosave. Save to keep it"));
                            }>"Recover"</button>
                        </div>
                    </div>
                </div>
            }
            .into_any()
        }
    }
    }
}

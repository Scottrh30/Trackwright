//! Trackwright UI (Leptos, client-side rendered inside Tauri).
//!
//! © Scott Hambach. All rights reserved.

mod draw;
mod tauri;

use draw::{draw_timeline, draw_viewport, frame_image, Overlays, Scene, Tool, View};
use leptos::ev;
use leptos::html;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;
use trackwright_core::geometry::{MotionModel, Point};
use trackwright_core::{ExportFormat, FrameTrack, Layer, Project};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct AppSettings {
    reelforge_dir: String,
    forma3d_dir: String,
    tidefire_dir: String,
    default_format: ExportFormat,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            reelforge_dir: String::new(),
            forma3d_dir: String::new(),
            tidefire_dir: String::new(),
            default_format: ExportFormat::Reelforge,
        }
    }
}

#[derive(Clone, Deserialize)]
struct TrackFrameEvent {
    layer_id: u32,
    frame: usize,
    track: FrameTrack,
}

#[derive(Clone, Deserialize)]
struct TrackDoneEvent {
    #[allow(dead_code)]
    layer_id: u32,
    ok: bool,
    #[allow(dead_code)]
    cancelled: bool,
    message: String,
}

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    None,
    Pan { start: Point, pan: Point },
    ShapePoint(usize),
    SurfaceCorner(usize),
}

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(App);
}

fn dist2(a: Point, b: Point) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn format_id(f: ExportFormat) -> String {
    serde_json::to_value(f)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default()
}

fn format_from_id(s: &str) -> ExportFormat {
    serde_json::from_value(json!(s)).unwrap_or(ExportFormat::Reelforge)
}

fn model_id(m: MotionModel) -> String {
    serde_json::to_value(m)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default()
}

#[component]
fn App() -> impl IntoView {
    let project = RwSignal::new(Project::default());
    let frame = RwSignal::new(0usize);
    let selected = RwSignal::new(None::<u32>);
    let tool = RwSignal::new(Tool::Pan);
    let draft = RwSignal::new(Vec::<Point>::new());
    let hover = RwSignal::new(None::<Point>);
    let view = RwSignal::new(View::default());
    let redraw = RwSignal::new(0u32);
    let tracking = RwSignal::new(false);
    let status = RwSignal::new(String::from("Ready"));
    let status_err = RwSignal::new(false);
    let show_surface = RwSignal::new(true);
    let show_grid = RwSignal::new(false);
    let show_points = RwSignal::new(true);
    let show_shapes = RwSignal::new(true);
    let settings = RwSignal::new(AppSettings::default());
    let export_format = RwSignal::new(ExportFormat::Reelforge);
    let about_open = RwSignal::new(false);
    let playing = RwSignal::new(false);
    let drag = StoredValue::new(Drag::None);
    let play_handle = StoredValue::new(None::<IntervalHandle>);

    let viewport_ref = NodeRef::<html::Canvas>::new();
    let timeline_ref = NodeRef::<html::Canvas>::new();

    let say = move |msg: String, err: bool| {
        status.set(msg);
        status_err.set(err);
    };

    let frame_count = move || project.with(|p| p.footage.as_ref().map_or(0, |f| f.frames.len()));

    let fit_view = move || {
        let Some(c) = viewport_ref.get_untracked() else {
            return;
        };
        project.with_untracked(|p| {
            if let Some(f) = &p.footage {
                view.set(View::fit(
                    f.width as f64,
                    f.height as f64,
                    c.client_width() as f64,
                    c.client_height() as f64,
                ));
            }
        });
    };

    // Apply a project returned by the backend.
    let apply_project = move |p: Project| {
        let n = p.footage.as_ref().map_or(0, |f| f.frames.len());
        project.set(p);
        if frame.get_untracked() >= n {
            frame.set(n.saturating_sub(1));
        }
        let ids: Vec<u32> = project.with_untracked(|p| p.layers.iter().map(|l| l.id).collect());
        if let Some(s) = selected.get_untracked() {
            if !ids.contains(&s) {
                selected.set(ids.last().copied());
            }
        }
    };

    // Run a command that returns a Project.
    let run_project_cmd =
        move |cmd: &'static str, args: serde_json::Value, ok_msg: Option<&'static str>| {
            spawn_local(async move {
                match tauri::call::<Project>(cmd, args).await {
                    Ok(p) => {
                        apply_project(p);
                        if let Some(m) = ok_msg {
                            say(m.into(), false);
                        }
                    }
                    Err(e) => say(e, true),
                }
            });
        };

    // Edit the selected layer locally and push it to the backend.
    let edit_layer = move |f: Box<dyn FnOnce(&mut Layer)>| {
        let Some(id) = selected.get_untracked() else {
            return;
        };
        let mut edited = None;
        project.update(|p| {
            if let Some(l) = p.layer_mut(id) {
                f(l);
                edited = Some(l.clone());
            }
        });
        if let Some(l) = edited {
            run_project_cmd("update_layer", json!({ "layer": l }), None);
        }
    };

    // ---- startup ----
    spawn_local(async move {
        if !tauri::in_tauri() {
            say(
                "Preview mode: start with `cargo tauri dev` to load footage".into(),
                true,
            );
            return;
        }
        if let Ok(p) = tauri::call::<Project>("get_project", json!({})).await {
            apply_project(p);
        }
        if let Ok(s) = tauri::call::<AppSettings>("get_settings", json!({})).await {
            export_format.set(s.default_format);
            settings.set(s);
        }
    });

    tauri::listen::<TrackFrameEvent>("track-frame", move |e| {
        project.update(|p| {
            if let Some(l) = p.layer_mut(e.layer_id) {
                l.tracks.insert(e.frame, e.track.clone());
            }
        });
        frame.set(e.frame);
        status.set(format!(
            "Tracking… frame {}  ({} points, {:.2}px error)",
            e.frame, e.track.inliers, e.track.rms_error
        ));
        status_err.set(false);
    });
    tauri::listen::<Project>("project-changed", move |p| {
        apply_project(p);
        say(
            "Footage loaded. Choose Draw Shape (D) and outline a flat surface".into(),
            false,
        );
    });
    tauri::listen::<String>("app-error", move |m| say(m, true));
    tauri::listen::<String>("app-status", move |m| say(m, false));
    tauri::listen::<TrackDoneEvent>("track-done", move |e| {
        tracking.set(false);
        say(e.message, !e.ok);
        spawn_local(async move {
            if let Ok(p) = tauri::call::<Project>("get_project", json!({})).await {
                apply_project(p);
            }
        });
    });

    // ---- rendering ----
    Effect::new(move |_| {
        redraw.track();
        let Some(canvas) = viewport_ref.get() else {
            return;
        };
        let fr = frame.get();
        let draft_pts = draft.get();
        let scene_view = view.get();
        let (sel, t, hv) = (selected.get(), tool.get(), hover.get());
        let ov = Overlays {
            surface: show_surface.get(),
            grid: show_grid.get(),
            points: show_points.get(),
            shapes: show_shapes.get(),
        };
        project.with(|p| {
            let bump = move || redraw.update(|n| *n = n.wrapping_add(1));
            let img = frame_image(p, fr, bump);
            // Preload upcoming frames for smooth scrubbing and playback.
            for k in 1..=3 {
                let _ = frame_image(p, fr + k, || {});
            }
            if fr > 0 {
                let _ = frame_image(p, fr - 1, || {});
            }
            let scene = Scene {
                project: p,
                frame: fr,
                selected: sel,
                tool: t,
                draft: &draft_pts,
                hover: hv,
                view: scene_view,
                overlays: ov,
            };
            draw_viewport(&canvas, &scene, img);
        });
    });
    Effect::new(move |_| {
        redraw.track();
        let Some(canvas) = timeline_ref.get() else {
            return;
        };
        let (fr, sel) = (frame.get(), selected.get());
        project.with(|p| draw_timeline(&canvas, p, fr, sel));
    });
    let _resize = window_event_listener(ev::resize, move |_| {
        redraw.update(|n| *n = n.wrapping_add(1))
    });

    // Fit the view whenever new footage is loaded.
    Effect::new(move |prev: Option<String>| {
        let src = project.with(|p| {
            p.footage
                .as_ref()
                .map(|f| f.source.clone())
                .unwrap_or_default()
        });
        if prev.as_deref() != Some(src.as_str()) && !src.is_empty() {
            fit_view();
        }
        src
    });

    // ---- actions ----
    let open_footage = move |kind: &'static str| {
        say(
            "Opening footage… (video is decoded to frames first)".into(),
            false,
        );
        run_project_cmd(
            "open_footage",
            json!({ "kind": kind, "path": null }),
            Some("Footage loaded. Choose Draw Shape (D) and outline a flat surface"),
        );
    };

    let start_track = move |direction: i32, steps: Option<usize>| {
        if tracking.get_untracked() {
            return;
        }
        let Some(id) = selected.get_untracked() else {
            say("Select or draw a layer to track".into(), true);
            return;
        };
        let fr = frame.get_untracked();
        let tracked = project.with_untracked(|p| {
            p.layer(id)
                .map(|l| l.tracks.contains_key(&fr))
                .unwrap_or(false)
        });
        if !tracked {
            say("This frame isn't tracked for the layer. Move to a tracked frame (coloured on the timeline)".into(), true);
            return;
        }
        tracking.set(true);
        say("Tracking…".into(), false);
        spawn_local(async move {
            let args = json!({ "id": id, "from": fr, "direction": direction, "steps": steps });
            if let Err(e) = tauri::call::<()>("start_tracking", args).await {
                tracking.set(false);
                say(e, true);
            }
        });
    };
    let stop_track = move || {
        spawn_local(async move {
            let _ = tauri::call::<()>("stop_tracking", json!({})).await;
        });
    };

    let finish_draft = move || {
        let pts = draft.get_untracked();
        if pts.len() < 3 {
            say("A shape needs at least 3 points".into(), true);
            return;
        }
        draft.set(vec![]);
        let fr = frame.get_untracked();
        spawn_local(async move {
            match tauri::call::<Project>("add_layer", json!({ "frame": fr, "points": pts })).await {
                Ok(p) => {
                    let id = p.layers.last().map(|l| l.id);
                    apply_project(p);
                    selected.set(id);
                    tool.set(Tool::Surface);
                    say("Layer created. Adjust the blue surface corners if needed, then Track Forward (Shift+→ steps)".into(), false);
                }
                Err(e) => say(e, true),
            }
        });
    };

    let step_frame = move |d: i64| {
        let n = frame_count() as i64;
        if n == 0 {
            return;
        }
        frame.update(|f| *f = (*f as i64 + d).clamp(0, n - 1) as usize);
    };

    let toggle_play = move || {
        if playing.get_untracked() {
            playing.set(false);
            if let Some(h) = play_handle.get_value() {
                h.clear();
            }
            play_handle.set_value(None);
            return;
        }
        let fps = project
            .with_untracked(|p| p.footage.as_ref().map_or(24.0, |f| f.fps))
            .clamp(1.0, 120.0);
        playing.set(true);
        let h = set_interval_with_handle(
            move || {
                let n = frame_count();
                if n > 0 {
                    frame.update(|f| *f = (*f + 1) % n);
                }
            },
            Duration::from_millis((1000.0 / fps) as u64),
        );
        play_handle.set_value(h.ok());
    };

    let save = move |save_as: bool| {
        spawn_local(async move {
            match tauri::call::<Option<String>>("save_project", json!({ "saveAs": save_as })).await
            {
                Ok(Some(p)) => say(format!("Saved {p}"), false),
                Ok(None) => {}
                Err(e) => say(e, true),
            }
        });
    };

    let do_export = move |destination: &'static str| {
        let Some(id) = selected.get_untracked() else {
            say("Select a tracked layer to export".into(), true);
            return;
        };
        let fmt = export_format.get_untracked();
        spawn_local(async move {
            match tauri::call::<Option<String>>(
                "export_layer",
                json!({ "id": id, "format": fmt, "destination": destination }),
            )
            .await
            {
                Ok(Some(p)) => say(format!("Exported to {p}"), false),
                Ok(None) => {}
                Err(e) => say(e, true),
            }
        });
    };

    let change_dir = move |which: &'static str| {
        spawn_local(async move {
            let title = format!("Choose the {which} import folder");
            if let Ok(Some(dir)) =
                tauri::call::<Option<String>>("pick_folder", json!({ "title": title })).await
            {
                let mut s = settings.get_untracked();
                match which {
                    "Reelforge" => s.reelforge_dir = dir,
                    "Forma3D" => s.forma3d_dir = dir,
                    _ => s.tidefire_dir = dir,
                }
                match tauri::call::<AppSettings>("set_settings", json!({ "settings": s })).await {
                    Ok(s) => settings.set(s),
                    Err(e) => say(e, true),
                }
            }
        });
    };

    // ---- keyboard ----
    let _keys = window_event_listener(ev::keydown, move |e: web_sys::KeyboardEvent| {
        let in_field = e
            .target()
            .and_then(|t| wasm_bindgen::JsCast::dyn_into::<web_sys::Element>(t).ok())
            .map(|el| matches!(el.tag_name().as_str(), "INPUT" | "SELECT" | "TEXTAREA"))
            .unwrap_or(false);
        if in_field {
            return;
        }
        let key = e.key();
        match key.as_str() {
            "ArrowRight" if e.shift_key() => start_track(1, Some(1)),
            "ArrowLeft" if e.shift_key() => start_track(-1, Some(1)),
            "ArrowRight" => step_frame(1),
            "ArrowLeft" => step_frame(-1),
            "Home" => frame.set(0),
            "End" => frame.set(frame_count().saturating_sub(1)),
            " " => {
                e.prevent_default();
                toggle_play();
            }
            "Escape" => {
                if tracking.get_untracked() {
                    stop_track();
                }
                draft.set(vec![]);
            }
            "Enter" if !draft.get_untracked().is_empty() => finish_draft(),
            "d" | "D" => tool.set(Tool::Draw),
            "e" | "E" => tool.set(Tool::EditShape),
            "s" | "S" if !e.ctrl_key() => tool.set(Tool::Surface),
            "h" | "H" => tool.set(Tool::Pan),
            "f" | "F" => fit_view(),
            "s" if e.ctrl_key() => {
                e.prevent_default();
                save(false);
            }
            _ => {}
        }
    });

    // ---- viewport mouse ----
    let mouse_pos = |e: &web_sys::MouseEvent| [e.offset_x() as f64, e.offset_y() as f64];

    let on_down = move |e: web_sys::MouseEvent| {
        let s = mouse_pos(&e);
        let v = view.get_untracked();
        let ip = v.to_image(s);
        if e.button() == 1 || e.button() == 2 || tool.get_untracked() == Tool::Pan || e.alt_key() {
            if e.button() == 2 && !draft.get_untracked().is_empty() {
                draft.update(|d| {
                    d.pop();
                });
                return;
            }
            drag.set_value(Drag::Pan {
                start: s,
                pan: v.pan,
            });
            return;
        }
        if e.button() != 0 || project.with_untracked(|p| p.footage.is_none()) {
            return;
        }
        let fr = frame.get_untracked();
        let hit_r2 = 9.0f64.powi(2);
        match tool.get_untracked() {
            Tool::Draw => {
                let d = draft.get_untracked();
                if d.len() >= 3 && dist2(v.to_screen(d[0]), s) < hit_r2 * 1.5 {
                    finish_draft();
                } else {
                    draft.update(|d| d.push(ip));
                }
            }
            Tool::EditShape | Tool::Surface => {
                if tracking.get_untracked() {
                    return;
                }
                let Some(id) = selected.get_untracked() else {
                    return;
                };
                let hit = project.with_untracked(|p| {
                    let l = p.layer(id)?;
                    if tool.get_untracked() == Tool::EditShape {
                        let sh = l.shape_at(fr)?;
                        sh.points
                            .iter()
                            .position(|&q| dist2(v.to_screen(q), s) < hit_r2)
                            .map(Drag::ShapePoint)
                    } else {
                        let c = l.surface_at(fr)?;
                        c.iter()
                            .position(|&q| dist2(v.to_screen(q), s) < hit_r2 * 1.4)
                            .map(Drag::SurfaceCorner)
                    }
                });
                match hit {
                    Some(d) => drag.set_value(d),
                    None => drag.set_value(Drag::Pan {
                        start: s,
                        pan: v.pan,
                    }),
                }
            }
            Tool::Pan => {}
        }
    };

    let on_move = move |e: web_sys::MouseEvent| {
        let s = mouse_pos(&e);
        let v = view.get_untracked();
        let ip = v.to_image(s);
        if tool.get_untracked() == Tool::Draw {
            hover.set(Some(ip));
        }
        let fr = frame.get_untracked();
        match drag.get_value() {
            Drag::None => {}
            Drag::Pan { start, pan } => {
                view.update(|v| v.pan = [pan[0] + s[0] - start[0], pan[1] + s[1] - start[1]])
            }
            Drag::ShapePoint(i) => {
                let Some(id) = selected.get_untracked() else {
                    return;
                };
                project.update(|p| {
                    if let Some(l) = p.layer_mut(id) {
                        if let Some(mut sh) = l.shape_at(fr) {
                            if i < sh.points.len() {
                                sh.points[i] = ip;
                                l.set_shape_at(fr, sh);
                            }
                        }
                    }
                });
            }
            Drag::SurfaceCorner(i) => {
                let Some(id) = selected.get_untracked() else {
                    return;
                };
                project.update(|p| {
                    if let Some(l) = p.layer_mut(id) {
                        l.set_surface_corner(fr, i, ip);
                    }
                });
            }
        }
    };

    let on_up = move |_e: web_sys::MouseEvent| {
        let d = drag.get_value();
        drag.set_value(Drag::None);
        if matches!(d, Drag::ShapePoint(_) | Drag::SurfaceCorner(_)) {
            edit_layer(Box::new(|_| {}));
        }
    };

    let on_wheel = move |e: web_sys::WheelEvent| {
        e.prevent_default();
        let s = [e.offset_x() as f64, e.offset_y() as f64];
        view.update(|v| {
            let before = v.to_image(s);
            let factor = (-e.delta_y() * 0.0015).exp();
            v.zoom = (v.zoom * factor).clamp(0.05, 32.0);
            let after = v.to_screen(before);
            v.pan[0] += s[0] - after[0];
            v.pan[1] += s[1] - after[1];
        });
    };

    let on_dbl = move |_e: web_sys::MouseEvent| {
        if tool.get_untracked() == Tool::Draw && draft.with_untracked(|d| d.len() >= 3) {
            draft.update(|d| {
                d.pop(); // the double-click's second point
            });
            finish_draft();
        }
    };

    let on_timeline = move |e: web_sys::MouseEvent| {
        if e.buttons() & 1 == 0 && e.type_() != "mousedown" {
            return;
        }
        let Some(c) = timeline_ref.get_untracked() else {
            return;
        };
        let n = frame_count();
        if n == 0 {
            return;
        }
        let f = (e.offset_x() as f64 / c.client_width().max(1) as f64 * n as f64).floor() as i64;
        frame.set(f.clamp(0, n as i64 - 1) as usize);
    };

    // ---- derived values ----
    let sel_layer = move || {
        selected
            .get()
            .and_then(|id| project.with(|p| p.layer(id).cloned()))
    };
    let frame_label = move || {
        let f = frame.get();
        project.with(|p| p.footage.as_ref().map(|ft| ft.frame_number(f)).unwrap_or(0))
    };
    let tool_btn = move |t: Tool, label: &'static str, key: &'static str| {
        view! {
            <button class="tool" class:active=move || tool.get() == t title=format!("{label} ({key})")
                on:click=move |_| { tool.set(t); if t != Tool::Draw { draft.set(vec![]); hover.set(None); } }>
                {label}<kbd>{key}</kbd>
            </button>
        }
    };
    let toggle = move |sig: RwSignal<bool>, label: &'static str| {
        view! {
            <label class="toggle"><input type="checkbox" prop:checked=move || sig.get() on:change=move |_| sig.update(|v| *v = !*v) />{label}</label>
        }
    };

    view! {
        <div class="app">
            <header class="topbar">
                <div class="brand"><span class="logo">"◆"</span>"Trackwright"</div>
                <div class="group">
                    <button on:click=move |_| run_project_cmd("new_project", json!({}), Some("New project"))>"New"</button>
                    <button on:click=move |_| open_footage("file")>"Open Footage…"</button>
                    <button on:click=move |_| open_footage("folder")>"Open Sequence Folder…"</button>
                    <button on:click=move |_| run_project_cmd("open_project", json!({}), Some("Project opened"))>"Open Project…"</button>
                    <button on:click=move |_| save(false)>"Save"</button>
                    <button on:click=move |_| save(true)>"Save As…"</button>
                </div>
                <div class="group tools">
                    {tool_btn(Tool::Pan, "Pan", "H")}
                    {tool_btn(Tool::Draw, "Draw Shape", "D")}
                    {tool_btn(Tool::EditShape, "Edit Shape", "E")}
                    {tool_btn(Tool::Surface, "Surface", "S")}
                </div>
                <div class="group">
                    {toggle(show_shapes, "Shapes")}
                    {toggle(show_surface, "Surface")}
                    {toggle(show_grid, "Grid")}
                    {toggle(show_points, "Points")}
                    <button on:click=move |_| fit_view() title="Fit (F)">"Fit"</button>
                    <button on:click=move |_| about_open.set(true)>"About"</button>
                </div>
            </header>

            <aside class="panel left">
                <section>
                    <h3>"Layers"
                        <button class="small" on:click=move |_| { tool.set(Tool::Draw); say("Click points around a flat surface; click the first point (or Enter) to close".into(), false); }>"+ New"</button>
                    </h3>
                    <ul class="layers">
                        {move || project.with(|p| p.layers.iter().map(|l| {
                            let id = l.id;
                            let name = l.name.clone();
                            let color = l.color.clone();
                            let vis = l.visible;
                            let tracked = l.tracks.len();
                            view! {
                                <li class:selected=move || selected.get() == Some(id) on:click=move |_| selected.set(Some(id))>
                                    <span class="swatch" style=format!("background:{color}")></span>
                                    <span class="lname">{name}</span>
                                    <span class="muted">{format!("{tracked}f")}</span>
                                    <button class="icon" title="Show / hide" on:click=move |ev| {
                                        ev.stop_propagation();
                                        selected.set(Some(id));
                                        edit_layer(Box::new(|l| l.visible = !l.visible));
                                    }>{if vis { "👁" } else { "–" }}</button>
                                    <button class="icon" title="Delete layer" on:click=move |ev| {
                                        ev.stop_propagation();
                                        run_project_cmd("delete_layer", json!({ "id": id }), Some("Layer deleted"));
                                    }>"✕"</button>
                                </li>
                            }
                        }).collect_view())}
                    </ul>
                </section>

                {move || sel_layer().map(|l| {
                    let s = l.settings.clone();
                    let model = model_id(s.model);
                    let name = l.name.clone();
                    let ref_label = project.with_untracked(|p| p.footage.as_ref().map_or(0, |f| f.frame_number(l.ref_frame)));
                    let num = move |label: &'static str, value: String, step: &'static str, apply: fn(&mut Layer, f64)| {
                        view! {
                            <label class="field">{label}
                                <input type="number" step=step prop:value=value on:change=move |ev| {
                                    if let Ok(v) = event_target_value(&ev).parse::<f64>() {
                                        edit_layer(Box::new(move |l| apply(l, v)));
                                    }
                                } />
                            </label>
                        }
                    };
                    view! {
                        <section>
                            <h3>"Layer"</h3>
                            <label class="field">"Name"
                                <input type="text" prop:value=name on:change=move |ev| { let v = event_target_value(&ev); edit_layer(Box::new(move |l| l.name = v)); } />
                            </label>
                            <p class="muted">{format!("Reference frame {ref_label}")}</p>
                            <div class="row">
                                <button on:click=move |_| {
                                    let Some(id) = selected.get_untracked() else { return };
                                    run_project_cmd("set_reference_frame", json!({ "id": id, "frame": frame.get_untracked() }), Some("Reference frame set"));
                                }>"Set Reference Here"</button>
                            </div>
                            <div class="row">
                                <button on:click=move |_| {
                                    let Some(id) = selected.get_untracked() else { return };
                                    run_project_cmd("clear_track", json!({ "id": id, "frame": frame.get_untracked(), "forward": false }), Some("Cleared track before this frame"));
                                }>"Clear ◀"</button>
                                <button on:click=move |_| {
                                    let Some(id) = selected.get_untracked() else { return };
                                    run_project_cmd("clear_track", json!({ "id": id, "frame": null, "forward": true }), Some("Track cleared"));
                                }>"Clear All"</button>
                                <button on:click=move |_| {
                                    let Some(id) = selected.get_untracked() else { return };
                                    run_project_cmd("clear_track", json!({ "id": id, "frame": frame.get_untracked(), "forward": true }), Some("Cleared track after this frame"));
                                }>"Clear ▶"</button>
                            </div>
                            <label class="toggle">
                                <input type="checkbox" prop:checked=l.shape.smooth on:change=move |_| edit_layer(Box::new(|l| l.shape.smooth = !l.shape.smooth)) />
                                "Smooth (spline) outline"
                            </label>
                            <button class="wide" on:click=move |_| edit_layer(Box::new(|l| l.surface = trackwright_core::shape::default_surface(&l.shape)))>"Reset Surface to Shape Bounds"</button>
                        </section>
                        <section>
                            <h3>"Track Settings"</h3>
                            <label class="field">"Motion"
                                <select on:change=move |ev| {
                                    let v = event_target_value(&ev);
                                    if let Ok(m) = serde_json::from_value::<MotionModel>(json!(v)) {
                                        edit_layer(Box::new(move |l| l.settings.model = m));
                                    }
                                }>
                                    {MotionModel::ALL.iter().map(|m| {
                                        let id = model_id(*m);
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
                                <input type="checkbox" prop:checked=s.refine_against_reference on:change=move |_| edit_layer(Box::new(|l| l.settings.refine_against_reference = !l.settings.refine_against_reference)) />
                                "Anti-drift (re-solve against reference)"
                            </label>
                        </section>
                    }
                })}
            </aside>

            <main class="viewport">
                <canvas node_ref=viewport_ref class="view"
                    class:draw=move || tool.get() == Tool::Draw
                    on:mousedown=on_down on:mousemove=on_move on:mouseup=on_up
                    on:mouseleave=move |e| { hover.set(None); on_up(e) }
                    on:wheel=on_wheel on:dblclick=on_dbl
                    on:contextmenu=|e| e.prevent_default()></canvas>
            </main>

            <aside class="panel right">
                <section>
                    <h3>"Footage"</h3>
                    {move || project.with(|p| match &p.footage {
                        None => view! { <p class="muted">"No footage loaded. Drag a video, image or folder onto the window."</p> }.into_any(),
                        Some(f) => {
                            let name = std::path::Path::new(&f.source).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                            let (fps, first) = (f.fps, f.first_frame);
                            view! {
                                <p class="fname" title=f.source.clone()>{name}</p>
                                <p class="muted">{format!("{} × {} · {} frames", f.width, f.height, f.frames.len())}</p>
                                <div class="row">
                                    <label class="field">"FPS"
                                        <input type="number" step="0.001" prop:value=format!("{fps:.3}") on:change=move |ev| {
                                            if let Ok(v) = event_target_value(&ev).parse::<f64>() {
                                                let first = project.with_untracked(|p| p.footage.as_ref().map_or(0, |f| f.first_frame));
                                                run_project_cmd("set_footage_timing", json!({ "fps": v, "firstFrame": first }), None);
                                            }
                                        } />
                                    </label>
                                    <label class="field">"First frame"
                                        <input type="number" step="1" prop:value=first.to_string() on:change=move |ev| {
                                            if let Ok(v) = event_target_value(&ev).parse::<i64>() {
                                                let fps = project.with_untracked(|p| p.footage.as_ref().map_or(24.0, |f| f.fps));
                                                run_project_cmd("set_footage_timing", json!({ "fps": fps, "firstFrame": v }), None);
                                            }
                                        } />
                                    </label>
                                </div>
                            }.into_any()
                        }
                    })}
                </section>

                <section>
                    <h3>"Export Track"</h3>
                    <label class="field">"Format"
                        <select on:change=move |ev| export_format.set(format_from_id(&event_target_value(&ev)))>
                            {move || ExportFormat::ALL.iter().map(|f| {
                                let id = format_id(*f);
                                let sel = *f == export_format.get();
                                view! { <option value=id selected=sel>{f.label()}</option> }
                            }).collect_view()}
                        </select>
                    </label>
                    <div class="export-grid">
                        <button class="primary" on:click=move |_| do_export("desktop")>"Save to Desktop"</button>
                        <button on:click=move |_| do_export("reelforge")>"Send to Reelforge"</button>
                        <button on:click=move |_| do_export("forma3d")>"Send to Forma3D"</button>
                        <button on:click=move |_| do_export("tidefire")>"Send to Tidefire"</button>
                        <button on:click=move |_| do_export("choose")>"Save As…"</button>
                    </div>
                    <p class="hint">"Tip: pick the matching format (Reelforge track for Reelforge, etc.). Nuke and After Effects corner pins are available too."</p>
                </section>

                <section>
                    <h3>"App Folders"</h3>
                    {move || {
                        let s = settings.get();
                        [("Reelforge", s.reelforge_dir), ("Forma3D", s.forma3d_dir), ("Tidefire", s.tidefire_dir)].into_iter().map(|(app, dir)| view! {
                            <div class="dir">
                                <div class="dir-head"><strong>{app}</strong><button class="small" on:click=move |_| change_dir(app)>"Change…"</button></div>
                                <code title=dir.clone()>{dir.clone()}</code>
                            </div>
                        }).collect_view()
                    }}
                </section>

                <section>
                    <h3>"Shortcuts"</h3>
                    <dl class="keys">
                        <dt>"← / →"</dt><dd>"Previous / next frame"</dd>
                        <dt>"Shift + ← / →"</dt><dd>"Track one frame"</dd>
                        <dt>"Space"</dt><dd>"Play / pause"</dd>
                        <dt>"Enter / dbl-click"</dt><dd>"Close shape"</dd>
                        <dt>"Right-click"</dt><dd>"Undo last shape point"</dd>
                        <dt>"Esc"</dt><dd>"Stop tracking / cancel shape"</dd>
                        <dt>"Wheel / middle-drag"</dt><dd>"Zoom / pan"</dd>
                    </dl>
                </section>
            </aside>

            <footer class="bottom">
                <div class="transport">
                    <button title="First frame (Home)" on:click=move |_| frame.set(0)>"⏮"</button>
                    <button class="track" title="Track backward" disabled=move || tracking.get() on:click=move |_| start_track(-1, None)>"◀◀ Track"</button>
                    <button class="track" title="Track one frame backward (Shift+←)" disabled=move || tracking.get() on:click=move |_| start_track(-1, Some(1))>"◀|"</button>
                    <button class="stop" title="Stop (Esc)" disabled=move || !tracking.get() on:click=move |_| stop_track()>"■"</button>
                    <button class="track" title="Track one frame forward (Shift+→)" disabled=move || tracking.get() on:click=move |_| start_track(1, Some(1))>"|▶"</button>
                    <button class="track" title="Track forward" disabled=move || tracking.get() on:click=move |_| start_track(1, None)>"Track ▶▶"</button>
                    <button title="Last frame (End)" on:click=move |_| frame.set(frame_count().saturating_sub(1))>"⏭"</button>
                    <span class="sep"></span>
                    <button title="Previous frame (←)" on:click=move |_| step_frame(-1)>"‹"</button>
                    <button title="Play / pause (Space)" on:click=move |_| toggle_play()>{move || if playing.get() { "❚❚" } else { "▶" }}</button>
                    <button title="Next frame (→)" on:click=move |_| step_frame(1)>"›"</button>
                    <span class="frame">{move || format!("Frame {}", frame_label())}</span>
                    <span class="muted">{move || format!("{} / {}", frame.get() + 1, frame_count())}</span>
                    <span class="sep"></span>
                    <span class="status" class:error=move || status_err.get()>{move || status.get()}</span>
                </div>
                <div class="timeline-wrap"><canvas node_ref=timeline_ref class="timeline" on:mousedown=on_timeline on:mousemove=on_timeline></canvas></div>
            </footer>

            <Show when=move || about_open.get()>
                <div class="modal-bg" on:click=move |_| about_open.set(false)>
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
                        <button class="primary" on:click=move |_| about_open.set(false)>"Close"</button>
                    </div>
                </div>
            </Show>
        </div>
    }
}

//! Viewport and timeline rendering on 2D canvases.

use std::cell::RefCell;
use std::collections::HashMap;
use trackwright_core::geometry::{apply, homography_from_corners, Point};
use trackwright_core::Project;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, HtmlImageElement};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub zoom: f64,
    pub pan: [f64; 2],
}

impl Default for View {
    fn default() -> Self {
        Self {
            zoom: 0.5,
            pan: [20.0, 20.0],
        }
    }
}

impl View {
    pub fn to_screen(&self, p: Point) -> Point {
        [
            p[0] * self.zoom + self.pan[0],
            p[1] * self.zoom + self.pan[1],
        ]
    }
    pub fn to_image(&self, s: Point) -> Point {
        [
            (s[0] - self.pan[0]) / self.zoom,
            (s[1] - self.pan[1]) / self.zoom,
        ]
    }
    pub fn fit(w: f64, h: f64, cw: f64, ch: f64) -> Self {
        let zoom = (cw / w).min(ch / h) * 0.94;
        Self {
            zoom,
            pan: [(cw - w * zoom) / 2.0, (ch - h * zoom) / 2.0],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Pan,
    Draw,
    EditShape,
    Surface,
}

pub struct Overlays {
    pub surface: bool,
    pub grid: bool,
    pub points: bool,
    pub shapes: bool,
}

pub struct Scene<'a> {
    pub project: &'a Project,
    pub frame: usize,
    pub selected: Option<u32>,
    pub tool: Tool,
    pub draft: &'a [Point],
    pub hover: Option<Point>,
    pub view: View,
    pub overlays: Overlays,
}

thread_local! {
    static IMAGES: RefCell<(String, HashMap<usize, HtmlImageElement>)> = RefCell::new((String::new(), HashMap::new()));
    static LAST_GOOD: RefCell<Option<HtmlImageElement>> = const { RefCell::new(None) };
}

/// Get (and start loading) the image for `frame`. `on_load` fires once it is ready.
pub fn frame_image(
    project: &Project,
    frame: usize,
    on_load: impl Fn() + 'static + Clone,
) -> Option<HtmlImageElement> {
    let f = project.footage.as_ref()?;
    let path = f.frames.get(frame)?.clone();
    IMAGES.with(|cell| {
        let mut c = cell.borrow_mut();
        if c.0 != f.source {
            c.0 = f.source.clone();
            c.1.clear();
        }
        // Evict far-away frames to bound memory.
        if c.1.len() > 48 {
            c.1.retain(|&k, _| k.abs_diff(frame) < 16);
        }
        if let Some(img) = c.1.get(&frame) {
            return Some(img.clone());
        }
        let img = HtmlImageElement::new().ok()?;
        let cb = Closure::<dyn Fn()>::new(on_load);
        img.set_onload(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
        img.set_src(&crate::tauri::convert_file_src(&path));
        c.1.insert(frame, img.clone());
        Some(img)
    })
}

fn ctx_of(canvas: &HtmlCanvasElement) -> Option<CanvasRenderingContext2d> {
    canvas.get_context("2d").ok()??.dyn_into().ok()
}

/// Match the backing store to the element's CSS size; returns (css_w, css_h, dpr).
fn fit_canvas(canvas: &HtmlCanvasElement) -> (f64, f64, f64) {
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0);
    let (cw, ch) = (
        canvas.client_width().max(1) as f64,
        canvas.client_height().max(1) as f64,
    );
    let (bw, bh) = ((cw * dpr).round() as u32, (ch * dpr).round() as u32);
    if canvas.width() != bw || canvas.height() != bh {
        canvas.set_width(bw);
        canvas.set_height(bh);
    }
    (cw, ch, dpr)
}

fn path(ctx: &CanvasRenderingContext2d, pts: &[Point], close: bool) {
    ctx.begin_path();
    for (i, p) in pts.iter().enumerate() {
        if i == 0 {
            ctx.move_to(p[0], p[1]);
        } else {
            ctx.line_to(p[0], p[1]);
        }
    }
    if close {
        ctx.close_path();
    }
}

fn handle(ctx: &CanvasRenderingContext2d, p: Point, r: f64, fill: &str, stroke: &str) {
    ctx.begin_path();
    let _ = ctx.arc(p[0], p[1], r, 0.0, std::f64::consts::TAU);
    ctx.set_fill_style_str(fill);
    ctx.fill();
    ctx.set_stroke_style_str(stroke);
    ctx.stroke();
}

pub fn draw_viewport(canvas: &HtmlCanvasElement, scene: &Scene, image: Option<HtmlImageElement>) {
    let Some(ctx) = ctx_of(canvas) else { return };
    let (cw, ch, dpr) = fit_canvas(canvas);
    let _ = ctx.set_transform(dpr, 0.0, 0.0, dpr, 0.0, 0.0);
    ctx.set_fill_style_str("#0d0e11");
    ctx.fill_rect(0.0, 0.0, cw, ch);

    let Some(footage) = scene.project.footage.as_ref() else {
        ctx.set_fill_style_str("#6b7280");
        ctx.set_font("15px system-ui, sans-serif");
        ctx.set_text_align("center");
        let _ = ctx.fill_text(
            "Open a video or image sequence to start tracking",
            cw / 2.0,
            ch / 2.0,
        );
        return;
    };

    let v = scene.view;
    let z = v.zoom;
    let _ = ctx.set_transform(dpr * z, 0.0, 0.0, dpr * z, dpr * v.pan[0], dpr * v.pan[1]);
    let (w, h) = (footage.width as f64, footage.height as f64);
    let img = image.filter(|i| i.complete() && i.natural_width() > 0);
    let img = match img {
        Some(i) => {
            LAST_GOOD.with(|l| *l.borrow_mut() = Some(i.clone()));
            Some(i)
        }
        None => LAST_GOOD.with(|l| l.borrow().clone()),
    };
    if let Some(i) = img {
        let _ = ctx.draw_image_with_html_image_element_and_dw_and_dh(&i, 0.0, 0.0, w, h);
    } else {
        ctx.set_fill_style_str("#1b1d22");
        ctx.fill_rect(0.0, 0.0, w, h);
    }
    ctx.set_stroke_style_str("#2c2f36");
    ctx.set_line_width(1.0 / z);
    ctx.stroke_rect(0.0, 0.0, w, h);

    let px = 1.0 / z; // one screen pixel in image units
    for layer in scene.project.layers.iter().filter(|l| l.visible) {
        let selected = scene.selected == Some(layer.id);
        let Some(shape) = layer.shape_at(scene.frame) else {
            continue;
        };
        let track = layer.tracks.get(&scene.frame);

        if scene.overlays.shapes {
            let poly = shape.flatten(10);
            path(&ctx, &poly, true);
            if selected {
                ctx.set_global_alpha(0.14);
                ctx.set_fill_style_str(&layer.color);
                ctx.fill();
                ctx.set_global_alpha(1.0);
            }
            ctx.set_stroke_style_str(&layer.color);
            ctx.set_line_width(if selected { 2.0 } else { 1.2 } * px);
            ctx.stroke();
            if selected && scene.tool == Tool::EditShape {
                ctx.set_line_width(1.5 * px);
                for p in &shape.points {
                    handle(&ctx, *p, 5.0 * px, "#15171c", &layer.color);
                }
            }
        }

        if scene.overlays.points && selected {
            if let Some(t) = track {
                ctx.set_fill_style_str("#7dff6a");
                for p in &t.points {
                    ctx.fill_rect(p[0] - 1.5 * px, p[1] - 1.5 * px, 3.0 * px, 3.0 * px);
                }
            }
        }

        let show_surface = scene.overlays.surface && (selected || scene.tool == Tool::Surface);
        if let Some(corners) = layer.surface_at(scene.frame) {
            if scene.overlays.grid && selected {
                draw_grid(&ctx, &corners, px);
            }
            if show_surface {
                path(&ctx, &corners, true);
                ctx.set_stroke_style_str("#3fd0ff");
                ctx.set_line_width(1.6 * px);
                ctx.stroke();
                // Diagonals mark the centre like a corner pin.
                ctx.set_global_alpha(0.5);
                path(&ctx, &[corners[0], corners[2]], false);
                ctx.stroke();
                path(&ctx, &[corners[1], corners[3]], false);
                ctx.stroke();
                ctx.set_global_alpha(1.0);
                let labels = ["1", "2", "3", "4"];
                ctx.set_font(&format!("{}px system-ui, sans-serif", 11.0 * px));
                ctx.set_text_align("center");
                for (i, c) in corners.iter().enumerate() {
                    handle(&ctx, *c, 6.0 * px, "#3fd0ff", "#0d0e11");
                    ctx.set_fill_style_str("#0d0e11");
                    let _ = ctx.fill_text(labels[i], c[0], c[1] + 4.0 * px);
                }
            }
        }
    }

    // Shape being drawn.
    if !scene.draft.is_empty() {
        let mut pts = scene.draft.to_vec();
        if let Some(hv) = scene.hover {
            pts.push(hv);
        }
        path(&ctx, &pts, false);
        ctx.set_stroke_style_str("#ffb020");
        ctx.set_line_width(1.5 * px);
        let _ = ctx.set_line_dash(&js_sys::Array::of2(&(6.0 * px).into(), &(4.0 * px).into()));
        ctx.stroke();
        let _ = ctx.set_line_dash(&js_sys::Array::new());
        for (i, p) in scene.draft.iter().enumerate() {
            let r = if i == 0 { 7.0 } else { 4.5 } * px;
            handle(
                &ctx,
                *p,
                r,
                if i == 0 { "#ffb020" } else { "#15171c" },
                "#ffb020",
            );
        }
    }
}

/// Checkerboard-ish grid warped onto the surface to judge track quality (like Mocha's insert).
fn draw_grid(ctx: &CanvasRenderingContext2d, corners: &[Point; 4], px: f64) {
    let unit = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let Some(h) = homography_from_corners(&unit, corners) else {
        return;
    };
    let n = 8;
    // Alternate filled cells.
    ctx.set_global_alpha(0.28);
    for i in 0..n {
        for j in 0..n {
            if (i + j) % 2 == 0 {
                continue;
            }
            let (a, b) = (i as f64 / n as f64, j as f64 / n as f64);
            let d = 1.0 / n as f64;
            let cell = [[a, b], [a + d, b], [a + d, b + d], [a, b + d]].map(|p| apply(&h, p));
            path(ctx, &cell, true);
            ctx.set_fill_style_str("#ff5fa2");
            ctx.fill();
        }
    }
    ctx.set_global_alpha(0.8);
    ctx.set_stroke_style_str("#ff5fa2");
    ctx.set_line_width(1.0 * px);
    for i in 0..=n {
        let t = i as f64 / n as f64;
        path(ctx, &[apply(&h, [t, 0.0]), apply(&h, [t, 1.0])], false);
        ctx.stroke();
        path(ctx, &[apply(&h, [0.0, t]), apply(&h, [1.0, t])], false);
        ctx.stroke();
    }
    ctx.set_global_alpha(1.0);
}

pub fn draw_timeline(
    canvas: &HtmlCanvasElement,
    project: &Project,
    frame: usize,
    selected: Option<u32>,
) {
    let Some(ctx) = ctx_of(canvas) else { return };
    let (cw, ch, dpr) = fit_canvas(canvas);
    let _ = ctx.set_transform(dpr, 0.0, 0.0, dpr, 0.0, 0.0);
    ctx.set_fill_style_str("#111317");
    ctx.fill_rect(0.0, 0.0, cw, ch);
    let n = project.footage.as_ref().map_or(0, |f| f.frames.len());
    if n == 0 {
        return;
    }
    let fw = cw / n as f64;
    let x_of = |f: usize| (f as f64 + 0.5) * fw;

    // Tracked-frame bars for every layer, selected layer drawn tallest.
    let lane_h = 6.0;
    for (li, layer) in project.layers.iter().enumerate() {
        let sel = selected == Some(layer.id);
        let y = 4.0 + li as f64 * (lane_h + 2.0);
        if y + lane_h > ch - 14.0 {
            break;
        }
        ctx.set_global_alpha(if sel { 1.0 } else { 0.45 });
        ctx.set_fill_style_str(&layer.color);
        for &f in layer.tracks.keys() {
            ctx.fill_rect((f as f64 * fw).floor(), y, fw.ceil().max(1.0) + 1.0, lane_h);
        }
        ctx.set_global_alpha(1.0);
        if sel {
            ctx.set_fill_style_str("#ffffff");
            let x = x_of(layer.ref_frame);
            path(
                &ctx,
                &[[x - 4.0, y - 2.0], [x + 4.0, y - 2.0], [x, y + lane_h]],
                true,
            );
            ctx.fill();
        }
    }

    // Ticks.
    ctx.set_fill_style_str("#7a8090");
    let step = (n as f64 / (cw / 60.0)).ceil().max(1.0) as usize;
    let step = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
        .into_iter()
        .find(|&s| s >= step)
        .unwrap_or(step);
    ctx.set_font("10px system-ui, sans-serif");
    ctx.set_text_align("left");
    let first = project.footage.as_ref().map_or(0, |f| f.first_frame);
    for f in (0..n).step_by(step) {
        let x = f as f64 * fw;
        ctx.fill_rect(x, ch - 12.0, 1.0, 12.0);
        let _ = ctx.fill_text(&(first + f as i64).to_string(), x + 3.0, ch - 2.0);
    }

    // Playhead.
    ctx.set_fill_style_str("#ffb020");
    ctx.fill_rect(x_of(frame) - 1.0, 0.0, 2.0, ch);
}

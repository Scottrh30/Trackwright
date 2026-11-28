//! Viewport and timeline rendering on 2D canvases.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use trackwright_core::geometry::{apply, homography_from_corners, Point};
use trackwright_core::{exclusion_polys, Problem, Project};
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
    /// Adjust Track: per-frame corner corrections.
    Adjust,
    /// Lens calibration: click points along edges that should be straight.
    LensLine,
}

pub struct Overlays {
    pub surface: bool,
    pub grid: bool,
    pub points: bool,
    pub shapes: bool,
    pub inserts: bool,
    /// Camera solve: 2D point tracks (trails).
    pub tracks: bool,
    /// Camera solve: solved 3D points projected into the frame.
    pub cloud: bool,
    /// Camera solve: ground grid of the oriented scene.
    pub ground: bool,
    /// Lens: distortion grid.
    pub lens_grid: bool,
}

pub struct Scene<'a> {
    pub project: &'a Project,
    pub frame: usize,
    pub selected: Option<u32>,
    pub sel_point: Option<usize>,
    pub tool: Tool,
    pub draft: &'a [Point],
    pub hover: Option<Point>,
    pub view: View,
    pub overlays: Overlays,
    /// Loaded insert images for this frame, by layer id.
    pub inserts: Vec<(u32, HtmlImageElement)>,
}

const IMAGE_CACHE: usize = 96;

thread_local! {
    static IMAGES: RefCell<(HashMap<String, HtmlImageElement>, VecDeque<String>)> = RefCell::new((HashMap::new(), VecDeque::new()));
    static LAST_GOOD: RefCell<Option<HtmlImageElement>> = const { RefCell::new(None) };
}

/// Get (and start loading) the image at `path`. `on_load` fires once it is ready.
pub fn image_for(path: &str, on_load: impl Fn() + 'static) -> Option<HtmlImageElement> {
    IMAGES.with(|cell| {
        let mut c = cell.borrow_mut();
        if let Some(img) = c.0.get(path) {
            return Some(img.clone());
        }
        let img = HtmlImageElement::new().ok()?;
        let cb = Closure::<dyn Fn()>::new(on_load);
        img.set_onload(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
        img.set_src(&crate::tauri::convert_file_src(path));
        c.0.insert(path.to_string(), img.clone());
        c.1.push_back(path.to_string());
        while c.1.len() > IMAGE_CACHE {
            if let Some(old) = c.1.pop_front() {
                c.0.remove(&old);
            }
        }
        Some(img)
    })
}

pub fn is_ready(img: &HtmlImageElement) -> bool {
    img.complete() && img.natural_width() > 0
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
            "Open a video or image sequence to start tracking (or drag it here)",
            cw / 2.0,
            ch / 2.0,
        );
        return;
    };

    let v = scene.view;
    let z = v.zoom;
    let _ = ctx.set_transform(dpr * z, 0.0, 0.0, dpr * z, dpr * v.pan[0], dpr * v.pan[1]);
    let (w, h) = (footage.width as f64, footage.height as f64);
    let img = match image.filter(is_ready) {
        Some(i) => {
            LAST_GOOD.with(|l| *l.borrow_mut() = Some(i.clone()));
            Some(i)
        }
        None => LAST_GOOD.with(|l| l.borrow().clone()),
    };
    if let Some(i) = img {
        // Proxies are smaller than the footage; always draw at footage size.
        let _ = ctx.draw_image_with_html_image_element_and_dw_and_dh(&i, 0.0, 0.0, w, h);
    } else {
        ctx.set_fill_style_str("#1b1d22");
        ctx.fill_rect(0.0, 0.0, w, h);
    }
    ctx.set_stroke_style_str("#2c2f36");
    ctx.set_line_width(1.0 / z);
    ctx.stroke_rect(0.0, 0.0, w, h);

    let px = 1.0 / z; // one screen pixel in image units

    // Inserts sit on the footage, under every overlay.
    if scene.overlays.inserts {
        for (id, ins_img) in &scene.inserts {
            let Some(layer) = scene.project.layer(*id) else {
                continue;
            };
            if !layer.visible || !is_ready(ins_img) {
                continue;
            }
            let (Some(corners), Some(ins)) = (layer.surface_at(scene.frame), layer.insert.as_ref())
            else {
                continue;
            };
            let mesh = layer.mesh_at(scene.frame);
            match (
                mesh.as_ref(),
                trackwright_core::mesh::grid_side(mesh.as_ref().map_or(0, |m| m.len())),
            ) {
                (Some(m), Some(n)) => {
                    // Mesh-tracked: the insert bends with the surface.
                    let (sw, sh) = (
                        ins_img.natural_width() as f64,
                        ins_img.natural_height() as f64,
                    );
                    let rect = [[0.0, 0.0], [sw, 0.0], [sw, sh], [0.0, sh]];
                    if let Some(hs) = homography_from_corners(&layer.surface, &rect) {
                        let uv: Vec<Point> = trackwright_core::mesh::grid(&layer.surface, n)
                            .iter()
                            .map(|p| apply(&hs, *p))
                            .collect();
                        draw_grid_warp(&ctx, ins_img, n, &uv, m, ins.opacity, px);
                    }
                }
                _ => draw_insert(&ctx, ins_img, &corners, ins.opacity, px),
            }
        }
    }

    // Areas the selected layer ignores: red hatching.
    if let Some(sel) = scene.selected.and_then(|id| scene.project.layer(id)) {
        for poly in exclusion_polys(&scene.project.layers, sel, scene.frame) {
            path(&ctx, &poly, true);
            ctx.save();
            ctx.clip();
            ctx.set_global_alpha(0.16);
            ctx.set_fill_style_str("#ff4d4d");
            ctx.fill_rect(0.0, 0.0, w, h);
            ctx.set_global_alpha(0.55);
            ctx.set_stroke_style_str("#ff4d4d");
            ctx.set_line_width(1.0 * px);
            let step = 10.0 * px;
            let mut x = -h;
            while x < w {
                path(&ctx, &[[x, 0.0], [x + h, h]], false);
                ctx.stroke();
                x += step;
            }
            ctx.restore();
            path(&ctx, &poly, true);
            ctx.set_stroke_style_str("#ff4d4d");
            ctx.set_line_width(1.5 * px);
            ctx.stroke();
        }
    }

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
            if layer.link.is_some() {
                let _ =
                    ctx.set_line_dash(&js_sys::Array::of2(&(5.0 * px).into(), &(3.0 * px).into()));
            }
            ctx.stroke();
            let _ = ctx.set_line_dash(&js_sys::Array::new());
            if selected && scene.tool == Tool::EditShape {
                ctx.set_line_width(1.5 * px);
                for (i, p) in shape.points.iter().enumerate() {
                    let on = scene.sel_point == Some(i);
                    handle(
                        &ctx,
                        *p,
                        if on { 6.5 } else { 5.0 } * px,
                        if on { &layer.color } else { "#15171c" },
                        &layer.color,
                    );
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

        // Mesh tracking: the grid as it bends; pupil: a ring and cross.
        if let Some(t) = track {
            if let (Some(m), true) = (
                layer.mesh_at(scene.frame),
                scene.overlays.surface && (selected || scene.overlays.grid),
            ) {
                if let Some(n) = trackwright_core::mesh::grid_side(m.len()) {
                    ctx.set_stroke_style_str(&layer.color);
                    ctx.set_global_alpha(0.75);
                    ctx.set_line_width(1.0 * px);
                    let s = n + 1;
                    for j in 0..s {
                        let row: Vec<Point> = (0..s).map(|i| m[j * s + i]).collect();
                        path(&ctx, &row, false);
                        ctx.stroke();
                        let col: Vec<Point> = (0..s).map(|i| m[i * s + j]).collect();
                        path(&ctx, &col, false);
                        ctx.stroke();
                    }
                    ctx.set_global_alpha(1.0);
                }
            }
            if let Some([x, y, r]) = t.pupil {
                ctx.set_stroke_style_str("#7dff6a");
                ctx.set_line_width(1.5 * px);
                ctx.begin_path();
                let _ = ctx.arc(x, y, r, 0.0, std::f64::consts::TAU);
                ctx.stroke();
                let k = (r * 0.35).max(4.0 * px);
                path(&ctx, &[[x - k, y], [x + k, y]], false);
                ctx.stroke();
                path(&ctx, &[[x, y - k], [x, y + k]], false);
                ctx.stroke();
            }
        }

        let adjusting = scene.tool == Tool::Adjust && selected;
        let show_surface =
            adjusting || (scene.overlays.surface && (selected || scene.tool == Tool::Surface));
        let adjust_key_here = layer.adjust_keys.contains_key(&scene.frame);
        if let Some(corners) = layer.surface_at(scene.frame) {
            if scene.overlays.grid && selected {
                draw_grid(&ctx, &corners, px);
            }
            if show_surface {
                if adjusting {
                    // Ghost of the raw tracked corners, to see how much is corrected.
                    if let Some(raw) = layer.tracked_surface_at(scene.frame) {
                        path(&ctx, &raw, true);
                        ctx.set_stroke_style_str("#ff7a45");
                        ctx.set_line_width(1.0 * px);
                        let _ = ctx.set_line_dash(&js_sys::Array::of2(
                            &(4.0 * px).into(),
                            &(3.0 * px).into(),
                        ));
                        ctx.stroke();
                        let _ = ctx.set_line_dash(&js_sys::Array::new());
                    }
                }
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
                    let fill = if adjusting && adjust_key_here {
                        "#ff7a45"
                    } else {
                        "#3fd0ff"
                    };
                    handle(&ctx, *c, 6.0 * px, fill, "#0d0e11");
                    ctx.set_fill_style_str("#0d0e11");
                    let _ = ctx.fill_text(labels[i], c[0], c[1] + 4.0 * px);
                }
            }
        }
    }

    draw_camera_overlays(&ctx, scene, w, h, px);

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

/// Affine transform (canvas a..f) mapping triangle `s` onto triangle `d`.
fn tri_affine(s: [Point; 3], d: [Point; 3]) -> Option<[f64; 6]> {
    let den = s[0][0] * (s[1][1] - s[2][1])
        + s[1][0] * (s[2][1] - s[0][1])
        + s[2][0] * (s[0][1] - s[1][1]);
    if den.abs() < 1e-12 {
        return None;
    }
    let row = |k: usize| {
        let a = (d[0][k] * (s[1][1] - s[2][1])
            + d[1][k] * (s[2][1] - s[0][1])
            + d[2][k] * (s[0][1] - s[1][1]))
            / den;
        let c = (d[0][k] * (s[2][0] - s[1][0])
            + d[1][k] * (s[0][0] - s[2][0])
            + d[2][k] * (s[1][0] - s[0][0]))
            / den;
        let e = (d[0][k] * (s[1][0] * s[2][1] - s[2][0] * s[1][1])
            + d[1][k] * (s[2][0] * s[0][1] - s[0][0] * s[2][1])
            + d[2][k] * (s[0][0] * s[1][1] - s[1][0] * s[0][1]))
            / den;
        (a, c, e)
    };
    let (a, c, e) = row(0);
    let (b, d_, f) = row(1);
    Some([a, b, c, d_, e, f])
}

/// Draw `img` perspective-warped onto `corners` (TL, TR, BR, BL) by subdividing the
/// quad into small affine triangles, which 2D canvas can draw exactly.
fn draw_insert(
    ctx: &CanvasRenderingContext2d,
    img: &HtmlImageElement,
    corners: &[Point; 4],
    opacity: f64,
    px: f64,
) {
    let unit = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let Some(h) = homography_from_corners(&unit, corners) else {
        return;
    };
    let (sw, sh) = (img.natural_width() as f64, img.natural_height() as f64);
    const N: usize = 12;
    let mut uv = Vec::with_capacity((N + 1) * (N + 1));
    let mut grid = Vec::with_capacity((N + 1) * (N + 1));
    for j in 0..=N {
        for i in 0..=N {
            let (u, v) = (i as f64 / N as f64, j as f64 / N as f64);
            uv.push([u * sw, v * sh]);
            grid.push(apply(&h, [u, v]));
        }
    }
    draw_grid_warp(ctx, img, N, &uv, &grid, opacity, px);
}

/// Draw `img` warped by a grid: vertex k of the `n` x `n` cell grid takes image point
/// `uv[k]` to canvas point `dst[k]`, one affine triangle at a time.
fn draw_grid_warp(
    ctx: &CanvasRenderingContext2d,
    img: &HtmlImageElement,
    n: usize,
    uv: &[Point],
    dst: &[Point],
    opacity: f64,
    px: f64,
) {
    let (sw, sh) = (img.natural_width() as f64, img.natural_height() as f64);
    if uv.len() != (n + 1) * (n + 1) || dst.len() != uv.len() {
        return;
    }
    ctx.save();
    ctx.set_global_alpha(opacity.clamp(0.0, 1.0));
    for t in trackwright_core::mesh::triangles(n) {
        let s = t.map(|k| uv[k]);
        let d = t.map(|k| dst[k]);
        let Some(m) = tri_affine(s, d) else { continue };
        // Grow the clip slightly so neighbouring triangles overlap (no seams).
        let c = [
            (d[0][0] + d[1][0] + d[2][0]) / 3.0,
            (d[0][1] + d[1][1] + d[2][1]) / 3.0,
        ];
        let grow = |p: Point| {
            let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
            let l = (dx * dx + dy * dy).sqrt().max(1e-9);
            [p[0] + dx / l * 0.8 * px, p[1] + dy / l * 0.8 * px]
        };
        ctx.save();
        path(ctx, &d.map(grow), true);
        ctx.clip();
        let _ = ctx.transform(m[0], m[1], m[2], m[3], m[4], m[5]);
        let x0 = (s.iter().map(|p| p[0]).fold(f64::MAX, f64::min) - 1.0).max(0.0);
        let y0 = (s.iter().map(|p| p[1]).fold(f64::MAX, f64::min) - 1.0).max(0.0);
        let x1 = (s.iter().map(|p| p[0]).fold(f64::MIN, f64::max) + 1.0).min(sw);
        let y1 = (s.iter().map(|p| p[1]).fold(f64::MIN, f64::max) + 1.0).min(sh);
        if x1 > x0 && y1 > y0 {
            let _ = ctx
                .draw_image_with_html_image_element_and_sw_and_sh_and_dx_and_dy_and_dw_and_dh(
                    img,
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                );
        }
        ctx.restore();
    }
    ctx.restore();
}

/// Checkerboard grid warped onto the surface to judge track quality.
fn draw_grid(ctx: &CanvasRenderingContext2d, corners: &[Point; 4], px: f64) {
    let unit = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let Some(h) = homography_from_corners(&unit, corners) else {
        return;
    };
    let n = 8;
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

fn diamond(ctx: &CanvasRenderingContext2d, x: f64, y: f64, r: f64, fill: &str) {
    path(ctx, &[[x, y - r], [x + r, y], [x, y + r], [x - r, y]], true);
    ctx.set_fill_style_str(fill);
    ctx.fill();
}

/// Timeline: layer lanes, keyframes, In/Out range, the selected layer's quality graph,
/// frame ticks and the playhead.
pub fn draw_timeline(
    canvas: &HtmlCanvasElement,
    project: &Project,
    frame: usize,
    selected: Option<u32>,
    in_pt: Option<usize>,
    out_pt: Option<usize>,
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
    let x0_of = |f: usize| (f as f64 * fw).floor();
    let tick_h = 14.0;

    // Lanes.
    let lane_h = 6.0;
    let mut lanes_bottom = 4.0;
    for (li, layer) in project.layers.iter().enumerate() {
        let sel = selected == Some(layer.id);
        let y = 4.0 + li as f64 * (lane_h + 2.0);
        if y + lane_h > 34.0 {
            break;
        }
        lanes_bottom = y + lane_h;
        ctx.set_global_alpha(if sel { 1.0 } else { 0.45 });
        ctx.set_fill_style_str(&layer.color);
        for &f in layer.tracks.keys() {
            ctx.fill_rect(x0_of(f), y, fw.ceil().max(1.0) + 1.0, lane_h);
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

    // Quality graph for the selected layer: inlier bars, error line, problem marks.
    let gy0 = lanes_bottom + 6.0;
    let gy1 = ch - tick_h - 2.0;
    if let Some(layer) = selected.and_then(|id| project.layer(id)) {
        let gh = gy1 - gy0;
        if gh > 12.0 {
            ctx.set_fill_style_str("#181b21");
            ctx.fill_rect(0.0, gy0, cw, gh);
            let max_inl = layer
                .tracks
                .values()
                .map(|t| t.inliers)
                .max()
                .unwrap_or(1)
                .max(1) as f64;
            ctx.set_fill_style_str("rgba(125,255,106,0.28)");
            for (&f, t) in &layer.tracks {
                let bh = gh * t.inliers as f64 / max_inl;
                ctx.fill_rect(x0_of(f), gy1 - bh, fw.ceil().max(1.0), bh);
            }
            // Error line: full graph height = 2 px of error. Dashed guide at 1 px.
            let y_err = |e: f64| gy1 - gh * (e / 2.0).min(1.0);
            ctx.set_stroke_style_str("rgba(255,255,255,0.18)");
            ctx.set_line_width(1.0);
            let _ = ctx.set_line_dash(&js_sys::Array::of2(&3.0.into(), &3.0.into()));
            path(&ctx, &[[0.0, y_err(1.0)], [cw, y_err(1.0)]], false);
            ctx.stroke();
            let _ = ctx.set_line_dash(&js_sys::Array::new());
            ctx.set_stroke_style_str("#ffd24a");
            ctx.set_line_width(1.3);
            let mut prev: Option<(usize, Point)> = None;
            ctx.begin_path();
            for (&f, t) in &layer.tracks {
                let p = [x_of(f), y_err(t.rms_error)];
                match prev {
                    Some((pf, _)) if pf + 1 == f => ctx.line_to(p[0], p[1]),
                    _ => ctx.move_to(p[0], p[1]),
                }
                prev = Some((f, p));
            }
            ctx.stroke();
            for &f in layer.tracks.keys() {
                if let Some(pr) = layer.problem_at(f) {
                    ctx.set_fill_style_str(match pr {
                        Problem::HighError => "#ff4d4d",
                        Problem::FewPoints => "#ff9a3c",
                        Problem::Retried => "#b28bff",
                        Problem::Predicted => "#4dd8ff",
                    });
                    ctx.fill_rect(x0_of(f), gy0, fw.ceil().max(1.5), 4.0);
                }
            }
            ctx.set_fill_style_str("#7a8090");
            ctx.set_font("9px system-ui, sans-serif");
            ctx.set_text_align("left");
            let _ = ctx.fill_text("error (line) / points (bars)", 4.0, gy0 + 13.0);
        }
        // Keyframes: shape keys (white) and Adjust Track keys (orange) on the lane.
        if let Some(li) = project.layers.iter().position(|l| l.id == layer.id) {
            let y = 4.0 + li as f64 * (lane_h + 2.0) + lane_h / 2.0;
            if y < 34.0 {
                for &f in layer.shape_keys.keys() {
                    diamond(&ctx, x_of(f), y, 5.5, "#ffffff");
                }
                for &f in layer.adjust_keys.keys() {
                    diamond(&ctx, x_of(f), y, 5.0, "#ff7a45");
                }
            }
        }
    }

    // In/Out: dim everything outside the range.
    if in_pt.is_some() || out_pt.is_some() {
        let a = in_pt.unwrap_or(0);
        let b = out_pt.unwrap_or(n - 1);
        ctx.set_fill_style_str("rgba(0,0,0,0.45)");
        ctx.fill_rect(0.0, 0.0, x0_of(a), ch);
        ctx.fill_rect(x0_of(b + 1), 0.0, cw - x0_of(b + 1), ch);
        ctx.set_fill_style_str("#46d17a");
        for (f, left) in [(in_pt, true), (out_pt, false)] {
            if let Some(f) = f {
                let x = if left { x0_of(f) } else { x0_of(f + 1) - 2.0 };
                ctx.fill_rect(x, 0.0, 2.0, ch - tick_h);
            }
        }
    }

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

    ctx.set_fill_style_str("#ffb020");
    ctx.fill_rect(x_of(frame) - 1.0, 0.0, 2.0, ch);
}

/// Camera solve and lens overlays: point trails, solved points, ground grid, lens lines and grid.
fn draw_camera_overlays(ctx: &CanvasRenderingContext2d, scene: &Scene, w: f64, h: f64, px: f64) {
    let p = scene.project;
    let f = scene.frame;
    let cam = &p.camera;

    // Lens lines (always shown while calibrating, otherwise with the lens grid).
    if scene.tool == Tool::LensLine || scene.overlays.lens_grid {
        for (_, line) in &p.lens_lines {
            path(ctx, line, false);
            ctx.set_stroke_style_str("#ffe14d");
            ctx.set_line_width(1.5 * px);
            ctx.stroke();
            for q in line {
                handle(ctx, *q, 3.5 * px, "#ffe14d", "#0d0e11");
            }
            // Where the line would be after undistorting: straight chord through the ends.
            if !p.lens.is_identity() && line.len() >= 2 {
                let und: Vec<Point> = line.iter().map(|q| p.lens.undistort(*q, w, h)).collect();
                path(ctx, &und, false);
                ctx.set_stroke_style_str("rgba(120,255,160,0.85)");
                ctx.set_line_width(1.0 * px);
                ctx.stroke();
            }
        }
    }
    if scene.overlays.lens_grid && !p.lens.is_identity() {
        // A straight grid in the undistorted image, drawn bent as the lens sees it.
        ctx.set_stroke_style_str("rgba(255,225,77,0.45)");
        ctx.set_line_width(1.0 * px);
        let n = 12;
        for i in 0..=n {
            let t = i as f64 / n as f64;
            let v: Vec<Point> = (0..=40)
                .map(|k| p.lens.distort([t * w, k as f64 / 40.0 * h], w, h))
                .collect();
            path(ctx, &v, false);
            ctx.stroke();
            let hz: Vec<Point> = (0..=40)
                .map(|k| p.lens.distort([k as f64 / 40.0 * w, t * h], w, h))
                .collect();
            path(ctx, &hz, false);
            ctx.stroke();
        }
    }

    // Point-track trails (last few frames) for the camera solve.
    if scene.overlays.tracks && !cam.tracks.is_empty() {
        let solved: std::collections::HashMap<u32, f64> = cam
            .solve
            .as_ref()
            .map(|s| s.points.iter().map(|q| (q.id, q.error)).collect())
            .unwrap_or_default();
        for t in &cam.tracks {
            let Some(now) = t.points.get(&f) else {
                continue;
            };
            let trail: Vec<Point> = t
                .points
                .range(f.saturating_sub(8)..=f)
                .map(|(_, q)| *q)
                .collect();
            let color = match solved.get(&t.id) {
                Some(e) if *e < 1.0 => "#5cff8a",
                Some(e) if *e < 3.0 => "#ffd24a",
                Some(_) => "#ff5f5f",
                None => "#8fb3ff",
            };
            if trail.len() > 1 {
                path(ctx, &trail, false);
                ctx.set_stroke_style_str(color);
                ctx.set_line_width(1.0 * px);
                ctx.stroke();
            }
            ctx.set_fill_style_str(color);
            ctx.fill_rect(now[0] - 1.5 * px, now[1] - 1.5 * px, 3.0 * px, 3.0 * px);
        }
    }

    let Some(solve) = cam.solve.as_ref() else {
        return;
    };
    if !solve.poses.contains_key(&f) {
        return;
    }
    // Solved 3D points projected back: small crosses.
    if scene.overlays.cloud {
        ctx.set_stroke_style_str("rgba(255,80,220,0.9)");
        ctx.set_line_width(1.0 * px);
        for sp in &solve.points {
            if let Some(q) = solve.project(f, sp.pos) {
                if q[0] < 0.0 || q[1] < 0.0 || q[0] > w || q[1] > h {
                    continue;
                }
                let r = 3.0 * px;
                path(ctx, &[[q[0] - r, q[1]], [q[0] + r, q[1]]], false);
                ctx.stroke();
                path(ctx, &[[q[0], q[1] - r], [q[0], q[1] + r]], false);
                ctx.stroke();
            }
        }
    }
    // Ground grid on the oriented scene's XZ plane (y = 0), with X red and Z blue axes.
    // Only meaningful once the scene is oriented (Set Ground); before that y = 0 runs through the camera.
    if scene.overlays.ground && solve.world != trackwright_core::sfm::World::default() {
        let n = 10i32;
        let step = 0.1;
        let seg = |a: [f64; 3], b: [f64; 3]| -> Vec<Point> {
            (0..=20)
                .filter_map(|k| {
                    let t = k as f64 / 20.0;
                    solve.project_user(
                        f,
                        [
                            a[0] + (b[0] - a[0]) * t,
                            a[1] + (b[1] - a[1]) * t,
                            a[2] + (b[2] - a[2]) * t,
                        ],
                    )
                })
                .collect()
        };
        for i in -n..=n {
            let c = i as f64 * step;
            let e = n as f64 * step;
            ctx.set_stroke_style_str(if i == 0 {
                "rgba(255,255,255,0.7)"
            } else {
                "rgba(140,200,255,0.35)"
            });
            ctx.set_line_width(1.0 * px);
            for line in [
                seg([c, 0.0, -e], [c, 0.0, e]),
                seg([-e, 0.0, c], [e, 0.0, c]),
            ] {
                if line.len() > 1 {
                    path(ctx, &line, false);
                    ctx.stroke();
                }
            }
        }
        for (end, color) in [
            ([0.3, 0.0, 0.0], "#ff4d4d"),
            ([0.0, 0.3, 0.0], "#5cff8a"),
            ([0.0, 0.0, 0.3], "#4d8dff"),
        ] {
            let line = seg([0.0, 0.0, 0.0], end);
            if line.len() > 1 {
                path(ctx, &line, false);
                ctx.set_stroke_style_str(color);
                ctx.set_line_width(2.5 * px);
                ctx.stroke();
            }
        }
    }
}

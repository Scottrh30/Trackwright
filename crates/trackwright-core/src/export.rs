//! Exporters for track data.

use crate::geometry::{Mat3, Point};
use crate::project::{Footage, Layer};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// Corner pin + transform track for the Reelforge video editor.
    Reelforge,
    /// Planar track for Forma3D (plane in normalised image space).
    Forma3d,
    /// Planar track for Tidefire.
    Tidefire,
    /// Nuke CornerPin2D node.
    NukeCornerPin,
    /// After Effects Corner Pin keyframe data (paste into AE).
    AfterEffectsCornerPin,
    /// Generic Trackwright JSON.
    Json,
    Csv,
}

impl ExportFormat {
    pub const ALL: [ExportFormat; 7] = [
        ExportFormat::Reelforge,
        ExportFormat::Forma3d,
        ExportFormat::Tidefire,
        ExportFormat::NukeCornerPin,
        ExportFormat::AfterEffectsCornerPin,
        ExportFormat::Json,
        ExportFormat::Csv,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ExportFormat::Reelforge => "Reelforge track (.rftrack.json)",
            ExportFormat::Forma3d => "Forma3D planar track (.f3dtrack.json)",
            ExportFormat::Tidefire => "Tidefire planar track (.tftrack.json)",
            ExportFormat::NukeCornerPin => "Nuke CornerPin2D (.nk)",
            ExportFormat::AfterEffectsCornerPin => "After Effects Corner Pin (.txt)",
            ExportFormat::Json => "Trackwright JSON (.json)",
            ExportFormat::Csv => "CSV corners (.csv)",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Reelforge => "rftrack.json",
            ExportFormat::Forma3d => "f3dtrack.json",
            ExportFormat::Tidefire => "tftrack.json",
            ExportFormat::NukeCornerPin => "nk",
            ExportFormat::AfterEffectsCornerPin => "txt",
            ExportFormat::Json => "json",
            ExportFormat::Csv => "csv",
        }
    }
}

/// Tracked frames of a layer in order with their surface corners.
fn frames(layer: &Layer) -> Vec<(usize, [Point; 4], Mat3, u32, f64)> {
    layer
        .tracks
        .iter()
        .map(|(&f, t)| {
            (
                f,
                layer.surface_at(f).unwrap(),
                layer.final_h_at(f).unwrap_or(t.h_from_ref),
                t.inliers,
                t.rms_error,
            )
        })
        .collect()
}

fn dist(a: Point, b: Point) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Position / rotation / scale of the surface relative to the reference surface.
pub fn decompose(reference: &[Point; 4], c: &[Point; 4]) -> (Point, f64, Point) {
    let centre = [
        (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4.0,
        (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4.0,
    ];
    let ang = |a: Point, b: Point| (b[1] - a[1]).atan2(b[0] - a[0]);
    let mut rot = (ang(c[0], c[1]) - ang(reference[0], reference[1])).to_degrees();
    if rot > 180.0 {
        rot -= 360.0;
    } else if rot < -180.0 {
        rot += 360.0;
    }
    let sx = dist(c[0], c[1]) / dist(reference[0], reference[1]).max(1e-9);
    let sy = dist(c[0], c[3]) / dist(reference[0], reference[3]).max(1e-9);
    (centre, rot, [sx, sy])
}

fn corners_obj(c: &[Point; 4]) -> serde_json::Value {
    json!({ "tl": c[0], "tr": c[1], "br": c[2], "bl": c[3] })
}

fn r4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

fn r4p(p: Point) -> Point {
    [r4(p[0]), r4(p[1])]
}

pub fn export(format: ExportFormat, footage: &Footage, layer: &Layer) -> String {
    match format {
        ExportFormat::Reelforge => reelforge(footage, layer),
        ExportFormat::Forma3d => planar_3d(footage, layer, "forma3d"),
        ExportFormat::Tidefire => planar_3d(footage, layer, "tidefire"),
        ExportFormat::NukeCornerPin => nuke(footage, layer),
        ExportFormat::AfterEffectsCornerPin => after_effects(footage, layer),
        ExportFormat::Json => generic_json(footage, layer),
        ExportFormat::Csv => csv(footage, layer),
    }
}

fn source_json(f: &Footage) -> serde_json::Value {
    json!({ "path": f.source, "width": f.width, "height": f.height, "fps": f.fps, "first_frame": f.first_frame, "frame_count": f.frames.len() })
}

fn reelforge(f: &Footage, l: &Layer) -> String {
    let frames: Vec<_> = frames(l)
        .into_iter()
        .map(|(i, c, h, inl, rms)| {
            let (pos, rot, scale) = decompose(&l.surface, &c);
            json!({
                "frame": f.frame_number(i),
                "time": if f.fps > 0.0 { r4(i as f64 / f.fps) } else { 0.0 },
                "corners": corners_obj(&c.map(r4p)),
                "transform": { "position": r4p(pos), "rotation_deg": r4(rot), "scale": r4p(scale) },
                "homography": h,
                "confidence": { "inliers": inl, "rms_px": r4(rms) },
            })
        })
        .collect();
    serde_json::to_string_pretty(&json!({
        "format": "trackwright.reelforge.v1",
        "generator": concat!("Trackwright ", env!("CARGO_PKG_VERSION")),
        "source": source_json(f),
        "layer": { "name": l.name, "reference_frame": f.frame_number(l.ref_frame), "reference_corners": corners_obj(&l.surface.map(r4p)) },
        "coordinate_space": "pixels, origin top-left, y down",
        "frames": frames,
    }))
    .unwrap()
}

/// For 3D apps: corners in normalised device coords (-1..1, y up) so the plane
/// can be placed on a camera image plane regardless of resolution.
fn planar_3d(f: &Footage, l: &Layer, target: &str) -> String {
    let (w, h) = (f.width.max(1) as f64, f.height.max(1) as f64);
    let ndc = |p: Point| r4p([p[0] / w * 2.0 - 1.0, 1.0 - p[1] / h * 2.0]);
    let frames: Vec<_> = frames(l)
        .into_iter()
        .map(|(i, c, hm, inl, rms)| {
            json!({
                "frame": f.frame_number(i),
                "corners_px": c.map(r4p),
                "corners_ndc": c.map(ndc),
                "homography": hm,
                "inliers": inl,
                "rms_px": r4(rms),
            })
        })
        .collect();
    serde_json::to_string_pretty(&json!({
        "format": "trackwright.planar.v1",
        "target": target,
        "generator": concat!("Trackwright ", env!("CARGO_PKG_VERSION")),
        "source": source_json(f),
        "aspect": w / h,
        "plane": {
            "name": l.name,
            "reference_frame": f.frame_number(l.ref_frame),
            "corners_px": l.surface.map(r4p),
            "corner_order": ["tl", "tr", "br", "bl"],
        },
        "camera": serde_json::Value::Null,
        "frames": frames,
    }))
    .unwrap()
}

fn generic_json(f: &Footage, l: &Layer) -> String {
    serde_json::to_string_pretty(&json!({
        "format": "trackwright.track.v1",
        "source": source_json(f),
        "layer": l,
    }))
    .unwrap()
}

fn csv(f: &Footage, l: &Layer) -> String {
    let mut out = String::from("frame,tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y,inliers,rms_px\n");
    for (i, c, _, inl, rms) in frames(l) {
        out += &format!(
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4}\n",
            f.frame_number(i),
            c[0][0],
            c[0][1],
            c[1][0],
            c[1][1],
            c[2][0],
            c[2][1],
            c[3][0],
            c[3][1],
            inl,
            rms
        );
    }
    out
}

fn nuke_name(s: &str) -> String {
    let n: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("Trackwright_{n}")
}

fn nuke(f: &Footage, l: &Layer) -> String {
    let h = f.height as f64;
    let fr = frames(l);
    // Nuke: to1 = bottom-left, to2 = bottom-right, to3 = top-right, to4 = top-left; y is up.
    let order = [3usize, 2, 1, 0];
    let curve = |corner: usize, axis: usize| {
        let mut s = String::from("{curve");
        for (i, c, ..) in &fr {
            let v = if axis == 0 {
                c[corner][0]
            } else {
                h - c[corner][1]
            };
            s += &format!(" x{} {:.4}", f.frame_number(*i), v);
        }
        s + "}"
    };
    let mut out = String::from("set cut_paste_input [stack 0]\nversion 13.0 v1\nCornerPin2D {\n");
    for (k, &corner) in order.iter().enumerate() {
        out += &format!(
            " to{} {{{} {}}}\n",
            k + 1,
            curve(corner, 0),
            curve(corner, 1)
        );
    }
    out += " invert false\n";
    for (k, &corner) in order.iter().enumerate() {
        let p = l.surface[corner];
        out += &format!(" from{} {{{:.4} {:.4}}}\n", k + 1, p[0], h - p[1]);
    }
    out += &format!(" name {}\n selected true\n}}\n", nuke_name(&l.name));
    out
}

fn after_effects(f: &Footage, l: &Layer) -> String {
    let fr = frames(l);
    let fps = if f.fps > 0.0 { f.fps } else { 24.0 };
    let mut out = format!(
        "Adobe After Effects 8.0 Keyframe Data\r\n\r\n\tUnits Per Second\t{}\r\n\tSource Width\t{}\r\n\tSource Height\t{}\r\n\tSource Pixel Aspect Ratio\t1\r\n\tComp Pixel Aspect Ratio\t1\r\n\r\n",
        fps, f.width, f.height
    );
    // AE Corner Pin params: 0001 Upper Left, 0002 Upper Right, 0003 Lower Left, 0004 Lower Right.
    let order = [0usize, 1, 3, 2];
    for (k, &corner) in order.iter().enumerate() {
        out += &format!(
            "Effects\tCorner Pin #1\tCorner Pin-{:04}\r\n\tFrame\tX pixels\tY pixels\t\r\n",
            k + 1
        );
        for (i, c, ..) in &fr {
            // AE keyframe times are relative to the layer start.
            out += &format!("\t{}\t{:.4}\t{:.4}\t\r\n", i, c[corner][0], c[corner][1]);
        }
        out += "\r\n";
    }
    out += "\r\nEnd of Keyframe Data\r\n";
    out
}

/// Safe file stem for a layer export.
pub fn file_name(footage: &Footage, layer: &Layer, format: ExportFormat) -> String {
    let stem = std::path::Path::new(&footage.source)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "footage".into());
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
    format!(
        "{}_{}.{}",
        clean(&stem),
        clean(&layer.name),
        format.extension()
    )
}

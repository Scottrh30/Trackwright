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
    /// Fusion / DaVinci Resolve (Fusion page) CornerPositioner, paste-able .setting.
    FusionCornerPin,
    /// Blender Python script: four corner tracks (+ plane track) in a Movie Clip.
    Blender,
    /// Shake 4.x four-point tracker file, imported by Silhouette (and Shake).
    SilhouetteShake,
    /// Animated mask (roto shape per frame) for Reelforge.
    ReelforgeMask,
    /// HitFilm composite shot: a plane with a Quad Warp corner pin keyed per frame.
    HitFilm,
    /// Generic Trackwright JSON.
    Json,
    Csv,
}

impl ExportFormat {
    pub const ALL: [ExportFormat; 12] = [
        ExportFormat::Reelforge,
        ExportFormat::Forma3d,
        ExportFormat::Tidefire,
        ExportFormat::NukeCornerPin,
        ExportFormat::AfterEffectsCornerPin,
        ExportFormat::FusionCornerPin,
        ExportFormat::Blender,
        ExportFormat::SilhouetteShake,
        ExportFormat::HitFilm,
        ExportFormat::ReelforgeMask,
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
            ExportFormat::FusionCornerPin => "Fusion / DaVinci Resolve CornerPositioner (.setting)",
            ExportFormat::Blender => "Blender plane-track script (.py)",
            ExportFormat::SilhouetteShake => "Silhouette / Shake 4-point tracker (.txt)",
            ExportFormat::HitFilm => "HitFilm composite shot, Quad Warp corner pin (.hfcs)",
            ExportFormat::ReelforgeMask => "Reelforge animated mask (.rfmask.json)",
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
            ExportFormat::FusionCornerPin => "setting",
            ExportFormat::Blender => "py",
            ExportFormat::SilhouetteShake => "shake.txt",
            ExportFormat::HitFilm => "hfcs",
            ExportFormat::ReelforgeMask => "rfmask.json",
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
        ExportFormat::FusionCornerPin => fusion(footage, layer),
        ExportFormat::Blender => blender(footage, layer),
        ExportFormat::SilhouetteShake => shake_trackers(footage, layer),
        ExportFormat::HitFilm => hitfilm(footage, layer),
        ExportFormat::ReelforgeMask => reelforge_mask(footage, layer),
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
                "pupil": pupil_json(l, i),
                "mesh": mesh_json(l, i),
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
                "pupil": pupil_json(l, i),
                "mesh": mesh_json(l, i),
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

/// The pupil on frame `i` (eye layers with Find pupil on): centre and radius, pixels.
fn pupil_json(l: &Layer, i: usize) -> serde_json::Value {
    match l.tracks.get(&i).and_then(|t| t.pupil) {
        Some(p) => json!({ "x": r4(p[0]), "y": r4(p[1]), "radius": r4(p[2]) }),
        None => serde_json::Value::Null,
    }
}

/// Mesh-tracked layers: the grid on frame `i` (cells across, vertices row by row).
fn mesh_json(l: &Layer, i: usize) -> serde_json::Value {
    let Some(m) = l.mesh_at(i) else {
        return serde_json::Value::Null;
    };
    match crate::mesh::grid_side(m.len()) {
        Some(n) => json!({ "cells": n, "points": m.iter().map(|p| r4p(*p)).collect::<Vec<_>>() }),
        None => serde_json::Value::Null,
    }
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

fn ident(s: &str) -> String {
    let n: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("Trackwright_{n}")
}

/// Fusion comp snippet (copy the file contents and paste into the Fusion node view,
/// or File > Import). Points are normalised 0..1 with the origin at bottom-left; time
/// is the clip frame index starting at 0, as on Resolve's Fusion page.
fn fusion(f: &Footage, l: &Layer) -> String {
    let (w, h) = (f.width.max(1) as f64, f.height.max(1) as f64);
    let fr = frames(l);
    let name = ident(&l.name);
    // Fusion inputs, in our corner order TL, TR, BR, BL.
    let inputs = ["TopLeft", "TopRight", "BottomRight", "BottomLeft"];
    let mut out = String::from("{\n\tTools = ordered() {\n");
    out += &format!("\t\t{name} = CornerPositioner {{\n\t\t\tInputs = {{\n");
    for inp in inputs {
        out += &format!(
            "\t\t\t\t{inp} = Input {{ SourceOp = \"{name}{inp}\", Source = \"Value\", }},\n"
        );
    }
    out += "\t\t\t},\n\t\t\tViewInfo = OperatorInfo { Pos = { 0, 0 } },\n\t\t},\n";
    for (ci, inp) in inputs.iter().enumerate() {
        out += &format!(
            "\t\t{name}{inp} = XYPath {{\n\t\t\tShowKeyPoints = false,\n\t\t\tDrawMode = \"ModifyOnly\",\n\t\t\tInputs = {{\n\t\t\t\tX = Input {{ SourceOp = \"{name}{inp}X\", Source = \"Value\", }},\n\t\t\t\tY = Input {{ SourceOp = \"{name}{inp}Y\", Source = \"Value\", }},\n\t\t\t}},\n\t\t}},\n"
        );
        for axis in 0..2 {
            let suffix = if axis == 0 { "X" } else { "Y" };
            out += &format!("\t\t{name}{inp}{suffix} = BezierSpline {{\n\t\t\tSplineColor = {{ Red = 255, Green = 176, Blue = 32 }},\n\t\t\tNameSet = true,\n\t\t\tKeyFrames = {{\n");
            for (i, c, ..) in &fr {
                let v = if axis == 0 {
                    c[ci][0] / w
                } else {
                    1.0 - c[ci][1] / h
                };
                out += &format!("\t\t\t\t[{i}] = {{ {v:.6}, Flags = {{ Linear = true }} }},\n");
            }
            out += "\t\t\t}\n\t\t},\n";
        }
    }
    out += "\t},\n\tActiveTool = \"";
    out += &name;
    out += "\"\n}\n";
    out
}

/// Python for Blender's Text Editor (Run Script): loads the footage as a Movie Clip,
/// adds four corner tracks and, when possible, a plane track from them.
fn blender(f: &Footage, l: &Layer) -> String {
    let fr = frames(l);
    let video_exts = [
        "mp4", "mov", "mkv", "avi", "webm", "m4v", "mxf", "mpg", "mpeg", "wmv",
    ];
    let is_video = std::path::Path::new(&f.source)
        .extension()
        .map(|e| video_exts.contains(&e.to_string_lossy().to_lowercase().as_str()))
        .unwrap_or(false);
    let clip_path = if is_video {
        f.source.clone()
    } else {
        f.frames.first().cloned().unwrap_or_default()
    };
    let mut corners = String::new();
    for (i, c, ..) in &fr {
        corners += &format!(
            "    {}: (({:.4}, {:.4}), ({:.4}, {:.4}), ({:.4}, {:.4}), ({:.4}, {:.4})),\n",
            i + 1,
            c[0][0],
            c[0][1],
            c[1][0],
            c[1][1],
            c[2][0],
            c[2][1],
            c[3][0],
            c[3][1]
        );
    }
    format!(
        r#"# Trackwright -> Blender
# Open Blender's Scripting workspace, open this file in the Text Editor and press Run Script.
# It loads the footage as a Movie Clip and adds four corner tracks for layer "{layer}".
# A plane track is created from them when a Movie Clip Editor is open; otherwise select the
# four tracks in the Movie Clip Editor and choose Track > Create Plane Track.
import bpy

FOOTAGE = r"{path}"
WIDTH, HEIGHT = {w}, {h}
LAYER = "{ident}"
# clip frame (1 = first frame): (top-left, top-right, bottom-right, bottom-left), pixels, y down
CORNERS = {{
{corners}}}
NAMES = ("TL", "TR", "BR", "BL")


def main():
    clip = bpy.data.movieclips.load(FOOTAGE, check_existing=True)
    tracks_coll = clip.tracking.objects.active.tracks
    first = min(CORNERS)
    tracks = []
    for i, name in enumerate(NAMES):
        track = tracks_coll.new(name=LAYER + "_" + name, frame=first)
        for frame, c in CORNERS.items():
            x, y = c[i]
            track.markers.insert_frame(frame, co=(x / WIDTH, 1.0 - y / HEIGHT))
        tracks.append(track)
    scene = bpy.context.scene
    scene.active_clip = clip
    scene.frame_start = first
    scene.frame_end = max(CORNERS)
    for t in tracks_coll:
        t.select = t in tracks
    made_plane = False
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "CLIP_EDITOR":
                area.spaces.active.clip = clip
                region = next(r for r in area.regions if r.type == "WINDOW")
                with bpy.context.temp_override(window=window, area=area, region=region):
                    try:
                        bpy.ops.clip.create_plane_track()
                        made_plane = True
                    except RuntimeError as err:
                        print("Could not create the plane track automatically:", err)
                break
    print("Trackwright: added", len(tracks), "tracks to", clip.name,
          "and a plane track" if made_plane else "(select them and use Track > Create Plane Track)")


main()
"#,
        layer = l.name.replace('"', "'"),
        path = clip_path.replace('"', "\\\""),
        w = f.width,
        h = f.height,
        ident = ident(&l.name),
        corners = corners,
    )
}

/// Shake 4.x tracker text file: four trackers (one per corner), origin bottom-left,
/// frames numbered from 1. Silhouette imports this as point trackers.
fn shake_trackers(f: &Footage, l: &Layer) -> String {
    let h = f.height as f64;
    let fr = frames(l);
    let names = ["1_TopLeft", "2_TopRight", "3_BottomRight", "4_BottomLeft"];
    let mut out = String::new();
    for (ci, name) in names.iter().enumerate() {
        out += &format!(
            "TrackName {}_{}\n   Frame             X             Y   Correlation\n",
            ident(&l.name),
            name
        );
        for (i, c, ..) in &fr {
            out += &format!(
                "   {:.2}   {:.3}   {:.3}   {:.3}\n",
                (*i + 1) as f64,
                c[ci][0],
                h - c[ci][1],
                1.0
            );
        }
        out += "\n";
    }
    out
}

/// Roto shape per frame: control points, the drawn outline (polygon) and the feather.
fn reelforge_mask(f: &Footage, l: &Layer) -> String {
    let frames: Vec<_> = l
        .tracks
        .keys()
        .filter_map(|&i| {
            let shape = l.shape_at(i)?;
            Some(json!({
                "frame": f.frame_number(i),
                "time": if f.fps > 0.0 { r4(i as f64 / f.fps) } else { 0.0 },
                "points": shape.points.iter().map(|p| r4p(*p)).collect::<Vec<_>>(),
                "polygon": shape.flatten(10).iter().map(|p| r4p(*p)).collect::<Vec<_>>(),
            }))
        })
        .collect();
    serde_json::to_string_pretty(&json!({
        "format": "trackwright.mask.v1",
        "generator": concat!("Trackwright ", env!("CARGO_PKG_VERSION")),
        "source": source_json(f),
        "mask": { "name": l.name, "smooth": l.shape.smooth, "feather_px": l.feather },
        "coordinate_space": "pixels, origin top-left, y down",
        "frames": frames,
    }))
    .unwrap()
}

/// A GUID-shaped id, stable for the same seed text (HitFilm only needs them unique
/// within the file).
fn guid(seed: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut b = [0u8; 16];
    for (i, byte) in seed.bytes().chain(std::iter::repeat_n(0u8, 16)).enumerate() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
        b[i % 16] ^= (h >> 24) as u8;
        if i > seed.len() + 16 {
            break;
        }
    }
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let x: String = b.iter().map(|v| format!("{v:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &x[0..8], &x[8..12], &x[12..16], &x[16..20], &x[20..32])
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Layer settings HitFilm writes for a plane (lighting, shadows, materials), copied
/// from a composite shot saved by HitFilm Pro.
const HITFILM_LAYER_DEFAULTS: &str = r#"<Prop CanInterpT="1" Type="1" Spatial="0"><Name>aoSampleRadius</Name><Default V="4"><i>15</i></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>castShadowsIfLayerDisabled</Name><Default V="4"><b>0</b></Default></Prop><Prop CanInterpT="1" Type="0" Spatial="0"><Name>audioLevel</Name><Default V="4"><fl>0</fl></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>receiveReflections</Name><Default V="4"><b>0</b></Default></Prop><Prop CanInterpT="1" Type="1" Spatial="0"><Name>speed</Name><Default V="4"><db>1</db></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>receiveShadows</Name><Default V="4"><b>1</b></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>billboardedLights</Name><Default V="4"><b>0</b></Default></Prop><Prop CanInterpT="1" Type="0" Spatial="0"><Name>shadowColor</Name><Default V="4"><cl G="0" B="0" R="0" A="1"/></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>castAmbientOcclusion</Name><Default V="4"><b>0</b></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>castShadows</Name><Default V="4"><b>0</b></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>billboardedShadows</Name><Default V="4"><b>0</b></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>receiveAmbientOcclusion</Name><Default V="4"><b>0</b></Default></Prop><Prop CanInterpT="1" Type="0" Spatial="0"><Name>matAmbient</Name><Default V="4"><fl>100</fl></Default></Prop><Prop CanInterpT="1" Type="0" Spatial="0"><Name>matDiffuse</Name><Default V="4"><fl>50</fl></Default></Prop><Prop CanInterpT="1" Type="1" Spatial="0"><Name>aoDepthFallOff</Name><Default V="4"><fl>10</fl></Default></Prop><Prop CanInterpT="1" Type="0" Spatial="0"><Name>matSpecular</Name><Default V="4"><fl>50</fl></Default></Prop><Prop CanInterpT="1" Type="0" Spatial="0"><Name>matShininess</Name><Default V="4"><fl>100</fl></Default></Prop><Prop CanInterpT="1" Type="0" Spatial="0"><Name>matEmissiveColor</Name><Default V="4"><cl G="0" B="0" R="0" A="1"/></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>castReflections</Name><Default V="4"><b>0</b></Default></Prop><Prop CanInterpT="1" Type="1" Spatial="0"><Name>aoDepthScale</Name><Default V="4"><fl>10</fl></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>castReflectionsIfLayerDisabled</Name><Default V="4"><b>0</b></Default></Prop><Prop CanInterpT="1" Type="0" Spatial="0"><Name>specularReflectivity</Name><Default V="4"><fl>100</fl></Default></Prop><Prop CanInterpT="1" Type="1" Spatial="0"><Name>aoSamples</Name><Default V="4"><i>4</i></Default></Prop><Prop CanInterpT="1" Type="1" Spatial="0"><Name>aoIntensity</Name><Default V="4"><fl>50</fl></Default></Prop><Prop CanInterpT="1" Type="1" Spatial="0"><Name>aoBias</Name><Default V="4"><fl>0.01</fl></Default></Prop><Prop CanInterpT="1" Type="1" Spatial="0"><Name>aoBlurRadius</Name><Default V="4"><i>5</i></Default></Prop>"#;

/// The message slots HitFilm keeps on every effect.
const HITFILM_EFFECT_MESSAGES: &str = r#"<Prop CanInterpT="0" Type="1" Spatial="0"><Name>HF_Persistant_Message_Error</Name><Default V="4"><st></st></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>HF_Persistant_Message_Warning</Name><Default V="4"><st></st></Default></Prop><Prop CanInterpT="0" Type="1" Spatial="0"><Name>HF_Persistant_Message_Info</Name><Default V="4"><st></st></Default></Prop>"#;

/// HitFilm composite shot (.hfcs), the format HitFilm's Import › Composite Shot
/// reads (and Mocha HitFilm writes): a composite shot the size and frame rate of the
/// footage holding a plane with HitFilm's Quad Warp (corner pin) effect, its four
/// corners keyed on every tracked frame. Layout taken from a file saved by HitFilm
/// Pro 2025 (18.0): key times are milliseconds from the start of the shot, corner
/// positions are layer pixels from the centre with Y up.
fn hitfilm(f: &Footage, l: &Layer) -> String {
    let fps = if f.fps > 0.0 { f.fps } else { 24.0 };
    let (w, h) = (f.width.max(1) as f64, f.height.max(1) as f64);
    let frames_total = f.frames.len().max(1);
    let name = xml_escape(&l.name);
    let seed = format!("{}#{}#{}", f.source, l.id, l.name);
    let (asset_id, comp_id, layer_id, effect_id, view_id) = (
        guid(&format!("{seed}/asset")),
        guid(&format!("{seed}/comp")),
        guid(&format!("{seed}/layer")),
        guid(&format!("{seed}/effect")),
        guid(&format!("{seed}/view")),
    );
    let fr = frames(l);
    // Pixels (origin top-left, y down) -> HitFilm layer space (origin centre, y up).
    let hf = |p: Point| [r4(p[0] - w / 2.0), r4(h / 2.0 - p[1])];
    let num = |v: f64| {
        if v == v.trunc() && v.abs() < 1e15 {
            format!("{}", v as i64)
        } else {
            format!("{v}")
        }
    };
    let ms = |i: usize| ((i as f64) / fps * 1000.0).round() as i64;
    let corner_prop = |prop: &str, corner: usize, default: Point| -> String {
        let d = hf(default);
        let mut o = format!(
            "<Prop CanInterpT=\"1\" Type=\"0\" Spatial=\"0\"><Name>{prop}</Name><Default V=\"4\"><p2 X=\"{}\" Y=\"{}\"/></Default><Static V=\"4\"><p2 X=\"{}\" Y=\"{}\"/></Static><Animation>",
            num(d[0]), num(d[1]), num(d[0]), num(d[1])
        );
        for (i, c, ..) in &fr {
            let p = hf(c[corner]);
            o += &format!(
                "<Key Ti=\"{}\" V=\"5\" Tp=\"1\" STp=\"2\"><TLk>0</TLk><SLk>0</SLk><Vl V=\"4\"><p2 X=\"{}\" Y=\"{}\"/></Vl></Key>",
                ms(*i),
                num(p[0]),
                num(p[1])
            );
        }
        o += "</Animation></Prop>";
        o
    };
    let use_layer = |prop: &str| {
        format!("<Prop CanInterpT=\"0\" Type=\"1\" Spatial=\"0\"><Name>{prop}</Name><Default V=\"4\"><id>00000000-0000-0000-0000-000000000000</id></Default></Prop>")
    };
    // The plane's own corners: the whole frame.
    let full = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let mut quad = String::new();
    quad += &use_layer("topLeftUseLayer");
    quad += &corner_prop("topLeftPoint", 0, full[0]);
    quad += "<Prop CanInterpT=\"1\" Type=\"1\" Spatial=\"0\"><Name>pluginPreset</Name><Default V=\"4\"><i>0</i></Default></Prop>";
    quad += &corner_prop("topRightPoint", 1, full[1]);
    quad += &use_layer("topRightUseLayer");
    quad += HITFILM_EFFECT_MESSAGES;
    quad += &corner_prop("bottomLeftPoint", 3, full[3]);
    quad += &use_layer("bottomLeftUseLayer");
    quad += &corner_prop("bottomRightPoint", 2, full[2]);
    quad += &use_layer("bottomRightUseLayer");
    quad += "<Prop CanInterpT=\"0\" Type=\"0\" Spatial=\"0\"><Name>isPerspective</Name><Default V=\"4\"><b>1</b></Default></Prop>";
    quad += "<Prop CanInterpT=\"0\" Type=\"0\" Spatial=\"0\"><Name>flipOrder</Name><Default V=\"4\"><b>0</b></Default></Prop>";
    let ui = ["topLeftUseLayer", "topLeftPoint", "topRightPoint", "topRightUseLayer", "bottomLeftPoint", "bottomLeftUseLayer", "bottomRightPoint", "bottomRightUseLayer", "isPerspective", "flipOrder"]
        .iter()
        .map(|n| format!("<Property vis=\"1\" name=\"{n}\"/>"))
        .collect::<String>();
    let fl = |n: &str, v: &str| format!("<Prop CanInterpT=\"1\" Type=\"0\" Spatial=\"0\"><Name>{n}</Name><Default V=\"4\"><fl>{v}</fl></Default></Prop>");
    let b1 = |n: &str, v: u8| format!("<Prop CanInterpT=\"0\" Type=\"1\" Spatial=\"0\"><Name>{n}</Name><Default V=\"4\"><b>{v}</b></Default></Prop>");
    let layer_props = [
        fl("rotationY", "0"),
        "<Prop CanInterpT=\"1\" Type=\"0\" Spatial=\"1\"><Name>anchorPoint</Name><Default V=\"4\"><p3 X=\"0\" Z=\"0\" Y=\"0\"/></Default></Prop>".to_string(),
        "<Prop CanInterpT=\"1\" Type=\"0\" Spatial=\"1\"><Name>position</Name><Default V=\"4\"><p3 X=\"0\" Z=\"0\" Y=\"0\"/></Default></Prop>".to_string(),
        b1("illuminated", 1),
        "<Prop CanInterpT=\"1\" Type=\"0\" Spatial=\"1\"><Name>orientation</Name><Default V=\"4\"><or X=\"0\" Z=\"0\" Y=\"0\"/></Default></Prop>".to_string(),
        "<Prop CanInterpT=\"1\" Type=\"0\" Spatial=\"0\"><Name>scale</Name><Default V=\"4\"><sc X=\"100\" Z=\"100\" Y=\"100\"/></Default></Prop>".to_string(),
        b1("scaleLinked", 1),
        fl("rotationX", "0"),
        fl("rotationZ", "0"),
        fl("opacity", "100"),
        // The rest of a plane's settings, as HitFilm writes them.
        HITFILM_LAYER_DEFAULTS.to_string(),
    ]
    .concat();
    let end = frames_total.saturating_sub(1).max(1);
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out += "<BiffCompositeShot AppVersion=\"18.0.12110.52955\" AppEdition=\"2000\" Version=\"1\">";
    out += &format!(
        "<Assets><SolidAsset Version=\"1\"><ID>{asset_id}</ID><Name>{name} plane</Name><Color G=\"0.5\" B=\"0.9\" R=\"0.3\" A=\"1\"/><Width>{}</Width><Height>{}</Height><PAR>0</PAR><ParentFolderID>00000000-0000-0000-0000-000000000000</ParentFolderID><IsHidden>0</IsHidden><Instances><Instance ID=\"{layer_id}\" Type=\"2\" CompID=\"{comp_id}\"/></Instances></SolidAsset></Assets>",
        f.width, f.height
    );
    out += &format!(
        "<CompositionAsset Version=\"19\"><ID>{comp_id}</ID><Name>{name} (Trackwright)</Name><ParentFolderID>00000000-0000-0000-0000-000000000000</ParentFolderID><IsHidden>0</IsHidden><CTI>0</CTI><InPoint>0</InPoint><OutPoint>{end}</OutPoint><TimelineZoom>0</TimelineZoom><SplitterPosition>440</SplitterPosition><TimelineTimeFormat>1000</TimelineTimeFormat><TimelineSnapMode>1000</TimelineSnapMode><IsPrimary>0</IsPrimary>"
    );
    out += &format!(
        "<AudioVideoSettings Version=\"1\"><FrameCount>{frames_total}</FrameCount><AudioSampleRate>48000</AudioSampleRate><Width>{}</Width><Height>{}</Height><PAR>0</PAR><PARCustom>1</PARCustom><FrameRate>{}</FrameRate></AudioVideoSettings>",
        f.width, f.height, num(r4(fps))
    );
    out += "<RenderSettings Version=\"2\"><FogEnabled>0</FogEnabled><FogNearDistance>900</FogNearDistance><FogFarDistance>2000</FogFarDistance><FogDensity>1</FogDensity><FogColor G=\"0\" B=\"0\" R=\"0\" A=\"1\"/><FogFalloff>0</FogFalloff><MotionBlurEnabled>1</MotionBlurEnabled><ShutterAngle>180</ShutterAngle><ShutterPhase>-90</ShutterPhase><MaxNumOfSamples>20</MaxNumOfSamples><UseAdaptiveSamples>1</UseAdaptiveSamples></RenderSettings>";
    out += &format!(
        "<Layers><AssetLayer Version=\"17\"><AssetID>{asset_id}</AssetID><AssetInstanceStart>0</AssetInstanceStart><AssetHasAudio>0</AssetHasAudio><AssetHasVideo>1</AssetHasVideo><AssetHadAudio>0</AssetHadAudio><AssetHadVideo>1</AssetHadVideo><AssetType>2</AssetType><Dimensions>0</Dimensions><AutoOrient>0</AutoOrient><AutoOrientPathAxis>1</AutoOrientPathAxis><AutoOrientLayer>00000000-0000-0000-0000-000000000000</AutoOrientLayer><PromoteLights>0</PromoteLights><IncludeInDepthMap>1</IncludeInDepthMap><WaveformIsVisible>0</WaveformIsVisible><UseProxyMedia>1</UseProxyMedia><Trackers/>"
    );
    out += &format!(
        "<Effects><Instance Version=\"8\"><ID>{effect_id}</ID><Name></Name><PluginID>65e96f64-7f4a-4141-a02a-76eb1fb42996</PluginID><PluginName>Quad Warp</PluginName><PluginVersion Mi=\"2\" Bu=\"0\" Re=\"0\" Ma=\"2\"/><ParentID>{layer_id}</ParentID><ParentTimelineID>{comp_id}</ParentTimelineID><Enabled>1</Enabled><CreatedInComp>1</CreatedInComp><InitializedUIState>1</InitializedUIState><ParentLayerLastSize X=\"{}\" Y=\"{}\"/><PropertyManager Version=\"7\">{quad}</PropertyManager><UIState>{ui}</UIState></Instance></Effects><Masks/>",
        f.width, f.height
    );
    out += &format!(
        "<LayerBase Version=\"2\"><ID>{layer_id}</ID><Name>{name}</Name><CompID>{comp_id}</CompID><ParentLayerID>00000000-0000-0000-0000-000000000000</ParentLayerID><StartFrame>0</StartFrame><EndFrame>{end}</EndFrame><BlendMode>0</BlendMode><Visible>1</Visible><Muted>0</Muted><Locked>0</Locked><MotionBlurOn>0</MotionBlurOn><Label>-1</Label><BehaviorEffects/><PropertyManager Version=\"7\">{layer_props}</PropertyManager></LayerBase></AssetLayer></Layers><Instances/>"
    );
    out += &format!(
        "<ViewerState Version=\"5\"><CurrentLayout>0</CurrentLayout><Layout0 Pos0=\"{view_id}\"/><Layout1/><Layout2 Sp1=\"0\" Sp0=\"0.5\" Sp2=\"0\"/><Layout3 Sp1=\"0\" Sp0=\"0.5\" Sp2=\"0.5\"/><Layout4 Sp1=\"0.5\" Sp0=\"0.5\" Sp2=\"0.5\"/><Views><State Active=\"1\" TextureScale=\"0.5\" ID=\"{view_id}\" CameraScale=\"1\" Type=\"0\" Layout=\"0\"><TexturePosition X=\"0\" Y=\"0\"/><TextureAnchorPoint X=\"0\" Y=\"0\"/><CameraPosition X=\"0\" Z=\"0\" Y=\"0\"/><CameraRotation X=\"0\" Z=\"0\" Y=\"0\"/><Options Gz=\"0\" TrPa=\"1\" Ch=\"0\" Vol=\"1\" BiTo=\"1\" TrCb=\"0\" MoPa=\"1\" WoAx=\"1\"><BGColor G=\"0\" B=\"0\" R=\"0\" A=\"1\"/></Options></State></Views></ViewerState>"
    );
    out += "<CompositionTemplate Version=\"1\"><IsEditable>1</IsEditable></CompositionTemplate><TemplateParent>00000000-0000-0000-0000-000000000000</TemplateParent></CompositionAsset></BiffCompositeShot>";
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::FrameTrack;
    use crate::shape::Shape;

    #[test]
    fn hitfilm_composite_shot_matches_hitfilms_layout() {
        let footage = Footage {
            source: "/tmp/shot.mp4".into(),
            frames: vec![String::new(); 30],
            width: 2732,
            height: 1440,
            fps: 25.0,
            first_frame: 0,
        };
        let mut l = Layer::new(1, 0, Shape::rect(100.0, 100.0, 200.0, 100.0));
        l.name = "Screen <1>".into();
        l.surface = [[100.0, 100.0], [300.0, 100.0], [300.0, 200.0], [100.0, 200.0]];
        let mut t = FrameTrack::reference();
        t.keyframe = false;
        t.h_from_ref = [1.0, 0.0, 10.0, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0];
        l.tracks.insert(10, t);
        let x = export(ExportFormat::HitFilm, &footage, &l);
        // Frame 10 at 25 fps is 400 ms; top left (110, 95) is (-1256, 625) from the
        // centre with y up.
        assert!(x.contains(r#"<Key Ti="400" V="5" Tp="1" STp="2"><TLk>0</TLk><SLk>0</SLk><Vl V="4"><p2 X="-1256" Y="625"/></Vl></Key>"#), "{x}");
        assert!(x.contains("<PluginName>Quad Warp</PluginName>"));
        assert!(x.contains("Screen &lt;1&gt;"));
        assert!(x.starts_with("<?xml"));
        std::fs::write("/tmp/tw_test.hfcs", &x).unwrap();
    }
}

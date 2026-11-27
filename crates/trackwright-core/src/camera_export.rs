//! Exporters for solved cameras and point clouds.
//!
//! All formats use the solve's scene orientation (`World`): Y up, right-handed, metres
//! or whatever unit the scene was scaled to. Cameras follow the OpenGL convention
//! (looking down -Z, +Y up).

use crate::geometry::{Mat3, Point};
use crate::linalg::V3;
use crate::project::Footage;
use crate::sfm::CameraSolve;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CameraFormat {
    Forma3d,
    Tidefire,
    Blender,
    Nuke,
    Collada,
    PointCloudObj,
    Chan,
}

impl CameraFormat {
    pub const ALL: [CameraFormat; 7] = [
        CameraFormat::Forma3d,
        CameraFormat::Tidefire,
        CameraFormat::Blender,
        CameraFormat::Nuke,
        CameraFormat::Collada,
        CameraFormat::PointCloudObj,
        CameraFormat::Chan,
    ];

    pub fn label(self) -> &'static str {
        match self {
            CameraFormat::Forma3d => "Forma3D camera + points (.f3dcam.json)",
            CameraFormat::Tidefire => "Tidefire camera + points (.tfcam.json)",
            CameraFormat::Blender => "Blender camera + points script (.py)",
            CameraFormat::Nuke => "Nuke Camera2 (.nk)",
            CameraFormat::Collada => "Collada camera (.dae) for Cinema 4D / Maya / Houdini",
            CameraFormat::PointCloudObj => "Point cloud (.obj)",
            CameraFormat::Chan => "Camera .chan (XYZ rotation order)",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            CameraFormat::Forma3d => "f3dcam.json",
            CameraFormat::Tidefire => "tfcam.json",
            CameraFormat::Blender => "camera.py",
            CameraFormat::Nuke => "camera.nk",
            CameraFormat::Collada => "dae",
            CameraFormat::PointCloudObj => "points.obj",
            CameraFormat::Chan => "chan",
        }
    }
}

struct Frame {
    index: usize,
    number: i64,
    pos: V3,
    /// Columns: camera right, up, back (world coordinates).
    rot: Mat3,
}

fn frames(f: &Footage, s: &CameraSolve) -> Vec<Frame> {
    s.poses
        .iter()
        .map(|(&i, p)| {
            let (pos, rot) = s.world.camera(p);
            Frame {
                index: i,
                number: f.frame_number(i),
                pos,
                rot,
            }
        })
        .collect()
}

fn points(s: &CameraSolve, max_error: f64) -> Vec<V3> {
    s.points
        .iter()
        .filter(|p| p.error <= max_error)
        .map(|p| s.world.to_user(p.pos))
        .collect()
}

/// Quaternion [x, y, z, w] of a rotation matrix.
pub fn quaternion(m: &Mat3) -> [f64; 4] {
    let tr = m[0] + m[4] + m[8];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [
            (m[7] - m[5]) / s,
            (m[2] - m[6]) / s,
            (m[3] - m[1]) / s,
            0.25 * s,
        ]
    } else if m[0] > m[4] && m[0] > m[8] {
        let s = (1.0 + m[0] - m[4] - m[8]).sqrt() * 2.0;
        [
            0.25 * s,
            (m[1] + m[3]) / s,
            (m[2] + m[6]) / s,
            (m[7] - m[5]) / s,
        ]
    } else if m[4] > m[8] {
        let s = (1.0 + m[4] - m[0] - m[8]).sqrt() * 2.0;
        [
            (m[1] + m[3]) / s,
            0.25 * s,
            (m[5] + m[7]) / s,
            (m[2] - m[6]) / s,
        ]
    } else {
        let s = (1.0 + m[8] - m[0] - m[4]).sqrt() * 2.0;
        [
            (m[2] + m[6]) / s,
            (m[5] + m[7]) / s,
            0.25 * s,
            (m[3] - m[1]) / s,
        ]
    };
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    q.map(|v| v / n)
}

/// Euler angles (degrees) for rotation order XYZ: R = Rz * Ry * Rx (X applied first).
pub fn euler_xyz(m: &Mat3) -> V3 {
    let sy = (-m[6]).clamp(-1.0, 1.0);
    let y = sy.asin();
    let (x, z) = if sy.abs() < 0.999_999 {
        (m[7].atan2(m[8]), m[3].atan2(m[0]))
    } else {
        ((-m[5]).atan2(m[4]), 0.0)
    };
    [x.to_degrees(), y.to_degrees(), z.to_degrees()]
}

/// Keep successive Euler angles continuous (no 360° jumps).
fn unwrap_deg(prev: Option<V3>, mut cur: V3) -> V3 {
    if let Some(p) = prev {
        for i in 0..3 {
            while cur[i] - p[i] > 180.0 {
                cur[i] -= 360.0;
            }
            while cur[i] - p[i] < -180.0 {
                cur[i] += 360.0;
            }
        }
    }
    cur
}

fn r5(v: f64) -> f64 {
    (v * 100_000.0).round() / 100_000.0
}

pub fn export_camera(
    format: CameraFormat,
    f: &Footage,
    s: &CameraSolve,
    sensor_mm: f64,
    name: &str,
) -> String {
    match format {
        CameraFormat::Forma3d => app_json(f, s, sensor_mm, "forma3d"),
        CameraFormat::Tidefire => app_json(f, s, sensor_mm, "tidefire"),
        CameraFormat::Blender => blender(f, s, sensor_mm, name),
        CameraFormat::Nuke => nuke(f, s, sensor_mm, name),
        CameraFormat::Collada => collada(f, s, sensor_mm, name),
        CameraFormat::PointCloudObj => obj(s),
        CameraFormat::Chan => chan(f, s, sensor_mm),
    }
}

fn app_json(f: &Footage, s: &CameraSolve, sensor_mm: f64, target: &str) -> String {
    let fov_y = 2.0 * (s.height / 2.0 / s.focal_px).atan();
    let mut prev_q: Option<[f64; 4]> = None;
    let fr: Vec<_> = frames(f, s)
        .iter()
        .map(|fr| {
            let mut q = quaternion(&fr.rot);
            if let Some(p) = prev_q {
                if p.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
                    q = q.map(|v| -v);
                }
            }
            prev_q = Some(q);
            let m = fr.rot;
            // Column-major 4x4 (glTF / OpenGL order).
            let mat = [
                m[0], m[3], m[6], 0.0, m[1], m[4], m[7], 0.0, m[2], m[5], m[8], 0.0, fr.pos[0],
                fr.pos[1], fr.pos[2], 1.0,
            ];
            json!({
                "frame": fr.number,
                "time": if f.fps > 0.0 { r5(fr.index as f64 / f.fps) } else { 0.0 },
                "position": fr.pos.map(r5),
                "rotation_quat_xyzw": q.map(r5),
                "matrix_col_major": mat.map(r5),
            })
        })
        .collect();
    serde_json::to_string_pretty(&json!({
        "format": "trackwright.camera.v1",
        "target": target,
        "generator": concat!("Trackwright ", env!("CARGO_PKG_VERSION")),
        "source": { "path": f.source, "width": f.width, "height": f.height, "fps": f.fps, "first_frame": f.first_frame },
        "coordinate_system": "Y up, right-handed; camera looks down -Z",
        "camera": {
            "focal_mm": r5(s.focal_px * sensor_mm / s.width),
            "focal_px": r5(s.focal_px),
            "sensor_width_mm": sensor_mm,
            "sensor_height_mm": r5(sensor_mm * s.height / s.width),
            "fov_y_deg": r5(fov_y.to_degrees()),
            "lens_k1": s.lens.k1,
            "lens_k2": s.lens.k2,
            "tripod": s.tripod,
            "solve_rms_px": r5(s.rms_px),
        },
        "frames": fr,
        "points": points(s, 3.0).iter().map(|p| p.map(r5)).collect::<Vec<_>>(),
    }))
    .unwrap()
}

/// Y-up to Blender's Z-up: (x, y, z) -> (x, -z, y).
fn to_z_up(v: V3) -> V3 {
    [v[0], -v[2], v[1]]
}

fn to_z_up_m(m: &Mat3) -> Mat3 {
    let a = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
    crate::geometry::mul(&a, m)
}

fn blender(f: &Footage, s: &CameraSolve, sensor_mm: f64, name: &str) -> String {
    let video_exts = [
        "mp4", "mov", "mkv", "avi", "webm", "m4v", "mxf", "mpg", "mpeg", "wmv",
    ];
    let is_video = std::path::Path::new(&f.source)
        .extension()
        .map(|e| video_exts.contains(&e.to_string_lossy().to_lowercase().as_str()))
        .unwrap_or(false);
    let clip = if is_video {
        f.source.clone()
    } else {
        f.frames.first().cloned().unwrap_or_default()
    };
    let mut keys = String::new();
    let mut prev_q: Option<[f64; 4]> = None;
    for fr in frames(f, s) {
        let p = to_z_up(fr.pos);
        let mut q = quaternion(&to_z_up_m(&fr.rot));
        if let Some(pq) = prev_q {
            if pq.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
                q = q.map(|v| -v);
            }
        }
        prev_q = Some(q);
        // Blender quaternions are (w, x, y, z).
        keys += &format!(
            "    ({}, ({:.6}, {:.6}, {:.6}), ({:.7}, {:.7}, {:.7}, {:.7})),\n",
            fr.index + 1,
            p[0],
            p[1],
            p[2],
            q[3],
            q[0],
            q[1],
            q[2]
        );
    }
    let mut pts = String::new();
    for p in points(s, 3.0) {
        let b = to_z_up(p);
        pts += &format!("({:.5}, {:.5}, {:.5}), ", b[0], b[1], b[2]);
    }
    format!(
        r#"# Trackwright camera -> Blender
# Scripting workspace > open this file > Run Script. Creates an animated camera, the
# solved point cloud and the footage as the camera background.
import bpy

NAME = "{name}"
FOOTAGE = r"{clip}"
WIDTH, HEIGHT, FPS = {w}, {h}, {fps}
FOCAL_MM, SENSOR_MM = {focal_mm:.6}, {sensor:.4}
# (frame, location, rotation quaternion w x y z) in Blender's Z-up world
KEYS = [
{keys}]
POINTS = [{pts}]


def main():
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.fps = max(1, round(FPS))
    scene.frame_start, scene.frame_end = KEYS[0][0], KEYS[-1][0]
    cam_data = bpy.data.cameras.new(NAME)
    cam_data.lens = FOCAL_MM
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = SENSOR_MM
    cam = bpy.data.objects.new(NAME, cam_data)
    scene.collection.objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    for frame, loc, quat in KEYS:
        cam.location = loc
        cam.rotation_quaternion = quat
        cam.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("rotation_quaternion", frame=frame)
    scene.camera = cam
    try:
        clip = bpy.data.movieclips.load(FOOTAGE, check_existing=True)
        cam_data.show_background_images = True
        bg = cam_data.background_images.new()
        bg.source = "MOVIE_CLIP"
        bg.clip = clip
    except Exception as err:
        print("Background footage not loaded:", err)
    if POINTS:
        mesh = bpy.data.meshes.new(NAME + "_points")
        mesh.from_pydata(POINTS, [], [])
        obj = bpy.data.objects.new(NAME + "_points", mesh)
        scene.collection.objects.link(obj)
    print("Trackwright: camera", NAME, "with", len(KEYS), "keys and", len(POINTS), "points")


main()
"#,
        name = name.replace('"', "'"),
        clip = clip.replace('"', "'"),
        w = f.width,
        h = f.height,
        fps = if f.fps > 0.0 { f.fps } else { 24.0 },
        focal_mm = s.focal_px * sensor_mm / s.width,
        sensor = sensor_mm,
        keys = keys,
        pts = pts,
    )
}

fn nuke(f: &Footage, s: &CameraSolve, sensor_mm: f64, name: &str) -> String {
    let fr = frames(f, s);
    let mut tr = [String::new(), String::new(), String::new()];
    let mut ro = [String::new(), String::new(), String::new()];
    let mut prev = None;
    for x in &fr {
        let e = unwrap_deg(prev, euler_xyz(&x.rot));
        prev = Some(e);
        for i in 0..3 {
            tr[i] += &format!(" x{} {:.6}", x.number, x.pos[i]);
            ro[i] += &format!(" x{} {:.6}", x.number, e[i]);
        }
    }
    let ident: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!(
        "set cut_paste_input [stack 0]\nversion 13.0 v1\nCamera2 {{\n rot_order XYZ\n translate {{{{curve{}}} {{curve{}}} {{curve{}}}}}\n rotate {{{{curve{}}} {{curve{}}} {{curve{}}}}}\n focal {:.6}\n haperture {:.6}\n vaperture {:.6}\n name Trackwright_{}\n selected true\n}}\n",
        tr[0], tr[1], tr[2], ro[0], ro[1], ro[2],
        s.focal_px * sensor_mm / s.width,
        sensor_mm,
        sensor_mm * s.height / s.width,
        ident
    )
}

fn collada(f: &Footage, s: &CameraSolve, sensor_mm: f64, name: &str) -> String {
    let fr = frames(f, s);
    let fps = if f.fps > 0.0 { f.fps } else { 24.0 };
    let xfov = 2.0 * (s.width / 2.0 / s.focal_px).atan().to_degrees();
    let times: Vec<String> = fr
        .iter()
        .map(|x| format!("{:.6}", x.index as f64 / fps))
        .collect();
    let mut mats = String::new();
    for x in &fr {
        let m = x.rot;
        let p = x.pos;
        let row = [
            m[0], m[1], m[2], p[0], m[3], m[4], m[5], p[1], m[6], m[7], m[8], p[2], 0.0, 0.0, 0.0,
            1.0,
        ];
        mats += &row
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(" ");
        mats.push(' ');
    }
    let first = fr
        .first()
        .map(|x| {
            let m = x.rot;
            let p = x.pos;
            [
                m[0], m[1], m[2], p[0], m[3], m[4], m[5], p[1], m[6], m[7], m[8], p[2], 0.0, 0.0,
                0.0, 1.0,
            ]
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(" ")
        })
        .unwrap_or_default();
    let n = fr.len();
    let interp = vec!["LINEAR"; n].join(" ");
    let id: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let _ = sensor_mm;
    format!(
        r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor><authoring_tool>Trackwright</authoring_tool></contributor>
    <unit name="meter" meter="1"/>
    <up_axis>Y_UP</up_axis>
  </asset>
  <library_cameras>
    <camera id="{id}-cam" name="{id}">
      <optics><technique_common><perspective>
        <xfov>{xfov:.6}</xfov><aspect_ratio>{aspect:.6}</aspect_ratio><znear>0.01</znear><zfar>10000</zfar>
      </perspective></technique_common></optics>
    </camera>
  </library_cameras>
  <library_animations>
    <animation id="{id}-anim">
      <source id="{id}-in"><float_array id="{id}-in-arr" count="{n}">{times}</float_array>
        <technique_common><accessor source="#{id}-in-arr" count="{n}" stride="1"><param name="TIME" type="float"/></accessor></technique_common></source>
      <source id="{id}-out"><float_array id="{id}-out-arr" count="{n16}">{mats}</float_array>
        <technique_common><accessor source="#{id}-out-arr" count="{n}" stride="16"><param name="TRANSFORM" type="float4x4"/></accessor></technique_common></source>
      <source id="{id}-interp"><Name_array id="{id}-interp-arr" count="{n}">{interp}</Name_array>
        <technique_common><accessor source="#{id}-interp-arr" count="{n}" stride="1"><param name="INTERPOLATION" type="name"/></accessor></technique_common></source>
      <sampler id="{id}-sampler">
        <input semantic="INPUT" source="#{id}-in"/><input semantic="OUTPUT" source="#{id}-out"/><input semantic="INTERPOLATION" source="#{id}-interp"/>
      </sampler>
      <channel source="#{id}-sampler" target="{id}/transform"/>
    </animation>
  </library_animations>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="{id}" name="{id}" type="NODE">
        <matrix sid="transform">{first}</matrix>
        <instance_camera url="#{id}-cam"/>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
"##,
        aspect = s.width / s.height,
        n16 = n * 16,
        times = times.join(" "),
    )
}

fn obj(s: &CameraSolve) -> String {
    let mut out = String::from("# Trackwright point cloud (Y up)\n");
    for p in points(s, 3.0) {
        out += &format!("v {:.6} {:.6} {:.6}\n", p[0], p[1], p[2]);
    }
    out
}

/// Nuke/Maya style .chan: frame tx ty tz rx ry rz vertical-fov (XYZ rotation order).
fn chan(f: &Footage, s: &CameraSolve, _sensor_mm: f64) -> String {
    let vfov = 2.0 * (s.height / 2.0 / s.focal_px).atan().to_degrees();
    let mut out = String::new();
    let mut prev = None;
    for x in frames(f, s) {
        let e = unwrap_deg(prev, euler_xyz(&x.rot));
        prev = Some(e);
        out += &format!(
            "{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
            x.number, x.pos[0], x.pos[1], x.pos[2], e[0], e[1], e[2], vfov
        );
    }
    out
}

/// Points used for the viewport overlay: footage pixels of solved points at `frame`.
pub fn reprojected(s: &CameraSolve, frame: usize) -> Vec<(u32, Point, f64)> {
    s.points
        .iter()
        .filter_map(|p| s.project(frame, p.pos).map(|q| (p.id, q, p.error)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::rodrigues;

    #[test]
    fn euler_and_quaternion_match_matrix() {
        let m = rodrigues([0.3, -0.5, 0.2]);
        let e = euler_xyz(&m).map(|d| d.to_radians());
        let rx = rodrigues([e[0], 0.0, 0.0]);
        let ry = rodrigues([0.0, e[1], 0.0]);
        let rz = rodrigues([0.0, 0.0, e[2]]);
        let back = crate::geometry::mul(&rz, &crate::geometry::mul(&ry, &rx));
        assert!((0..9).all(|i| (back[i] - m[i]).abs() < 1e-9));
        let q = quaternion(&m);
        // Rotate a vector with the quaternion and compare.
        let v = [1.0, 2.0, 3.0];
        let qv = [q[0], q[1], q[2]];
        let t = crate::linalg::scale(crate::linalg::cross(qv, v), 2.0);
        let rv = crate::linalg::add(
            crate::linalg::add(v, crate::linalg::scale(t, q[3])),
            crate::linalg::cross(qv, t),
        );
        let mv = crate::linalg::mat_vec(&m, v);
        assert!((0..3).all(|i| (rv[i] - mv[i]).abs() < 1e-9));
    }
}

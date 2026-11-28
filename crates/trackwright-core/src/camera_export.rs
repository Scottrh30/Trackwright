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
    Fbx,
    Alembic,
    MayaAscii,
}

impl CameraFormat {
    pub const ALL: [CameraFormat; 10] = [
        CameraFormat::Forma3d,
        CameraFormat::Tidefire,
        CameraFormat::Blender,
        CameraFormat::Nuke,
        CameraFormat::Collada,
        CameraFormat::PointCloudObj,
        CameraFormat::Fbx,
        CameraFormat::Alembic,
        CameraFormat::MayaAscii,
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
            CameraFormat::Chan => "Camera .chan (Nuke ZXY rotation order)",
            CameraFormat::Fbx => {
                "FBX camera + points (.fbx) for Maya / Cinema 4D / Unreal / Blender"
            }
            CameraFormat::Alembic => "Alembic camera + points (.abc) for Houdini / Maya / Blender",
            CameraFormat::MayaAscii => "HitFilm / Maya camera + points (.ma)",
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
            CameraFormat::Fbx => "fbx",
            CameraFormat::Alembic => "abc",
            CameraFormat::MayaAscii => "ma",
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

/// Euler angles (degrees) about X, Y, Z for Nuke's ZXY order: R = Ry * Rx * Rz.
pub fn euler_zxy(m: &Mat3) -> V3 {
    let rx = (-m[5]).clamp(-1.0, 1.0).asin();
    let (ry, rz) = if rx.cos().abs() > 1e-9 {
        (m[2].atan2(m[8]), m[3].atan2(m[4]))
    } else {
        ((-m[6]).atan2(m[0]), 0.0)
    };
    [rx.to_degrees(), ry.to_degrees(), rz.to_degrees()]
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
) -> Vec<u8> {
    match format {
        CameraFormat::Forma3d => app_json(f, s, sensor_mm, "forma3d").into_bytes(),
        CameraFormat::Tidefire => app_json(f, s, sensor_mm, "tidefire").into_bytes(),
        CameraFormat::Blender => blender(f, s, sensor_mm, name).into_bytes(),
        CameraFormat::Nuke => nuke(f, s, sensor_mm, name).into_bytes(),
        CameraFormat::Collada => collada(f, s, sensor_mm, name).into_bytes(),
        CameraFormat::PointCloudObj => obj(s).into_bytes(),
        CameraFormat::Chan => chan(f, s, sensor_mm).into_bytes(),
        CameraFormat::Fbx => fbx_camera(f, s, sensor_mm, name),
        CameraFormat::Alembic => alembic_camera(f, s, sensor_mm, name),
        CameraFormat::MayaAscii => maya_ascii(f, s, sensor_mm, name).into_bytes(),
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

/// Nuke/Maya style .chan: frame tx ty tz rx ry rz vertical-fov, in Nuke's
/// default ZXY rotation order (what Nuke, Blender and Reelforge assume).
fn chan(f: &Footage, s: &CameraSolve, _sensor_mm: f64) -> String {
    let vfov = 2.0 * (s.height / 2.0 / s.focal_px).atan().to_degrees();
    let mut out = String::new();
    let mut prev = None;
    for x in frames(f, s) {
        let e = unwrap_deg(prev, euler_zxy(&x.rot));
        prev = Some(e);
        out += &format!(
            "{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
            x.number, x.pos[0], x.pos[1], x.pos[2], e[0], e[1], e[2], vfov
        );
    }
    out
}

/// Maya ASCII scene: an animated camera and a locator per solved point. This is the
/// format HitFilm's Import › 3D Camera Tracking Data reads (as made by SynthEyes,
/// Boujou, PFTrack), and Maya opens it too. Maya cameras look down -Z with Y up,
/// like ours; rotations use Maya's default XYZ order. Keys are on footage frame
/// numbers; HitFilm asks for the frame rate when importing.
fn maya_ascii(f: &Footage, s: &CameraSolve, sensor_mm: f64, name: &str) -> String {
    let ident = |s: &str| -> String {
        let mut o: String = s
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        if o.is_empty() || o.starts_with(|c: char| c.is_ascii_digit()) {
            o.insert_str(0, "cam_");
        }
        o
    };
    let cam = ident(&format!("{name}_camera"));
    let fr = frames(f, s);
    let fps = fps_of(f);
    let time_unit = match (fps * 1000.0).round() as i64 {
        24000 => "film".to_string(),
        25000 => "pal".to_string(),
        30000 => "ntsc".to_string(),
        48000 => "show".to_string(),
        50000 => "palf".to_string(),
        60000 => "ntscf".to_string(),
        _ => format!("{}fps", r5(fps)),
    };
    let film_w = sensor_mm / 25.4;
    let film_h = film_w * s.height / s.width;
    let mut out = String::new();
    out += "//Maya ASCII 2018 scene\n";
    out += &format!("//Name: {cam}.ma\n");
    out += &format!(
        "//Codeset: UTF-8\n//Made by Trackwright {}\n",
        env!("CARGO_PKG_VERSION")
    );
    out += "requires maya \"2018\";\n";
    out += &format!("currentUnit -l centimeter -a degree -t {time_unit};\n");
    out += &format!(
        "fileInfo \"resolution\" \"{} {}\";\n",
        s.width as i64, s.height as i64
    );
    let mut rots = Vec::new();
    let mut prev = None;
    for x in &fr {
        let e = unwrap_deg(prev, euler_xyz(&x.rot));
        prev = Some(e);
        rots.push(e);
    }
    let (p0, r0) = (
        fr.first().map_or([0.0; 3], |x| x.pos),
        rots.first().copied().unwrap_or([0.0; 3]),
    );
    out += &format!("createNode transform -n \"{cam}\";\n");
    out += &format!(
        "\tsetAttr \".t\" -type \"double3\" {} {} {} ;\n",
        r5(p0[0]),
        r5(p0[1]),
        r5(p0[2])
    );
    out += &format!(
        "\tsetAttr \".r\" -type \"double3\" {} {} {} ;\n",
        r5(r0[0]),
        r5(r0[1]),
        r5(r0[2])
    );
    out += &format!("createNode camera -n \"{cam}Shape\" -p \"{cam}\";\n");
    out += "\tsetAttr -k off \".v\";\n";
    out += &format!(
        "\tsetAttr \".cap\" -type \"double2\" {} {} ;\n",
        r5(film_w),
        r5(film_h)
    );
    out += "\tsetAttr \".ff\" 1;\n";
    out += &format!("\tsetAttr \".fl\" {};\n", r5(s.focal_mm(sensor_mm)));
    out += "\tsetAttr \".coi\" 10;\n";
    out += &format!("\tsetAttr \".imn\" -type \"string\" \"{cam}\";\n");
    out += "\tsetAttr \".den\" -type \"string\" \"depth\";\n";
    out += "\tsetAttr \".man\" -type \"string\" \"mask\";\n";
    // Animation curves, one per channel.
    let channels: [(&str, &str, &str, Box<dyn Fn(usize) -> f64>); 6] = [
        (
            "translateX",
            "tx",
            "animCurveTL",
            Box::new(|i| fr[i].pos[0]),
        ),
        (
            "translateY",
            "ty",
            "animCurveTL",
            Box::new(|i| fr[i].pos[1]),
        ),
        (
            "translateZ",
            "tz",
            "animCurveTL",
            Box::new(|i| fr[i].pos[2]),
        ),
        ("rotateX", "rx", "animCurveTA", Box::new(|i| rots[i][0])),
        ("rotateY", "ry", "animCurveTA", Box::new(|i| rots[i][1])),
        ("rotateZ", "rz", "animCurveTA", Box::new(|i| rots[i][2])),
    ];
    if fr.len() > 1 {
        for (long, short, kind, val) in &channels {
            let node = format!("{cam}_{long}");
            out += &format!("createNode {kind} -n \"{node}\";\n");
            out += "\tsetAttr \".tan\" 18;\n";
            out += "\tsetAttr \".wgt\" no;\n";
            out += &format!("\tsetAttr -s {} \".ktv[0:{}]\" ", fr.len(), fr.len() - 1);
            for (i, x) in fr.iter().enumerate() {
                out += &format!(" {} {}", x.number, r5(val(i)));
            }
            out += ";\n";
            let _ = short;
        }
    }
    // Solved points as locators.
    for (i, p) in points(s, 3.0).iter().take(5000).enumerate() {
        let n = format!("trackPoint{}", i + 1);
        out += &format!("createNode transform -n \"{n}\";\n");
        out += &format!(
            "\tsetAttr \".t\" -type \"double3\" {} {} {} ;\n",
            r5(p[0]),
            r5(p[1]),
            r5(p[2])
        );
        out += &format!("createNode locator -n \"{n}Shape\" -p \"{n}\";\n");
        out += "\tsetAttr -k off \".v\";\n";
    }
    if fr.len() > 1 {
        for (long, short, _, _) in &channels {
            out += &format!("connectAttr \"{cam}_{long}.o\" \"{cam}.{short}\";\n");
        }
    }
    out += &format!(
        "playbackOptions -min {} -max {} -ast {} -aet {};\n",
        fr.first().map_or(0, |x| x.number),
        fr.last().map_or(0, |x| x.number),
        fr.first().map_or(0, |x| x.number),
        fr.last().map_or(0, |x| x.number)
    );
    out += &format!("// End of {cam}.ma\n");
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

fn fps_of(f: &Footage) -> f64 {
    if f.fps > 0.0 {
        f.fps
    } else {
        24.0
    }
}

/// Binary FBX 7.4: an animated camera (Y up, 1 unit = 1 metre) and the solved
/// points as a vertex-only mesh.
fn fbx_camera(f: &Footage, s: &CameraSolve, sensor_mm: f64, name: &str) -> Vec<u8> {
    use crate::fbx::{obj_name, p, Node, Prop, KTIME_PER_SECOND};
    let fr = frames(f, s);
    let fps = fps_of(f);
    let ktime = |number: i64| (number as f64 / fps * KTIME_PER_SECOND).round() as i64;
    let (t0, t1) = (
        fr.first().map_or(0, |x| ktime(x.number)),
        fr.last().map_or(0, |x| ktime(x.number)),
    );
    let focal_mm = s.focal_mm(sensor_mm);
    let film_w = sensor_mm / 25.4;
    let film_h = film_w * s.height / s.width;
    let fov_x = 2.0 * (s.width / 2.0 / s.focal_px).atan().to_degrees();
    let fov_y = 2.0 * (s.height / 2.0 / s.focal_px).atan().to_degrees();
    // FBX cameras look down +X: turn ours (looking down -Z) a quarter turn about Y.
    let to_fbx = |m: &Mat3| -> Mat3 {
        let c = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
        let mut r = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = (0..3).map(|k| m[i * 3 + k] * c[k * 3 + j]).sum();
            }
        }
        r
    };
    let mut rots = Vec::new();
    let mut prev = None;
    for x in &fr {
        let e = unwrap_deg(prev, euler_xyz(&to_fbx(&x.rot)));
        prev = Some(e);
        rots.push(e);
    }
    let (first_pos, first_rot) = (
        fr.first().map_or([0.0; 3], |x| x.pos),
        rots.first().copied().unwrap_or([0.0; 3]),
    );
    let pts = points(s, 3.0);
    let has_points = !pts.is_empty();

    let (id_attr, id_model, id_stack, id_layer, id_tnode, id_rnode) = (
        1_000_001i64,
        1_000_002,
        1_000_003,
        1_000_004,
        1_000_005,
        1_000_006,
    );
    let id_curve = |k: i64| 1_000_100 + k;
    let (id_geom, id_pmodel) = (1_000_201i64, 1_000_202);
    let keys: Vec<i64> = fr.iter().map(|x| ktime(x.number)).collect();
    let curve = |id: i64, vals: Vec<f32>| {
        Node::new(
            "AnimationCurve",
            vec![
                Prop::I64(id),
                obj_name("", "AnimCurve"),
                Prop::Str(String::new()),
            ],
        )
        .with(vec![
            Node::new(
                "Default",
                vec![Prop::F64(vals.first().copied().unwrap_or(0.0) as f64)],
            ),
            Node::new("KeyVer", vec![Prop::I32(4008)]),
            Node::new("KeyTime", vec![Prop::ArrI64(keys.clone())]),
            Node::new("KeyValueFloat", vec![Prop::ArrF32(vals)]),
            // Linear keys, as Blender writes them.
            Node::new("KeyAttrFlags", vec![Prop::ArrI32(vec![24836])]),
            Node::new(
                "KeyAttrDataFloat",
                vec![Prop::ArrF32(vec![0.0, 0.0, 9.419963e-30, 0.0])],
            ),
            Node::new(
                "KeyAttrRefCount",
                vec![Prop::ArrI32(vec![keys.len() as i32])],
            ),
        ])
    };
    let mut objects = vec![
        Node::new(
            "NodeAttribute",
            vec![
                Prop::I64(id_attr),
                obj_name(name, "NodeAttribute"),
                Prop::Str("Camera".into()),
            ],
        )
        .with(vec![
            Node::new("Properties70", vec![]).with(vec![
                p("AspectRatioMode", "enum", "", "", vec![Prop::I32(2)]),
                p(
                    "AspectWidth",
                    "double",
                    "Number",
                    "",
                    vec![Prop::F64(s.width)],
                ),
                p(
                    "AspectHeight",
                    "double",
                    "Number",
                    "",
                    vec![Prop::F64(s.height)],
                ),
                p("FilmWidth", "double", "Number", "", vec![Prop::F64(film_w)]),
                p(
                    "FilmHeight",
                    "double",
                    "Number",
                    "",
                    vec![Prop::F64(film_h)],
                ),
                p(
                    "FilmAspectRatio",
                    "double",
                    "Number",
                    "",
                    vec![Prop::F64(film_w / film_h)],
                ),
                p("ApertureMode", "enum", "", "", vec![Prop::I32(3)]),
                p("GateFit", "enum", "", "", vec![Prop::I32(2)]),
                p(
                    "FieldOfView",
                    "FieldOfView",
                    "",
                    "A",
                    vec![Prop::F64(fov_x)],
                ),
                p(
                    "FieldOfViewX",
                    "FieldOfViewX",
                    "",
                    "A",
                    vec![Prop::F64(fov_x)],
                ),
                p(
                    "FieldOfViewY",
                    "FieldOfViewY",
                    "",
                    "A",
                    vec![Prop::F64(fov_y)],
                ),
                p(
                    "FocalLength",
                    "double",
                    "Number",
                    "A",
                    vec![Prop::F64(focal_mm)],
                ),
                p("NearPlane", "double", "Number", "", vec![Prop::F64(1.0)]),
                p(
                    "FarPlane",
                    "double",
                    "Number",
                    "",
                    vec![Prop::F64(100000.0)],
                ),
            ]),
            Node::new("TypeFlags", vec![Prop::Str("Camera".into())]),
            Node::new("GeometryVersion", vec![Prop::I32(124)]),
            Node::new(
                "Position",
                vec![Prop::F64(0.0), Prop::F64(0.0), Prop::F64(0.0)],
            ),
            Node::new("Up", vec![Prop::F64(0.0), Prop::F64(1.0), Prop::F64(0.0)]),
            Node::new(
                "LookAt",
                vec![Prop::F64(1.0), Prop::F64(0.0), Prop::F64(0.0)],
            ),
            Node::new("ShowInfoOnMoving", vec![Prop::I32(1)]),
            Node::new("ShowAudio", vec![Prop::I32(0)]),
            Node::new(
                "AudioColor",
                vec![Prop::F64(0.0), Prop::F64(1.0), Prop::F64(0.0)],
            ),
            Node::new("CameraOrthoZoom", vec![Prop::F64(1.0)]),
        ]),
        Node::new(
            "Model",
            vec![
                Prop::I64(id_model),
                obj_name(name, "Model"),
                Prop::Str("Camera".into()),
            ],
        )
        .with(vec![
            Node::new("Version", vec![Prop::I32(232)]),
            Node::new("Properties70", vec![]).with(vec![
                p(
                    "Lcl Translation",
                    "Lcl Translation",
                    "",
                    "A+",
                    first_pos.iter().map(|v| Prop::F64(*v)).collect(),
                ),
                p(
                    "Lcl Rotation",
                    "Lcl Rotation",
                    "",
                    "A+",
                    first_rot.iter().map(|v| Prop::F64(*v)).collect(),
                ),
                p(
                    "Lcl Scaling",
                    "Lcl Scaling",
                    "",
                    "A+",
                    vec![Prop::F64(1.0), Prop::F64(1.0), Prop::F64(1.0)],
                ),
                p(
                    "DefaultAttributeIndex",
                    "int",
                    "Integer",
                    "",
                    vec![Prop::I32(0)],
                ),
                p("InheritType", "enum", "", "", vec![Prop::I32(1)]),
                p("RotationOrder", "enum", "", "", vec![Prop::I32(0)]),
                p("AspectW", "double", "Number", "", vec![Prop::F64(s.width)]),
                p("AspectH", "double", "Number", "", vec![Prop::F64(s.height)]),
            ]),
            Node::new("MultiLayer", vec![Prop::I32(0)]),
            Node::new("MultiTake", vec![Prop::I32(0)]),
            Node::new("Shading", vec![Prop::Bool(true)]),
            Node::new("Culling", vec![Prop::Str("CullingOff".into())]),
        ]),
    ];
    if has_points {
        let verts: Vec<f64> = pts.iter().flat_map(|q| q.iter().copied()).collect();
        objects.push(
            Node::new(
                "Geometry",
                vec![
                    Prop::I64(id_geom),
                    obj_name("TrackwrightPoints", "Geometry"),
                    Prop::Str("Mesh".into()),
                ],
            )
            .with(vec![
                Node::new("GeometryVersion", vec![Prop::I32(124)]),
                Node::new("Vertices", vec![Prop::ArrF64(verts)]),
                Node::new("PolygonVertexIndex", vec![Prop::ArrI32(Vec::new())]),
            ]),
        );
        objects.push(
            Node::new(
                "Model",
                vec![
                    Prop::I64(id_pmodel),
                    obj_name("TrackwrightPoints", "Model"),
                    Prop::Str("Mesh".into()),
                ],
            )
            .with(vec![
                Node::new("Version", vec![Prop::I32(232)]),
                Node::new("Properties70", vec![]).with(vec![
                    p(
                        "DefaultAttributeIndex",
                        "int",
                        "Integer",
                        "",
                        vec![Prop::I32(0)],
                    ),
                    p("InheritType", "enum", "", "", vec![Prop::I32(1)]),
                ]),
                Node::new("MultiLayer", vec![Prop::I32(0)]),
                Node::new("MultiTake", vec![Prop::I32(0)]),
                Node::new("Shading", vec![Prop::Bool(true)]),
                Node::new("Culling", vec![Prop::Str("CullingOff".into())]),
            ]),
        );
    }
    let take = format!("{name}|Take");
    objects.push(
        Node::new(
            "AnimationStack",
            vec![
                Prop::I64(id_stack),
                obj_name(&take, "AnimStack"),
                Prop::Str(String::new()),
            ],
        )
        .with(vec![Node::new("Properties70", vec![]).with(vec![
            p("LocalStart", "KTime", "Time", "", vec![Prop::I64(t0)]),
            p("LocalStop", "KTime", "Time", "", vec![Prop::I64(t1)]),
            p("ReferenceStart", "KTime", "Time", "", vec![Prop::I64(t0)]),
            p("ReferenceStop", "KTime", "Time", "", vec![Prop::I64(t1)]),
        ])]),
    );
    objects.push(Node::new(
        "AnimationLayer",
        vec![
            Prop::I64(id_layer),
            obj_name(&take, "AnimLayer"),
            Prop::Str(String::new()),
        ],
    ));
    for (id, label, v) in [(id_tnode, "T", first_pos), (id_rnode, "R", first_rot)] {
        objects.push(
            Node::new(
                "AnimationCurveNode",
                vec![
                    Prop::I64(id),
                    obj_name(label, "AnimCurveNode"),
                    Prop::Str(String::new()),
                ],
            )
            .with(vec![Node::new("Properties70", vec![]).with(vec![
                p("d|X", "Number", "", "A", vec![Prop::F64(v[0])]),
                p("d|Y", "Number", "", "A", vec![Prop::F64(v[1])]),
                p("d|Z", "Number", "", "A", vec![Prop::F64(v[2])]),
            ])]),
        );
    }
    for axis in 0..3 {
        objects.push(curve(
            id_curve(axis as i64),
            fr.iter().map(|x| x.pos[axis] as f32).collect(),
        ));
        objects.push(curve(
            id_curve(3 + axis as i64),
            rots.iter().map(|r| r[axis] as f32).collect(),
        ));
    }

    let oo = |a: i64, b: i64| {
        Node::new(
            "C",
            vec![Prop::Str("OO".into()), Prop::I64(a), Prop::I64(b)],
        )
    };
    let op = |a: i64, b: i64, prop: &str| {
        Node::new(
            "C",
            vec![
                Prop::Str("OP".into()),
                Prop::I64(a),
                Prop::I64(b),
                Prop::Str(prop.into()),
            ],
        )
    };
    let mut conns = vec![oo(id_model, 0), oo(id_attr, id_model)];
    if has_points {
        conns.push(oo(id_pmodel, 0));
        conns.push(oo(id_geom, id_pmodel));
    }
    conns.extend([
        oo(id_layer, id_stack),
        oo(id_tnode, id_layer),
        op(id_tnode, id_model, "Lcl Translation"),
        oo(id_rnode, id_layer),
        op(id_rnode, id_model, "Lcl Rotation"),
    ]);
    for (axis, d) in ["d|X", "d|Y", "d|Z"].iter().enumerate() {
        conns.push(op(id_curve(axis as i64), id_tnode, d));
        conns.push(op(id_curve(3 + axis as i64), id_rnode, d));
    }

    let creator = concat!("Trackwright ", env!("CARGO_PKG_VERSION"));
    let count = |n: i32| Node::new("Count", vec![Prop::I32(n)]);
    let object_type =
        |ty: &str, n: i32| Node::new("ObjectType", vec![Prop::Str(ty.into())]).with(vec![count(n)]);
    let mut defs = vec![
        Node::new("Version", vec![Prop::I32(100)]),
        count(if has_points { 12 } else { 10 }),
        object_type("GlobalSettings", 1),
        object_type("NodeAttribute", 1),
        object_type("Model", if has_points { 2 } else { 1 }),
        object_type("AnimationStack", 1),
        object_type("AnimationLayer", 1),
        object_type("AnimationCurveNode", 2),
        object_type("AnimationCurve", 6),
    ];
    if has_points {
        defs.push(object_type("Geometry", 1));
    }
    let nodes = vec![
        Node::new("FBXHeaderExtension", vec![]).with(vec![
            Node::new("FBXHeaderVersion", vec![Prop::I32(1003)]),
            Node::new("FBXVersion", vec![Prop::I32(7400)]),
            Node::new("EncryptionType", vec![Prop::I32(0)]),
            Node::new("Creator", vec![Prop::Str(creator.into())]),
        ]),
        Node::new("FileId", vec![Prop::Raw(crate::fbx::FILE_ID.to_vec())]),
        Node::new(
            "CreationTime",
            vec![Prop::Str(crate::fbx::CREATION_TIME.into())],
        ),
        Node::new("Creator", vec![Prop::Str(creator.into())]),
        Node::new("GlobalSettings", vec![]).with(vec![
            Node::new("Version", vec![Prop::I32(1000)]),
            Node::new("Properties70", vec![]).with(vec![
                p("UpAxis", "int", "Integer", "", vec![Prop::I32(1)]),
                p("UpAxisSign", "int", "Integer", "", vec![Prop::I32(1)]),
                p("FrontAxis", "int", "Integer", "", vec![Prop::I32(2)]),
                p("FrontAxisSign", "int", "Integer", "", vec![Prop::I32(1)]),
                p("CoordAxis", "int", "Integer", "", vec![Prop::I32(0)]),
                p("CoordAxisSign", "int", "Integer", "", vec![Prop::I32(1)]),
                p("OriginalUpAxis", "int", "Integer", "", vec![Prop::I32(1)]),
                p(
                    "OriginalUpAxisSign",
                    "int",
                    "Integer",
                    "",
                    vec![Prop::I32(1)],
                ),
                // 1 unit = 100 cm.
                p(
                    "UnitScaleFactor",
                    "double",
                    "Number",
                    "",
                    vec![Prop::F64(100.0)],
                ),
                p(
                    "OriginalUnitScaleFactor",
                    "double",
                    "Number",
                    "",
                    vec![Prop::F64(100.0)],
                ),
                p("TimeMode", "enum", "", "", vec![Prop::I32(14)]),
                p("TimeSpanStart", "KTime", "Time", "", vec![Prop::I64(t0)]),
                p("TimeSpanStop", "KTime", "Time", "", vec![Prop::I64(t1)]),
                p(
                    "CustomFrameRate",
                    "double",
                    "Number",
                    "",
                    vec![Prop::F64(fps)],
                ),
            ]),
        ]),
        Node::new("Documents", vec![]).with(vec![
            count(1),
            Node::new(
                "Document",
                vec![
                    Prop::I64(1_000_000),
                    Prop::Str("Scene".into()),
                    Prop::Str("Scene".into()),
                ],
            )
            .with(vec![
                Node::new("Properties70", vec![]).with(vec![
                    p("SourceObject", "object", "", "", vec![]),
                    p(
                        "ActiveAnimStackName",
                        "KString",
                        "",
                        "",
                        vec![Prop::Str(take.clone())],
                    ),
                ]),
                Node::new("RootNode", vec![Prop::I64(0)]),
            ]),
        ]),
        Node::new("References", vec![]),
        Node::new("Definitions", vec![]).with(defs),
        Node::new("Objects", vec![]).with(objects),
        Node::new("Connections", vec![]).with(conns),
        Node::new("Takes", vec![]).with(vec![
            Node::new("Current", vec![Prop::Str(take.clone())]),
            Node::new("Take", vec![Prop::Str(take.clone())]).with(vec![
                Node::new("FileName", vec![Prop::Str(format!("{take}.tak"))]),
                Node::new("LocalTime", vec![Prop::I64(t0), Prop::I64(t1)]),
                Node::new("ReferenceTime", vec![Prop::I64(t0), Prop::I64(t1)]),
            ]),
        ]),
    ];
    crate::fbx::write(&nodes)
}

/// Alembic (Ogawa): an Xform with a Camera under it, sampled every frame, and
/// the solved points as a Points object.
fn alembic_camera(f: &Footage, s: &CameraSolve, sensor_mm: f64, name: &str) -> Vec<u8> {
    use crate::alembic::{f32s, f64s, write, Object, Pod, Property, TimeSampling};
    let fr = frames(f, s);
    let fps = fps_of(f);
    let n = fr.len() as u32;
    let start = fr.first().map_or(0.0, |x| x.number as f64 / fps);
    let mats: Vec<Vec<u8>> = fr
        .iter()
        .map(|x| {
            let m = x.rot;
            // Imath row-vector matrix: rows are the images of X, Y, Z, then the translation.
            f64s(&[
                m[0], m[3], m[6], 0.0, m[1], m[4], m[7], 0.0, m[2], m[5], m[8], 0.0, x.pos[0],
                x.pos[1], x.pos[2], 1.0,
            ])
        })
        .collect();
    // Channels that change.
    let anim: Vec<u32> = (0..16u32)
        .filter(|&c| {
            mats.windows(2).any(|w| {
                w[0][c as usize * 8..c as usize * 8 + 8] != w[1][c as usize * 8..c as usize * 8 + 8]
            })
        })
        .collect();
    let scalar =
        |name: &str, pod: Pod, extent: u8, ts: u32, samples: Vec<Vec<u8>>| Property::Scalar {
            name: name.into(),
            meta: String::new(),
            pod,
            extent,
            ts,
            samples,
        };
    let xform_props = vec![
        scalar(".inherits", Pod::Bool, 1, 1, vec![vec![1]; n as usize]),
        scalar(".ops", Pod::U8, 1, 1, vec![vec![0x30]; n as usize]), // one matrix op
        scalar(".vals", Pod::F64, 16, 1, mats),
        scalar("isNotConstantIdentity", Pod::Bool, 1, 0, vec![vec![1]]),
        Property::Array {
            name: ".animChans".into(),
            meta: String::new(),
            pod: Pod::U32,
            extent: 1,
            ts: 1,
            samples: vec![
                (
                    anim.iter().flat_map(|c| c.to_le_bytes()).collect(),
                    anim.len() as u64
                );
                n as usize
            ],
        },
    ];
    let h_ap = sensor_mm / 10.0; // centimetres
    let v_ap = h_ap * s.height / s.width;
    let core = f64s(&[
        s.focal_mm(sensor_mm),
        h_ap,
        0.0,
        v_ap,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        5.6,
        5.0,
        0.0,
        1.0 / (2.0 * fps),
        0.1,
        100000.0,
    ]);
    let camera = Object {
        name: format!("{name}Shape"),
        meta: "schema=AbcGeom_Camera_v1;schemaObjTitle=AbcGeom_Camera_v1:.geom".into(),
        props: vec![Property::Compound {
            name: ".geom".into(),
            meta: "schema=AbcGeom_Camera_v1".into(),
            children: vec![scalar(".core", Pod::F64, 16, 1, vec![core; n as usize])],
        }],
        children: vec![],
    };
    let xform = Object {
        name: name.into(),
        meta: "schema=AbcGeom_Xform_v3;schemaObjTitle=AbcGeom_Xform_v3:.xform".into(),
        props: vec![Property::Compound {
            name: ".xform".into(),
            meta: "schema=AbcGeom_Xform_v3".into(),
            children: xform_props,
        }],
        children: vec![camera],
    };
    let mut top = vec![xform];
    let pts = points(s, 3.0);
    if !pts.is_empty() {
        let p: Vec<f32> = pts
            .iter()
            .flat_map(|q| q.iter().map(|v| *v as f32))
            .collect();
        let ids: Vec<u8> = (0..pts.len() as u64)
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for q in &pts {
            for k in 0..3 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        top.push(Object {
            name: "TrackwrightPoints".into(),
            meta: "schema=AbcGeom_Points_v1;schemaObjTitle=AbcGeom_Points_v1:.geom".into(),
            props: vec![Property::Compound {
                name: ".geom".into(),
                meta: "schema=AbcGeom_Points_v1".into(),
                children: vec![
                    Property::Scalar {
                        name: ".selfBnds".into(),
                        meta: "interpretation=box".into(),
                        pod: Pod::F64,
                        extent: 6,
                        ts: 0,
                        samples: vec![f64s(&[lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]])],
                    },
                    Property::Array {
                        name: "P".into(),
                        meta: "geoScope=var;interpretation=point".into(),
                        pod: Pod::F32,
                        extent: 3,
                        ts: 0,
                        samples: vec![(f32s(&p), pts.len() as u64)],
                    },
                    Property::Array {
                        name: ".pointIds".into(),
                        meta: "geoScope=var".into(),
                        pod: Pod::U64,
                        extent: 1,
                        ts: 0,
                        samples: vec![(ids, pts.len() as u64)],
                    },
                ],
            }],
            children: vec![],
        });
    }
    let sampling = TimeSampling {
        fps,
        start,
        count: n,
    };
    write(
        top,
        &sampling,
        &format!(
            "_ai_Description=Camera solved by Trackwright from {}",
            f.source.replace(';', ",")
        ),
    )
}

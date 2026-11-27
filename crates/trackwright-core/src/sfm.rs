//! Camera solve (structure from motion): turns 2D point tracks into a moving camera,
//! its focal length and lens distortion, and a 3D point cloud.

use crate::epipolar::{
    decompose_essential, focal_from_fundamental, pnp_dlt, ransac_fundamental, triangulate,
};
use crate::geometry::{mul, ransac, Mat3, MotionModel, Point, XorShift};
use crate::lens::Lens;
use crate::linalg::{
    self, add, cross, dot, log_rotation, mat_vec, normalized, rodrigues, scale, skew, solve_spd,
    sub, transpose, V3,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};

/// A 2D feature tracked through the footage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointTrack {
    pub id: u32,
    pub points: BTreeMap<usize, Point>,
}

/// World -> camera transform (OpenCV convention: x right, y down, z forward).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Pose {
    /// Rotation as an angle-axis vector.
    pub r: V3,
    pub t: V3,
}

impl Pose {
    pub fn rot(&self) -> Mat3 {
        rodrigues(self.r)
    }
    pub fn apply(&self, x: V3) -> V3 {
        add(mat_vec(&self.rot(), x), self.t)
    }
    /// Camera centre in world coordinates.
    pub fn center(&self) -> V3 {
        scale(mat_vec(&transpose(&self.rot()), self.t), -1.0)
    }
    fn from_rt(r: &Mat3, t: V3) -> Self {
        Self {
            r: log_rotation(r),
            t,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum FocalMode {
    /// Solve the focal length from the footage.
    #[default]
    Auto,
    /// Focal length is known (millimetres, with `sensor_mm`).
    Known { mm: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SolveMode {
    /// Free camera unless the shot has no parallax, then tripod.
    #[default]
    Auto,
    Free,
    /// Camera only rotates (nodal pan / tilt).
    Tripod,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraSettings {
    pub focal: FocalMode,
    /// Sensor (film back) width in mm, used to convert focal lengths.
    pub sensor_mm: f64,
    pub mode: SolveMode,
    /// Solve radial distortion k1 along with the camera.
    pub solve_distortion: bool,
    pub max_points: usize,
    /// Minimum spacing between auto-tracked points, in pixels at 1920 wide.
    pub min_distance: f64,
    /// Layers whose areas are ignored (moving objects, people, sky...).
    pub exclude_layers: Vec<u32>,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            focal: FocalMode::Auto,
            sensor_mm: 36.0,
            mode: SolveMode::Auto,
            solve_distortion: false,
            max_points: 500,
            min_distance: 22.0,
            exclude_layers: vec![],
        }
    }
}

/// Maps solve coordinates to the user's scene (Y up, right-handed).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct World {
    pub rotation: Mat3,
    pub translation: V3,
    pub scale: f64,
}

impl Default for World {
    fn default() -> Self {
        // First camera's frame with Y flipped up and Z pointing back (OpenGL style).
        Self {
            rotation: [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
            translation: [0.0; 3],
            scale: 1.0,
        }
    }
}

impl World {
    pub fn to_user(&self, p: V3) -> V3 {
        add(
            scale(mat_vec(&self.rotation, p), self.scale),
            self.translation,
        )
    }
    pub fn from_user(&self, q: V3) -> V3 {
        mat_vec(
            &transpose(&self.rotation),
            scale(sub(q, self.translation), 1.0 / self.scale),
        )
    }
    /// Camera position and orientation in the user's scene. The rotation's columns are the
    /// camera's right, up and backward axes (OpenGL camera: looks down -Z).
    pub fn camera(&self, pose: &Pose) -> (V3, Mat3) {
        let c = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
        let r = mul(&self.rotation, &mul(&transpose(&pose.rot()), &c));
        (self.to_user(pose.center()), r)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolvedPoint {
    pub id: u32,
    pub pos: V3,
    /// RMS reprojection error of this point (pixels).
    pub error: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraSolve {
    pub focal_px: f64,
    pub lens: Lens,
    pub width: f64,
    pub height: f64,
    pub poses: BTreeMap<usize, Pose>,
    pub points: Vec<SolvedPoint>,
    pub rms_px: f64,
    pub keyframes: Vec<usize>,
    pub tripod: bool,
    pub world: World,
}

impl CameraSolve {
    /// Ideal (undistorted) pixel of a world point, if it is in front of the camera.
    pub fn project_ideal(&self, pose: &Pose, x: V3) -> Option<Point> {
        let c = pose.apply(x);
        (c[2] > 1e-6).then(|| {
            [
                self.focal_px * c[0] / c[2] + self.width / 2.0,
                self.focal_px * c[1] / c[2] + self.height / 2.0,
            ]
        })
    }
    /// Where a world point appears in the footage at `frame`.
    pub fn project(&self, frame: usize, x: V3) -> Option<Point> {
        let p = self.project_ideal(self.poses.get(&frame)?, x)?;
        Some(self.lens.distort(p, self.width, self.height))
    }
    /// Project a point given in the user's scene coordinates.
    pub fn project_user(&self, frame: usize, q: V3) -> Option<Point> {
        self.project(frame, self.world.from_user(q))
    }
    pub fn focal_mm(&self, sensor_mm: f64) -> f64 {
        self.focal_px * sensor_mm / self.width
    }
}

/// Everything about the camera track stored in the project.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct CameraData {
    pub settings: CameraSettings,
    pub tracks: Vec<PointTrack>,
    pub solve: Option<CameraSolve>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveError(pub String);

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SolveError {}

fn err<T>(s: impl Into<String>) -> Result<T, SolveError> {
    Err(SolveError(s.into()))
}

// ---------------------------------------------------------------------------
// Bundle adjustment

struct Ba<'a> {
    cams: Vec<Pose>,
    fixed: Vec<bool>,
    points: Vec<V3>,
    /// (camera, point, observed footage pixel)
    obs: &'a [(usize, usize, Point)],
    focal: f64,
    lens: Lens,
    w: f64,
    h: f64,
    solve_focal: bool,
    solve_k1: bool,
    tripod: bool,
    huber: f64,
}

struct ObsJac {
    r: [f64; 2],
    jc: [[f64; 6]; 2],
    ji: [[f64; 2]; 2],
    jp: [[f64; 3]; 2],
}

impl<'a> Ba<'a> {
    fn cam_dim(&self) -> usize {
        if self.tripod {
            3
        } else {
            6
        }
    }

    fn intr_dim(&self) -> usize {
        self.solve_focal as usize + self.solve_k1 as usize
    }

    fn residual(
        &self,
        cams: &[Pose],
        points: &[V3],
        focal: f64,
        lens: &Lens,
        o: &(usize, usize, Point),
    ) -> Option<[f64; 2]> {
        let c = cams[o.0].apply(points[o.1]);
        if c[2] <= 1e-6 {
            return None;
        }
        let ideal = [
            focal * c[0] / c[2] + self.w / 2.0,
            focal * c[1] / c[2] + self.h / 2.0,
        ];
        let d = lens.distort(ideal, self.w, self.h);
        Some([d[0] - o.2[0], d[1] - o.2[1]])
    }

    fn huber_w(&self, r: &[f64; 2]) -> f64 {
        let e = (r[0] * r[0] + r[1] * r[1]).sqrt();
        if e <= self.huber {
            1.0
        } else {
            self.huber / e
        }
    }

    fn cost(&self, cams: &[Pose], points: &[V3], focal: f64, lens: &Lens) -> f64 {
        self.obs
            .iter()
            .map(|o| match self.residual(cams, points, focal, lens, o) {
                Some(r) => {
                    let e = (r[0] * r[0] + r[1] * r[1]).sqrt();
                    if e <= self.huber {
                        e * e
                    } else {
                        2.0 * self.huber * e - self.huber * self.huber
                    }
                }
                None => 1e6,
            })
            .sum()
    }

    fn jacobian(&self, o: &(usize, usize, Point)) -> Option<ObsJac> {
        let pose = &self.cams[o.0];
        let rm = pose.rot();
        let rx = mat_vec(&rm, self.points[o.1]);
        let c = add(rx, pose.t);
        if c[2] <= 1e-6 {
            return None;
        }
        let (xn, yn) = (c[0] / c[2], c[1] / c[2]);
        let f = self.focal;
        let ideal = [f * xn + self.w / 2.0, f * yn + self.h / 2.0];
        let d = self.lens.distort(ideal, self.w, self.h);
        let r = [d[0] - o.2[0], d[1] - o.2[1]];
        // Distortion Jacobian at the ideal point.
        let rr = 0.5 * (self.w * self.w + self.h * self.h).sqrt();
        let u = [
            (ideal[0] - self.w / 2.0) / rr,
            (ideal[1] - self.h / 2.0) / rr,
        ];
        let r2 = u[0] * u[0] + u[1] * u[1];
        let fac = 1.0 + self.lens.k1 * r2 + self.lens.k2 * r2 * r2;
        let dfac = self.lens.k1 + 2.0 * self.lens.k2 * r2;
        let dd = [
            [fac + 2.0 * dfac * u[0] * u[0], 2.0 * dfac * u[0] * u[1]],
            [2.0 * dfac * u[0] * u[1], fac + 2.0 * dfac * u[1] * u[1]],
        ];
        // d ideal / d camera point.
        let iz = f / c[2];
        let dic = [[iz, 0.0, -iz * xn], [0.0, iz, -iz * yn]];
        // Combined d footage-pixel / d camera point.
        let mut a = [[0.0; 3]; 2];
        for i in 0..2 {
            for k in 0..3 {
                a[i][k] = dd[i][0] * dic[0][k] + dd[i][1] * dic[1][k];
            }
        }
        let mut jp = [[0.0; 3]; 2];
        let mut jc = [[0.0; 6]; 2];
        let neg_skew = skew(scale(rx, -1.0));
        for i in 0..2 {
            for k in 0..3 {
                jp[i][k] = (0..3).map(|m| a[i][m] * rm[m * 3 + k]).sum();
                jc[i][k] = (0..3).map(|m| a[i][m] * neg_skew[m * 3 + k]).sum();
                jc[i][3 + k] = a[i][k];
            }
        }
        let mut ji = [[0.0; 2]; 2];
        let mut col = 0;
        if self.solve_focal {
            for i in 0..2 {
                ji[i][col] = dd[i][0] * xn + dd[i][1] * yn;
            }
            col += 1;
        }
        if self.solve_k1 {
            ji[0][col] = rr * u[0] * r2;
            ji[1][col] = rr * u[1] * r2;
        }
        Some(ObsJac { r, jc, ji, jp })
    }

    /// Levenberg-Marquardt with the Schur complement over points. Returns RMS error (px).
    fn run(&mut self, iterations: usize, cancel: &AtomicBool) -> f64 {
        let cd = self.cam_dim();
        let ni = self.intr_dim();
        // Free-camera parameter offsets.
        let mut cam_off = vec![usize::MAX; self.cams.len()];
        let mut nc = 0;
        for (i, f) in self.fixed.iter().enumerate() {
            if !f {
                cam_off[i] = nc;
                nc += cd;
            }
        }
        let side = nc + ni;
        let mut by_point: Vec<Vec<usize>> = vec![vec![]; self.points.len()];
        for (k, o) in self.obs.iter().enumerate() {
            by_point[o.1].push(k);
        }
        let mut lambda = 1e-3;
        let mut cost = self.cost(&self.cams, &self.points, self.focal, &self.lens);
        for _ in 0..iterations {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let mut u = vec![0.0; side * side];
            let mut bs = vec![0.0; side];
            let mut vblocks = vec![[0.0; 9]; self.points.len()];
            let mut bp = vec![[0.0; 3]; self.points.len()];
            // Per point: sparse W rows (side index -> 3 values).
            let mut wrows: Vec<HashMap<usize, [f64; 3]>> = vec![HashMap::new(); self.points.len()];
            for (k, o) in self.obs.iter().enumerate() {
                let _ = k;
                let Some(j) = self.jacobian(o) else { continue };
                let w = self.huber_w(&j.r);
                // Side Jacobian as (index, row0, row1).
                let mut sj: Vec<(usize, f64, f64)> = Vec::with_capacity(cd + ni);
                if cam_off[o.0] != usize::MAX {
                    let base = cam_off[o.0];
                    for p in 0..cd {
                        sj.push((base + p, j.jc[0][p], j.jc[1][p]));
                    }
                }
                for p in 0..ni {
                    sj.push((nc + p, j.ji[0][p], j.ji[1][p]));
                }
                for &(a, a0, a1) in &sj {
                    for &(b, b0, b1) in &sj {
                        u[a * side + b] += w * (a0 * b0 + a1 * b1);
                    }
                    bs[a] -= w * (a0 * j.r[0] + a1 * j.r[1]);
                }
                let pi = o.1;
                for a in 0..3 {
                    for b in 0..3 {
                        vblocks[pi][a * 3 + b] +=
                            w * (j.jp[0][a] * j.jp[0][b] + j.jp[1][a] * j.jp[1][b]);
                    }
                    bp[pi][a] -= w * (j.jp[0][a] * j.r[0] + j.jp[1][a] * j.r[1]);
                }
                for &(a, a0, a1) in &sj {
                    let e = wrows[pi].entry(a).or_insert([0.0; 3]);
                    for b in 0..3 {
                        e[b] += w * (a0 * j.jp[0][b] + a1 * j.jp[1][b]);
                    }
                }
            }
            let mut accepted = false;
            for _attempt in 0..8 {
                // Damped copies.
                let mut s = u.clone();
                let umax = (0..side)
                    .map(|i| u[i * side + i])
                    .fold(0.0, f64::max)
                    .max(1e-12);
                for i in 0..side {
                    s[i * side + i] += lambda * u[i * side + i] + 1e-9 * umax;
                }
                let mut rhs = bs.clone();
                let mut vinv = vec![[0.0; 9]; self.points.len()];
                for p in 0..self.points.len() {
                    if wrows[p].is_empty() {
                        continue;
                    }
                    let mut v = vblocks[p];
                    let tr = v[0] + v[4] + v[8];
                    for d in 0..3 {
                        v[d * 4] += lambda * v[d * 4] + 1e-9 * tr.max(1e-12);
                    }
                    let Some(vi) = crate::geometry::invert(&v) else {
                        continue;
                    };
                    vinv[p] = vi;
                    let rows: Vec<(usize, [f64; 3])> =
                        wrows[p].iter().map(|(k, v)| (*k, *v)).collect();
                    let vb = mat_vec(&vi, bp[p]);
                    for &(a, wa) in &rows {
                        let wv = mat_vec(&vi, wa); // V⁻¹ wₐ (V symmetric)
                        for &(b, wb) in &rows {
                            s[a * side + b] -= dot(wv, wb);
                        }
                        rhs[a] -= dot(wa, vb);
                    }
                }
                let ds = if side == 0 {
                    Some(vec![])
                } else {
                    solve_spd(&s, &rhs, side)
                };
                let Some(ds) = ds else {
                    lambda *= 10.0;
                    continue;
                };
                // Back-substitute points.
                let mut new_points = self.points.clone();
                for p in 0..self.points.len() {
                    if wrows[p].is_empty() {
                        continue;
                    }
                    let mut b = bp[p];
                    for (a, wa) in &wrows[p] {
                        for k in 0..3 {
                            b[k] -= wa[k] * ds[*a];
                        }
                    }
                    new_points[p] = add(self.points[p], mat_vec(&vinv[p], b));
                }
                let mut new_cams = self.cams.clone();
                for (i, c) in new_cams.iter_mut().enumerate() {
                    if cam_off[i] == usize::MAX {
                        continue;
                    }
                    let o = cam_off[i];
                    let dr = [ds[o], ds[o + 1], ds[o + 2]];
                    let rn = mul(&rodrigues(dr), &c.rot());
                    c.r = log_rotation(&rn);
                    if !self.tripod {
                        c.t = add(c.t, [ds[o + 3], ds[o + 4], ds[o + 5]]);
                    }
                }
                let mut nf = self.focal;
                let mut nl = self.lens;
                let mut col = nc;
                if self.solve_focal {
                    nf += ds[col];
                    col += 1;
                }
                if self.solve_k1 {
                    nl.k1 += ds[col];
                }
                if !(nf.is_finite() && nf > 10.0) {
                    lambda *= 10.0;
                    continue;
                }
                let nc_cost = self.cost(&new_cams, &new_points, nf, &nl);
                if nc_cost < cost {
                    let rel = (cost - nc_cost) / cost.max(1e-30);
                    self.cams = new_cams;
                    self.points = new_points;
                    self.focal = nf;
                    self.lens = nl;
                    cost = nc_cost;
                    lambda = (lambda * 0.3).max(1e-12);
                    accepted = true;
                    if rel < 1e-9 {
                        return self.rms();
                    }
                    break;
                }
                lambda *= 10.0;
            }
            if !accepted {
                break;
            }
        }
        self.rms()
    }

    fn rms(&self) -> f64 {
        let mut s = 0.0;
        let mut n = 0;
        for o in self.obs {
            if let Some(r) = self.residual(&self.cams, &self.points, self.focal, &self.lens, o) {
                s += r[0] * r[0] + r[1] * r[1];
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            (s / n as f64).sqrt()
        }
    }
}

// ---------------------------------------------------------------------------
// Pose refinement for one camera with fixed points (used for every frame).

fn refine_pose(
    pose: Pose,
    pts: &[(V3, Point)],
    focal: f64,
    lens: Lens,
    w: f64,
    h: f64,
    tripod: bool,
) -> (Pose, f64) {
    if pts.len() < 4 {
        return (pose, f64::INFINITY);
    }
    let obs: Vec<(usize, usize, Point)> = pts
        .iter()
        .enumerate()
        .map(|(i, (_, p))| (0, i, *p))
        .collect();
    // Points fixed: put them in, but mark them as not optimised by giving every point
    // a single observation (the BA skips nothing, so use a dedicated loop instead).
    let mut ba = Ba {
        cams: vec![pose],
        fixed: vec![false],
        points: pts.iter().map(|(x, _)| *x).collect(),
        obs: &obs,
        focal,
        lens,
        w,
        h,
        solve_focal: false,
        solve_k1: false,
        tripod,
        huber: 2.0,
    };
    let cd = ba.cam_dim();
    let mut lambda = 1e-3;
    let mut cost = ba.cost(&ba.cams, &ba.points, focal, &lens);
    for _ in 0..20 {
        let mut jtj = vec![0.0; cd * cd];
        let mut jtr = vec![0.0; cd];
        for o in &obs {
            let Some(j) = ba.jacobian(o) else { continue };
            let wgt = ba.huber_w(&j.r);
            for a in 0..cd {
                for b in 0..cd {
                    jtj[a * cd + b] += wgt * (j.jc[0][a] * j.jc[0][b] + j.jc[1][a] * j.jc[1][b]);
                }
                jtr[a] -= wgt * (j.jc[0][a] * j.r[0] + j.jc[1][a] * j.r[1]);
            }
        }
        let mut ok = false;
        for _ in 0..8 {
            let mut m = jtj.clone();
            for a in 0..cd {
                m[a * cd + a] += lambda * jtj[a * cd + a] + 1e-12;
            }
            let Some(d) = solve_spd(&m, &jtr, cd) else {
                lambda *= 10.0;
                continue;
            };
            let mut np = ba.cams[0];
            np.r = log_rotation(&mul(&rodrigues([d[0], d[1], d[2]]), &np.rot()));
            if !tripod {
                np.t = add(np.t, [d[3], d[4], d[5]]);
            }
            let c = ba.cost(&[np], &ba.points, focal, &lens);
            if c < cost {
                let rel = (cost - c) / cost.max(1e-30);
                ba.cams[0] = np;
                cost = c;
                lambda = (lambda * 0.3).max(1e-12);
                ok = rel > 1e-10;
                break;
            }
            lambda *= 10.0;
        }
        if !ok {
            break;
        }
    }
    let rms = ba.rms();
    (ba.cams[0], rms)
}

// ---------------------------------------------------------------------------
// The solver

pub struct SolveOptions {
    pub width: f64,
    pub height: f64,
    pub focal: FocalMode,
    pub sensor_mm: f64,
    pub mode: SolveMode,
    pub solve_distortion: bool,
    /// Known lens distortion (from the Lens panel); used as the starting point.
    pub lens: Lens,
}

struct Obs {
    /// frame -> list of (track index, footage pixel)
    by_frame: BTreeMap<usize, Vec<(usize, Point)>>,
}

impl Obs {
    fn common(&self, a: usize, b: usize) -> Vec<(usize, Point, Point)> {
        let (Some(fa), Some(fb)) = (self.by_frame.get(&a), self.by_frame.get(&b)) else {
            return vec![];
        };
        let mb: HashMap<usize, Point> = fb.iter().copied().collect();
        fa.iter()
            .filter_map(|(t, p)| mb.get(t).map(|q| (*t, *p, *q)))
            .collect()
    }
}

/// Solve the camera for `frames` (sorted, contiguous) from `tracks`.
pub fn solve_camera(
    tracks: &[PointTrack],
    frames: &[usize],
    opts: &SolveOptions,
    cancel: &AtomicBool,
    mut progress: impl FnMut(&str, f64),
) -> Result<CameraSolve, SolveError> {
    let (w, h) = (opts.width, opts.height);
    if frames.len() < 3 {
        return err("Need at least 3 frames to solve a camera");
    }
    let mut by_frame: BTreeMap<usize, Vec<(usize, Point)>> = BTreeMap::new();
    for (ti, t) in tracks.iter().enumerate() {
        if t.points.len() < 3 {
            continue;
        }
        for (&f, &p) in &t.points {
            if frames.binary_search(&f).is_ok() {
                by_frame.entry(f).or_default().push((ti, p));
            }
        }
    }
    let obs = Obs { by_frame };
    let lens0 = opts.lens;
    let und = |p: Point| lens0.undistort(p, w, h);

    // 1. Keyframes: a new key whenever too many tracks have changed or every 12 frames.
    progress("Choosing keyframes", 0.02);
    let mut keys = vec![frames[0]];
    for (i, &f) in frames.iter().enumerate().skip(1) {
        let last = *keys.last().unwrap();
        let n_last = obs.by_frame.get(&last).map_or(0, |v| v.len()).max(1);
        let common = obs.common(last, f).len();
        let is_end = i + 1 == frames.len();
        if common < 40 {
            if f - last <= 1 {
                return err(format!("Too few tracked points carry over between frames {last} and {f} ({common}). Track points again or shorten the range"));
            }
            // Use the previous frame instead.
            let prev = frames[i - 1];
            if prev != last {
                keys.push(prev);
            }
        }
        let last = *keys.last().unwrap();
        if (common as f64) < 0.7 * n_last as f64 || f - last >= 12 || is_end {
            if f != last {
                keys.push(f);
            }
        }
    }
    keys.dedup();
    if keys.len() < 2 {
        return err("Not enough keyframes");
    }

    // 2. Initial pair with the most parallax that still shares many tracks.
    progress("Finding the starting frames", 0.06);
    let mut best: Option<(f64, usize, Mat3, Vec<(usize, Point, Point)>, Vec<bool>, f64)> = None;
    for j in 1..keys.len().min(10) {
        let com = obs.common(keys[0], keys[j]);
        if com.len() < 30 {
            break;
        }
        let a: Vec<Point> = com.iter().map(|c| und(c.1)).collect();
        let b: Vec<Point> = com.iter().map(|c| und(c.2)).collect();
        let Some((fm, inl)) = ransac_fundamental(&a, &b, 1.5, j as u64) else {
            continue;
        };
        let n_in = inl.iter().filter(|&&x| x).count();
        let hr = ransac(MotionModel::Perspective, &a, &b, 1.5, 7 + j as u64)
            .map_or(1.0, |r| r.inlier_count as f64 / com.len() as f64);
        let mut flows: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(p, q)| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt())
            .collect();
        flows.sort_by(f64::total_cmp);
        let parallax = flows[flows.len() / 2] / w;
        let score = n_in as f64 * (1.0 - hr).max(0.0) * parallax.min(0.2);
        if best.as_ref().map_or(true, |b| score > b.0) {
            best = Some((score, j, fm, com, inl, hr));
        }
    }
    let Some((score, j0, fmat, com0, inl0, hratio)) = best else {
        return err(
            "Could not match points between the first keyframes. Try tracking points again",
        );
    };
    let tripod = match opts.mode {
        SolveMode::Tripod => true,
        SolveMode::Free => false,
        SolveMode::Auto => hratio > 0.92 || score < 1e-3,
    };

    // 3. Focal length.
    let known_focal = match opts.focal {
        FocalMode::Known { mm } => Some(mm / opts.sensor_mm * w),
        FocalMode::Auto => None,
    };
    let focal0 = if let Some(f) = known_focal {
        f
    } else if tripod {
        focal_from_rotations(&obs, &keys, &und, w, h).unwrap_or(w)
    } else {
        focal_from_fundamental(&fmat, w, h)
    };
    if !(focal0.is_finite() && focal0 > 0.1 * w && focal0 < 20.0 * w) {
        return err(
            "Could not estimate the focal length. Enter it under Focal length (mm) and solve again",
        );
    }
    let (cx, cy) = (w / 2.0, h / 2.0);
    let norm_pt = |p: Point, f: f64| [(p[0] - cx) / f, (p[1] - cy) / f];

    // Shared solve state.
    let mut poses: BTreeMap<usize, Pose> = BTreeMap::new();
    let mut points: HashMap<usize, V3> = HashMap::new();
    let mut focal = focal0;
    let mut lens = lens0;
    let ident = Pose::default();
    poses.insert(keys[0], ident);

    if tripod {
        // Rotations chained from homographies between consecutive keyframes.
        let kinv = [
            1.0 / focal,
            0.0,
            -cx / focal,
            0.0,
            1.0 / focal,
            -cy / focal,
            0.0,
            0.0,
            1.0,
        ];
        let kmat = [focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0];
        let mut r_acc = rodrigues([0.0; 3]);
        for k in 1..keys.len() {
            let com = obs.common(keys[k - 1], keys[k]);
            let a: Vec<Point> = com.iter().map(|c| und(c.1)).collect();
            let b: Vec<Point> = com.iter().map(|c| und(c.2)).collect();
            let Some(hfit) = ransac(MotionModel::Perspective, &a, &b, 2.0, k as u64) else {
                return err(format!(
                    "Tripod solve: could not match frames {} and {}",
                    keys[k - 1],
                    keys[k]
                ));
            };
            let rrel = linalg::nearest_rotation(&mul(&kinv, &mul(&hfit.model, &kmat)));
            r_acc = mul(&rrel, &r_acc);
            poses.insert(keys[k], Pose::from_rt(&r_acc, [0.0; 3]));
        }
        // Points as directions at a fixed distance.
        for (ti, t) in tracks.iter().enumerate() {
            if let Some((&f, &p)) = t.points.iter().find(|(f, _)| poses.contains_key(f)) {
                let q = norm_pt(und(p), focal);
                let d = normalized([q[0], q[1], 1.0]);
                let world = mat_vec(&transpose(&poses[&f].rot()), d);
                points.insert(ti, scale(world, 100.0));
            }
        }
    } else {
        // 4. Two-view start.
        progress("Solving the first two keyframes", 0.1);
        let kmat = [focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0];
        let e = mul(&transpose(&kmat), &mul(&fmat, &kmat));
        let inliers: Vec<&(usize, Point, Point)> = com0
            .iter()
            .zip(&inl0)
            .filter(|(_, &i)| i)
            .map(|(c, _)| c)
            .collect();
        let i3 = rodrigues([0.0; 3]);
        let mut best_pose: Option<(usize, Mat3, V3)> = None;
        for (r, t) in decompose_essential(&e) {
            let mut good = 0;
            for c in &inliers {
                let a = norm_pt(und(c.1), focal);
                let b = norm_pt(und(c.2), focal);
                if let Some(x) = triangulate(&[(i3, [0.0; 3], a), (r, t, b)]) {
                    let xb = add(mat_vec(&r, x), t);
                    if x[2] > 0.0 && xb[2] > 0.0 {
                        good += 1;
                    }
                }
            }
            if best_pose.as_ref().map_or(true, |b| good > b.0) {
                best_pose = Some((good, r, t));
            }
        }
        let (_, r1, t1) = best_pose.unwrap();
        poses.insert(keys[j0], Pose::from_rt(&r1, t1));
        triangulate_new(&obs, &poses, &mut points, focal, &und, cx, cy, 1.0);
        if points.len() < 20 {
            return err("Too few 3D points from the starting frames. The camera may not move enough; try Tripod mode");
        }
    }

    // 5. Add the remaining keyframes outward from the solved ones.
    let solve_focal_now = known_focal.is_none();
    let mut since_ba = 0;
    let total_keys = keys.len();
    let ba_run = |poses: &mut BTreeMap<usize, Pose>,
                  points: &mut HashMap<usize, V3>,
                  focal: &mut f64,
                  lens: &mut Lens,
                  solve_k1: bool,
                  iters: usize| {
        let cam_frames: Vec<usize> = poses.keys().copied().collect();
        let cam_index: HashMap<usize, usize> = cam_frames
            .iter()
            .enumerate()
            .map(|(i, f)| (*f, i))
            .collect();
        let pt_ids: Vec<usize> = points.keys().copied().collect();
        let pt_index: HashMap<usize, usize> =
            pt_ids.iter().enumerate().map(|(i, t)| (*t, i)).collect();
        let mut ob = Vec::new();
        for (&f, list) in &obs.by_frame {
            let Some(&ci) = cam_index.get(&f) else {
                continue;
            };
            if !keys.contains(&f) {
                continue;
            }
            for (t, p) in list {
                if let Some(&pi) = pt_index.get(t) {
                    ob.push((ci, pi, *p));
                }
            }
        }
        let mut ba = Ba {
            cams: cam_frames.iter().map(|f| poses[f]).collect(),
            fixed: cam_frames.iter().map(|f| *f == keys[0]).collect(),
            points: pt_ids.iter().map(|t| points[t]).collect(),
            obs: &ob,
            focal: *focal,
            lens: *lens,
            w,
            h,
            solve_focal: solve_focal_now,
            solve_k1,
            tripod,
            huber: 2.0,
        };
        let rms = ba.run(iters, cancel);
        for (i, f) in cam_frames.iter().enumerate() {
            poses.insert(*f, ba.cams[i]);
        }
        for (i, t) in pt_ids.iter().enumerate() {
            points.insert(*t, ba.points[i]);
        }
        *focal = ba.focal;
        *lens = ba.lens;
        rms
    };
    ba_run(&mut poses, &mut points, &mut focal, &mut lens, false, 30);

    loop {
        if cancel.load(Ordering::Relaxed) {
            return err("Solve cancelled");
        }
        // Next keyframe adjacent to the solved set.
        let solved: Vec<usize> = keys
            .iter()
            .copied()
            .filter(|k| poses.contains_key(k))
            .collect();
        if solved.len() == keys.len() {
            break;
        }
        let first_i = keys.iter().position(|k| poses.contains_key(k)).unwrap();
        let mut next = None;
        for (i, k) in keys.iter().enumerate() {
            if poses.contains_key(k) {
                continue;
            }
            let neighbour_solved = (i > 0 && poses.contains_key(&keys[i - 1]))
                || (i + 1 < keys.len() && poses.contains_key(&keys[i + 1]));
            if neighbour_solved {
                next = Some(i);
                if i > first_i {
                    break;
                }
            }
        }
        let Some(ni) = next else { break };
        let kf = keys[ni];
        progress(
            &format!("Solving keyframe {} of {}", solved.len() + 1, total_keys),
            0.1 + 0.6 * solved.len() as f64 / total_keys as f64,
        );
        let neighbour = if ni > 0 && poses.contains_key(&keys[ni - 1]) {
            keys[ni - 1]
        } else {
            keys[ni + 1]
        };
        let pose = resection(&obs, kf, &points, poses[&neighbour], focal, lens, &und, w, h, tripod)
            .ok_or_else(|| SolveError(format!("Lost the camera at frame {kf}: too few known 3D points are visible. Track more points or exclude moving objects")))?;
        poses.insert(kf, pose);
        if !tripod {
            triangulate_new(&obs, &poses, &mut points, focal, &und, cx, cy, 1.5);
        } else {
            add_tripod_points(&obs, &poses, &mut points, kf, focal, &und, cx, cy);
        }
        since_ba += 1;
        if since_ba >= 3 || solved.len() < 6 {
            ba_run(&mut poses, &mut points, &mut focal, &mut lens, false, 10);
            remove_outlier_points(&obs, &poses, &mut points, focal, lens, w, h, 4.0);
            since_ba = 0;
        }
    }

    // 6. Final adjustment over keyframes (with distortion when asked).
    progress("Refining", 0.75);
    ba_run(&mut poses, &mut points, &mut focal, &mut lens, false, 40);
    remove_outlier_points(&obs, &poses, &mut points, focal, lens, w, h, 3.0);
    ba_run(
        &mut poses,
        &mut points,
        &mut focal,
        &mut lens,
        opts.solve_distortion,
        60,
    );

    // 7. Every other frame: interpolate from the keyframes, then refine with points fixed.
    progress("Solving in-between frames", 0.88);
    let keyset: Vec<usize> = poses.keys().copied().collect();
    for &f in frames {
        if poses.contains_key(&f) && keys.contains(&f) {
            continue;
        }
        let before = keyset.iter().copied().filter(|k| *k <= f).next_back();
        let after = keyset.iter().copied().find(|k| *k >= f);
        let init = match (before, after) {
            (Some(a), Some(b)) if a != b => {
                interpolate_pose(&poses[&a], &poses[&b], (f - a) as f64 / (b - a) as f64)
            }
            (Some(a), _) => poses[&a],
            (None, Some(b)) => poses[&b],
            _ => continue,
        };
        let pts: Vec<(V3, Point)> = obs.by_frame.get(&f).map_or(vec![], |l| {
            l.iter()
                .filter_map(|(t, p)| points.get(t).map(|x| (*x, *p)))
                .collect()
        });
        let (pose, _) = refine_pose(init, &pts, focal, lens, w, h, tripod);
        poses.insert(f, pose);
    }

    // 8. Points seen only in in-between frames, then final statistics.
    if !tripod {
        triangulate_new(
            &obs,
            &poses,
            &mut points,
            focal,
            &|p| lens.undistort(p, w, h),
            cx,
            cy,
            2.0,
        );
    }
    progress("Measuring", 0.97);
    let mut solved_points = Vec::new();
    let (mut tot, mut cnt) = (0.0, 0usize);
    let tmp = CameraSolve {
        focal_px: focal,
        lens,
        width: w,
        height: h,
        poses: poses.clone(),
        points: vec![],
        rms_px: 0.0,
        keyframes: keys.clone(),
        tripod,
        world: World::default(),
    };
    for (ti, t) in tracks.iter().enumerate() {
        let Some(&x) = points.get(&ti) else { continue };
        let (mut s, mut n) = (0.0, 0);
        for (&f, &p) in &t.points {
            if let Some(q) = tmp.project(f, x) {
                s += (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2);
                n += 1;
            }
        }
        if n == 0 {
            continue;
        }
        let e = (s / n as f64).sqrt();
        if e < 4.0 {
            tot += s;
            cnt += n;
        }
        solved_points.push(SolvedPoint {
            id: t.id,
            pos: x,
            error: e,
        });
    }
    let rms = if cnt > 0 {
        (tot / cnt as f64).sqrt()
    } else {
        f64::INFINITY
    };
    progress("Done", 1.0);
    Ok(CameraSolve {
        points: solved_points,
        rms_px: rms,
        ..tmp
    })
}

/// Focal length for a rotating camera: the K that makes K⁻¹ H K closest to a rotation.
fn focal_from_rotations(
    obs: &Obs,
    keys: &[usize],
    und: &dyn Fn(Point) -> Point,
    w: f64,
    h: f64,
) -> Option<f64> {
    let (cx, cy) = (w / 2.0, h / 2.0);
    let mut hs = Vec::new();
    for k in 1..keys.len() {
        let com = obs.common(keys[0], keys[k]);
        if com.len() < 20 {
            break;
        }
        let a: Vec<Point> = com.iter().map(|c| und(c.1)).collect();
        let b: Vec<Point> = com.iter().map(|c| und(c.2)).collect();
        if let Some(r) = ransac(MotionModel::Perspective, &a, &b, 2.0, 99 + k as u64) {
            hs.push(r.model);
        }
    }
    if hs.is_empty() {
        return None;
    }
    let cost = |f: f64| {
        let k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0];
        let ki = [1.0 / f, 0.0, -cx / f, 0.0, 1.0 / f, -cy / f, 0.0, 0.0, 1.0];
        hs.iter()
            .map(|hm| {
                let m = mul(&ki, &mul(hm, &k));
                let d = linalg::det3(&m).abs().cbrt().max(1e-12);
                let m = m.map(|v| v / d);
                let mtm = mul(&transpose(&m), &m);
                (0..9)
                    .map(|i| (mtm[i] - if i % 4 == 0 { 1.0 } else { 0.0 }).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
    };
    let (lo, hi) = ((0.3 * w).ln(), (5.0 * w).ln());
    let mut best = (f64::INFINITY, w);
    for i in 0..=200 {
        let f = (lo + (hi - lo) * i as f64 / 200.0).exp();
        let c = cost(f);
        if c < best.0 {
            best = (c, f);
        }
    }
    Some(best.1)
}

#[allow(clippy::too_many_arguments)]
fn triangulate_new(
    obs: &Obs,
    poses: &BTreeMap<usize, Pose>,
    points: &mut HashMap<usize, V3>,
    focal: f64,
    und: &dyn Fn(Point) -> Point,
    cx: f64,
    cy: f64,
    min_angle_deg: f64,
) {
    let mut views: HashMap<usize, Vec<(usize, Point)>> = HashMap::new();
    for (&f, list) in &obs.by_frame {
        if !poses.contains_key(&f) {
            continue;
        }
        for (t, p) in list {
            if !points.contains_key(t) {
                views.entry(*t).or_default().push((f, *p));
            }
        }
    }
    let min_angle = min_angle_deg.to_radians();
    for (t, v) in views {
        if v.len() < 2 {
            continue;
        }
        // Use up to 8 views spread over the track.
        let step = (v.len() / 8).max(1);
        let chosen: Vec<&(usize, Point)> = v.iter().step_by(step).collect();
        let mut tv = Vec::new();
        for (f, p) in &chosen {
            let q = und(*p);
            let pose = &poses[f];
            tv.push((
                pose.rot(),
                pose.t,
                [(q[0] - cx) / focal, (q[1] - cy) / focal],
            ));
        }
        let Some(x) = triangulate(&tv) else { continue };
        // Cheirality, reprojection and baseline angle checks.
        let mut ok = true;
        let mut rays = Vec::new();
        for (f, p) in &chosen {
            let pose = &poses[f];
            let c = pose.apply(x);
            if c[2] <= 1e-6 {
                ok = false;
                break;
            }
            let q = und(*p);
            let u = [focal * c[0] / c[2] + cx, focal * c[1] / c[2] + cy];
            if (u[0] - q[0]).powi(2) + (u[1] - q[1]).powi(2) > 16.0 {
                ok = false;
                break;
            }
            rays.push(normalized(sub(x, pose.center())));
        }
        if !ok {
            continue;
        }
        let mut max_angle: f64 = 0.0;
        for a in &rays {
            for b in &rays {
                max_angle = max_angle.max(dot(*a, *b).clamp(-1.0, 1.0).acos());
            }
        }
        if max_angle >= min_angle {
            points.insert(t, x);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn add_tripod_points(
    obs: &Obs,
    poses: &BTreeMap<usize, Pose>,
    points: &mut HashMap<usize, V3>,
    frame: usize,
    focal: f64,
    und: &dyn Fn(Point) -> Point,
    cx: f64,
    cy: f64,
) {
    let Some(list) = obs.by_frame.get(&frame) else {
        return;
    };
    let pose = &poses[&frame];
    for (t, p) in list {
        if points.contains_key(t) {
            continue;
        }
        let q = und(*p);
        let d = normalized([(q[0] - cx) / focal, (q[1] - cy) / focal, 1.0]);
        points.insert(*t, scale(mat_vec(&transpose(&pose.rot()), d), 100.0));
    }
}

#[allow(clippy::too_many_arguments)]
fn resection(
    obs: &Obs,
    frame: usize,
    points: &HashMap<usize, V3>,
    guess: Pose,
    focal: f64,
    lens: Lens,
    und: &dyn Fn(Point) -> Point,
    w: f64,
    h: f64,
    tripod: bool,
) -> Option<Pose> {
    let list = obs.by_frame.get(&frame)?;
    let pts: Vec<(V3, Point)> = list
        .iter()
        .filter_map(|(t, p)| points.get(t).map(|x| (*x, *p)))
        .collect();
    if pts.len() < 8 {
        return None;
    }
    let (cx, cy) = (w / 2.0, h / 2.0);
    let mut init = guess;
    if !tripod {
        // RANSAC DLT for a robust start; fall back to the neighbour's pose.
        let mut rng = XorShift::new(frame as u64 * 31 + 7);
        let normd: Vec<Point> = pts
            .iter()
            .map(|(_, p)| {
                let q = und(*p);
                [(q[0] - cx) / focal, (q[1] - cy) / focal]
            })
            .collect();
        let mut best = (0usize, None);
        for _ in 0..200 {
            let mut idx = Vec::new();
            while idx.len() < 6 {
                let c = rng.below(pts.len());
                if !idx.contains(&c) {
                    idx.push(c);
                }
            }
            let wpts: Vec<V3> = idx.iter().map(|&i| pts[i].0).collect();
            let ipts: Vec<Point> = idx.iter().map(|&i| normd[i]).collect();
            let Some((r, t)) = pnp_dlt(&wpts, &ipts) else {
                continue;
            };
            let pose = Pose::from_rt(&r, t);
            let count = pts
                .iter()
                .zip(&normd)
                .filter(|((x, _), q)| {
                    let c = pose.apply(*x);
                    c[2] > 0.0
                        && ((c[0] / c[2] - q[0]).powi(2) + (c[1] / c[2] - q[1]).powi(2)).sqrt()
                            * focal
                            < 3.0
                })
                .count();
            if count > best.0 {
                best = (count, Some(pose));
            }
        }
        // Prefer the DLT start only if it explains clearly more points than the guess.
        let guess_count = pts
            .iter()
            .zip(&normd)
            .filter(|((x, _), q)| {
                let c = guess.apply(*x);
                c[2] > 0.0
                    && ((c[0] / c[2] - q[0]).powi(2) + (c[1] / c[2] - q[1]).powi(2)).sqrt() * focal
                        < 3.0
            })
            .count();
        if let (n, Some(p)) = best {
            if n > guess_count {
                init = p;
            }
        }
    }
    let (pose, rms) = refine_pose(init, &pts, focal, lens, w, h, tripod);
    (rms.is_finite() && rms < 20.0).then_some(pose)
}

#[allow(clippy::too_many_arguments)]
fn remove_outlier_points(
    obs: &Obs,
    poses: &BTreeMap<usize, Pose>,
    points: &mut HashMap<usize, V3>,
    focal: f64,
    lens: Lens,
    w: f64,
    h: f64,
    max_px: f64,
) {
    let mut err: HashMap<usize, (f64, usize)> = HashMap::new();
    for (&f, list) in &obs.by_frame {
        let Some(pose) = poses.get(&f) else { continue };
        for (t, p) in list {
            let Some(x) = points.get(t) else { continue };
            let c = pose.apply(*x);
            let e = if c[2] <= 1e-6 {
                1e6
            } else {
                let u = lens.distort(
                    [focal * c[0] / c[2] + w / 2.0, focal * c[1] / c[2] + h / 2.0],
                    w,
                    h,
                );
                (u[0] - p[0]).powi(2) + (u[1] - p[1]).powi(2)
            };
            let en = err.entry(*t).or_insert((0.0, 0));
            en.0 += e;
            en.1 += 1;
        }
    }
    for (t, (s, n)) in err {
        if (s / n as f64).sqrt() > max_px {
            points.remove(&t);
        }
    }
}

fn interpolate_pose(a: &Pose, b: &Pose, t: f64) -> Pose {
    let ra = a.rot();
    let rb = b.rot();
    let delta = log_rotation(&mul(&rb, &transpose(&ra)));
    let r = mul(&rodrigues(scale(delta, t)), &ra);
    let c = add(scale(a.center(), 1.0 - t), scale(b.center(), t));
    let t_vec = scale(mat_vec(&r, c), -1.0);
    Pose {
        r: log_rotation(&r),
        t: t_vec,
    }
}

// ---------------------------------------------------------------------------
// Scene orientation

/// Make the plane through `pts` (solve coordinates) the ground: origin at their centre,
/// +Y the plane normal pointing towards the camera at `cam_frame`, +X along `x_hint`
/// (a direction in solve coordinates, e.g. the top edge of a planar layer).
/// `scale_to` rescales so that `scale_ref` (a length in solve units) becomes that many units.
pub fn orient_to_plane(
    solve: &CameraSolve,
    pts: &[V3],
    cam_frame: usize,
    x_hint: Option<V3>,
    scale_ref: Option<(f64, f64)>,
) -> Option<World> {
    if pts.len() < 3 {
        return None;
    }
    let n = pts.len() as f64;
    let c = scale(pts.iter().fold([0.0; 3], |a, p| add(a, *p)), 1.0 / n);
    let mut cov = [0.0; 9];
    for p in pts {
        let d = sub(*p, c);
        for i in 0..3 {
            for j in 0..3 {
                cov[i * 3 + j] += d[i] * d[j];
            }
        }
    }
    let (_, vecs) = linalg::sym_eigen(&cov, 3);
    let mut up = normalized([vecs[0][0], vecs[0][1], vecs[0][2]]);
    let cam = solve.poses.get(&cam_frame)?.center();
    if dot(up, sub(cam, c)) < 0.0 {
        up = scale(up, -1.0);
    }
    let hint = x_hint.unwrap_or([vecs[2][0], vecs[2][1], vecs[2][2]]);
    let x = normalized(sub(hint, scale(up, dot(hint, up))));
    let z = cross(x, up);
    // Rows are the new axes expressed in solve coordinates.
    let rot = [x[0], x[1], x[2], up[0], up[1], up[2], z[0], z[1], z[2]];
    let s = match scale_ref {
        Some((measured, wanted)) if measured > 1e-12 => wanted / measured,
        _ => solve.world.scale,
    };
    let t = scale(mat_vec(&rot, c), -s);
    Some(World {
        rotation: rot,
        translation: t,
        scale: s,
    })
}

/// Intersect the camera ray through footage pixel `p` at `frame` with the plane through
/// `origin` with `normal` (solve coordinates).
pub fn ray_plane(
    solve: &CameraSolve,
    frame: usize,
    p: Point,
    origin: V3,
    normal: V3,
) -> Option<V3> {
    let pose = solve.poses.get(&frame)?;
    let q = solve.lens.undistort(p, solve.width, solve.height);
    let d_cam = [
        (q[0] - solve.width / 2.0) / solve.focal_px,
        (q[1] - solve.height / 2.0) / solve.focal_px,
        1.0,
    ];
    let d = mat_vec(&transpose(&pose.rot()), d_cam);
    let c = pose.center();
    let den = dot(d, normal);
    if den.abs() < 1e-12 {
        return None;
    }
    let t = dot(sub(origin, c), normal) / den;
    (t > 0.0).then(|| add(c, scale(d, t)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::norm;

    /// Synthetic shot: random 3D points, a camera that dollies and pans, 0.25 px noise.
    fn synth(tripod: bool, lens: Lens) -> (Vec<PointTrack>, Vec<usize>, Vec<Pose>, f64) {
        let (w, h, f) = (1920.0, 1080.0, 1350.0);
        let mut rng = XorShift::new(42);
        let mut rnd = || rng.below(100_000) as f64 / 100_000.0;
        let pts: Vec<V3> = (0..900)
            .map(|_| [rnd() * 24.0 - 12.0, rnd() * 6.0 - 4.0, 6.0 + rnd() * 22.0])
            .collect();
        let n = 60;
        let poses: Vec<Pose> = (0..n)
            .map(|k| {
                let t = k as f64 / n as f64;
                let r = rodrigues([0.03 * t, 0.25 * t - 0.1, 0.02]);
                let c = if tripod {
                    [0.0; 3]
                } else {
                    [3.0 * t, -0.3 * t, 1.5 * t]
                };
                Pose {
                    r: log_rotation(&r),
                    t: scale(mat_vec(&r, c), -1.0),
                }
            })
            .collect();
        let mut noise_rng = XorShift::new(7);
        let mut noise = || (noise_rng.below(10_000) as f64 / 10_000.0 - 0.5) * 0.5;
        let solve_proj = CameraSolve {
            focal_px: f,
            lens,
            width: w,
            height: h,
            poses: poses.iter().enumerate().map(|(i, p)| (i, *p)).collect(),
            points: vec![],
            rms_px: 0.0,
            keyframes: vec![],
            tripod,
            world: World::default(),
        };
        let mut tracks = Vec::new();
        for (i, x) in pts.iter().enumerate() {
            let mut t = PointTrack {
                id: i as u32,
                points: BTreeMap::new(),
            };
            for k in 0..n {
                if let Some(p) = solve_proj.project(k, *x) {
                    if p[0] > 5.0 && p[1] > 5.0 && p[0] < w - 5.0 && p[1] < h - 5.0 {
                        t.points.insert(k, [p[0] + noise(), p[1] + noise()]);
                    }
                }
            }
            if t.points.len() >= 3 {
                tracks.push(t);
            }
        }
        (tracks, (0..n).collect(), poses, f)
    }

    /// Align solved camera centres to the truth with a similarity and return the
    /// worst position error relative to the path length.
    fn path_error(solve: &CameraSolve, truth: &[Pose]) -> f64 {
        let a: Vec<V3> = (0..truth.len()).map(|k| solve.poses[&k].center()).collect();
        let b: Vec<V3> = truth.iter().map(|p| p.center()).collect();
        let n = a.len() as f64;
        let ca = scale(a.iter().fold([0.0; 3], |s, p| add(s, *p)), 1.0 / n);
        let cb = scale(b.iter().fold([0.0; 3], |s, p| add(s, *p)), 1.0 / n);
        let mut cov = [0.0; 9];
        let mut va = 0.0;
        for (p, q) in a.iter().zip(&b) {
            let (da, db) = (sub(*p, ca), sub(*q, cb));
            va += dot(da, da);
            for i in 0..3 {
                for j in 0..3 {
                    cov[i * 3 + j] += db[i] * da[j];
                }
            }
        }
        let r = linalg::nearest_rotation(&cov);
        let (_, s, _) = linalg::svd3(&cov);
        let sc = (s[0] + s[1] + s[2]) / va;
        let len: f64 = b
            .windows(2)
            .map(|w| norm(sub(w[1], w[0])))
            .sum::<f64>()
            .max(1e-9);
        a.iter()
            .zip(&b)
            .map(|(p, q)| norm(sub(add(scale(mat_vec(&r, sub(*p, ca)), sc), cb), *q)))
            .fold(0.0, f64::max)
            / len
    }

    fn opts(lens: Lens, distortion: bool) -> SolveOptions {
        SolveOptions {
            width: 1920.0,
            height: 1080.0,
            focal: FocalMode::Auto,
            sensor_mm: 36.0,
            mode: SolveMode::Auto,
            solve_distortion: distortion,
            lens,
        }
    }

    #[test]
    fn solves_moving_camera_and_focal() {
        let (tracks, frames, truth, f) = synth(false, Lens::default());
        let cancel = AtomicBool::new(false);
        let s = solve_camera(
            &tracks,
            &frames,
            &opts(Lens::default(), false),
            &cancel,
            |_, _| {},
        )
        .unwrap();
        let pe = path_error(&s, &truth);
        println!(
            "free: focal {:.1} (true {f}), rms {:.3}px, path error {:.4}, keys {}",
            s.focal_px,
            s.rms_px,
            pe,
            s.keyframes.len()
        );
        assert!(!s.tripod);
        assert!((s.focal_px - f).abs() / f < 0.01);
        assert!(s.rms_px < 0.3);
        assert!(pe < 0.01);
        assert_eq!(s.poses.len(), frames.len());
    }

    #[test]
    fn solves_lens_distortion() {
        let truth_lens = Lens { k1: -0.08, k2: 0.0 };
        let (tracks, frames, truth, f) = synth(false, truth_lens);
        let cancel = AtomicBool::new(false);
        let s = solve_camera(
            &tracks,
            &frames,
            &opts(Lens::default(), true),
            &cancel,
            |_, _| {},
        )
        .unwrap();
        let pe = path_error(&s, &truth);
        println!(
            "distortion: k1 {:.4} (true -0.08), focal {:.1}, rms {:.3}px, path error {:.4}",
            s.lens.k1, s.focal_px, s.rms_px, pe
        );
        assert!((s.lens.k1 - truth_lens.k1).abs() < 0.01);
        assert!((s.focal_px - f).abs() / f < 0.02);
        assert!(s.rms_px < 0.35);
    }

    #[test]
    fn solves_tripod_pan() {
        let (tracks, frames, truth, f) = synth(true, Lens::default());
        let cancel = AtomicBool::new(false);
        let s = solve_camera(
            &tracks,
            &frames,
            &opts(Lens::default(), false),
            &cancel,
            |_, _| {},
        )
        .unwrap();
        // Compare rotations relative to the first frame.
        let mut worst: f64 = 0.0;
        for k in 0..frames.len() {
            let rs = mul(&s.poses[&k].rot(), &transpose(&s.poses[&0].rot()));
            let rt = mul(&truth[k].rot(), &transpose(&truth[0].rot()));
            worst = worst.max(norm(log_rotation(&mul(&rs, &transpose(&rt)))).to_degrees());
        }
        println!(
            "tripod: focal {:.1} (true {f}), rms {:.3}px, worst rotation error {:.4} deg",
            s.focal_px, s.rms_px, worst
        );
        assert!(s.tripod);
        assert!((s.focal_px - f).abs() / f < 0.02);
        assert!(worst < 0.05);
    }

    #[test]
    fn orient_makes_plane_horizontal() {
        let (tracks, frames, _, _) = synth(false, Lens::default());
        let cancel = AtomicBool::new(false);
        let s = solve_camera(
            &tracks,
            &frames,
            &opts(Lens::default(), false),
            &cancel,
            |_, _| {},
        )
        .unwrap();
        // Three points on some plane in solve space.
        let pts = vec![
            [0.0, 1.0, 10.0],
            [2.0, 1.2, 12.0],
            [-1.0, 0.9, 14.0],
            [1.0, 1.1, 11.0],
        ];
        let wld = orient_to_plane(&s, &pts, 0, None, Some((2.0, 4.0))).unwrap();
        let mean_y = pts.iter().map(|p| wld.to_user(*p)[1]).sum::<f64>() / 4.0;
        assert!(mean_y.abs() < 1e-9);
        assert!((wld.scale - 2.0).abs() < 1e-12);
        let back = wld.from_user(wld.to_user([1.0, 2.0, 3.0]));
        assert!(norm(sub(back, [1.0, 2.0, 3.0])) < 1e-9);
    }
}

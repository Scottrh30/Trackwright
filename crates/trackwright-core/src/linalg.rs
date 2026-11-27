//! Small dense linear algebra for the camera solver.

use crate::geometry::Mat3;

pub type V3 = [f64; 3];

/// Eigen-decomposition of a symmetric `n x n` matrix (row-major) by cyclic Jacobi.
/// Returns eigenvalues ascending and the matching eigenvectors (each a `Vec` of length n).
pub fn sym_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| m[i * n + j].powi(2))
            .sum();
        let scale: f64 = (0..n)
            .map(|i| m[i * n + i].powi(2))
            .sum::<f64>()
            .max(1e-300);
        if off < 1e-24 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| m[i * n + i].total_cmp(&m[j * n + j]));
    let vals = idx.iter().map(|&i| m[i * n + i]).collect();
    let vecs = idx
        .iter()
        .map(|&i| (0..n).map(|k| v[k * n + i]).collect())
        .collect();
    (vals, vecs)
}

/// Unit vector `x` minimising `|A x|` for the rows of `A` (each of length `n`).
pub fn null_vector(rows: &[Vec<f64>], n: usize) -> Vec<f64> {
    let mut ata = vec![0.0; n * n];
    for r in rows {
        for i in 0..n {
            if r[i] == 0.0 {
                continue;
            }
            for j in 0..n {
                ata[i * n + j] += r[i] * r[j];
            }
        }
    }
    let (_, vecs) = sym_eigen(&ata, n);
    vecs.into_iter().next().unwrap()
}

/// Solve `A x = b` for symmetric positive-definite `A` (Cholesky). `None` if not SPD.
pub fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn normalized(a: V3) -> V3 {
    let n = norm(a).max(1e-300);
    scale(a, 1.0 / n)
}

pub fn mat_vec(m: &Mat3, v: V3) -> V3 {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

pub fn transpose(m: &Mat3) -> Mat3 {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

pub fn det3(m: &Mat3) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

pub fn skew(v: V3) -> Mat3 {
    [0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0]
}

/// Rotation matrix from an angle-axis vector.
pub fn rodrigues(w: V3) -> Mat3 {
    let th = norm(w);
    if th < 1e-12 {
        let k = skew(w);
        return [
            1.0 + k[0],
            k[1],
            k[2],
            k[3],
            1.0 + k[4],
            k[5],
            k[6],
            k[7],
            1.0 + k[8],
        ];
    }
    let k = scale(w, 1.0 / th);
    let (s, c) = th.sin_cos();
    let kx = skew(k);
    let kx2 = crate::geometry::mul(&kx, &kx);
    let mut r = [0.0; 9];
    for i in 0..9 {
        let id = if i % 4 == 0 { 1.0 } else { 0.0 };
        r[i] = id + s * kx[i] + (1.0 - c) * kx2[i];
    }
    r
}

/// Angle-axis vector of a rotation matrix.
pub fn log_rotation(r: &Mat3) -> V3 {
    let cos = ((r[0] + r[4] + r[8] - 1.0) / 2.0).clamp(-1.0, 1.0);
    let th = cos.acos();
    if th < 1e-9 {
        return [
            (r[7] - r[5]) / 2.0,
            (r[2] - r[6]) / 2.0,
            (r[3] - r[1]) / 2.0,
        ];
    }
    if (std::f64::consts::PI - th).abs() < 1e-6 {
        // Near 180 degrees: axis from the diagonal.
        let xx = ((r[0] + 1.0) / 2.0).max(0.0).sqrt();
        let yy = ((r[4] + 1.0) / 2.0).max(0.0).sqrt();
        let zz = ((r[8] + 1.0) / 2.0).max(0.0).sqrt();
        let axis = if xx >= yy && xx >= zz {
            [xx, r[1] / (2.0 * xx), r[2] / (2.0 * xx)]
        } else if yy >= zz {
            [r[1] / (2.0 * yy), yy, r[5] / (2.0 * yy)]
        } else {
            [r[2] / (2.0 * zz), r[5] / (2.0 * zz), zz]
        };
        return scale(normalized(axis), th);
    }
    let k = th / (2.0 * th.sin());
    [(r[7] - r[5]) * k, (r[2] - r[6]) * k, (r[3] - r[1]) * k]
}

/// SVD of a 3x3 matrix: returns (U, singular values descending, V) with M = U diag(S) Vᵀ.
pub fn svd3(m: &Mat3) -> (Mat3, V3, Mat3) {
    let mtm = crate::geometry::mul(&transpose(m), m);
    let (vals, vecs) = sym_eigen(&mtm, 3);
    // Descending order.
    let order = [2usize, 1, 0];
    let mut v = [0.0; 9];
    let mut s = [0.0; 3];
    for (c, &i) in order.iter().enumerate() {
        s[c] = vals[i].max(0.0).sqrt();
        for r in 0..3 {
            v[r * 3 + c] = vecs[i][r];
        }
    }
    let col = |mm: &Mat3, c: usize| [mm[c], mm[3 + c], mm[6 + c]];
    // Singular values this small relative to the largest are treated as zero.
    let tiny = 1e-9 * s[0].max(1e-300);
    let mut u_cols = [[0.0; 3]; 3];
    u_cols[0] = if s[0] > 1e-300 {
        normalized(mat_vec(m, col(&v, 0)))
    } else {
        [1.0, 0.0, 0.0]
    };
    u_cols[1] = if s[1] > tiny {
        // Gram-Schmidt against u0 for accuracy.
        let u1 = scale(mat_vec(m, col(&v, 1)), 1.0 / s[1]);
        normalized(sub(u1, scale(u_cols[0], dot(u1, u_cols[0]))))
    } else {
        let a = u_cols[0];
        let t = if a[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        normalized(cross(a, t))
    };
    let c01 = cross(u_cols[0], u_cols[1]);
    u_cols[2] = if s[2] > tiny {
        let u2 = scale(mat_vec(m, col(&v, 2)), 1.0 / s[2]);
        // Keep the sign implied by M while staying orthonormal.
        if dot(u2, c01) >= 0.0 {
            c01
        } else {
            scale(c01, -1.0)
        }
    } else {
        c01
    };
    let mut u = [0.0; 9];
    for c in 0..3 {
        for r in 0..3 {
            u[r * 3 + c] = u_cols[c][r];
        }
    }
    (u, s, v)
}

/// Closest rotation to `m` (orthogonal Procrustes).
pub fn nearest_rotation(m: &Mat3) -> Mat3 {
    let (u, _, v) = svd3(m);
    let mut r = crate::geometry::mul(&u, &transpose(&v));
    if det3(&r) < 0.0 {
        let mut u2 = u;
        for row in 0..3 {
            u2[row * 3 + 2] = -u2[row * 3 + 2];
        }
        r = crate::geometry::mul(&u2, &transpose(&v));
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::mul;

    #[test]
    fn eigen_and_svd() {
        let a = [4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0];
        let (vals, vecs) = sym_eigen(&a, 3);
        for (l, v) in vals.iter().zip(&vecs) {
            let av = mat_vec(&a, [v[0], v[1], v[2]]);
            for k in 0..3 {
                assert!((av[k] - l * v[k]).abs() < 1e-9);
            }
        }
        let m = [1.0, 2.0, 3.0, -1.0, 0.5, 2.0, 0.3, -0.2, 1.5];
        let (u, s, v) = svd3(&m);
        let us = [
            u[0] * s[0],
            u[1] * s[1],
            u[2] * s[2],
            u[3] * s[0],
            u[4] * s[1],
            u[5] * s[2],
            u[6] * s[0],
            u[7] * s[1],
            u[8] * s[2],
        ];
        let back = mul(&us, &transpose(&v));
        for i in 0..9 {
            assert!((back[i] - m[i]).abs() < 1e-9, "{back:?}");
        }
    }

    #[test]
    fn rotation_roundtrip() {
        for w in [[0.1, -0.2, 0.3], [1.2, 0.4, -2.0], [0.0, 0.0, 1e-8]] {
            let r = rodrigues(w);
            assert!((det3(&r) - 1.0).abs() < 1e-12);
            let w2 = log_rotation(&r);
            for k in 0..3 {
                assert!((w[k] - w2[k]).abs() < 1e-9);
            }
        }
        let a = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let x = solve_spd(&a, &[2.0, 3.0, 8.0], 3).unwrap();
        assert!((x[2] - 2.0).abs() < 1e-12);
    }
}

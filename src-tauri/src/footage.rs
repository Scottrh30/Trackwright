//! Footage import: image sequences, single images, folders and video (via ffmpeg).

use std::path::{Path, PathBuf};
use std::process::Command;
use trackwright_core::Footage;

const IMAGE_EXTS: [&str; 7] = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"];
const VIDEO_EXTS: [&str; 10] = [
    "mp4", "mov", "mkv", "avi", "webm", "m4v", "mxf", "mpg", "mpeg", "wmv",
];

fn ext_of(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

pub fn is_image(p: &Path) -> bool {
    IMAGE_EXTS.contains(&ext_of(p).as_str())
}

pub fn is_video(p: &Path) -> bool {
    VIDEO_EXTS.contains(&ext_of(p).as_str())
}

/// Split "plate_v2.1001" into ("plate_v2.", Some(1001), digit_count).
fn split_number(stem: &str) -> (String, Option<i64>, usize) {
    let digits: String = stem
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if digits.is_empty() {
        return (stem.to_string(), None, 0);
    }
    let prefix = stem[..stem.len() - digits.len()].to_string();
    (prefix, digits.parse().ok(), digits.len())
}

fn finish(
    source: &Path,
    frames: Vec<PathBuf>,
    first_frame: i64,
    fps: f64,
) -> Result<Footage, String> {
    let first = frames.first().ok_or("No frames found")?;
    let (w, h) = image::image_dimensions(first)
        .map_err(|e| format!("Cannot read {}: {e}", first.display()))?;
    Ok(Footage {
        source: source.to_string_lossy().to_string(),
        frames: frames
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
        width: w,
        height: h,
        fps,
        first_frame,
    })
}

/// Open one frame of a numbered sequence and gather its siblings.
pub fn open_sequence(file: &Path, fps: f64) -> Result<Footage, String> {
    let dir = file.parent().ok_or("File has no folder")?;
    let stem = file
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = ext_of(file);
    let (prefix, num, _) = split_number(&stem);
    if num.is_none() {
        return finish(file, vec![file.to_path_buf()], 0, fps);
    }
    let mut frames: Vec<(i64, PathBuf)> = std::fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| ext_of(p) == ext)
        .filter_map(|p| {
            let s = p.file_stem()?.to_string_lossy().to_string();
            let (pre, n, _) = split_number(&s);
            (pre == prefix).then_some((n?, p))
        })
        .collect();
    frames.sort_by_key(|(n, _)| *n);
    let first = frames.first().map(|f| f.0).unwrap_or(0);
    finish(file, frames.into_iter().map(|f| f.1).collect(), first, fps)
}

/// Every image in a folder, in natural (numeric-aware) order.
pub fn open_folder(dir: &Path, fps: f64) -> Result<Footage, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_image(p))
        .collect();
    if files.is_empty() {
        return Err("That folder has no supported images (png, jpg, tif, bmp, webp)".into());
    }
    files.sort_by_key(|p| {
        let s = p
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let (pre, n, _) = split_number(&s);
        (pre, n.unwrap_or(-1), s)
    });
    let first = files
        .first()
        .and_then(|p| split_number(&p.file_stem()?.to_string_lossy()).1)
        .unwrap_or(0);
    finish(dir, files, first, fps)
}

fn ffmpeg_bin(name: &str) -> String {
    // Allow a bundled/side-by-side ffmpeg next to the executable.
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe.with_file_name(if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.to_string()
        });
        if candidate.exists() {
            return candidate.to_string_lossy().to_string();
        }
    }
    name.to_string()
}

fn command(name: &str) -> Command {
    #[allow(unused_mut)]
    let mut c = Command::new(ffmpeg_bin(name));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    c
}

fn probe_fps(video: &Path) -> Option<f64> {
    let out = command("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
        ])
        .arg(video)
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let mut it = s.split('/');
    let n: f64 = it.next()?.trim().parse().ok()?;
    let d: f64 = it.next().unwrap_or("1").trim().parse().ok()?;
    (d > 0.0 && n > 0.0).then_some(n / d)
}

/// Decode a video into a JPEG frame cache under `cache_root` and open it as a sequence.
pub fn open_video(video: &Path, cache_root: &Path) -> Result<Footage, String> {
    let meta = std::fs::metadata(video).map_err(|e| e.to_string())?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let mut hash: u64 = 1469598103934665603;
    for b in video
        .to_string_lossy()
        .bytes()
        .chain(mtime.to_le_bytes())
        .chain(meta.len().to_le_bytes())
    {
        hash = (hash ^ b as u64).wrapping_mul(1099511628211);
    }
    let dir = cache_root.join(format!("{hash:016x}"));
    let done_marker = dir.join(".complete");
    let fps = probe_fps(video).unwrap_or(24.0);

    if !done_marker.exists() {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let out = command("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(video)
            .args(["-vsync", "0", "-q:v", "2", "-start_number", "0"])
            .arg(dir.join("frame_%06d.jpg"))
            .output()
            .map_err(|_| {
                "Video import needs ffmpeg. Install it (winget install ffmpeg) or put ffmpeg.exe next to Trackwright, or export your clip as an image sequence".to_string()
            })?;
        if !out.status.success() {
            return Err(format!(
                "ffmpeg could not decode the video: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        std::fs::write(&done_marker, b"ok").map_err(|e| e.to_string())?;
    }
    let mut f = open_sequence(&dir.join("frame_000000.jpg"), fps)?;
    f.source = video.to_string_lossy().to_string();
    f.first_frame = 0;
    Ok(f)
}

pub fn open_any(path: &Path, cache_root: &Path) -> Result<Footage, String> {
    if path.is_dir() {
        open_folder(path, 24.0)
    } else if is_video(path) {
        open_video(path, cache_root)
    } else if is_image(path) {
        open_sequence(path, 24.0)
    } else {
        Err(format!("Unsupported file type: {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::split_number;

    #[test]
    fn numbers() {
        assert_eq!(
            split_number("plate_v2.1001"),
            ("plate_v2.".into(), Some(1001), 4)
        );
        assert_eq!(split_number("shot"), ("shot".into(), None, 0));
    }
}

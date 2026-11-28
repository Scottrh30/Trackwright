//! Check the AI tracker against the reference: ai_check RUNTIME MODEL_DIR FRAMES_DIR F0 F1 x0,y0,x1,y1 px,py
use std::path::Path;
use trackwright_core::ai::PointTracker;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t = trackwright_ai::AllTracker::load(
        Path::new(&a[1]),
        &Path::new(&a[2]).join("at_encoder.onnx"),
        &Path::new(&a[2]).join("at_window.onnx"),
        2,
    )
    .unwrap();
    let (f0, f1): (usize, usize) = (a[4].parse().unwrap(), a[5].parse().unwrap());
    let b: Vec<u32> = a[6].split(',').map(|v| v.parse().unwrap()).collect();
    let p: Vec<f64> = a[7].split(',').map(|v| v.parse().unwrap()).collect();
    let (w, h) = (trackwright_ai::WIDTH as u32, trackwright_ai::HEIGHT as u32);
    let frames: Vec<Vec<f32>> = (f0..=f1)
        .map(|f| {
            let img = image::open(format!("{}/f_{f:06}.jpg", a[3]))
                .unwrap()
                .to_luma8();
            let c =
                image::imageops::crop_imm(&img, b[0], b[1], b[2] - b[0], b[3] - b[1]).to_image();
            let r = image::imageops::resize(&c, w, h, image::imageops::FilterType::Triangle);
            r.as_raw().iter().map(|v| *v as f32).collect()
        })
        .collect();
    let sx = w as f64 / (b[2] - b[0]) as f64;
    let sy = h as f64 / (b[3] - b[1]) as f64;
    let q = [(p[0] - b[0] as f64) * sx, (p[1] - b[1] as f64) * sy];
    let t0 = std::time::Instant::now();
    let r = t.track(&frames, &[q]).unwrap();
    println!("time {:.1}s", t0.elapsed().as_secs_f64());
    let dy: Vec<i64> = r
        .iter()
        .map(|f| ((f[0].0[1] - q[1]) / sy).round() as i64)
        .collect();
    let v: Vec<String> = r.iter().map(|f| format!("{:.2}", f[0].1)).collect();
    println!("dy {dy:?}");
    println!("vis {v:?}");
}

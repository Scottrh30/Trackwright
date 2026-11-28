//! `track_clip` with the AI point tracker registered: set TW_AI_RUNTIME (path of the
//! ONNX Runtime library) and TW_AI_MODELS (folder with at_encoder.onnx and
//! at_window.onnx), then pass ai=1.
#[path = "../../trackwright-core/examples/track_clip.rs"]
mod inner;

fn main() {
    if let (Ok(rt), Ok(dir)) = (
        std::env::var("TW_AI_RUNTIME"),
        std::env::var("TW_AI_MODELS"),
    ) {
        let dir = std::path::Path::new(&dir);
        let t = trackwright_ai::AllTracker::load(
            std::path::Path::new(&rt),
            &dir.join("at_encoder.onnx"),
            &dir.join("at_window.onnx"),
            2,
        )
        .expect("AI model");
        trackwright_core::ai::set_point_tracker(Some(std::sync::Arc::new(t)));
        eprintln!("AI tracker loaded");
    }
    inner::main()
}

use trackwright_core::export::{export, ExportFormat};
use trackwright_core::{Footage, FrameTrack, Layer, Shape};
use trackwright_import::{apply, CornerPinTrack};

fn layer() -> (Footage, Layer) {
    let mut l = Layer::new(1, 0, Shape::rect(100.0, 50.0, 400.0, 200.0));
    l.name = "Screen".into();
    for f in 1..10 {
        let t = f as f64;
        l.tracks.insert(
            f,
            FrameTrack {
                h_from_ref: [
                    1.0 + 0.01 * t,
                    0.02 * t,
                    10.0 * t,
                    -0.01 * t,
                    1.0,
                    5.0 * t,
                    1e-5 * t,
                    0.0,
                    1.0,
                ],
                ..FrameTrack::reference()
            },
        );
    }
    let footage = Footage {
        source: "/shots/plate.mov".into(),
        frames: vec![String::new(); 10],
        width: 1920,
        height: 1080,
        fps: 25.0,
        first_frame: 1001,
    };
    (footage, l)
}

#[test]
fn reads_reelforge_and_planar_exports() {
    let (footage, l) = layer();
    for fmt in [
        ExportFormat::Reelforge,
        ExportFormat::Forma3d,
        ExportFormat::Tidefire,
    ] {
        let track = CornerPinTrack::from_json(&export(fmt, &footage, &l)).unwrap();
        assert_eq!(track.frame_range(), (1001, 1010));
        assert_eq!(
            (track.width, track.height, track.fps),
            (1920.0, 1080.0, 25.0)
        );
        for f in 0..10usize {
            let expect = l.surface_at(f).unwrap();
            let got = track.corners_at(1001.0 + f as f64);
            for i in 0..4 {
                assert!(
                    (expect[i][0] - got[i][0]).abs() < 1e-3
                        && (expect[i][1] - got[i][1]).abs() < 1e-3
                );
            }
        }
    }
}

#[test]
fn interpolates_and_pins_layers() {
    let (footage, l) = layer();
    let track = CornerPinTrack::from_json(&export(ExportFormat::Reelforge, &footage, &l)).unwrap();
    // Halfway between frames 1003 and 1004.
    let a = track.corners_at(1003.0);
    let b = track.corners_at(1004.0);
    let m = track.corners_at(1003.5);
    assert!((m[0][0] - (a[0][0] + b[0][0]) / 2.0).abs() < 1e-9);
    // Time 0.1 s at 25 fps = frame 1003.5.
    assert_eq!(track.corners_at_time(0.1), m);
    // Held outside the tracked range.
    assert_eq!(track.corners_at(900.0), track.corners_at(1001.0));
    // A 640x360 layer's corners land on the surface corners.
    let h = track.pin_layer_at(1007.0, 640.0, 360.0).unwrap();
    let c = track.corners_at(1007.0);
    for (i, p) in [[0.0, 0.0], [640.0, 0.0], [640.0, 360.0], [0.0, 360.0]]
        .iter()
        .enumerate()
    {
        let q = apply(&h, *p);
        assert!((q[0] - c[i][0]).abs() < 1e-6 && (q[1] - c[i][1]).abs() < 1e-6);
    }
    // Transform is identity on the reference frame.
    let t = track.transform_at(1001.0);
    assert!(t.rotation_deg.abs() < 1e-9 && (t.scale[0] - 1.0).abs() < 1e-9);
    // Rescaling to a half-size proxy halves the corners.
    let half = track.rescaled(960.0, 540.0);
    let hc = half.corners_at(1007.0);
    assert!((hc[2][0] - c[2][0] / 2.0).abs() < 1e-9 && (hc[2][1] - c[2][1] / 2.0).abs() < 1e-9);
}

#[test]
fn rejects_other_files() {
    assert!(CornerPinTrack::from_json(
        "{\"format\":\"something.else\",\"source\":{\"width\":1,\"height\":1}}"
    )
    .is_err());
    assert!(CornerPinTrack::from_json("not json").is_err());
}

#[test]
fn reads_masks() {
    let (footage, mut l) = layer();
    l.feather = 3.0;
    let m = trackwright_import::MaskTrack::from_json(&export(
        ExportFormat::ReelforgeMask,
        &footage,
        &l,
    ))
    .unwrap();
    assert_eq!(m.frames.len(), 10);
    assert_eq!(m.feather_px, 3.0);
    let poly = m.polygon_at(1005.0);
    let expect = l.shape_at(4).unwrap().flatten(10);
    assert!((poly[0][0] - expect[0][0]).abs() < 1e-3);
    let a = m.rasterize(1001.0, 640, 360);
    assert_eq!(a[100 * 640 + 300], 255);
    assert_eq!(a[5 * 640 + 5], 0);
}

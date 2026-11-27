//! Alembic (Ogawa) writer, just enough for an animated camera and a point cloud.
//!
//! Ogawa is a tree of groups (lists of child offsets) and data blocks. On top
//! of it Alembic stores objects (a property compound, child objects, then a
//! block of child headers) and properties (compounds, and scalar or array
//! properties whose samples are data blocks starting with a 16-byte key).
//! The layout matches what the reference library (1.8) writes.

const DATA_FLAG: u64 = 1 << 63;

/// Ogawa file being built: data and groups are appended, the root last.
pub struct Ogawa {
    buf: Vec<u8>,
}

impl Default for Ogawa {
    fn default() -> Self {
        Self::new()
    }
}

impl Ogawa {
    pub fn new() -> Self {
        let mut buf = Vec::new();
        buf.extend(b"Ogawa");
        buf.push(0xff); // frozen (finished)
        buf.extend([0u8, 1]); // version
        buf.extend(0u64.to_le_bytes()); // root position, patched by `finish`
        Ogawa { buf }
    }

    /// A data block; returns its child entry.
    pub fn data(&mut self, bytes: &[u8]) -> u64 {
        if bytes.is_empty() {
            return DATA_FLAG;
        }
        let pos = self.buf.len() as u64;
        self.buf.extend((bytes.len() as u64).to_le_bytes());
        self.buf.extend(bytes);
        pos | DATA_FLAG
    }

    /// A group of `kids` (entries from `data` or `group`); returns its entry.
    pub fn group(&mut self, kids: &[u64]) -> u64 {
        if kids.is_empty() {
            return 0;
        }
        let pos = self.buf.len() as u64;
        self.buf.extend((kids.len() as u64).to_le_bytes());
        for k in kids {
            self.buf.extend(k.to_le_bytes());
        }
        pos
    }

    pub fn finish(mut self, root: u64) -> Vec<u8> {
        self.buf[8..16].copy_from_slice(&root.to_le_bytes());
        self.buf
    }
}

/// 128-bit content key for a sample (readers use it to share identical samples).
fn key(bytes: &[u8], salt: u64) -> [u8; 16] {
    let h = |seed: u64| {
        let mut x: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
        for b in bytes {
            x ^= *b as u64;
            x = x.wrapping_mul(0x0000_0100_0000_01b3);
        }
        x ^ (x >> 29)
    };
    let mut k = [0u8; 16];
    k[..8].copy_from_slice(&h(salt).to_le_bytes());
    k[8..].copy_from_slice(&h(salt ^ 0x9e37_79b9_7f4a_7c15).to_le_bytes());
    k
}

/// Alembic plain-old-data types (the numbers stored in property headers).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pod {
    Bool = 0,
    U8 = 1,
    U32 = 5,
    U64 = 7,
    F32 = 10,
    F64 = 11,
}

/// A property under a compound.
pub enum Property {
    Compound {
        name: String,
        meta: String,
        children: Vec<Property>,
    },
    /// `samples[i]` is the raw little-endian bytes of `extent` values.
    Scalar {
        name: String,
        meta: String,
        pod: Pod,
        extent: u8,
        ts: u32,
        samples: Vec<Vec<u8>>,
    },
    /// Each sample is `(raw values, count)`.
    Array {
        name: String,
        meta: String,
        pod: Pod,
        extent: u8,
        ts: u32,
        samples: Vec<(Vec<u8>, u64)>,
    },
}

/// An object: its properties (the top compound's children) and child objects.
pub struct Object {
    pub name: String,
    pub meta: String,
    pub props: Vec<Property>,
    pub children: Vec<Object>,
}

/// The metadata strings, numbered (0 is the empty string; up to 254 more).
struct MetaTable(Vec<String>);

impl MetaTable {
    /// Index for `m`, or `None` to write it inline.
    fn index(&mut self, m: &str) -> Option<u8> {
        if m.is_empty() {
            return Some(0);
        }
        if let Some(i) = self.0.iter().position(|x| x == m) {
            return Some(i as u8 + 1);
        }
        if self.0.len() < 254 && m.len() < 256 {
            self.0.push(m.to_string());
            return Some(self.0.len() as u8);
        }
        None
    }
}

/// Samples worth storing: one when every sample is the same.
fn changes<T: PartialEq>(samples: &[T]) -> (u32, u32) {
    let n = samples.len();
    if n <= 1 || samples.iter().all(|s| *s == samples[0]) {
        (0, 0)
    } else {
        // first change from sample 0, last change
        let first = (1..n).find(|&i| samples[i] != samples[i - 1]).unwrap_or(1);
        let last = (1..n)
            .rev()
            .find(|&i| samples[i] != samples[i - 1])
            .unwrap_or(n - 1);
        (first as u32, last as u32)
    }
}

/// The samples kept: the first, then from the first change to the last
/// (readers map other indices onto these).
fn stored<T>(samples: &[T], first: u32, last: u32) -> impl Iterator<Item = &T> {
    let rest = if first == 0 && last == 0 {
        &samples[0..0]
    } else {
        &samples[first as usize..=last as usize]
    };
    samples.iter().take(1).chain(rest)
}

struct Header {
    kind: u32, // 0 compound, 1 scalar, 2 array
    pod: u32,
    extent: u32,
    ts: u32,
    num_samples: u32,
    first: u32,
    last: u32,
    homogenous: bool,
    name: String,
    meta: String,
}

fn write_header(h: &Header, metas: &mut MetaTable, out: &mut Vec<u8>) {
    let meta_index = metas.index(&h.meta);
    let mut info: u32 = h.kind;
    let compound = h.kind == 0;
    let all_same = !compound && h.first == 0 && h.last == 0;
    let need_first_last = !compound && !all_same && (h.first != 1 || h.last + 1 != h.num_samples);
    if !compound {
        info |= h.pod << 4;
        if h.ts != 0 {
            info |= 0x0100;
        }
        if need_first_last {
            info |= 0x0200;
        }
        if h.homogenous {
            info |= 0x0400;
        }
        if all_same {
            info |= 0x0800;
        }
        info |= (h.extent & 0xff) << 12;
    }
    info |= (meta_index.unwrap_or(0xff) as u32) << 20;
    let mut biggest = h.name.len() as u32;
    if !compound {
        biggest = biggest
            .max(h.num_samples)
            .max(h.first)
            .max(h.last)
            .max(h.ts);
    }
    if meta_index.is_none() {
        biggest = biggest.max(h.meta.len() as u32);
    }
    let hint = if biggest < 256 {
        0
    } else if biggest < 65536 {
        1
    } else {
        2
    };
    info |= hint << 2;
    out.extend(info.to_le_bytes());
    let put = |v: u32, out: &mut Vec<u8>| match hint {
        0 => out.push(v as u8),
        1 => out.extend((v as u16).to_le_bytes()),
        _ => out.extend(v.to_le_bytes()),
    };
    if !compound {
        put(h.num_samples, out);
        if need_first_last {
            put(h.first, out);
            put(h.last, out);
        }
        if h.ts != 0 {
            put(h.ts, out);
        }
    }
    put(h.name.len() as u32, out);
    out.extend(h.name.as_bytes());
    if meta_index.is_none() {
        put(h.meta.len() as u32, out);
        out.extend(h.meta.as_bytes());
    }
}

fn pod_size(p: Pod) -> usize {
    match p {
        Pod::Bool | Pod::U8 => 1,
        Pod::U32 | Pod::F32 => 4,
        Pod::U64 | Pod::F64 => 8,
    }
}

/// Write a property's group; returns (entry, header).
fn write_property(o: &mut Ogawa, metas: &mut MetaTable, p: &Property) -> (u64, Header) {
    match p {
        Property::Compound {
            name,
            meta,
            children,
        } => {
            let g = write_compound(o, metas, children);
            (
                g,
                Header {
                    kind: 0,
                    pod: 0,
                    extent: 0,
                    ts: 0,
                    num_samples: 0,
                    first: 0,
                    last: 0,
                    homogenous: false,
                    name: name.clone(),
                    meta: meta.clone(),
                },
            )
        }
        Property::Scalar {
            name,
            meta,
            pod,
            extent,
            ts,
            samples,
        } => {
            let (first, last) = changes(samples);
            let mut kids = Vec::new();
            for s in stored(samples, first, last) {
                let mut b = key(s, *pod as u64).to_vec();
                b.extend(s);
                kids.push(o.data(&b));
            }
            let g = o.group(&kids);
            (
                g,
                Header {
                    kind: 1,
                    pod: *pod as u32,
                    extent: *extent as u32,
                    ts: *ts,
                    num_samples: samples.len() as u32,
                    first,
                    last,
                    homogenous: true,
                    name: name.clone(),
                    meta: meta.clone(),
                },
            )
        }
        Property::Array {
            name,
            meta,
            pod,
            extent,
            ts,
            samples,
        } => {
            let (first, last) = changes(samples);
            let mut kids = Vec::new();
            for (s, count) in stored(samples, first, last) {
                let mut b = key(s, *pod as u64 + 100).to_vec();
                b.extend(s);
                kids.push(o.data(&b));
                // Dimensions: left empty for a plain list (count from the data size).
                let plain = (*count as usize) * pod_size(*pod) * (*extent as usize) == s.len();
                kids.push(if plain {
                    o.data(&[])
                } else {
                    o.data(&count.to_le_bytes())
                });
            }
            let g = o.group(&kids);
            let homogenous = samples.windows(2).all(|w| w[0].1 == w[1].1);
            (
                g,
                Header {
                    kind: 2,
                    pod: *pod as u32,
                    extent: *extent as u32,
                    ts: *ts,
                    num_samples: samples.len() as u32,
                    first,
                    last,
                    homogenous,
                    name: name.clone(),
                    meta: meta.clone(),
                },
            )
        }
    }
}

fn write_compound(o: &mut Ogawa, metas: &mut MetaTable, children: &[Property]) -> u64 {
    let mut kids = Vec::new();
    let mut headers = Vec::new();
    for c in children {
        let (g, h) = write_property(o, metas, c);
        kids.push(g);
        write_header(&h, metas, &mut headers);
    }
    let h = o.data(&headers);
    kids.push(h);
    o.group(&kids)
}

fn write_object(o: &mut Ogawa, metas: &mut MetaTable, obj: &Object) -> u64 {
    let props = write_compound(o, metas, &obj.props);
    let mut kids = vec![props];
    let mut headers = Vec::new();
    for c in &obj.children {
        kids.push(write_object(o, metas, c));
        headers.extend((c.name.len() as u32).to_le_bytes());
        headers.extend(c.name.as_bytes());
        match metas.index(&c.meta) {
            Some(i) => headers.push(i),
            None => {
                headers.push(0xff);
                headers.extend((c.meta.len() as u32).to_le_bytes());
                headers.extend(c.meta.as_bytes());
            }
        }
    }
    // Property and children hashes (informational).
    headers.extend(key(&headers, 7));
    headers.extend(key(obj.name.as_bytes(), 9));
    kids.push(o.data(&headers));
    o.group(&kids)
}

/// Uniform time sampling: `count` samples `1 / fps` apart from `start` seconds.
pub struct TimeSampling {
    pub fps: f64,
    pub start: f64,
    pub count: u32,
}

/// The whole archive: `top` is the root object's children; time sampling 1 is `ts`.
pub fn write(top: Vec<Object>, ts: &TimeSampling, app_meta: &str) -> Vec<u8> {
    let mut o = Ogawa::new();
    let mut metas = MetaTable(Vec::new());
    let v0 = o.data(&0i32.to_le_bytes());
    let v1 = o.data(&10803i32.to_le_bytes()); // library version 1.8.3
    let root = Object {
        name: "ABC".into(),
        meta: String::new(),
        props: Vec::new(),
        children: top,
    };
    let top_obj = write_object(&mut o, &mut metas, &root);
    let archive_meta = format!(
        "FramesPerTimeUnit={:.6};_ai_Application=Trackwright;{app_meta}",
        ts.fps
    );
    let am = o.data(archive_meta.as_bytes());
    let mut tsb = Vec::new();
    // 0: identity (one sample at 0)
    tsb.extend(1u32.to_le_bytes());
    tsb.extend(1.0f64.to_le_bytes());
    tsb.extend(1u32.to_le_bytes());
    tsb.extend(0.0f64.to_le_bytes());
    // 1: uniform at the footage rate
    tsb.extend(ts.count.to_le_bytes());
    tsb.extend((1.0 / ts.fps).to_le_bytes());
    tsb.extend(1u32.to_le_bytes());
    tsb.extend(ts.start.to_le_bytes());
    let tsd = o.data(&tsb);
    let mut mb = Vec::new();
    for m in &metas.0 {
        mb.push(m.len() as u8);
        mb.extend(m.as_bytes());
    }
    let md = o.data(&mb);
    let root_group = o.group(&[v0, v1, top_obj, am, tsd, md]);
    o.finish(root_group)
}

/// Little-endian bytes of `f64`s.
pub fn f64s(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Little-endian bytes of `f32`s.
pub fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_header_matches_the_reference_library() {
        // `.vals` of an animated xform as Alembic 1.8 writes it: b1 05 01 00 03 01 05 ".vals"
        let mut m = MetaTable(Vec::new());
        let mut out = Vec::new();
        let h = Header {
            kind: 1,
            pod: 11,
            extent: 16,
            ts: 1,
            num_samples: 3,
            first: 1,
            last: 2,
            homogenous: true,
            name: ".vals".into(),
            meta: String::new(),
        };
        write_header(&h, &mut m, &mut out);
        assert_eq!(out, b"\xb1\x05\x01\x00\x03\x01\x05.vals");
        // A constant one: `.inherits` 01 1d 00 00 03 01 09 ".inherits"
        let mut out = Vec::new();
        let h = Header {
            kind: 1,
            pod: 0,
            extent: 1,
            ts: 1,
            num_samples: 3,
            first: 0,
            last: 0,
            homogenous: true,
            name: ".inherits".into(),
            meta: String::new(),
        };
        write_header(&h, &mut m, &mut out);
        assert_eq!(out, b"\x01\x1d\x00\x00\x03\x01\x09.inherits");
        // A compound with metadata: 00 00 30 00 06 ".xform" (metadata index 3)
        let mut m = MetaTable(vec![
            "a".into(),
            "b".into(),
            "schema=AbcGeom_Xform_v3".into(),
        ]);
        let mut out = Vec::new();
        let h = Header {
            kind: 0,
            pod: 0,
            extent: 0,
            ts: 0,
            num_samples: 0,
            first: 0,
            last: 0,
            homogenous: false,
            name: ".xform".into(),
            meta: "schema=AbcGeom_Xform_v3".into(),
        };
        write_header(&h, &mut m, &mut out);
        assert_eq!(out, b"\x00\x00\x30\x00\x06.xform");
    }
}

//! Binary FBX 7.4 writer, just enough for an animated camera and a point cloud.
//!
//! The layout follows what Autodesk's SDK and Blender write: header, nodes
//! (end offset, property count, property bytes, name, properties, children,
//! a 13-byte null record after children), then the fixed footer. Arrays are
//! stored uncompressed.

/// One property of a node.
pub enum Prop {
    I32(i32),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Raw(Vec<u8>),
    ArrF32(Vec<f32>),
    ArrF64(Vec<f64>),
    ArrI32(Vec<i32>),
    ArrI64(Vec<i64>),
}

/// A node with properties and children.
pub struct Node {
    pub name: &'static str,
    pub props: Vec<Prop>,
    pub children: Vec<Node>,
}

impl Node {
    pub fn new(name: &'static str, props: Vec<Prop>) -> Self {
        Node {
            name,
            props,
            children: Vec::new(),
        }
    }
    pub fn with(mut self, children: Vec<Node>) -> Self {
        self.children = children;
        self
    }
}

const HEADER: &[u8] = b"Kaydara FBX Binary  \x00\x1a\x00";
const VERSION: u32 = 7400;
const SENTINEL: [u8; 13] = [0; 13];
/// FileId and CreationTime as Blender writes them, so the footer id below matches.
pub const FILE_ID: [u8; 16] = [
    0x28, 0xb3, 0x2a, 0xeb, 0xb6, 0x24, 0xcc, 0xc2, 0xbf, 0xc8, 0xb0, 0x2a, 0xa9, 0x2b, 0xfc, 0xf1,
];
pub const CREATION_TIME: &str = "1970-01-01 10:00:00:000";
const FOOT_ID: [u8; 16] = [
    0xfa, 0xbc, 0xab, 0x09, 0xd0, 0xc8, 0xd4, 0x66, 0xb1, 0x76, 0xfb, 0x83, 0x1c, 0xf7, 0x26, 0x7e,
];
const FOOT_MAGIC: [u8; 16] = [
    0xf8, 0x5a, 0x8c, 0x6a, 0xde, 0xf5, 0xd9, 0x7e, 0xec, 0xe9, 0x0c, 0xe3, 0x75, 0x8f, 0x29, 0x0b,
];

fn prop_bytes(p: &Prop, out: &mut Vec<u8>) {
    fn arr<T: Copy>(out: &mut Vec<u8>, code: u8, v: &[T], f: impl Fn(T) -> Vec<u8>, size: usize) {
        out.push(code);
        out.extend((v.len() as u32).to_le_bytes());
        out.extend(0u32.to_le_bytes()); // not compressed
        out.extend(((v.len() * size) as u32).to_le_bytes());
        for x in v {
            out.extend(f(*x));
        }
    }
    match p {
        Prop::I32(v) => {
            out.push(b'I');
            out.extend(v.to_le_bytes());
        }
        Prop::I64(v) => {
            out.push(b'L');
            out.extend(v.to_le_bytes());
        }
        Prop::F64(v) => {
            out.push(b'D');
            out.extend(v.to_le_bytes());
        }
        Prop::Bool(v) => {
            out.push(b'C');
            out.push(*v as u8);
        }
        Prop::Str(s) => {
            out.push(b'S');
            out.extend((s.len() as u32).to_le_bytes());
            out.extend(s.as_bytes());
        }
        Prop::Raw(b) => {
            out.push(b'R');
            out.extend((b.len() as u32).to_le_bytes());
            out.extend(b);
        }
        Prop::ArrF32(v) => arr(out, b'f', v, |x| x.to_le_bytes().to_vec(), 4),
        Prop::ArrF64(v) => arr(out, b'd', v, |x| x.to_le_bytes().to_vec(), 8),
        Prop::ArrI32(v) => arr(out, b'i', v, |x| x.to_le_bytes().to_vec(), 4),
        Prop::ArrI64(v) => arr(out, b'l', v, |x| x.to_le_bytes().to_vec(), 8),
    }
}

fn write_node(n: &Node, is_last: bool, out: &mut Vec<u8>) {
    let start = out.len();
    out.extend([0u8; 12]); // end offset, property count, property length: patched below
    out.push(n.name.len() as u8);
    out.extend(n.name.as_bytes());
    let props_start = out.len();
    for p in &n.props {
        prop_bytes(p, out);
    }
    let props_len = (out.len() - props_start) as u32;
    if !n.children.is_empty() {
        let last = n.children.len() - 1;
        for (i, c) in n.children.iter().enumerate() {
            write_node(c, i == last, out);
        }
        out.extend(SENTINEL);
    } else if (n.props.is_empty() && !is_last)
        || n.name == "AnimationStack"
        || n.name == "AnimationLayer"
    {
        out.extend(SENTINEL);
    }
    let end = out.len() as u32;
    out[start..start + 4].copy_from_slice(&end.to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&(n.props.len() as u32).to_le_bytes());
    out[start + 8..start + 12].copy_from_slice(&props_len.to_le_bytes());
}

/// The whole file for top-level `nodes`.
pub fn write(nodes: &[Node]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(HEADER);
    out.extend(VERSION.to_le_bytes());
    let last = nodes.len().saturating_sub(1);
    for (i, n) in nodes.iter().enumerate() {
        write_node(n, i == last, &mut out);
    }
    out.extend(SENTINEL);
    out.extend(FOOT_ID);
    out.extend([0u8; 4]);
    let pad = match (16 - out.len() % 16) % 16 {
        0 => 16,
        p => p,
    };
    out.extend(vec![0u8; pad]);
    out.extend(VERSION.to_le_bytes());
    out.extend([0u8; 120]);
    out.extend(FOOT_MAGIC);
    out
}

/// A `P` entry of a `Properties70` block.
pub fn p(name: &str, ty: &str, label: &str, flags: &str, values: Vec<Prop>) -> Node {
    let mut props = vec![
        Prop::Str(name.into()),
        Prop::Str(ty.into()),
        Prop::Str(label.into()),
        Prop::Str(flags.into()),
    ];
    props.extend(values);
    Node::new("P", props)
}

/// FBX object name: `name\x00\x01Class`.
pub fn obj_name(name: &str, class: &str) -> Prop {
    Prop::Str(format!("{name}\x00\x01{class}"))
}

/// FBX time units per second.
pub const KTIME_PER_SECOND: f64 = 46_186_158_000.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_and_footer() {
        let b = write(&[Node::new("A", vec![Prop::I32(7)])
            .with(vec![Node::new("B", vec![Prop::Str("x".into())])])]);
        assert!(b.starts_with(HEADER));
        // A: 12 + 1 + 1 + 5 (I32) = 19, then B: 12 + 1 + 1 + 6 = 20, sentinel 13.
        let end_a = u32::from_le_bytes(b[27..31].try_into().unwrap()) as usize;
        assert_eq!(end_a, 27 + 19 + 20 + 13);
        assert_eq!(&b[b.len() - 16..], &FOOT_MAGIC);
        let fpos = end_a + 13;
        assert_eq!(&b[fpos..fpos + 16], &FOOT_ID);
    }
}

//! Grayscale float images and Gaussian pyramids with precomputed gradients.

/// Single-channel image with values in 0..255 stored as f32.
#[derive(Clone, Debug)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    pub fn from_luma8(width: usize, height: usize, pixels: &[u8]) -> Self {
        assert_eq!(
            pixels.len(),
            width * height,
            "pixel buffer does not match dimensions"
        );
        Self {
            width,
            height,
            data: pixels.iter().map(|&p| p as f32).collect(),
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    #[inline]
    fn get_clamped(&self, x: isize, y: isize) -> f32 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.data[y * self.width + x]
    }

    /// Bilinear sample with edge clamping.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let (xi, yi) = (x0 as isize, y0 as isize);
        if xi >= 0 && yi >= 0 && (xi as usize) + 1 < self.width && (yi as usize) + 1 < self.height {
            let i = yi as usize * self.width + xi as usize;
            let a = self.data[i];
            let b = self.data[i + 1];
            let c = self.data[i + self.width];
            let d = self.data[i + self.width + 1];
            return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
        }
        let a = self.get_clamped(xi, yi);
        let b = self.get_clamped(xi + 1, yi);
        let c = self.get_clamped(xi, yi + 1);
        let d = self.get_clamped(xi + 1, yi + 1);
        a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
    }

    /// Blur with a 5-tap binomial kernel and decimate by two.
    pub fn downsample(&self) -> GrayImage {
        const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
        let (w, h) = (self.width, self.height);
        let (nw, nh) = ((w + 1) / 2, (h + 1) / 2);
        // Horizontal pass, keeping only every second column.
        let mut tmp = GrayImage::new(nw, h);
        crate::par::rows(&mut tmp.data, nw, |y, row| {
            for (x, out) in row.iter_mut().enumerate() {
                let sx = 2 * x as isize;
                let mut s = 0.0;
                for (k, wk) in K.iter().enumerate() {
                    s += wk * self.get_clamped(sx + k as isize - 2, y as isize);
                }
                *out = s;
            }
        });
        let mut out = GrayImage::new(nw, nh);
        crate::par::rows(&mut out.data, nw, |y, row| {
            let sy = 2 * y as isize;
            for (x, o) in row.iter_mut().enumerate() {
                let mut s = 0.0;
                for (k, wk) in K.iter().enumerate() {
                    s += wk * tmp.get_clamped(x as isize, sy + k as isize - 2);
                }
                *o = s;
            }
        });
        out
    }

    /// Scharr gradients, normalised to intensity change per pixel.
    pub fn gradients(&self) -> (GrayImage, GrayImage) {
        let (w, h) = (self.width, self.height);
        let mut gx = GrayImage::new(w, h);
        let mut gy = GrayImage::new(w, h);
        crate::par::rows(&mut gx.data, w, |y, row| {
            let y = y as isize;
            for (x, o) in row.iter_mut().enumerate() {
                let x = x as isize;
                let p = |dx: isize, dy: isize| self.get_clamped(x + dx, y + dy);
                *o = (3.0 * (p(1, -1) - p(-1, -1))
                    + 10.0 * (p(1, 0) - p(-1, 0))
                    + 3.0 * (p(1, 1) - p(-1, 1)))
                    / 32.0;
            }
        });
        crate::par::rows(&mut gy.data, w, |y, row| {
            let y = y as isize;
            for (x, o) in row.iter_mut().enumerate() {
                let x = x as isize;
                let p = |dx: isize, dy: isize| self.get_clamped(x + dx, y + dy);
                *o = (3.0 * (p(-1, 1) - p(-1, -1))
                    + 10.0 * (p(0, 1) - p(0, -1))
                    + 3.0 * (p(1, 1) - p(1, -1)))
                    / 32.0;
            }
        });
        (gx, gy)
    }
}

/// Image pyramid; level 0 is full resolution.
#[derive(Clone, Debug)]
pub struct Pyramid {
    pub levels: Vec<GrayImage>,
    pub grad_x: Vec<GrayImage>,
    pub grad_y: Vec<GrayImage>,
}

impl Pyramid {
    pub fn build(base: GrayImage, max_levels: usize) -> Self {
        let mut levels = vec![base];
        while levels.len() < max_levels.max(1) {
            let last = levels.last().unwrap();
            if last.width < 48 || last.height < 48 {
                break;
            }
            let next = last.downsample();
            levels.push(next);
        }
        let (grad_x, grad_y) = levels.iter().map(|l| l.gradients()).unzip();
        Self {
            levels,
            grad_x,
            grad_y,
        }
    }

    pub fn width(&self) -> usize {
        self.levels[0].width
    }

    pub fn height(&self) -> usize {
        self.levels[0].height
    }
}

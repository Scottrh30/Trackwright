//! Tiny parallelism helpers: multi-threaded with the `parallel` feature, sequential otherwise.

/// Fill `data` (row-major, `width` per row) by calling `f(y, row)` for every row.
pub fn rows<F>(data: &mut [f32], width: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync + Send,
{
    if width == 0 {
        return;
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| f(y, row));
    }
    #[cfg(not(feature = "parallel"))]
    {
        data.chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| f(y, row));
    }
}

/// `items.iter().map(f).collect()`, in parallel when enabled.
pub fn map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        items.par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.iter().map(f).collect()
    }
}

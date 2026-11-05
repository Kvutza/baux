//! FP32 optimizer state, FP16 matrix storage, and FP32 tensor-core products.
pub use baux::semantics::muon as spec;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, launch_contract, thread};
pub use spec::{Polynomial, Shape, Step};

fn half(bits: u16) -> f32 {
    f16::from_bits(bits) as f32
}
fn bits(value: f32) -> u16 {
    (value as f16).to_bits()
}

unsafe fn sum(values: *mut f32, tid: u32, value: f32) -> f32 {
    unsafe {
        values.add(tid as usize).write(value);
    }
    thread::sync_threads();
    let mut stride = 128;
    while stride != 0 {
        if tid < stride {
            unsafe {
                values.add(tid as usize).write(
                    values.add(tid as usize).read() + values.add((tid + stride) as usize).read(),
                );
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    unsafe { values.read() }
}

unsafe fn tile_dot(left: *const f32, right: *const f32, ty: u32, tx: u32, limit: u32) -> f32 {
    let mut acc = 0.0_f32;
    let mut k = 0;
    while k < limit {
        unsafe {
            acc +=
                left.add((ty * 16 + k) as usize).read() * right.add((k * 16 + tx) as usize).read();
        }
        k += 1;
    }
    acc
}

#[cuda_module]
pub mod updates {
    use super::*;

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn init(
        weights: &[u16],
        mut master: DisjointSlice<f32>,
        mut momentum: DisjointSlice<f32>,
        count: u32,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i >= count {
            return;
        }
        unsafe {
            *master.get_unchecked_mut(i as usize) = half(weights[i as usize]);
            *momentum.get_unchecked_mut(i as usize) = 0.0;
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn momentum(
        mut gradient: DisjointSlice<f32>,
        mut momentum: DisjointSlice<f32>,
        mut partials: DisjointSlice<f32>,
        shape: Shape,
        step: Step,
    ) {
        static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let group = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let size = shape.rows * shape.cols;
        let chunks = size.div_ceil(8192);
        let matrix = group / chunks;
        if matrix >= shape.batch {
            return;
        }
        let first = (group % chunks) * 8192;
        let mut value = 0.0;
        let mut j = first + tid;
        while j < (first + 8192).min(size) {
            let i = (matrix * size + j) as usize;
            unsafe {
                let g = gradient.as_mut_ptr().add(i).read();
                let m =
                    step.momentum * momentum.as_mut_ptr().add(i).read() + (1.0 - step.momentum) * g;
                momentum.as_mut_ptr().add(i).write(m);
                let v = (1.0 - step.momentum) * g + step.momentum * m;
                gradient.as_mut_ptr().add(i).write(v);
                value += v * v;
            }
            j += 256;
        }
        let result = unsafe { sum(SharedArray::as_raw_mut_ptr(&raw mut SUMS), tid, value) };
        if tid == 0 {
            unsafe {
                *partials.get_unchecked_mut(group as usize) = result;
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn norm(partials: &[f32], mut norms: DisjointSlice<f32>, shape: Shape, step: Step) {
        static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let matrix = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if matrix >= shape.batch {
            return;
        }
        let chunks = (shape.rows * shape.cols).div_ceil(8192);
        let mut value = 0.0;
        let mut i = tid;
        while i < chunks {
            value += partials[(matrix * chunks + i) as usize];
            i += 256;
        }
        let result = unsafe { sum(SharedArray::as_raw_mut_ptr(&raw mut SUMS), tid, value) };
        if tid == 0 {
            unsafe {
                *norms.get_unchecked_mut(matrix as usize) =
                    1.0 / (1.01 * result.sqrt()).max(step.epsilon);
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn normalize(gradient: &[f32], norms: &[f32], mut x: DisjointSlice<u16>, shape: Shape) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        let size = shape.rows * shape.cols;
        if i >= size * shape.batch {
            return;
        }
        let j = i % size;
        let source = if shape.rows > shape.cols {
            (j % shape.rows) * shape.cols + j / shape.rows
        } else {
            j
        };
        unsafe {
            *x.get_unchecked_mut(i as usize) =
                bits(gradient[((i / size) * size + source) as usize] * norms[(i / size) as usize]);
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn transpose(x: &[u16], mut transposed: DisjointSlice<u16>, shape: Shape) {
        static mut TILE: SharedArray<u16, { 16 * 17 }> = SharedArray::UNINIT;
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let tid = thread::threadIdx_x();
        let cols = shape.cols.div_ceil(16);
        let tiles = shape.rows.div_ceil(16) * cols;
        let group = thread::blockIdx_x();
        let batch = group / tiles;
        if batch >= shape.batch {
            return;
        }
        let row_tile = (group % tiles) / cols * 16;
        let col_tile = (group % cols) * 16;
        let row = row_tile + tid / 16;
        let col = col_tile + tid % 16;
        let base = batch * shape.rows * shape.cols;
        let value = if row < shape.rows && col < shape.cols {
            x[(base + row * shape.cols + col) as usize]
        } else {
            0
        };
        unsafe {
            tile.add((tid / 16 * 17 + tid % 16) as usize).write(value);
        }
        thread::sync_threads();
        let row = row_tile + tid % 16;
        let col = col_tile + tid / 16;
        if row < shape.rows && col < shape.cols {
            unsafe {
                *transposed.get_unchecked_mut((base + col * shape.rows + row) as usize) =
                    tile.add((tid % 16 * 17 + tid / 16) as usize).read();
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn gemm(a: &[u16], b: &[u16], mut c: DisjointSlice<f32>, shape: Shape, inner: u32) {
        static mut LEFT: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut RIGHT: SharedArray<f32, 256> = SharedArray::UNINIT;
        let left = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LEFT) };
        let right = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RIGHT) };
        let tid = thread::threadIdx_x();
        let cols = shape.cols.div_ceil(16);
        let tiles = shape.rows.div_ceil(16) * cols;
        let group = thread::blockIdx_x();
        let batch = group / tiles;
        if batch >= shape.batch {
            return;
        }
        let ty = tid / 16;
        let tx = tid % 16;
        let row = (group % tiles) / cols * 16 + ty;
        let col = (group % cols) * 16 + tx;
        let a_base = batch * shape.rows * inner;
        let b_base = batch * inner * shape.cols;
        let mut acc = 0.0_f32;
        let mut start = 0;
        while start < inner {
            let av = if row < shape.rows && start + tx < inner {
                half(a[(a_base + row * inner + start + tx) as usize])
            } else {
                0.0
            };
            let bv = if start + ty < inner && col < shape.cols {
                half(b[(b_base + (start + ty) * shape.cols + col) as usize])
            } else {
                0.0
            };
            unsafe {
                left.add(tid as usize).write(av);
                right.add(tid as usize).write(bv);
            }
            thread::sync_threads();
            acc += unsafe { tile_dot(left, right, ty, tx, (inner - start).min(16)) };
            thread::sync_threads();
            start += 16;
        }
        if row < shape.rows && col < shape.cols {
            let out = (batch * shape.rows * shape.cols + row * shape.cols + col) as usize;
            unsafe { *c.get_unchecked_mut(out) = acc };
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn cast(product: &[f32], mut output: DisjointSlice<u16>, count: u32) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i < count {
            unsafe {
                *output.get_unchecked_mut(i as usize) = bits(product[i as usize]);
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn polynomial(
        gram: &[u16],
        square: &[f32],
        mut output: DisjointSlice<u16>,
        count: u32,
        c: Polynomial,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i < count {
            unsafe {
                *output.get_unchecked_mut(i as usize) =
                    bits(c.b * half(gram[i as usize]) + c.c * square[i as usize]);
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn axpy(
        x: &[u16],
        product: &[f32],
        mut next: DisjointSlice<u16>,
        count: u32,
        c: Polynomial,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i < count {
            unsafe {
                *next.get_unchecked_mut(i as usize) =
                    bits(c.a * half(x[i as usize]) + product[i as usize]);
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn update(
        direction: &[u16],
        mut master: DisjointSlice<f32>,
        mut weights: DisjointSlice<u16>,
        shape: Shape,
        step: Step,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        let size = shape.rows * shape.cols;
        if i >= size * shape.batch {
            return;
        }
        let j = i % size;
        let source = if shape.rows > shape.cols {
            (j % shape.cols) * shape.rows + j / shape.cols
        } else {
            j
        };
        unsafe {
            let value = (1.0 - step.rate * step.decay) * master.as_mut_ptr().add(i as usize).read()
                - step.rate * step.scale * half(direction[((i / size) * size + source) as usize]);
            *master.get_unchecked_mut(i as usize) = value;
            *weights.get_unchecked_mut(i as usize) = bits(value);
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn adam(
        gradient: &[f32],
        mut first: DisjointSlice<f32>,
        mut second: DisjointSlice<f32>,
        mut master: DisjointSlice<f32>,
        mut weights: DisjointSlice<u16>,
        count: u32,
        step: Step,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i >= count {
            return;
        }
        let i = i as usize;
        unsafe {
            let g = gradient[i];
            let m = step.momentum * first.as_mut_ptr().add(i).read() + (1.0 - step.momentum) * g;
            let v = step.beta2 * second.as_mut_ptr().add(i).read() + (1.0 - step.beta2) * g * g;
            *first.get_unchecked_mut(i) = m;
            *second.get_unchecked_mut(i) = v;
            let update = (m / step.bias1) / ((v / step.bias2).sqrt() + step.epsilon);
            let value = (1.0 - step.rate * step.decay) * master.as_mut_ptr().add(i).read()
                - step.rate * update;
            *master.get_unchecked_mut(i) = value;
            *weights.get_unchecked_mut(i) = bits(value);
        }
    }
}

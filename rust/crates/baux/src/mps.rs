//! Narrow FP32/FP16 MPS GEMM bridge. Buffers remain owned by the Metal graph.

use std::collections::HashMap;

use metal::foreign_types::ForeignTypeRef;
use metal::objc::{
    __send_message as send,
    rc::StrongPtr,
    runtime::{Class, NO, Object, Sel, YES},
};
use metal::{BufferRef, CommandBufferRef, DeviceRef, MTLCommandBufferStatus, MTLOrigin};

fn check_command(command: &CommandBufferRef) -> Result<(), String> {
    if !matches!(
        command.status(),
        MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
    ) {
        return Err("FBT cannot encode into a submitted command buffer".into());
    }
    Ok(())
}

// MPSMatrix is not wrapped by metal-rs 0.33. These selectors/types follow the
// public MPSCore/MPSMatrix SDK headers, not private GPU/compiler interfaces.
fn class(name: &str) -> Result<&'static Class, String> {
    Class::get(name).ok_or_else(|| format!("Missing MPS class {name}"))
}

unsafe fn owned(ptr: *mut Object) -> Result<StrongPtr, String> {
    if ptr.is_null() {
        Err("MPS object allocation failed".into())
    } else {
        Ok(unsafe { StrongPtr::new(ptr) })
    }
}

#[derive(Clone, Copy)]
pub struct Matrix<'a> {
    pub buffer: &'a BufferRef,
    pub rows: u32,
    pub cols: u32,
    pub batch: u32,
    /// Distance between matrices, in elements.
    pub stride: u64,
    /// Offset in elements, including submatrix row offsets.
    pub offset: u64,
    origin_row: u64,
    origin_col: u64,
    row_stride_cols: u32,
    data_type: u32,
    element_bytes: u64,
}

impl<'a> Matrix<'a> {
    pub fn new(buffer: &'a BufferRef, rows: u32, cols: u32) -> Self {
        Self {
            buffer,
            rows,
            cols,
            batch: 1,
            stride: u64::from(rows) * u64::from(cols),
            offset: 0,
            origin_row: 0,
            origin_col: 0,
            row_stride_cols: cols,
            data_type: metal::mps::MPSDataType::Float32 as u32,
            element_bytes: 4,
        }
    }

    pub fn half(buffer: &'a BufferRef, rows: u32, cols: u32) -> Self {
        Self {
            data_type: metal::mps::MPSDataType::Float16 as u32,
            element_bytes: 2,
            ..Self::new(buffer, rows, cols)
        }
    }

    pub fn layout(mut self, batch: u32, stride: u64, offset: u64) -> Self {
        self.batch = batch;
        self.stride = stride;
        self.offset = offset;
        self.origin_row = offset / u64::from(self.row_stride_cols);
        self.origin_col = offset % u64::from(self.row_stride_cols);
        self
    }

    pub fn row_view(mut self, rows: u32, offset: u64) -> Self {
        self.rows = rows;
        self.offset = offset;
        self.origin_row = offset / u64::from(self.row_stride_cols);
        self.origin_col = offset % u64::from(self.row_stride_cols);
        self
    }

    pub fn column_view(mut self, cols: u32, column: u32) -> Self {
        self.cols = cols;
        self.offset = u64::from(column);
        self.origin_row = 0;
        self.origin_col = u64::from(column);
        self
    }

    fn object(self, device: &DeviceRef) -> Result<StrongPtr, String> {
        let elements = u64::from(self.rows) * u64::from(self.cols);
        let row_stride = u64::from(self.row_stride_cols);
        let last_row = u64::from(self.rows.saturating_sub(1));
        let logical_end = self
            .origin_row
            .checked_add(last_row)
            .and_then(|row| row.checked_mul(row_stride))
            .and_then(|n| n.checked_add(self.origin_col))
            .and_then(|n| n.checked_add(u64::from(self.cols)))
            .ok_or("MPS matrix range overflow")?;
        let end_elements = if self.stride == 0 {
            logical_end
        } else {
            self.stride
                .checked_mul(u64::from(self.batch.saturating_sub(1)))
                .and_then(|n| n.checked_add(logical_end))
                .ok_or("MPS matrix range overflow")?
        };
        let end = end_elements
            .checked_mul(self.element_bytes)
            .ok_or("MPS matrix range overflow")?;
        let stride_bytes = self
            .stride
            .checked_mul(self.element_bytes)
            .ok_or("MPS matrix stride overflow")?;
        if self.rows == 0
            || self.cols == 0
            || self.batch == 0
            || (self.stride != 0 && self.stride < elements)
            || self.row_stride_cols < self.cols
            || self.origin_col + u64::from(self.cols) > row_stride
            || (self.stride != 0 && self.stride % row_stride != 0)
            || (self.batch > 1 && self.stride != 0 && self.offset + elements > self.stride)
            || end > self.buffer.length()
            || self.buffer.device().registry_id() != device.registry_id()
        {
            return Err(format!(
                "Invalid MPS matrix: rows={} cols={} batch={} stride={} elements={} offset={} \
                 end={} buf_len={}",
                self.rows,
                self.cols,
                self.batch,
                self.stride,
                elements,
                self.offset,
                end,
                self.buffer.length()
            ));
        }
        unsafe {
            // Use matrix origins for row views. The target MPS batched GEMM
            // path did not preserve nonzero buffer offsets in the sentinel
            // test.
            let batch_div = if self.stride == 0 {
                1
            } else {
                u64::from(self.batch)
            };
            // Spare buffer capacity is not padding between batch matrices.
            let full_rows = (if self.batch > 1 {
                self.stride / row_stride
            } else {
                0
            })
            .max(self.origin_row + u64::from(self.rows));
            let physical_bytes = full_rows
                .checked_mul(row_stride)
                .and_then(|n| n.checked_mul(self.element_bytes))
                .and_then(|n| n.checked_mul(batch_div))
                .ok_or("MPS physical extent overflow")?;
            if physical_bytes > self.buffer.length() {
                return Err("MPS physical extent exceeds allocation".into());
            }
            let matrix_bytes = if self.stride == 0 {
                0
            } else {
                stride_bytes.max(full_rows * row_stride * self.element_bytes)
            };
            let desc: *mut Object = send(
                class("MPSMatrixDescriptor")?,
                Sel::register(
                    "matrixDescriptorWithRows:columns:matrices:rowBytes:matrixBytes:dataType:",
                ),
                (
                    full_rows,
                    row_stride,
                    self.batch as u64,
                    row_stride * self.element_bytes,
                    matrix_bytes,
                    self.data_type,
                ),
            )
            .map_err(|e| e.to_string())?;
            if desc.is_null() {
                return Err("MPS matrix descriptor failed".into());
            }
            let object: *mut Object =
                send(class("MPSMatrix")?, Sel::register("alloc"), ()).map_err(|e| e.to_string())?;
            owned(
                send(
                    object,
                    Sel::register("initWithBuffer:descriptor:"),
                    (self.buffer, desc),
                )
                .map_err(|e| e.to_string())?,
            )
        }
    }
}

type MatrixKey = (usize, u32, u32, u32, u64, u64, u64, u64, u32, u32);

#[derive(Default)]
pub struct Matmul {
    kernels: HashMap<(u32, u32, u32, bool, bool, u64, u64, u32), StrongPtr>,
    matrices: HashMap<MatrixKey, StrongPtr>,
}

impl Matmul {
    pub fn encode(
        &mut self,
        device: &DeviceRef,
        command: &CommandBufferRef,
        a: Matrix<'_>,
        b: Matrix<'_>,
        c: Matrix<'_>,
        transpose_b: bool,
        alpha: f64,
    ) -> Result<(), String> {
        self.encode_flags(device, command, a, b, c, false, transpose_b, alpha, 0.0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_flags(
        &mut self,
        device: &DeviceRef,
        command: &CommandBufferRef,
        a: Matrix<'_>,
        b: Matrix<'_>,
        c: Matrix<'_>,
        transpose_a: bool,
        transpose_b: bool,
        alpha: f64,
        beta: f64,
    ) -> Result<(), String> {
        check_command(command)?;
        let (m, k, n) = (c.rows, if transpose_a { a.rows } else { a.cols }, c.cols);
        if c.rows != m
            || a.batch != b.batch
            || a.batch != c.batch
            || (a.rows, a.cols) != if transpose_a { (k, m) } else { (m, k) }
            || (b.rows, b.cols) != if transpose_b { (n, k) } else { (k, n) }
            || !alpha.is_finite()
            || !beta.is_finite()
            || a.data_type != b.data_type
            || a.element_bytes != b.element_bytes
            || std::ptr::eq(a.buffer, c.buffer)
            || std::ptr::eq(b.buffer, c.buffer)
        {
            return Err(format!(
                "MPS GEMM shape mismatch: a({}x{}, {}B), b({}x{}, {}B), c({}x{}, {}B), m={}, \
                 n={}, k={}, trans_a={}, trans_b={}, a_buf={:p}, b_buf={:p}, c_buf={:p}",
                a.rows,
                a.cols,
                a.element_bytes,
                b.rows,
                b.cols,
                b.element_bytes,
                c.rows,
                c.cols,
                c.element_bytes,
                m,
                n,
                k,
                transpose_a,
                transpose_b,
                a.buffer,
                b.buffer,
                c.buffer
            ));
        }
        let get_obj = |matrices: &mut HashMap<MatrixKey, StrongPtr>,
                       mat: Matrix<'_>|
         -> Result<StrongPtr, String> {
            let key = (
                mat.buffer.as_ptr() as usize,
                mat.rows,
                mat.cols,
                mat.batch,
                mat.stride,
                mat.offset,
                mat.origin_row,
                mat.origin_col,
                mat.row_stride_cols,
                mat.data_type,
            );
            if let Some(obj) = matrices.get(&key) {
                Ok(obj.clone())
            } else {
                let obj = mat.object(device)?;
                matrices.insert(key, obj.clone());
                Ok(obj)
            }
        };
        let a_obj = get_obj(&mut self.matrices, a)?;
        let b_obj = get_obj(&mut self.matrices, b)?;
        let c_obj = get_obj(&mut self.matrices, c)?;
        let key = (
            m,
            n,
            k,
            transpose_a,
            transpose_b,
            alpha.to_bits(),
            beta.to_bits(),
            a.data_type,
        );
        if let std::collections::hash_map::Entry::Vacant(entry) = self.kernels.entry(key) {
            let object = unsafe {
                let allocated: *mut Object = send(
                    class("MPSMatrixMultiplication")?,
                    Sel::register("alloc"),
                    (),
                )
                .map_err(|e| e.to_string())?;
                owned(
                    send(
                        allocated,
                        Sel::register(
                            "initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:\
                             interiorColumns:alpha:beta:",
                        ),
                        (
                            device,
                            if transpose_a { YES } else { NO },
                            if transpose_b { YES } else { NO },
                            m as u64,
                            n as u64,
                            k as u64,
                            alpha,
                            beta,
                        ),
                    )
                    .map_err(|e| e.to_string())?,
                )?
            };
            entry.insert(object);
        }
        let kernel = &self.kernels[&key];
        unsafe {
            for (selector, matrix) in [
                ("setLeftMatrixOrigin:", a),
                ("setRightMatrixOrigin:", b),
                ("setResultMatrixOrigin:", c),
            ] {
                let origin = MTLOrigin {
                    x: matrix.origin_row,
                    y: matrix.origin_col,
                    z: 0,
                };
                send::<Object, _, ()>(**kernel, Sel::register(selector), (origin,))
                    .map_err(|e| e.to_string())?;
            }
            send::<Object, _, ()>(**kernel, Sel::register("setBatchSize:"), (a.batch as u64,))
                .map_err(|e| e.to_string())?;
            send::<Object, _, ()>(
                **kernel,
                Sel::register("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"),
                (command, *a_obj, *b_obj, *c_obj),
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apple_gpu::Runtime;

    #[test]
    fn batch_parity() {
        metal::objc::rc::autoreleasepool(|| {
            let runtime = Runtime::shared().unwrap();
            let mut gemm = Matmul::default();
            for transpose in [false, true] {
                for m in [1, 256] {
                    let (n, k, batch) = (13, 17, 4);
                    let a_stride = 257 * k;
                    let a_offset = (257 - m) * k;
                    let c_stride = 257 * n;
                    let c_offset = (257 - m) * n;
                    let av: Vec<f32> = (0..a_stride * batch)
                        .map(|i| (i % 19) as f32 * 0.03 - 0.2)
                        .collect();
                    let bv: Vec<f32> = (0..n * k * batch)
                        .map(|i| (i % 23) as f32 * 0.02 - 0.1)
                        .collect();
                    let a = runtime.buffer_with(&av);
                    let b = runtime.buffer_with(&bv);
                    let c = runtime.buffer_with(&vec![-999.0f32; (c_stride * batch) as usize]);
                    let matrix_a =
                        Matrix::new(&a, m, k).layout(batch, a_stride as u64, a_offset as u64);
                    let matrix_b = Matrix {
                        batch,
                        ..Matrix::new(
                            &b,
                            if transpose { n } else { k },
                            if transpose { k } else { n },
                        )
                    };
                    let matrix_c =
                        Matrix::new(&c, m, n).layout(batch, c_stride as u64, c_offset as u64);
                    let command = runtime.queue.new_command_buffer();
                    gemm.encode(
                        &runtime.device,
                        command,
                        matrix_a,
                        matrix_b,
                        matrix_c,
                        transpose,
                        0.5,
                    )
                    .unwrap();
                    command.commit();
                    command.wait_until_completed();
                    assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                    let actual = unsafe {
                        std::slice::from_raw_parts(
                            c.contents().cast::<f32>(),
                            (c_stride * batch) as usize,
                        )
                    };
                    for sample in 0..batch {
                        for row in 0..257 - m {
                            assert_eq!(actual[(sample * c_stride + row * n) as usize], -999.0);
                        }
                        for row in 0..m {
                            for col in 0..n {
                                let expected: f64 = (0..k)
                                    .map(|j| {
                                        let ai = sample * a_stride + a_offset + row * k + j;
                                        let bi = sample * n * k
                                            + if transpose { col * k + j } else { j * n + col };
                                        f64::from(av[ai as usize])
                                            * f64::from(bv[bi as usize])
                                            * 0.5
                                    })
                                    .sum();
                                assert!(
                                    (f64::from(
                                        actual[(sample * c_stride + c_offset + row * n + col)
                                            as usize]
                                    ) - expected)
                                        .abs()
                                        < 1e-5
                                );
                            }
                        }
                    }
                }
            }
        });
    }
}

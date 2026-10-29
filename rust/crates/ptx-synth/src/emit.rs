//! Low-level instruction emission methods for PTX kernel builder.

use crate::ir::PtxKernelBuilder;
use crate::reg::{Reg32, Reg64, RegPred};

pub trait PtxEmitter {
    fn raw(&mut self, line: impl AsRef<str>);

    // Special register access
    fn get_tid_x(&mut self, dst: Reg32) {
        self.raw(format!("mov.u32 %r{}, %tid.x;", dst.0));
    }

    fn get_ntid_x(&mut self, dst: Reg32) {
        self.raw(format!("mov.u32 %r{}, %ntid.x;", dst.0));
    }

    fn get_ctaid_x(&mut self, dst: Reg32) {
        self.raw(format!("mov.u32 %r{}, %ctaid.x;", dst.0));
    }

    // Parameter loading
    fn ld_param_u64(&mut self, dst: Reg64, param_name: &str) {
        self.raw(format!("ld.param.u64 %rd{}, [{param_name}];", dst.0));
    }

    fn ld_param_u32(&mut self, dst: Reg32, param_name: &str) {
        self.raw(format!("ld.param.u32 %r{}, [{param_name}];", dst.0));
    }

    // Arithmetic
    fn add_u64(&mut self, dst: Reg64, a: Reg64, b: Reg64) {
        self.raw(format!("add.u64 %rd{}, %rd{}, %rd{};", dst.0, a.0, b.0));
    }

    fn mul_lo_u32(&mut self, dst: Reg32, a: Reg32, b: Reg32) {
        self.raw(format!("mul.lo.u32 %r{}, %r{}, %r{};", dst.0, a.0, b.0));
    }

    fn mad_lo_u32(&mut self, dst: Reg32, a: Reg32, b: Reg32, c: Reg32) {
        self.raw(format!(
            "mad.lo.u32 %r{}, %r{}, %r{}, %r{};",
            dst.0, a.0, b.0, c.0
        ));
    }

    fn shl_imm_u32(&mut self, dst: Reg32, src: Reg32, shift: u32) {
        self.raw(format!("shl.b32 %r{}, %r{}, {shift};", dst.0, src.0));
    }

    fn cvt_u64_u32(&mut self, dst: Reg64, src: Reg32) {
        self.raw(format!("cvt.u64.u32 %rd{}, %r{};", dst.0, src.0));
    }

    // Branching & predicates
    fn setp_ge_u32(&mut self, pred: RegPred, a: Reg32, b: Reg32) {
        self.raw(format!("setp.ge.u32 %p{}, %r{}, %r{};", pred.0, a.0, b.0));
    }

    fn branch_if(&mut self, pred: RegPred, target_label: &str) {
        self.raw(format!("@%p{} bra {target_label};", pred.0));
    }

    fn exit(&mut self) {
        self.raw("ret;");
    }
}

impl PtxEmitter for PtxKernelBuilder {
    fn raw(&mut self, line: impl AsRef<str>) {
        self.raw(line);
    }
}

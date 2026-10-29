//! Canonical compile-time flat weight layout and 7-step layer visit schedule
//! for mHC-4.
use super::probe::initialized::FAMILIES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub elements: u64,
    pub offset: u64,
}

impl Slot {
    pub const fn byte_offset(self) -> u64 {
        self.offset * 2
    }

    pub const fn byte_len(self) -> u64 {
        self.elements * 2
    }

    pub const fn end(self) -> u64 {
        self.offset + self.elements
    }

    pub const fn byte_end(self) -> u64 {
        (self.offset + self.elements) * 2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mhc4Layout {
    pub router: Slot,
    pub qkv: Slot,
    pub output: Slot,
    pub gate_up: Slot,
    pub down: Slot,
    pub attn_norm: Slot,
    pub ffn_norm: Slot,
    pub readout: Slot,
    pub attn_pred: Slot,
    pub attn_bias: Slot,
    pub attn_ctrl: Slot,
    pub moe_pred: Slot,
    pub moe_bias: Slot,
    pub moe_ctrl: Slot,
    pub final_norm: Slot,
    pub total: u64,
}

impl Mhc4Layout {
    pub const fn slots(self) -> [Slot; 15] {
        [
            self.router,
            self.qkv,
            self.output,
            self.gate_up,
            self.down,
            self.attn_norm,
            self.ffn_norm,
            self.readout,
            self.attn_pred,
            self.attn_bias,
            self.attn_ctrl,
            self.moe_pred,
            self.moe_bias,
            self.moe_ctrl,
            self.final_norm,
        ]
    }

    pub const fn byte_len(self) -> u64 {
        self.total * 2
    }
}

pub const LAYOUT: Mhc4Layout = {
    let mut slots = [Slot {
        elements: 0,
        offset: 0,
    }; 15];
    let mut cursor = 0;
    let mut index = 0;
    while index < 15 {
        let elements = FAMILIES[index].elements;
        slots[index] = Slot {
            elements,
            offset: cursor,
        };
        cursor += elements;
        index += 1;
    }
    Mhc4Layout {
        router: slots[0],
        qkv: slots[1],
        output: slots[2],
        gate_up: slots[3],
        down: slots[4],
        attn_norm: slots[5],
        ffn_norm: slots[6],
        readout: slots[7],
        attn_pred: slots[8],
        attn_bias: slots[9],
        attn_ctrl: slots[10],
        moe_pred: slots[11],
        moe_bias: slots[12],
        moe_ctrl: slots[13],
        final_norm: slots[14],
        total: cursor,
    }
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerVisit {
    pub layer: u32,
    pub pass: u32,
    pub step: u32,
}

pub const MHC4_VISITS: [LayerVisit; 7] = [
    LayerVisit {
        layer: 0,
        pass: 0,
        step: 0,
    },
    LayerVisit {
        layer: 1,
        pass: 0,
        step: 1,
    },
    LayerVisit {
        layer: 2,
        pass: 0,
        step: 2,
    },
    LayerVisit {
        layer: 1,
        pass: 1,
        step: 3,
    },
    LayerVisit {
        layer: 2,
        pass: 1,
        step: 4,
    },
    LayerVisit {
        layer: 3,
        pass: 0,
        step: 5,
    },
    LayerVisit {
        layer: 4,
        pass: 0,
        step: 6,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use baux::semantics::recurrence::RecurrentCore;
    use baux::semantics::shape::MHC4_PARAMS;

    #[test]
    fn layout_slots() {
        let slots = LAYOUT.slots();
        assert_eq!(slots.len(), FAMILIES.len());
        let mut cursor = 0;
        for (slot, family) in slots.iter().zip(FAMILIES.iter()) {
            assert_eq!(slot.offset, cursor);
            assert_eq!(slot.elements, family.elements);
            assert_eq!(slot.byte_offset(), cursor * 2);
            assert_eq!(slot.byte_len(), family.elements * 2);
            cursor = slot.end();
        }
        assert_eq!(cursor, LAYOUT.total);
        assert_eq!(LAYOUT.total, MHC4_PARAMS);
        assert_eq!(LAYOUT.total, 1_177_723_160);
        assert_eq!(LAYOUT.byte_len(), 2_355_446_320);
    }

    #[test]
    fn visit_schedule() {
        let visits = RecurrentCore::selective_fbt().layer_visits(5, 2).unwrap();
        assert_eq!(visits.len(), MHC4_VISITS.len());
        for (step, visit) in MHC4_VISITS.iter().zip(visits.iter()) {
            assert_eq!(step.layer as usize, visit.layer);
            assert_eq!(step.pass as usize, visit.visit);
            assert_eq!(step.step as usize, visit.execution);
        }
    }
}

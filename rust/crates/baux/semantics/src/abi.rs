//! Declare scalar Rust/CUDA records and their Metal source from one field list.

macro_rules! metal_kind {
    (u32) => {
        "uint"
    };
    (f32) => {
        "float"
    };
}

macro_rules! record {
    ($name:ident { $($field:ident: $kind:tt),* $(,)? }) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct $name { $(pub $field: $kind),* }

        // SAFETY: only repr(C) u32/f32 scalars and fixed arrays are accepted.
        #[cfg(feature = "cuda")]
        unsafe impl cuda_core::DeviceCopy for $name {}

        impl $name {
            /// Metal checks its actual layout against the compiling Rust target.
            pub fn metal_decl() -> String {
                let mut source = String::from(concat!(
                    "struct ", stringify!($name), " {\n",
                    $(metal_field!($field, $kind),)*
                    "};\n"
                ));
                source.push_str(&format!(
                    "static_assert(sizeof({}) == {}, \"{} size\");\n",
                    stringify!($name), size_of::<Self>(), stringify!($name)
                ));
                $(source.push_str(&format!(
                    "static_assert(__builtin_offsetof({}, {}) == {}, \"{} offset\");\n",
                    stringify!($name), stringify!($field),
                    core::mem::offset_of!(Self, $field), stringify!($field)
                ));)*
                source
            }
        }
    };
}

macro_rules! metal_field {
    ($field:ident, $kind:ident) => {
        concat!(metal_kind!($kind), " ", stringify!($field), ";\n")
    };
    ($field:ident,[$kind:ident; $count:literal]) => {
        concat!(
            metal_kind!($kind),
            " ",
            stringify!($field),
            "[",
            stringify!($count),
            "];\n"
        )
    };
}

pub(crate) use metal_field;
pub(crate) use metal_kind;
pub(crate) use record;

//! Deterministic CBOR storage using the repository's native Deser types.
//!
//! Encoding is separate from logical identity: existing run and protocol hashes
//! retain their original definition. Schema evolution still requires
//! migrations.

pub use deser_cbor::from_slice;

pub fn to_vec(value: &dyn deser::Serialize) -> Result<Vec<u8>, deser::Error> {
    deser_cbor::SerializerConfig::new()
        .canonical(true)
        .to_vec(value)
}

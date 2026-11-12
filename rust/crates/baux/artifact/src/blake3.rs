//! Official IETF/reference BLAKE3 cryptographic Merkle tree hash function.
//!
//! Implements the 7-round permutation and binary Merkle tree structure over
//! 1024-byte chunks with multi-threaded subtree reduction via
//! `std::thread::scope`.

use std::fs;
use std::path::Path;

pub const OUT_LEN: usize = 32;
pub const KEY_LEN: usize = 32;
const BLOCK_LEN: usize = 64;
const CHUNK_LEN: usize = 1024;
const PARALLEL_CHUNKS: usize = 64;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const KEYED_HASH: u32 = 1 << 4;
const DERIVE_CONTEXT: u32 = 1 << 5;
const DERIVE_MATERIAL: u32 = 1 << 6;

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

#[inline]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

#[inline]
fn round(state: &mut [u32; 16], m: &[u32; 16]) {
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

#[inline]
fn permute(m: &mut [u32; 16]) {
    let mut permuted = [0u32; 16];
    for i in 0..16 {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    *m = permuted;
}

fn compress(
    chaining_value: &[u32; 8],
    block_words: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let mut state = [
        chaining_value[0],
        chaining_value[1],
        chaining_value[2],
        chaining_value[3],
        chaining_value[4],
        chaining_value[5],
        chaining_value[6],
        chaining_value[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len,
        flags,
    ];
    let mut block = *block_words;
    for _ in 0..6 {
        round(&mut state, &block);
        permute(&mut block);
    }
    round(&mut state, &block);
    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
    state
}

#[inline]
fn first_words(output: [u32; 16]) -> [u32; 8] {
    [
        output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7],
    ]
}

fn block_words(bytes: &[u8]) -> [u32; 16] {
    let mut padded = [0u8; BLOCK_LEN];
    padded[..bytes.len()].copy_from_slice(bytes);
    let mut words = [0u32; 16];
    for (i, chunk) in padded.chunks_exact(4).enumerate() {
        words[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    words
}

fn words_bytes(words: &[u32; 8]) -> [u8; OUT_LEN] {
    let mut out = [0u8; OUT_LEN];
    for (i, &w) in words.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

#[derive(Clone, Copy)]
struct Output {
    input_chaining_value: [u32; 8],
    block_words: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        first_words(compress(
            &self.input_chaining_value,
            &self.block_words,
            self.counter,
            self.block_len,
            self.flags,
        ))
    }

    fn root_hash(&self) -> [u8; OUT_LEN] {
        let words = first_words(compress(
            &self.input_chaining_value,
            &self.block_words,
            0,
            self.block_len,
            self.flags | ROOT,
        ));
        words_bytes(&words)
    }
}

fn chunk_output(chunk: &[u8], chunk_counter: u64, key_words: &[u32; 8], flags: u32) -> Output {
    if chunk.is_empty() {
        return Output {
            input_chaining_value: *key_words,
            block_words: [0u32; 16],
            counter: chunk_counter,
            block_len: 0,
            flags: flags | CHUNK_START | CHUNK_END,
        };
    }
    let mut cv = *key_words;
    let blocks: Vec<&[u8]> = chunk.chunks(BLOCK_LEN).collect();
    let last_idx = blocks.len() - 1;
    for (idx, &block) in blocks[..last_idx].iter().enumerate() {
        let start_flag = if idx == 0 { CHUNK_START } else { 0 };
        let words = block_words(block);
        cv = first_words(compress(
            &cv,
            &words,
            chunk_counter,
            block.len() as u32,
            flags | start_flag,
        ));
    }
    let start_flag = if last_idx == 0 { CHUNK_START } else { 0 };
    let last_block = blocks[last_idx];
    Output {
        input_chaining_value: cv,
        block_words: block_words(last_block),
        counter: chunk_counter,
        block_len: last_block.len() as u32,
        flags: flags | start_flag | CHUNK_END,
    }
}

fn parent_output(
    left_cv: &[u32; 8],
    right_cv: &[u32; 8],
    key_words: &[u32; 8],
    flags: u32,
) -> Output {
    let mut block_words = [0u32; 16];
    block_words[..8].copy_from_slice(left_cv);
    block_words[8..].copy_from_slice(right_cv);
    Output {
        input_chaining_value: *key_words,
        block_words,
        counter: 0,
        block_len: BLOCK_LEN as u32,
        flags: flags | PARENT,
    }
}

fn reduce_cvs(cvs: &[[u32; 8]], key_words: &[u32; 8], flags: u32) -> [u32; 8] {
    if cvs.len() == 1 {
        return cvs[0];
    }
    let left_len = 1usize << (usize::BITS - 1 - (cvs.len() - 1).leading_zeros());
    let left_cv = reduce_cvs(&cvs[..left_len], key_words, flags);
    let right_cv = reduce_cvs(&cvs[left_len..], key_words, flags);
    parent_output(&left_cv, &right_cv, key_words, flags).chaining_value()
}

fn batch_cvs(batch: &[&[u8]], base: usize, key_words: &[u32; 8], flags: u32) -> Vec<[u32; 8]> {
    let mut out = Vec::with_capacity(batch.len());
    for (i, chunk) in batch.iter().enumerate() {
        out.push(chunk_output(chunk, (base + i) as u64, key_words, flags).chaining_value());
    }
    out
}

fn keyed_digest(input: &[u8], key_words: &[u32; 8], flags: u32) -> [u8; OUT_LEN] {
    if input.len() <= CHUNK_LEN {
        return chunk_output(input, 0, key_words, flags).root_hash();
    }
    let chunks: Vec<&[u8]> = input.chunks(CHUNK_LEN).collect();
    let cvs: Vec<[u32; 8]> = if chunks.len() >= PARALLEL_CHUNKS {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(chunks.len());
        let batch_size = chunks.len().div_ceil(threads);
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(threads);
            for (batch_idx, batch) in chunks.chunks(batch_size).enumerate() {
                let base = batch_idx * batch_size;
                handles.push(s.spawn(move || batch_cvs(batch, base, key_words, flags)));
            }
            let mut out = Vec::with_capacity(chunks.len());
            for h in handles {
                out.extend(h.join().expect("blake3 worker thread"));
            }
            out
        })
    } else {
        batch_cvs(&chunks, 0, key_words, flags)
    };
    let left_len = 1usize << (usize::BITS - 1 - (cvs.len() - 1).leading_zeros());
    let left_cv = reduce_cvs(&cvs[..left_len], key_words, flags);
    let right_cv = reduce_cvs(&cvs[left_len..], key_words, flags);
    parent_output(&left_cv, &right_cv, key_words, flags).root_hash()
}

/// Compute the 32-byte BLAKE3 digest of an in-memory byte slice.
pub fn hash(input: &[u8]) -> [u8; OUT_LEN] {
    keyed_digest(input, &IV, 0)
}

/// Compute the 64-character lowercase hexadecimal BLAKE3 digest of `input`.
pub fn hash_hex(input: &[u8]) -> String {
    hex_encode(&hash(input))
}

/// Compute the 32-byte keyed BLAKE3 MAC using a 32-byte key.
pub fn keyed_hash(key: &[u8; KEY_LEN], input: &[u8]) -> [u8; OUT_LEN] {
    let mut key_words = [0u32; 8];
    for (i, chunk) in key.chunks_exact(4).enumerate() {
        key_words[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    keyed_digest(input, &key_words, KEYED_HASH)
}

/// Derive a 32-byte subkey using BLAKE3's domain-separated KDF mode.
pub fn derive_key(context: &str, key_material: &[u8]) -> [u8; OUT_LEN] {
    let context_bytes = keyed_digest(context.as_bytes(), &IV, DERIVE_CONTEXT);
    let mut context_words = [0u32; 8];
    for (i, chunk) in context_bytes.chunks_exact(4).enumerate() {
        context_words[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    keyed_digest(key_material, &context_words, DERIVE_MATERIAL)
}

/// Compute the 64-character lowercase hex BLAKE3 digest of a file on disk.
pub fn hash_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(hash_hex(&bytes))
}

/// Compute the 64-character lowercase hexadecimal SHA-256 digest of `input`.
pub fn sha256_hex(input: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest: [u8; 32] = Sha256::digest(input).into();
    hex_encode(&digest)
}

/// Compute the standard base64-encoded SHA-384 digest for Subresource Integrity
/// (`sha384-...`).
pub fn sha384_sri(input: &[u8]) -> String {
    use sha2::{Digest, Sha384};
    let digest = Sha384::digest(input);
    base64_encode(&digest)
}

fn base64_encode(bytes: &[u8]) -> String {
    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut i = 0;
    while i < bytes.len() {
        let b0 = bytes[i];
        let b1 = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
        let b2 = if i + 2 < bytes.len() { bytes[i + 2] } else { 0 };
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);
        out.push(B64[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64[((n >> 12) & 0x3f) as usize] as char);
        out.push(if i + 1 < bytes.len() {
            B64[((n >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if i + 2 < bytes.len() {
            B64[(n & 0x3f) as usize] as char
        } else {
            '='
        });
        i += 3;
    }
    out
}

fn hex_encode(bytes: &[u8; OUT_LEN]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(OUT_LEN * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
#[path = "blake3/tests.rs"]
mod tests;

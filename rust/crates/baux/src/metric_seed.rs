//! Ported from yubo-research/enn, commit
//! 506e98c506eeb849cffbf53d9ddf3a3a799c6830. NumPy `SeedSequence` (pool size 4)
//! for integer entropy.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_LEFT: u32 = 0xca01_f9dd;
const MIX_RIGHT: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

fn hashmix(mut value: u32, hash_const: &mut u32) -> u32 {
    value ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let mut result = MIX_LEFT
        .wrapping_mul(x)
        .wrapping_sub(MIX_RIGHT.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

fn entropy_words(entropy: u64) -> ([u32; 2], usize) {
    if entropy == 0 {
        return ([0, 0], 1);
    }
    let lo = entropy as u32;
    let hi = (entropy >> 32) as u32;
    if hi == 0 { ([lo, 0], 1) } else { ([lo, hi], 2) }
}

fn mix_entropy(pool: &mut [u32], entropy: &[u32]) {
    let mut hash_const = INIT_A;
    for (i, slot) in pool.iter_mut().enumerate() {
        let word = if i < entropy.len() { entropy[i] } else { 0 };
        *slot = hashmix(word, &mut hash_const);
    }
    let n = pool.len();
    for i_src in 0..n {
        for i_dst in 0..n {
            if i_src != i_dst {
                let mixed = hashmix(pool[i_src], &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], mixed);
            }
        }
    }
    for &word in entropy.iter().skip(n) {
        for slot in pool.iter_mut() {
            let mixed = hashmix(word, &mut hash_const);
            *slot = mix(*slot, mixed);
        }
    }
}

/// Stack-allocated `SeedSequence(entropy).generate_state(4, dtype=uint64)`.
pub fn seed_quad(entropy: u64) -> [u64; 4] {
    let mut pool = [0u32; 4];
    let (words, len) = entropy_words(entropy);
    mix_entropy(&mut pool, &words[..len]);
    let mut state = [0u32; 8];
    let mut hash_const = INIT_B;
    for (i, slot) in state.iter_mut().enumerate() {
        let mut data = pool[i % pool.len()] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        data = data.wrapping_mul(hash_const);
        *slot = data ^ (data >> XSHIFT);
    }
    [
        u64::from(state[0]) | (u64::from(state[1]) << 32),
        u64::from(state[2]) | (u64::from(state[3]) << 32),
        u64::from(state[4]) | (u64::from(state[5]) << 32),
        u64::from(state[6]) | (u64::from(state[7]) << 32),
    ]
}

/// `SeedSequence(entropy).generate_state(n_u64, dtype=uint64)`.
pub fn seed_words(entropy: u64, n_u64: usize) -> Vec<u64> {
    let mut pool = [0u32; 4];
    let (words, len) = entropy_words(entropy);
    mix_entropy(&mut pool, &words[..len]);
    let n_u32 = n_u64 * 2;
    let mut state = vec![0u32; n_u32];
    let mut hash_const = INIT_B;
    for (i, slot) in state.iter_mut().enumerate() {
        let mut data = pool[i % pool.len()] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        data = data.wrapping_mul(hash_const);
        *slot = data ^ (data >> XSHIFT);
    }
    state
        .chunks(2)
        .map(|c| u64::from(c[0]) | (u64::from(c[1]) << 32))
        .collect()
}

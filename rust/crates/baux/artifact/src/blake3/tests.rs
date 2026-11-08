use super::{derive_key, hash_file, hash_hex, hex_encode, keyed_hash};

fn pattern_input(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

#[test]
fn official_vectors() {
    assert_eq!(
        hash_hex(b""),
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    );
    assert_eq!(
        hash_hex(b"abc"),
        "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
    );
    let cases: &[(usize, &str)] = &[
        (
            1,
            "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
        ),
        (
            64,
            "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98",
        ),
        (
            65,
            "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee",
        ),
        (
            1024,
            "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7",
        ),
        (
            1025,
            "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444",
        ),
        (
            2048,
            "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a",
        ),
        (
            2049,
            "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030",
        ),
        (
            65536,
            "68d647e619a930e7b1082f74f334b0c65a315725569bdc123f0ee11881717bfe",
        ),
    ];
    for &(len, expected) in cases {
        let input = pattern_input(len);
        assert_eq!(hash_hex(&input), expected, "mismatch at len={len}");
    }
}

#[test]
fn keyed_modes() {
    let key = *b"whats the Elvish word for friend";
    let input = pattern_input(2049);
    let mac = hex_encode(&keyed_hash(&key, &input));
    assert_eq!(
        mac,
        "9f29700902f7c86e514ddc4df1e3049f258b2472b6dd5267f61bf13983b78dd5"
    );
    let derived = hex_encode(&derive_key(
        "BLAKE3 2019-12-27 16:29:52 test vectors context",
        &input,
    ));
    assert_eq!(
        derived,
        "2ea477c5515cc3dd606512ee72bb3e0e758cfae7232826f35fb98ca1bcbdf273"
    );
}

#[test]
fn file_hashing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let empty_path = dir.path().join("empty.bin");
    std::fs::write(&empty_path, b"").expect("write empty");
    assert_eq!(hash_file(&empty_path).expect("hash empty"), hash_hex(b""));

    let big_path = dir.path().join("big.bin");
    let payload = pattern_input(33_000);
    std::fs::write(&big_path, &payload).expect("write big");
    assert_eq!(hash_file(&big_path).expect("hash big"), hash_hex(&payload));
}

#[test]
fn upstream_locks() {
    let Ok(root) = std::env::current_dir() else {
        return;
    };
    if !root.join("research/upstream/muon.lock.toml").is_file() {
        return;
    }
    for (name, rel) in [
        ("muon", ".cache/baux/upstream/Muon"),
        ("polar_express", ".cache/baux/upstream/PolarExpress"),
        ("mhc", ".cache/baux/upstream/mHC"),
        ("acme", ".cache/baux/upstream/acme"),
        ("hnet", ".cache/baux/upstream/hnet"),
        (
            "torax",
            ".pixi/envs/baux-py314/lib/python3.14/site-packages/torax",
        ),
    ] {
        let dir = root.join(rel);
        if dir.is_dir() {
            crate::artifact::merkle::verify_lock(&root, name, None)
                .unwrap_or_else(|e| panic!("verify_lock({name}) failed: {e}"));
        }
    }
}

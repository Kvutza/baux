use super::*;

const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("baux-sha256-{name}-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn pinned_bytes() {
    let directory = Scratch::new("pins");
    let path = directory.0.join("tokenizer.json");
    fs::write(&path, b"abc").unwrap();
    assert_eq!(file_hash(&path).unwrap(), ABC_SHA256);
    verify(&path, Some(ABC_SHA256)).unwrap();

    // Both digest algorithms produce 64 hexadecimal characters. Comparing a
    // BLAKE3 catalogue identity to a SHA-256 pin must still fail.
    let catalogue = baux::artifact::runs::file_digest(&path).unwrap();
    assert_ne!(catalogue, ABC_SHA256);
    assert!(verify(&path, Some(&catalogue)).is_err());

    fs::write(&path, b"abd").unwrap();
    assert!(verify(&path, Some(ABC_SHA256)).is_err());
    assert!(verify(&path, None).is_err());
}

#[test]
fn streaming_digest() {
    let directory = Scratch::new("stream");
    let path = directory.0.join("corpus");
    // NIST's one-million-'a' SHA-256 vector crosses the streaming boundary.
    fs::write(&path, vec![b'a'; 1_000_000]).unwrap();
    assert_eq!(
        file_hash(&path).unwrap(),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
    fs::write(&path, []).unwrap();
    assert_eq!(
        file_hash(&path).unwrap(),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

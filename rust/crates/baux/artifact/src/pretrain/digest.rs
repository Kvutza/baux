//! SHA-256 byte identities required by pinned corpus recipes and manifests.

use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

/// Hash original bytes without loading an entire pretraining corpus into RAM.
/// This contract is independent of BLAKE3 identities in the run catalogue.
pub fn sha_digest(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65_536];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            return Ok(format!("{:x}", hash.finalize()));
        }
        hash.update(&buffer[..count]);
    }
}

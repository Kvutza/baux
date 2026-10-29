//! Conflict decisions for epochs of GPU dispatches.
//!
//! Buffer identifiers denote host resource object identity. Distinct objects
//! sharing physical storage need a separate alias contract.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Access {
    pub buffer: u64,
    pub start: u64,
    pub end: u64,
    pub write: bool,
}

/// RAW, WAR or WAW conflict on at least one byte; shared reads and empty ranges
/// are not hazards. Invalid reversed ranges are empty for this predicate.
pub fn conflict(first: Access, second: Access) -> bool {
    first.buffer == second.buffer
        && (first.write || second.write)
        && baux::semantics::memory::overlaps(first.start, first.end, second.start, second.end)
}

fn against(active: &[Access], pending: Access) -> bool {
    let mut prior = 0;
    while prior < active.len() {
        if conflict(active[prior], pending) {
            return true;
        }
        prior += 1;
    }
    false
}

/// Whether any new dispatch access conflicts with any access in the open epoch.
/// This does not reject aliases inside one dispatch; those are kernel
/// contracts.
pub fn flush(active: &[Access], pending: &[Access]) -> bool {
    let mut next = 0;
    while next < pending.len() {
        if against(active, pending[next]) {
            return true;
        }
        next += 1;
    }
    false
}

/// Keep the production epoch's separate read/write arrays within 32 entries.
/// Subtraction follows range checks, so even arbitrary u32 inputs cannot wrap.
pub fn capacity(reads: u32, writes: u32, add_reads: u32, add_writes: u32) -> bool {
    if reads > 32 || writes > 32 {
        return true;
    }
    add_reads > 32 - reads || add_writes > 32 - writes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes() {
        for first_write in [false, true] {
            for second_write in [false, true] {
                let first = Access {
                    buffer: 1,
                    start: 0,
                    end: 8,
                    write: first_write,
                };
                let second = Access {
                    buffer: 1,
                    start: 4,
                    end: 12,
                    write: second_write,
                };
                assert_eq!(flush(&[first], &[second]), first_write || second_write);
                assert!(!conflict(
                    first,
                    Access {
                        buffer: 2,
                        ..second
                    }
                ));
                assert!(!conflict(first, Access { start: 8, ..second }));
                assert!(!conflict(
                    first,
                    Access {
                        start: 4,
                        end: 4,
                        ..second
                    }
                ));
            }
        }
    }

    #[test]
    fn scan() {
        let read = Access {
            buffer: 1,
            start: 0,
            end: 8,
            write: false,
        };
        let write = Access {
            write: true,
            ..read
        };
        let other = Access { buffer: 2, ..read };
        assert!(!flush(&[], &[write]));
        assert!(!flush(&[write], &[]));
        assert!(flush(&[other, read], &[other, write]));
        assert!(flush(&[write], &[read]));
        assert!(flush(&[write], &[write]));
        assert!(!flush(&[read, other], &[read, other]));
        assert!(!capacity(32, 32, 0, 0));
        assert!(capacity(32, 0, 1, 0));
        assert!(capacity(0, 32, 0, 1));
        assert!(capacity(u32::MAX, 0, 0, 0));
        assert!(capacity(0, 0, u32::MAX, 0));
    }
}

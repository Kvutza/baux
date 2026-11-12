//! Same ownership contract for each native runtime; handles never mix.
use super::*;
use crate::portable::dl::Library;
use std::ffi::CString;

pub struct Api {
    pub metal: bool,
    pub close: unsafe extern "C" fn(*mut c_void),
    pub retain: unsafe extern "C" fn(*mut c_void),
    pub release: unsafe extern "C" fn(*mut c_void),
    pub put: unsafe extern "C" fn(
        *mut c_void,
        *const usize,
        usize,
        u32,
        *const u8,
        usize,
        *mut *mut c_void,
        *mut c_char,
        usize,
    ) -> c_int,
    pub read:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u8, usize, *mut c_char, usize) -> c_int,
    pub call: unsafe extern "C" fn(
        *mut c_void,
        *const *mut c_void,
        usize,
        *mut *mut c_void,
        usize,
        *mut c_char,
        usize,
    ) -> c_int,
    pub matches: unsafe extern "C" fn(*mut c_void, *const usize, usize, u32) -> c_int,
}

impl Api {
    pub(super) unsafe fn resolve(
        lib: &Library,
        prefix: &str,
        metal: bool,
    ) -> std::result::Result<Self, String> {
        let sym_name =
            |suffix: &str| CString::new(format!("{prefix}_{suffix}")).expect("ASCII symbol name");
        Ok(Self {
            metal,
            close: unsafe { lib.sym(&sym_name("close"))? },
            retain: unsafe { lib.sym(&sym_name("retain"))? },
            release: unsafe { lib.sym(&sym_name("release"))? },
            put: unsafe { lib.sym(&sym_name("put"))? },
            read: unsafe { lib.sym(&sym_name("read"))? },
            call: unsafe { lib.sym(&sym_name("call"))? },
            matches: unsafe { lib.sym(&sym_name("matches"))? },
        })
    }
}

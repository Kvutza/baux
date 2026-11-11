//! Same ownership contract for each native runtime; handles never mix.
use super::*;

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

macro_rules! runtime {
    (
        $api:ident,
        $metal:expr,
        $close:ident,
        $retain:ident,
        $release:ident,
        $put:ident,
        $read:ident,
        $call:ident,
        $matches:ident
    ) => {
        unsafe extern "C" {
            fn $close(context: *mut c_void);
            fn $retain(view: *mut c_void);
            fn $release(view: *mut c_void);
            fn $put(
                context: *mut c_void,
                shape: *const usize,
                rank: usize,
                dtype: u32,
                data: *const u8,
                length: usize,
                output: *mut *mut c_void,
                error: *mut c_char,
                capacity: usize,
            ) -> c_int;
            fn $read(
                context: *mut c_void,
                view: *mut c_void,
                output: *mut u8,
                length: usize,
                error: *mut c_char,
                capacity: usize,
            ) -> c_int;
            fn $call(
                context: *mut c_void,
                inputs: *const *mut c_void,
                count: usize,
                outputs: *mut *mut c_void,
                capacity: usize,
                error: *mut c_char,
                error_capacity: usize,
            ) -> c_int;
            fn $matches(view: *mut c_void, shape: *const usize, rank: usize, dtype: u32) -> c_int;
        }
        pub static $api: Api = Api {
            metal: $metal,
            close: $close,
            retain: $retain,
            release: $release,
            put: $put,
            read: $read,
            call: $call,
            matches: $matches,
        };
    };
}

runtime!(
    IREE,
    true,
    bauxiree_close,
    bauxiree_retain,
    bauxiree_release,
    bauxiree_put,
    bauxiree_read,
    bauxiree_call,
    bauxiree_matches
);
runtime!(
    TVM,
    false,
    bauxtvm_close,
    bauxtvm_retain,
    bauxtvm_release,
    bauxtvm_put,
    bauxtvm_read,
    bauxtvm_call,
    bauxtvm_matches
);

runtime!(
    PJRT,
    false,
    bauxpjrt_close,
    bauxpjrt_retain,
    bauxpjrt_release,
    bauxpjrt_put,
    bauxpjrt_read,
    bauxpjrt_call,
    bauxpjrt_matches
);

//! Adapter for the shared TEPL semantic constraint solver (ABI version 1).
use super::{BuiltinError, BuiltinResult};
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::OnceLock;
pub type List = Vec<u64>;
#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    Found,
    Invalid,
    Unknown,
}
#[derive(Debug, PartialEq, Eq)]
pub enum Method {
    Ordered,
    ConstraintsZ3,
    Z3,
    Unavailable,
}
pub struct Solution {
    pub status: Status,
    pub fields: Vec<List>,
    pub method: Method,
    pub reason: String,
}
#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}
type Solve = unsafe extern "C" fn(*const c_char) -> *const c_char;
fn solver() -> Option<Solve> {
    static ADDRESS: OnceLock<usize> = OnceLock::new();
    let address = *ADDRESS.get_or_init(|| {
        #[cfg(unix)]
        {
            let path = std::env::var("TEPL_SEMANTIC_RUNTIME")
                .unwrap_or_else(|_| "libtepl_semantic_runtime.so".into());
            let Ok(path) = CString::new(path) else {
                return 0;
            };
            // The library remains loaded for the lifetime of the process.
            unsafe {
                let handle = dlopen(path.as_ptr(), 2);
                if handle.is_null() {
                    return 0;
                }
                dlsym(handle, c"tepl_semantic_solve_v1".as_ptr()) as usize
            }
        }
        #[cfg(not(unix))]
        {
            0
        }
    });
    if address == 0 {
        None
    } else {
        Some(unsafe { std::mem::transmute::<usize, Solve>(address) })
    }
}
pub fn solve(plan: &str, shapes: &[List], known: &[List], pinned: &[List]) -> Solution {
    let unknown = || Solution {
        status: Status::Unknown,
        fields: vec![],
        method: Method::Unavailable,
        reason: "semantic runtime unavailable or malformed response".into(),
    };
    let Some(solver) = solver() else {
        return unknown();
    };
    let mut request = plan.to_string();
    for lists in [shapes, known, pinned] {
        request.push_str(&format!("{} ", lists.len()));
        for list in lists {
            request.push_str(&format!("{} ", list.len()));
            for v in list {
                request.push_str(&format!("{v} "));
            }
        }
    }
    let Ok(request) = CString::new(request) else {
        return unknown();
    };
    let response = unsafe { solver(request.as_ptr()) };
    if response.is_null() {
        return unknown();
    }
    let Ok(response) = unsafe { CStr::from_ptr(response) }.to_str() else {
        return unknown();
    };
    let mut tokens = response.split_whitespace();
    let Some(status) = tokens.next() else {
        return unknown();
    };
    let Some(count) = tokens.next().and_then(|x| x.parse::<usize>().ok()) else {
        return unknown();
    };
    let mut fields = Vec::new();
    for _ in 0..count {
        let Some(len) = tokens.next().and_then(|x| x.parse::<usize>().ok()) else {
            return unknown();
        };
        let mut list = Vec::new();
        for _ in 0..len {
            let Some(v) = tokens.next().and_then(|x| x.parse::<u64>().ok()) else {
                return unknown();
            };
            list.push(v);
        }
        fields.push(list);
    }
    let method = match tokens.next() {
        Some("ordered") => Method::Ordered,
        Some("constraints-z3") => Method::ConstraintsZ3,
        Some("z3") => Method::Z3,
        _ => return unknown(),
    };
    let reason = match tokens.next() {
        Some("-") => String::new(),
        None => return unknown(),
        Some(encoded) => {
            if encoded.len() > 65536 || encoded.len() % 2 != 0 {
                return unknown();
            }
            let mut bytes = Vec::new();
            for pair in encoded.as_bytes().chunks_exact(2) {
                let digit = |c: u8| (c as char).to_digit(16);
                let (Some(a), Some(b)) = (digit(pair[0]), digit(pair[1])) else {
                    return unknown();
                };
                bytes.push((a * 16 + b) as u8);
            }
            let Ok(reason) = String::from_utf8(bytes) else {
                return unknown();
            };
            reason
        }
    };
    Solution {
        status: match status {
            "0" => Status::Found,
            "1" => Status::Invalid,
            _ => Status::Unknown,
        },
        fields,
        method,
        reason,
    }
}
pub fn construct(
    plan: &str,
    shapes: &[List],
    known: &[List],
    pinned: &[List],
) -> BuiltinResult<Vec<List>> {
    let solution = solve(plan, shapes, known, pinned);
    match solution.status {
        Status::Found => Ok(solution.fields),
        Status::Invalid => Err(BuiltinError::Assertion(
            "semantic inference invalid at matched metadata",
        )),
        Status::Unknown => Err(BuiltinError::Assertion(
            "semantic inference unknown or runtime unavailable",
        )),
    }
}

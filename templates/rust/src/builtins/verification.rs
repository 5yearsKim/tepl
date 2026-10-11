//! Bounded early rejection under ideal polynomial arithmetic.
//! UNSAT covers all values only at the concrete metadata that was checked.
use super::super::rewriting::AttrVar;
use super::super::{DType, OpAttrs, TensorInfo};
use super::semantic::List;
use egg::Var;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationStatus {
    Counterexample,
    NoCounterexampleFound,
    Unknown,
}
#[derive(Clone, Debug)]
pub struct VerificationOptions {
    pub max_rank: usize,
    pub dimensions: Vec<u64>,
    pub max_configurations: usize,
    pub attributes_per_shape: usize,
    pub timeout_ms: u64,
    pub query_timeout_ms: u32,
    pub work_budget: usize,
    pub max_degree: usize,
    /// Optional concrete shape tuples, in the rule's semantic capture order.
    pub configurations: Vec<Vec<List>>,
    /// Guards execute with this dtype unless a capture declares an exact dtype.
    pub dtype: DType,
}
impl Default for VerificationOptions {
    fn default() -> Self {
        Self {
            max_rank: 2,
            dimensions: vec![2, 1, 0, 3],
            max_configurations: 64,
            attributes_per_shape: 8,
            timeout_ms: 2000,
            query_timeout_ms: 100,
            work_budget: 100000,
            max_degree: 8,
            configurations: vec![],
            dtype: DType::F32,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseEntry {
    pub capture: usize,
    pub coordinates: List,
    pub value: u64,
}
#[derive(Clone, Debug)]
pub struct Counterexample {
    pub shapes: Vec<List>,
    /// Concrete fields used by both sides, including host-derived fields.
    pub attributes: Vec<List>,
    pub degree: usize,
    pub output: List,
    pub entries: Vec<SparseEntry>,
    pub lhs_value: String,
    pub rhs_value: String,
    pub reason: String,
}
#[derive(Clone, Debug)]
pub struct VerificationReport {
    pub status: VerificationStatus,
    pub checked: usize,
    pub skipped: usize,
    pub unknown: usize,
    pub reason: String,
    pub counterexample: Option<Counterexample>,
}
impl VerificationReport {
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self {
            status: VerificationStatus::Unknown,
            checked: 0,
            skipped: 0,
            unknown: 1,
            reason: reason.into(),
            counterexample: None,
        }
    }
    /// Suitable for build/tests: reject confirmed failures; retain coverage in the report.
    pub fn reject_counterexample(&self) -> Result<(), String> {
        if let Some(w) = &self.counterexample {
            return Err(format!(
                "{}; shapes={:?}; attributes={:?}; sparse entries={:?}; output={:?}; LHS={}; RHS={}",
                w.reason, w.shapes, w.attributes, w.entries, w.output, w.lhs_value, w.rhs_value
            ));
        }
        Ok(())
    }
}
#[doc(hidden)]
pub struct Sample {
    pub shapes: Vec<List>,
    pub known: Vec<List>,
    pub node_shapes: Vec<List>,
}
#[doc(hidden)]
pub enum MetadataCheck {
    Valid,
    Invalid(String),
    Unknown(String),
}
#[doc(hidden)]
pub struct Prepared {
    pub known: Vec<List>,
    pub pinned: Vec<List>,
    pub metadata: MetadataCheck,
}
/// Metadata-only context for generated guards and derivations. No e-graph is needed.
#[doc(hidden)]
#[derive(Default)]
pub struct ValidationContext {
    pub tensors: HashMap<Var, TensorInfo>,
    pub attributes: HashMap<AttrVar, OpAttrs>,
}
impl ValidationContext {
    pub fn tensor(&self, var: Var) -> Option<TensorInfo> {
        self.tensors.get(&var).cloned()
    }
    pub fn attrs(&self, var: AttrVar) -> Option<&OpAttrs> {
        self.attributes.get(&var)
    }
}
#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}
type Verify = unsafe extern "C" fn(*const c_char) -> *const c_char;
fn runtime() -> Option<Verify> {
    static ADDRESS: OnceLock<usize> = OnceLock::new();
    let address = *ADDRESS.get_or_init(|| {
        #[cfg(unix)]
        {
            let path = std::env::var("TEPL_SEMANTIC_RUNTIME")
                .unwrap_or_else(|_| "libtepl_semantic_runtime.so".into());
            let Ok(path) = CString::new(path) else {
                return 0;
            };
            unsafe {
                let handle = dlopen(path.as_ptr(), 2);
                if handle.is_null() {
                    return 0;
                }
                dlsym(handle, c"tepl_semantic_verify_v1".as_ptr()) as usize
            }
        }
        #[cfg(not(unix))]
        {
            0
        }
    });
    (address != 0).then(|| unsafe { std::mem::transmute::<usize, Verify>(address) })
}
fn invoke(request: String) -> Option<String> {
    let runtime = runtime()?;
    let request = CString::new(request).ok()?;
    let response = unsafe { runtime(request.as_ptr()) };
    if response.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(response) }
            .to_str()
            .ok()?
            .to_owned(),
    )
}
fn write_lists(request: &mut String, lists: &[List]) {
    request.push_str(&format!("{} ", lists.len()));
    for list in lists {
        request.push_str(&format!("{} ", list.len()));
        for x in list {
            request.push_str(&format!("{x} "));
        }
    }
}
struct Reader<'a>(std::str::SplitWhitespace<'a>);
impl<'a> Reader<'a> {
    fn number(&mut self) -> Option<usize> {
        self.0.next()?.parse().ok()
    }
    fn list(&mut self) -> Option<List> {
        let n = self.number()?;
        if n > 4096 {
            return None;
        }
        (0..n).map(|_| self.0.next()?.parse().ok()).collect()
    }
    fn lists(&mut self) -> Option<Vec<List>> {
        let n = self.number()?;
        if n > 10000 {
            return None;
        }
        (0..n).map(|_| self.list()).collect()
    }
    fn text(&mut self) -> Option<String> {
        let encoded = self.0.next()?;
        if encoded == "-" {
            return Some(String::new());
        }
        if encoded.len() > 65536 || encoded.len() % 2 != 0 {
            return None;
        }
        let bytes: Option<Vec<_>> = encoded
            .as_bytes()
            .chunks_exact(2)
            .map(|p| Some(((p[0] as char).to_digit(16)? * 16 + (p[1] as char).to_digit(16)?) as u8))
            .collect();
        String::from_utf8(bytes?).ok()
    }
}
fn prefix(operation: &str, plan: &str, options: &VerificationOptions, left: u32) -> String {
    format!(
        "{operation} {} {} {} {plan}",
        left.min(options.query_timeout_ms),
        options.work_budget,
        options.max_degree
    )
}
fn shapes(captures: usize, options: &VerificationOptions) -> Vec<Vec<List>> {
    if !options.configurations.is_empty() {
        return options
            .configurations
            .iter()
            .take(options.max_configurations)
            .cloned()
            .collect();
    }
    let mut candidates = vec![vec![]];
    for rank in 1..=options.max_rank.min(8) {
        for &extent in &options.dimensions {
            candidates.push(vec![extent; rank]);
        }
        if rank > 1 {
            for pair in options.dimensions.windows(2) {
                let mut dims = vec![pair[0]; rank];
                dims[rank - 1] = pair[1];
                candidates.push(dims);
            }
        }
    }
    let mut tuples = Vec::new();
    fn add(tuples: &mut Vec<Vec<List>>, tuple: Vec<List>, limit: usize) {
        if tuples.len() < limit && !tuples.contains(&tuple) {
            tuples.push(tuple);
        }
    }
    // Cover all small ranks before spending the budget on a Cartesian product.
    for shape in &candidates {
        add(
            &mut tuples,
            vec![shape.clone(); captures],
            options.max_configurations,
        );
        if tuples.len() >= options.max_configurations {
            return tuples;
        }
    }
    for base in &candidates {
        for capture in 0..captures {
            for shape in &candidates {
                let mut tuple = vec![base.clone(); captures];
                tuple[capture] = shape.clone();
                add(&mut tuples, tuple, options.max_configurations);
                if tuples.len() >= options.max_configurations {
                    return tuples;
                }
            }
        }
    }
    tuples
}
#[doc(hidden)]
pub fn run(
    plan: &str,
    captures: usize,
    options: &VerificationOptions,
    mut prepare: impl FnMut(&Sample) -> Option<Prepared>,
) -> VerificationReport {
    if options.max_configurations == 0
        || options.attributes_per_shape == 0
        || options.query_timeout_ms == 0
        || options.query_timeout_ms > 60000
        || options.work_budget == 0
        || options.work_budget > 10000000
        || options.max_degree > 64
        || options.max_configurations > 10000
        || options.attributes_per_shape > 10000
        || (options.configurations.is_empty()
            && (options.max_rank > 8 || options.dimensions.len() > 16))
    {
        return VerificationReport::unknown("invalid or empty verification budget");
    }
    let deadline = Instant::now() + Duration::from_millis(options.timeout_ms.min(60000));
    let remaining = || {
        deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .min(60000) as u32
    };
    let mut report = VerificationReport {
        status: VerificationStatus::Unknown,
        checked: 0,
        skipped: 0,
        unknown: 0,
        reason: String::new(),
        counterexample: None,
    };
    for tuple in shapes(captures, options) {
        if tuple.len() != captures {
            report.unknown += 1;
            report.reason = "configuration capture count mismatch".into();
            continue;
        }
        let mut blocked: Vec<Vec<List>> = Vec::new();
        for _ in 0..options.attributes_per_shape {
            if remaining() == 0 {
                report.unknown += 1;
                report.reason = "verification time budget exhausted".into();
                break;
            }
            let mut request = prefix("sample-v1", plan, options, remaining());
            write_lists(&mut request, &tuple);
            request.push_str(&format!("{} ", blocked.len()));
            for assignment in &blocked {
                write_lists(&mut request, assignment);
            }
            let Some(response) = invoke(request) else {
                return VerificationReport::unknown("semantic verification runtime unavailable");
            };
            let mut reader = Reader(response.split_whitespace());
            let parsed = (|| {
                Some((
                    reader.number()?,
                    reader.lists()?,
                    reader.lists()?,
                    reader.text()?,
                ))
            })();
            let Some((status, known, node_shapes, reason)) = parsed else {
                return VerificationReport::unknown("malformed source-sample response");
            };
            if status == 1 {
                break;
            }
            if status != 0 {
                report.unknown += 1;
                report.reason = reason;
                break;
            }
            blocked.push(known.clone());
            let sample = Sample {
                shapes: tuple.clone(),
                known,
                node_shapes,
            };
            let Some(prepared) = prepare(&sample) else {
                report.skipped += 1;
                continue;
            };
            match &prepared.metadata {
                MetadataCheck::Invalid(reason) => {
                    report.status = VerificationStatus::Counterexample;
                    report.reason = reason.clone();
                    report.counterexample = Some(Counterexample {
                        shapes: tuple.clone(),
                        attributes: prepared.known,
                        degree: 0,
                        output: vec![],
                        entries: vec![],
                        lhs_value: String::new(),
                        rhs_value: String::new(),
                        reason: reason.clone(),
                    });
                    return report;
                }
                MetadataCheck::Unknown(reason) => {
                    report.unknown += 1;
                    report.reason = reason.clone();
                    continue;
                }
                MetadataCheck::Valid => {}
            }
            if remaining() == 0 {
                report.unknown += 1;
                report.reason = "verification time budget exhausted".into();
                break;
            }
            let mut request = prefix("sparse-v1", plan, options, remaining());
            write_lists(&mut request, &tuple);
            write_lists(&mut request, &prepared.known);
            write_lists(&mut request, &prepared.pinned);
            let Some(response) = invoke(request) else {
                return VerificationReport::unknown("semantic verification runtime unavailable");
            };
            let mut reader = Reader(response.split_whitespace());
            let parsed = (|| {
                let status = reader.number()?;
                let degree = reader.number()?;
                let output = reader.list()?;
                let count = reader.number()?;
                if count > 64 {
                    return None;
                }
                let mut entries = Vec::new();
                for _ in 0..count {
                    entries.push(SparseEntry {
                        capture: reader.number()?,
                        value: reader.0.next()?.parse().ok()?,
                        coordinates: reader.list()?,
                    });
                }
                Some((
                    status,
                    degree,
                    output,
                    entries,
                    reader.text()?,
                    reader.text()?,
                    reader.text()?,
                ))
            })();
            let Some((status, degree, output, entries, lhs_value, rhs_value, reason)) = parsed
            else {
                return VerificationReport::unknown("malformed sparse-verification response");
            };
            match status {
                0 => {
                    report.status = VerificationStatus::Counterexample;
                    report.reason = reason.clone();
                    report.counterexample = Some(Counterexample {
                        shapes: tuple.clone(),
                        attributes: prepared.known,
                        degree,
                        output,
                        entries,
                        lhs_value,
                        rhs_value,
                        reason,
                    });
                    return report;
                }
                1 => report.checked += 1,
                2 => report.skipped += 1,
                _ => {
                    report.unknown += 1;
                    report.reason = reason;
                }
            }
        }
        if remaining() == 0 {
            break;
        }
    }
    if report.checked > 0 && report.unknown == 0 {
        report.status = VerificationStatus::NoCounterexampleFound;
        report.reason = format!(
            "{} concrete configurations verified for all values; higher ranks and untested metadata remain unverified",
            report.checked
        );
    } else if report.reason.is_empty() {
        report.reason = "no applicable configurations were verified".into();
    }
    report
}

use std::sync::atomic::{AtomicUsize, Ordering};
use tepl_generated::ir::builtins::verification::{VerificationOptions, VerificationStatus};
use tepl_generated::ir::dialects::tensor_lang::OpAttrs as TensorAttrs;
use tepl_generated::ir::rules::semantic_early::*;
use tepl_generated::ir::{OpAttrs, TensorInfo};
struct Host {
    enabled: bool,
    fail: bool,
    calls: AtomicUsize,
}
impl rule_transpose_host::HostFunctions for Host {
    fn eligible(&self, x: &TensorInfo) -> Option<bool> {
        Some(self.enabled && x.shape.len() == 2)
    }
    fn copy_attrs(&self, attrs: &OpAttrs) -> Option<OpAttrs> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return None;
        }
        Some(attrs.clone())
    }
}
struct WrongHost;
impl rule_transpose_host::HostFunctions for WrongHost {
    fn eligible(&self, x: &TensorInfo) -> Option<bool> {
        Some(x.shape.len() == 2)
    }
    fn copy_attrs(&self, attrs: &OpAttrs) -> Option<OpAttrs> {
        let OpAttrs::TensorLang(TensorAttrs::TransposeAttrs { permutation }) = attrs else {
            return None;
        };
        // First call changes the source permutation; second call keeps it.
        // This is intentionally incorrect for one of the sampled source choices.
        let _ = permutation;
        Some(OpAttrs::TensorLang(TensorAttrs::TransposeAttrs {
            permutation: vec![0, 1],
        }))
    }
}
struct DotHost;
impl rule_distribution_host::HostFunctions for DotHost {
    fn copy_dot(&self, attrs: &OpAttrs) -> Option<OpAttrs> {
        Some(attrs.clone())
    }
}
struct Disabled;
impl rule_guarded_false::HostFunctions for Disabled {
    fn enabled(&self, _: &TensorInfo) -> Option<bool> {
        Some(false)
    }
}
fn options(configurations: Vec<Vec<Vec<u64>>>) -> VerificationOptions {
    VerificationOptions {
        configurations,
        query_timeout_ms: 1000,
        timeout_ms: 10000,
        ..Default::default()
    }
}
#[test]
fn real_host_derivations_and_sparse_witness() {
    let config = options(vec![vec![vec![2, 2]]]);
    let host = Host {
        enabled: true,
        fail: false,
        calls: AtomicUsize::new(0),
    };
    let valid = rule_transpose_host::verify_small(&host, &config);
    assert_eq!(
        valid.status,
        VerificationStatus::NoCounterexampleFound,
        "{}",
        valid.reason
    );
    assert_eq!(valid.checked, 2);
    assert_eq!(host.calls.load(Ordering::SeqCst), 4);
    let invalid = rule_transpose_host::verify_small(&WrongHost, &config);
    assert_eq!(
        invalid.status,
        VerificationStatus::Counterexample,
        "{}",
        invalid.reason
    );
    let witness = invalid.counterexample.as_ref().unwrap();
    assert_eq!(witness.degree, 1);
    assert_eq!(witness.entries.len(), 1);
    assert_ne!(witness.lhs_value, witness.rhs_value);
    assert!(
        invalid
            .reject_counterexample()
            .unwrap_err()
            .contains("sparse entries")
    );
}
#[test]
fn host_guards_and_fallible_derivations_skip() {
    for (enabled, fail) in [(false, false), (true, true)] {
        let host = Host {
            enabled,
            fail,
            calls: AtomicUsize::new(0),
        };
        let report = rule_transpose_host::verify_small(&host, &options(vec![vec![vec![2, 2]]]));
        assert_eq!(report.status, VerificationStatus::Unknown);
        assert_eq!(report.checked, 0);
        assert_eq!(report.skipped, 2);
        assert_eq!(
            host.calls.load(Ordering::SeqCst),
            if enabled { 2 } else { 0 }
        );
    }
    let report =
        rule_guarded_false::verify_small(&Disabled, &options(vec![vec![vec![2], vec![2]]]));
    assert_eq!(report.checked, 0);
    assert!(report.counterexample.is_none());
}
#[test]
fn distribution_and_shape_declarations() {
    let valid = rule_distribution_host::verify_small(
        &DotHost,
        &options(vec![vec![vec![2, 2], vec![2, 2], vec![2, 2]]]),
    );
    assert_eq!(
        valid.status,
        VerificationStatus::NoCounterexampleFound,
        "{}",
        valid.reason
    );
    assert!(valid.checked > 0);
    let config = options(vec![vec![vec![2], vec![2]], vec![vec![2, 2], vec![2, 2]]]);
    let invalid = rule_dropped_operand::verify_small(&(), &config);
    assert_eq!(
        invalid.status,
        VerificationStatus::Counterexample,
        "{}",
        invalid.reason
    );
    assert_eq!(invalid.skipped, 1);
    assert_eq!(invalid.counterexample.unwrap().entries.len(), 1);
}
#[test]
fn automatic_and_empty_budgets_are_unknown() {
    let config = VerificationOptions::default();
    assert_eq!(
        rule_automatic_unsupported::verify_small(&(), &config).status,
        VerificationStatus::Unknown
    );
    let empty = VerificationOptions {
        max_configurations: 0,
        ..config
    };
    assert_eq!(
        rule_dropped_operand::verify_small(&(), &empty).status,
        VerificationStatus::Unknown
    );
}

impl rule_bound_intermediate_host::HostFunctions for Host {
    fn eligible(&self, x: &TensorInfo) -> Option<bool> {
        Some(self.enabled && x.shape.len() == 2)
    }
    fn copy_attrs(&self, attrs: &OpAttrs) -> Option<OpAttrs> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Some(attrs.clone())
    }
}
#[test]
fn source_dtype_legality_and_bound_metadata() {
    let config = options(vec![vec![vec![2, 2], vec![2, 2]]]);
    let unreachable = rule_unreachable_dtype::verify_small(&(), &config);
    assert_eq!(unreachable.status, VerificationStatus::Unknown);
    assert_eq!(unreachable.checked, 0);
    assert_eq!(unreachable.skipped, 1);
    let mixed = rule_unreachable_mixed_dtype::verify_small(&(), &config);
    assert_eq!(mixed.status, VerificationStatus::Unknown);
    assert_eq!(mixed.checked, 0);
    assert_eq!(mixed.skipped, 1);
    let wrong_dtype = rule_dtype_mismatch::verify_small(&(), &config);
    assert_eq!(wrong_dtype.status, VerificationStatus::Counterexample);
    assert!(wrong_dtype.reason.contains("metadata differs"));
    assert!(wrong_dtype.counterexample.unwrap().entries.is_empty());
    let host = Host {
        enabled: true,
        fail: false,
        calls: AtomicUsize::new(0),
    };
    let bound = rule_bound_intermediate_host::verify_small(&host, &options(vec![vec![vec![2, 2]]]));
    assert_eq!(
        bound.status,
        VerificationStatus::NoCounterexampleFound,
        "{}",
        bound.reason
    );
    assert_eq!(bound.checked, 2);
    assert_eq!(host.calls.load(Ordering::SeqCst), 2);
}

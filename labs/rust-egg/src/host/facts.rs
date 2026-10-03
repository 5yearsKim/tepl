use egg::DidMerge;

use crate::ir::analysis::Inference;
use crate::ir::pattern::TensorInfo;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Observed {
    None,
    One(TensorInfo),
    Conflict,
}

/// Facts about every alternative in an e-class, including shape and dtype.
/// Unknown alternatives block metadata access without erasing known evidence.
/// Evidence only accumulates: supplying missing host metadata later requires a
/// fresh graph. This deliberately conservative sample does not retract facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorFacts {
    // Remember known metadata even when another alternative is unknown.
    // This lets a later disagreement still be detected.
    observed: Observed,
    // At least one alternative could not be inferred.
    has_unknown: bool,
    // At least one alternative failed inference.
    has_invalid: bool,
}

impl TensorFacts {
    pub fn from_inference(result: Inference<TensorInfo>) -> Self {
        match result {
            Inference::Known(info) => Self {
                observed: Observed::One(info),
                has_unknown: false,
                has_invalid: false,
            },
            Inference::Unknown => Self {
                observed: Observed::None,
                has_unknown: true,
                has_invalid: false,
            },
            Inference::Invalid(_) => Self {
                observed: Observed::None,
                has_unknown: false,
                has_invalid: true,
            },
        }
    }

    pub fn info(&self) -> Option<&TensorInfo> {
        if self.has_unknown || self.is_invalid() {
            return None;
        }
        match &self.observed {
            Observed::One(info) => Some(info),
            _ => None,
        }
    }

    pub fn is_invalid(&self) -> bool {
        self.has_invalid || self.observed == Observed::Conflict
    }

    pub fn is_unknown(&self) -> bool {
        !self.is_invalid() && self.info().is_none()
    }

    /// Associative, commutative, idempotent join; also reusable by custom analyses.
    pub fn merge(&mut self, incoming: Self) -> DidMerge {
        // Combine known evidence: equal metadata agrees, different metadata conflicts.
        let observed = match (&self.observed, &incoming.observed) {
            (Observed::Conflict, _) | (_, Observed::Conflict) => Observed::Conflict,
            (Observed::None, other) | (other, Observed::None) => other.clone(),
            (Observed::One(a), Observed::One(b)) if a == b => Observed::One(a.clone()),
            (Observed::One(_), Observed::One(_)) => Observed::Conflict,
        };
        let merged = Self {
            observed,
            has_unknown: self.has_unknown || incoming.has_unknown,
            has_invalid: self.has_invalid || incoming.has_invalid,
        };
        // egg needs to know whether either side gained information.
        let changed = DidMerge(*self != merged, incoming != merged);
        *self = merged;
        changed
    }
}

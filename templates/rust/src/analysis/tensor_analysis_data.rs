use ::egg::DidMerge;

use super::Inference;
use super::TensorInfo;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Observed {
    None,
    One(TensorInfo),
    Conflict,
}

/// Facts about every alternative in an e-class, including shape and dtype.
/// Unknown alternatives block metadata access without erasing known evidence.
/// Evidence accumulates; new host metadata requires a fresh graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorAnalysisData {
    // Retained across unknown alternatives to detect later conflicts.
    observed: Observed,
    has_unknown: bool,
    has_invalid: bool,
}

impl TensorAnalysisData {
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

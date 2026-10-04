//! Errors shared by TEPL expression evaluators.
use ::std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltinError {
    Assertion(&'static str),
    Overflow,
    IndexOutOfBounds,
    InvalidRange,
    NonPositiveDivisor,
    IncompatibleBroadcast,
    CapacityExceeded,
    DivisionByZero,
    InvalidDimension,
}

impl BuiltinError {
    // Keep the original shape diagnostics stable; rules propagate failure as None.
    pub fn message(self) -> &'static str {
        match self {
            Self::Assertion(message) => message,
            Self::Overflow => "shape arithmetic overflow",
            Self::IndexOutOfBounds => "shape index out of bounds",
            Self::InvalidRange => "range bound must be nonnegative and fit usize",
            Self::NonPositiveDivisor => "shape division requires a positive divisor",
            Self::IncompatibleBroadcast => "incompatible broadcast dimensions",
            Self::CapacityExceeded => "shape list capacity exceeded",
            Self::DivisionByZero => "shape division by zero",
            Self::InvalidDimension => "output dimension must be nonnegative and fit u64",
        }
    }
}

impl fmt::Display for BuiltinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl ::std::error::Error for BuiltinError {}

pub type BuiltinResult<T> = Result<T, BuiltinError>;

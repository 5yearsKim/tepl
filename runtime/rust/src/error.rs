use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalError {
    Unavailable(&'static str),
    Invalid(String),
    Conflict(String),
    Limit(&'static str),
}
pub type EvalResult<T> = Result<T, EvalError>;
impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(s) | Self::Limit(s) => f.write_str(s),
            Self::Invalid(s) | Self::Conflict(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for EvalError {}
pub fn require(ok: bool, message: &str) -> EvalResult<()> {
    if ok {
        Ok(())
    } else {
        Err(EvalError::Invalid(message.into()))
    }
}

/// Missing inference is different from a failed assertion or invalid metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Inference<T> {
    Known(T),
    Unknown,
    Invalid(&'static str),
}

impl<T> Inference<T> {
    pub fn into_option(self) -> Option<T> {
        match self {
            Self::Known(value) => Some(value),
            Self::Unknown | Self::Invalid(_) => None,
        }
    }
}

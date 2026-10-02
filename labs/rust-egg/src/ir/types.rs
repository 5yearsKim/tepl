//! Tensor element formats. Storage type does not determine numerical policy.

use std::{fmt, str::FromStr};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    pub const ALL: [Self; 13] = [
        Self::Bool,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::F16,
        Self::BF16,
        Self::F32,
        Self::F64,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// Integer and boolean literals must be in range. Floating spellings are
    /// preserved; format rounding and representability are host responsibilities.
    pub(crate) fn accepts_literal(self, value: &str) -> bool {
        if matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64) {
            return true;
        }
        let negative = value.starts_with('-');
        let digits = value.strip_prefix(['+', '-']).unwrap_or(value);
        let digits = digits.trim_start_matches('0');
        let digits = if digits.is_empty() { "0" } else { digits };
        let (unsigned, limit) = match self {
            Self::Bool => (true, "1"),
            Self::I8 => (false, if negative { "128" } else { "127" }),
            Self::I16 => (false, if negative { "32768" } else { "32767" }),
            Self::I32 => (false, if negative { "2147483648" } else { "2147483647" }),
            Self::I64 => (
                false,
                if negative {
                    "9223372036854775808"
                } else {
                    "9223372036854775807"
                },
            ),
            Self::U8 => (true, "255"),
            Self::U16 => (true, "65535"),
            Self::U32 => (true, "4294967295"),
            Self::U64 => (true, "18446744073709551615"),
            _ => unreachable!(),
        };
        !digits.contains('.')
            && !(unsigned && negative)
            && (digits.len() < limit.len() || (digits.len() == limit.len() && digits <= limit))
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for DType {
    type Err = String;
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|dtype| dtype.name() == name)
            .ok_or_else(|| format!("unknown dtype '{name}'"))
    }
}

use crate::error::{EvalError, EvalResult, require};

#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    pub owner: usize,
    pub group: usize,
    pub member: usize,
    pub extent: i128,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Integer(i128),
    Float(f64),
    Boolean(bool),
    String(String),
    List(Vec<Value>),
    Axis(Axis),
    Tensor(egg::Id),
    Descriptor { schema: usize, fields: Vec<Value> },
    Opaque(crate::types::OpaqueValue),
}
impl Value {
    pub fn integer(&self) -> EvalResult<i128> {
        match self {
            Self::Integer(x) => Ok(*x),
            _ => Err(invalid("expected integer")),
        }
    }
    pub fn boolean(&self) -> EvalResult<bool> {
        match self {
            Self::Boolean(x) => Ok(*x),
            _ => Err(invalid("expected bool")),
        }
    }
    pub fn list(&self) -> EvalResult<&[Value]> {
        match self {
            Self::List(x) => Ok(x),
            _ => Err(invalid("expected list")),
        }
    }
    pub fn string(&self) -> EvalResult<&str> {
        match self {
            Self::String(x) => Ok(x),
            _ => Err(invalid("expected string")),
        }
    }
    pub fn position(&self) -> EvalResult<usize> {
        usize::try_from(self.integer()?).map_err(|_| invalid("invalid index"))
    }
    pub fn field(&self, field: usize) -> EvalResult<Value> {
        match self {
            Self::Descriptor { fields, .. } => fields
                .get(field)
                .cloned()
                .ok_or_else(|| invalid("missing attribute field")),
            _ => Err(invalid("expected attributes")),
        }
    }
    pub fn dimensions(&self) -> EvalResult<Vec<u64>> {
        self.list()?
            .iter()
            .map(|x| u64::try_from(x.integer()?).map_err(|_| invalid("index overflow")))
            .collect()
    }
}
pub fn invalid(message: &str) -> EvalError {
    EvalError::Invalid(message.into())
}
pub fn integer(text: &str) -> EvalResult<Value> {
    text.parse::<i128>()
        .map(Value::Integer)
        .map_err(|_| invalid("integer overflow or invalid integer"))
}
pub fn floating(text: &str) -> EvalResult<Value> {
    let n = text.parse::<f64>().map_err(|_| invalid("invalid float"))?;
    require(n.is_finite(), "nonfinite floating value")?;
    Ok(Value::Float(n))
}
pub fn shape(xs: &[u64]) -> Value {
    Value::List(xs.iter().map(|x| Value::Integer((*x).into())).collect())
}
#[derive(Clone, Debug)]
pub struct Budget {
    pub steps: usize,
    pub max_list: usize,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            steps: 1_000_000,
            max_list: 1_000_000,
        }
    }
}
impl Budget {
    pub fn tick(&mut self) -> EvalResult<()> {
        self.charge(1)
    }
    pub fn charge(&mut self, n: usize) -> EvalResult<()> {
        self.steps = self
            .steps
            .checked_sub(n)
            .ok_or(EvalError::Limit("expression step limit"))?;
        Ok(())
    }
    pub fn size(&self, n: usize) -> EvalResult<()> {
        if n > self.max_list {
            Err(EvalError::Limit("list size limit"))
        } else {
            Ok(())
        }
    }
    pub fn arguments(&mut self, args: &[Value]) -> EvalResult<()> {
        self.tick()?;
        for a in args {
            if let Value::List(xs) = a {
                self.size(xs.len())?;
                self.charge(xs.len())?;
            }
        }
        Ok(())
    }
}

// Rust schema types convert only at the generated/runtime boundary.
pub trait AttributeValue: Sized {
    fn to_value(&self) -> Value;
    fn from_value(value: &Value) -> EvalResult<Self>;
}
impl AttributeValue for u64 {
    fn to_value(&self) -> Value {
        Value::Integer((*self).into())
    }
    fn from_value(v: &Value) -> EvalResult<Self> {
        Self::try_from(v.integer()?).map_err(|_| invalid("index overflow"))
    }
}
impl AttributeValue for i64 {
    fn to_value(&self) -> Value {
        Value::Integer((*self).into())
    }
    fn from_value(v: &Value) -> EvalResult<Self> {
        Self::try_from(v.integer()?).map_err(|_| invalid("i64 overflow"))
    }
}
impl AttributeValue for bool {
    fn to_value(&self) -> Value {
        Value::Boolean(*self)
    }
    fn from_value(v: &Value) -> EvalResult<Self> {
        v.boolean()
    }
}
impl AttributeValue for String {
    fn to_value(&self) -> Value {
        Value::String(self.clone())
    }
    fn from_value(v: &Value) -> EvalResult<Self> {
        Ok(v.string()?.into())
    }
}
impl<T: AttributeValue> AttributeValue for Vec<T> {
    fn to_value(&self) -> Value {
        Value::List(self.iter().map(T::to_value).collect())
    }
    fn from_value(v: &Value) -> EvalResult<Self> {
        v.list()?.iter().map(T::from_value).collect()
    }
}
impl<T: AttributeValue> AttributeValue for Option<T> {
    fn to_value(&self) -> Value {
        self.as_ref().map_or(Value::None, T::to_value)
    }
    fn from_value(v: &Value) -> EvalResult<Self> {
        if *v == Value::None {
            Ok(None)
        } else {
            T::from_value(v).map(Some)
        }
    }
}

#[derive(Clone, Copy)]
pub enum ScalarType {
    Tensor,
    Index,
    I64,
    F64,
    Bool,
    DType,
    String,
    Precision,
    Opaque,
    Descriptor(usize),
}
pub fn checked(
    value: Value,
    scalar: ScalarType,
    depth: usize,
    optional: bool,
) -> EvalResult<Value> {
    if optional && value == Value::None {
        return Ok(value);
    }
    if depth > 0 {
        for x in value.list()? {
            checked(x.clone(), scalar, depth - 1, false)?;
        }
        return Ok(value);
    }
    match scalar {
        ScalarType::Index => {
            u64::from_value(&value)?;
        }
        ScalarType::I64 => {
            i64::from_value(&value)?;
        }
        ScalarType::Bool => {
            value.boolean()?;
        }
        ScalarType::F64 => require(
            matches!(value, Value::Float(x) if x.is_finite()),
            "expected finite f64",
        )?,
        ScalarType::String => {
            value.string()?;
        }
        ScalarType::DType => {
            crate::types::DType::from_value(&value)?;
        }
        ScalarType::Precision => {
            crate::types::Precision::from_value(&value)?;
        }
        ScalarType::Tensor => require(matches!(value, Value::Tensor(_)), "expected tensor")?,
        ScalarType::Descriptor(id) => require(
            matches!(value, Value::Descriptor { schema, .. } if schema == id),
            "descriptor schema mismatch",
        )?,
        ScalarType::Opaque => require(matches!(value, Value::Opaque(_)), "expected opaque value")?,
    }
    Ok(value)
}

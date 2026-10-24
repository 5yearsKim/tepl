//! Implementations of the source-language builtin catalog.
use crate::error::{EvalResult, require};
use crate::values::{Budget, Value, invalid};

pub fn arithmetic(op: &str, a: i128, b: i128) -> EvalResult<i128> {
    match op {
        "+" => a.checked_add(b),
        "-" => a.checked_sub(b),
        "*" => a.checked_mul(b),
        "/" => a.checked_div(b),
        "%" => a.checked_rem(b),
        _ => None,
    }
    .ok_or_else(|| {
        invalid(if b == 0 && matches!(op, "/" | "%") {
            "division by zero"
        } else {
            "integer overflow"
        })
    })
}
pub fn contains(xs: &[Value], x: &Value, b: &mut Budget) -> EvalResult<bool> {
    b.charge(xs.len())?;
    Ok(xs.contains(x))
}
pub fn distinct(xs: &[Value], b: &mut Budget) -> EvalResult<bool> {
    for (i, x) in xs.iter().enumerate() {
        if contains(&xs[..i], x, b)? {
            return Ok(false);
        }
    }
    Ok(true)
}
pub fn valid(xs: &[Value], rank: i128, b: &mut Budget) -> EvalResult<bool> {
    if u64::try_from(rank).is_err() || !distinct(xs, b)? {
        return Ok(false);
    }
    for x in xs {
        let n = x.integer()?;
        if n < 0 || n >= rank {
            return Ok(false);
        }
    }
    Ok(true)
}
pub fn index(value: Value, at: Value, b: &mut Budget) -> EvalResult<Value> {
    b.arguments(&[value.clone(), at.clone()])?;
    value
        .list()?
        .get(at.position()?)
        .cloned()
        .ok_or_else(|| invalid("index out of bounds"))
}
pub fn operator(op: &str, a: Vec<Value>, b: &mut Budget) -> EvalResult<Value> {
    b.arguments(&a)?;
    if matches!(op, "==" | "!=") {
        require(a.len() == 2, "invalid operator arity")?;
        return Ok(Value::Boolean((a[0] == a[1]) == (op == "==")));
    }
    if op == "!" {
        require(a.len() == 1, "invalid operator arity")?;
        return Ok(Value::Boolean(!a[0].boolean()?));
    }
    require(
        a.len() == 2 || (a.len() == 1 && matches!(op, "+" | "-")),
        "invalid operator arity",
    )?;
    if let Value::Float(x) = a[0] {
        let y = if a.len() == 1 {
            0.0
        } else {
            match a[1] {
                Value::Float(y) => y,
                _ => return Err(invalid("expected float")),
            }
        };
        let comparison = match op {
            "<" => Some(x < y),
            "<=" => Some(x <= y),
            ">" => Some(x > y),
            ">=" => Some(x >= y),
            _ => None,
        };
        if let Some(c) = comparison {
            return Ok(Value::Boolean(c));
        }
        let n = if a.len() == 1 {
            if op == "-" { -x } else { x }
        } else {
            match op {
                "+" => x + y,
                "-" => x - y,
                "*" => x * y,
                "/" => x / y,
                _ => return Err(invalid("unsupported floating operator")),
            }
        };
        require(n.is_finite(), "nonfinite floating value")?;
        return Ok(Value::Float(n));
    }
    let x = a[0].integer()?;
    if a.len() == 1 {
        return Ok(Value::Integer(if op == "-" {
            arithmetic("-", 0, x)?
        } else {
            x
        }));
    }
    let y = a[1].integer()?;
    Ok(match op {
        "<" => Value::Boolean(x < y),
        "<=" => Value::Boolean(x <= y),
        ">" => Value::Boolean(x > y),
        ">=" => Value::Boolean(x >= y),
        _ => Value::Integer(arithmetic(op, x, y)?),
    })
}

pub fn call(name: &str, a: Vec<Value>, b: &mut Budget) -> EvalResult<Value> {
    let arity = match name {
        "concat" => None,
        "gather" | "exclude" | "contains" | "is_disjoint" | "is_valid_axis_list"
        | "broadcast_shape" | "min" | "max" | "floor_div" | "ceil_div" | "filter"
        | "difference" | "subsequence" => Some(2),
        "slice" | "replace" => Some(3),
        _ => Some(1),
    };
    require(
        arity.map_or(a.len() >= 2, |n| n == a.len()),
        "invalid builtin arity",
    )?;
    b.arguments(&a)?;
    match name {
        "len" => Ok(Value::Integer(a[0].list()?.len() as i128)),
        "range" => {
            let n = a[0].position()?;
            b.size(n)?;
            Ok(Value::List(
                (0..n).map(|i| Value::Integer(i as i128)).collect(),
            ))
        }
        "concat" => concat(&a, b),
        "gather" => {
            let src = a[0].list()?;
            Ok(Value::List(
                a[1].list()?
                    .iter()
                    .map(|p| {
                        src.get(p.position()?)
                            .cloned()
                            .ok_or_else(|| invalid("gather out of bounds"))
                    })
                    .collect::<EvalResult<_>>()?,
            ))
        }
        "slice" => {
            let xs = a[0].list()?;
            let start = a[1].position()?;
            let end = a[2].position()?;
            require(start <= end && end <= xs.len(), "invalid slice")?;
            Ok(Value::List(xs[start..end].to_vec()))
        }
        "replace" => {
            let mut xs = a[0].list()?.to_vec();
            let i = a[1].position()?;
            *xs.get_mut(i)
                .ok_or_else(|| invalid("replace out of bounds"))? = a[2].clone();
            Ok(Value::List(xs))
        }
        "contains" => Ok(Value::Boolean(contains(a[0].list()?, &a[1], b)?)),
        "filter" | "difference" => {
            let source = a[0].list()?;
            let members = a[1].list()?;
            for value in source.iter().chain(members) {
                require(
                    matches!(value, Value::Axis(_)),
                    "axis_filter expects logical axes",
                )?;
            }
            let mut out = Vec::new();
            for value in source {
                b.tick()?;
                if contains(members, value, b)? == (name == "filter") {
                    out.push(value.clone());
                }
            }
            Ok(Value::List(out))
        }
        "subsequence" => {
            let required = a[0].list()?;
            let mut index = 0;
            for value in a[1].list()? {
                b.tick()?;
                if required.get(index) == Some(value) {
                    index += 1;
                }
            }
            Ok(Value::Boolean(index == required.len()))
        }
        "exclude" => exclude(&a, b),
        "is_disjoint" => disjoint(&a, b),
        "all" | "any" => {
            for x in a[0].list()? {
                if x.boolean()? != (name == "all") {
                    return Ok(Value::Boolean(name == "any"));
                }
            }
            Ok(Value::Boolean(name == "all"))
        }
        "sum" | "product" => {
            let xs = a[0].list()?;
            if name == "product" {
                for x in xs {
                    if x.integer()? == 0 {
                        return Ok(Value::Integer(0));
                    }
                }
            }
            let mut n = if name == "sum" { 0 } else { 1 };
            for x in xs {
                n = arithmetic(if name == "sum" { "+" } else { "*" }, n, x.integer()?)?;
            }
            Ok(Value::Integer(n))
        }
        "min" | "max" => Ok(Value::Integer(if name == "min" {
            a[0].integer()?.min(a[1].integer()?)
        } else {
            a[0].integer()?.max(a[1].integer()?)
        })),
        "floor_div" | "ceil_div" => {
            let x = a[0].integer()?;
            let y = a[1].integer()?;
            require(y > 0, "rounded division requires positive divisor")?;
            let q = arithmetic("/", x, y)?;
            let r = arithmetic("%", x, y)?;
            Ok(Value::Integer(if r < 0 && name == "floor_div" {
                arithmetic("-", q, 1)?
            } else if r > 0 && name == "ceil_div" {
                arithmetic("+", q, 1)?
            } else {
                q
            }))
        }
        "is_valid_axis_list" => Ok(Value::Boolean(valid(a[0].list()?, a[1].integer()?, b)?)),
        "broadcast_shape" => {
            let x = a[0].list()?;
            let y = a[1].list()?;
            let mut out = Vec::with_capacity(x.len().max(y.len()));
            for i in 0..x.len().max(y.len()) {
                let l = if i < x.len() {
                    x[x.len() - 1 - i].integer()?
                } else {
                    1
                };
                let r = if i < y.len() {
                    y[y.len() - 1 - i].integer()?
                } else {
                    1
                };
                require(l >= 0 && r >= 0, "negative extent")?;
                require(l == r || l == 1 || r == 1, "incompatible broadcast")?;
                out.push(Value::Integer(if l == 1 { r } else { l }));
            }
            out.reverse();
            Ok(Value::List(out))
        }
        "is_float" | "is_integer" | "is_signed_integer" | "is_unsigned_integer" | "is_numeric" => {
            let s = a[0].string()?;
            require(s.parse::<crate::DType>().is_ok(), "unknown dtype")?;
            let f = s.starts_with('f') || s == "bf16";
            let i = s.starts_with('i');
            let u = s.starts_with('u');
            Ok(Value::Boolean(match name {
                "is_float" => f,
                "is_integer" => i || u,
                "is_signed_integer" => i,
                "is_unsigned_integer" => u,
                _ => f || i || u,
            }))
        }
        _ => Err(invalid("unknown builtin")),
    }
}

pub(crate) fn concat(a: &[Value], b: &mut Budget) -> EvalResult<Value> {
    let mut xs = vec![];
    for v in a {
        let part = v.list()?;
        b.size(
            xs.len()
                .checked_add(part.len())
                .ok_or_else(|| invalid("list size overflow"))?,
        )?;
        xs.extend_from_slice(part);
    }
    Ok(Value::List(xs))
}
pub(crate) fn exclude(a: &[Value], b: &mut Budget) -> EvalResult<Value> {
    let mut out = vec![];
    for x in a[0].list()? {
        if !contains(a[1].list()?, x, b)? {
            out.push(x.clone());
        }
    }
    Ok(Value::List(out))
}
pub(crate) fn disjoint(a: &[Value], b: &mut Budget) -> EvalResult<Value> {
    for x in a[0].list()? {
        if contains(a[1].list()?, x, b)? {
            return Ok(Value::Boolean(false));
        }
    }
    Ok(Value::Boolean(true))
}

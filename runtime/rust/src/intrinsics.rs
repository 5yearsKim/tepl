//! Compiler-generated metadata operations. These are not source builtins.
use crate::builtins::{arithmetic, call as builtin, contains, distinct, valid};
use crate::error::{EvalResult, require};
use crate::values::{Budget, Value, invalid};

pub fn call(name: &str, a: Vec<Value>, b: &mut Budget) -> EvalResult<Value> {
    let arity = match name {
        "empty_axes" => Some(0),
        "inverse"
        | "order_positions"
        | "is_valid_extents"
        | "is_positive_extents"
        | "distinct_axes"
        | "axis_extents"
        | "unique_axes" => Some(1),
        "replace_axes" | "place_axes" | "if" => Some(3),
        "concat" | "union_axes" => None,
        _ => Some(2),
    };
    require(
        arity.is_none_or(|n| n == a.len()),
        "invalid intrinsic arity",
    )?;
    if name != "index" {
        b.arguments(&a)?;
    }
    match name {
        "empty_axes" => Ok(Value::List(vec![])),
        "concat" => crate::builtins::concat(&a, b),
        "index" => crate::builtins::index(a[0].clone(), a[1].clone(), b),
        "if" => Ok(if a[0].boolean()? {
            a[1].clone()
        } else {
            a[2].clone()
        }),
        "axis_partition" => Ok(Value::Boolean(
            valid(a[0].list()?, a[1].integer()?, b)?
                && a[0].list()?.len() as i128 == a[1].integer()?,
        )),
        "inverse" => {
            let xs = a[0].list()?;
            require(valid(xs, xs.len() as i128, b)?, "invalid permutation")?;
            let mut out = vec![Value::None; xs.len()];
            for (i, x) in xs.iter().enumerate() {
                out[x.position()?] = Value::Integer(i as i128);
            }
            Ok(Value::List(out))
        }
        "order_positions" => {
            let keys = a[0]
                .list()?
                .iter()
                .map(Value::integer)
                .collect::<EvalResult<Vec<_>>>()?;
            let mut order: Vec<_> = (0..keys.len()).collect();
            order.sort_by_key(|i| keys[*i]);
            Ok(Value::List(
                order
                    .into_iter()
                    .map(|i| Value::Integer(i as i128))
                    .collect(),
            ))
        }
        "offset_positions" | "offset_extents" | "scale_extents" => Ok(Value::List(
            a[0].list()?
                .iter()
                .map(|x| {
                    Ok(Value::Integer(arithmetic(
                        if name == "scale_extents" { "*" } else { "+" },
                        x.integer()?,
                        a[1].integer()?,
                    )?))
                })
                .collect::<EvalResult<_>>()?,
        )),
        "extent_add" | "extent_sub" | "extent_mul" | "extent_floor_div" | "extent_le" => {
            let x = a[0].list()?;
            let y = a[1].list()?;
            require(x.len() == y.len(), "extent lengths differ")?;
            let mut out = vec![];
            let mut le = true;
            for (x, y) in x.iter().zip(y) {
                le &= x.integer()? <= y.integer()?;
                if name == "extent_floor_div" {
                    out.push(builtin("floor_div", vec![x.clone(), y.clone()], b)?);
                } else if name != "extent_le" {
                    out.push(Value::Integer(arithmetic(
                        match name {
                            "extent_add" => "+",
                            "extent_sub" => "-",
                            _ => "*",
                        },
                        x.integer()?,
                        y.integer()?,
                    )?));
                }
            }
            Ok(if name == "extent_le" {
                Value::Boolean(le)
            } else {
                Value::List(out)
            })
        }
        "is_valid_extents" | "is_positive_extents" => {
            for x in a[0].list()? {
                if if name == "is_valid_extents" {
                    x.integer()? < 0
                } else {
                    x.integer()? <= 0
                } {
                    return Ok(Value::Boolean(false));
                }
            }
            Ok(Value::Boolean(true))
        }
        "distinct_axes" => Ok(Value::Boolean(distinct(a[0].list()?, b)?)),
        "axis_extents" => Ok(Value::List(
            a[0].list()?
                .iter()
                .map(|x| match x {
                    Value::Axis(axis) => Ok(Value::Integer(axis.extent)),
                    _ => Err(invalid("expected axis")),
                })
                .collect::<EvalResult<_>>()?,
        )),
        "positions" => {
            let src = a[0].list()?;
            let wanted = a[1].list()?;
            require(
                distinct(src, b)? && distinct(wanted, b)?,
                "noninjective positions",
            )?;
            let mut out = vec![];
            for x in wanted {
                b.charge(src.len())?;
                out.push(Value::Integer(
                    src.iter()
                        .position(|v| v == x)
                        .ok_or_else(|| invalid("absent axis"))? as i128,
                ));
            }
            Ok(Value::List(out))
        }
        "membership_positions" | "matched_positions" => {
            let mut out = vec![];
            let y = a[1].list()?;
            for (i, x) in a[0].list()?.iter().enumerate() {
                b.charge(y.len())?;
                if let Some(j) = y.iter().position(|v| v == x) {
                    out.push(Value::Integer(
                        if name == "membership_positions" { i } else { j } as i128,
                    ));
                }
            }
            Ok(Value::List(out))
        }
        "union_axes" | "unique_axes" => {
            let mut out = vec![];
            for v in &a {
                for x in v.list()? {
                    if !contains(&out, x, b)? {
                        out.push(x.clone());
                    }
                }
            }
            Ok(Value::List(out))
        }
        "ordered_complement" | "difference_axes" => crate::builtins::exclude(&a, b),
        "disjoint" => crate::builtins::disjoint(&a, b),
        "filter_axes" | "intersection_axes" | "same_set" | "subset" => {
            let x = a[0].list()?;
            let y = a[1].list()?;
            let mut out = vec![];
            for v in x {
                if contains(y, v, b)? {
                    out.push(v.clone());
                }
            }
            if name == "subset" {
                return Ok(Value::Boolean(out.len() == x.len()));
            }
            if name == "same_set" {
                for v in y {
                    if !contains(x, v, b)? {
                        return Ok(Value::Boolean(false));
                    }
                }
                return Ok(Value::Boolean(out.len() == x.len()));
            }
            Ok(Value::List(out))
        }
        "replace_axes" => {
            let mut xs = a[0].list()?.to_vec();
            let from = a[1].list()?;
            let to = a[2].list()?;
            require(
                from.len() == to.len() && distinct(from, b)? && distinct(to, b)?,
                "invalid axis replacement",
            )?;
            for x in &mut xs {
                b.charge(from.len())?;
                if let Some(i) = from.iter().position(|v| v == x) {
                    *x = to[i].clone();
                }
            }
            Ok(Value::List(xs))
        }
        "place_axes" => {
            let left = a[0].list()?;
            let p = a[1].list()?;
            let right = a[2].list()?;
            let n = left
                .len()
                .checked_add(right.len())
                .ok_or_else(|| invalid("list size overflow"))?;
            require(
                left.len() == p.len() && valid(p, n as i128, b)?,
                "invalid axis placement",
            )?;
            b.size(n)?;
            let mut out = vec![None; n];
            for (x, pos) in left.iter().zip(p) {
                out[pos.position()?] = Some(x.clone());
            }
            let mut rest = right.iter();
            Ok(Value::List(
                out.into_iter()
                    .map(|x| x.unwrap_or_else(|| rest.next().unwrap().clone()))
                    .collect(),
            ))
        }
        _ => Err(invalid("unknown intrinsic")),
    }
}

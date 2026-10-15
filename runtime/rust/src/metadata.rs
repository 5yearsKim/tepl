use crate::values::invalid;
use crate::{Budget, EvalResult, TensorFacts, Value};
use std::collections::BTreeMap;

pub struct Context<'a> {
    operands: &'a [TensorFacts],
    attrs: &'a Value,
    pub locals: BTreeMap<usize, Value>,
    pub budget: &'a mut Budget,
}
impl<'a> Context<'a> {
    pub fn new(operands: &'a [TensorFacts], attrs: &'a Value, budget: &'a mut Budget) -> Self {
        Self {
            operands,
            attrs,
            locals: BTreeMap::new(),
            budget,
        }
    }
    pub fn parameter(
        &mut self,
        symbol: usize,
        operand: usize,
        variadic: bool,
        dtype: bool,
    ) -> EvalResult<()> {
        let input = |i: usize| -> EvalResult<Value> {
            let facts = self
                .operands
                .get(i)
                .ok_or_else(|| invalid("missing operand"))?;
            if dtype {
                facts.dtype_value()
            } else {
                facts.shape_value()
            }
        };
        let value = if variadic {
            Value::List(
                (operand..self.operands.len())
                    .map(input)
                    .collect::<EvalResult<_>>()?,
            )
        } else {
            input(operand)?
        };
        self.locals.insert(symbol, value);
        Ok(())
    }
    pub fn local(&self, id: usize) -> EvalResult<Value> {
        self.locals
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid("missing metadata local"))
    }
    pub fn field(&self, id: usize) -> EvalResult<Value> {
        self.attrs.field(id)
    }
}

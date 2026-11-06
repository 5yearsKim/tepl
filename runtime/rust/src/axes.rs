use crate::error::require;
use crate::values::{Axis, invalid};
use crate::{Budget, EvalResult, TensorFacts, Value};
use std::collections::BTreeSet;

/// Concrete operation contracts create fresh axis identities within this call.
pub struct AxisContext<'a> {
    operands: &'a [TensorFacts],
    attrs: &'a Value,
    groups: Vec<Value>,
    used: Vec<BTreeSet<usize>>,
    extents: Option<Vec<Value>>,
    pub budget: &'a mut Budget,
}
impl<'a> AxisContext<'a> {
    pub fn new(operands: &'a [TensorFacts], attrs: &'a Value, budget: &'a mut Budget) -> Self {
        Self {
            operands,
            attrs,
            groups: vec![],
            used: vec![BTreeSet::new(); operands.len()],
            extents: None,
            budget,
        }
    }
    pub fn operand(&self, i: usize) -> EvalResult<Value> {
        self.operands
            .get(i)
            .ok_or_else(|| invalid("missing operand"))?
            .shape_value()
    }
    pub fn field(&self, i: usize) -> EvalResult<Value> {
        self.attrs.field(i)
    }
    pub fn group(&self, i: usize) -> EvalResult<Value> {
        self.groups
            .get(i)
            .cloned()
            .ok_or_else(|| invalid("missing axis group"))
    }
    pub fn begin_group(&mut self) {
        self.extents = None;
    }
    pub fn bind(&mut self, operand: usize, positions: Value) -> EvalResult<()> {
        let shape = self.operand(operand)?;
        require(
            crate::builtins::valid(positions.list()?, shape.list()?.len() as i128, self.budget)?,
            "invalid axis binding",
        )?;
        for p in positions.list()? {
            require(
                self.used[operand].insert(p.position()?),
                "overlapping axis bindings",
            )?;
        }
        let selected = crate::builtins::call("gather", vec![shape, positions], self.budget)?
            .list()?
            .to_vec();
        if let Some(old) = &self.extents {
            require(old == &selected, "axis group extents differ")?;
        } else {
            self.extents = Some(selected);
        }
        Ok(())
    }
    pub fn finish_group(&mut self, domain: Option<Value>) -> EvalResult<()> {
        if let Some(v) = domain {
            self.extents = Some(v.list()?.to_vec());
        }
        let extents = self
            .extents
            .take()
            .ok_or_else(|| invalid("axis group has no domain"))?;
        let mut axes = vec![];
        for (member, extent) in extents.iter().enumerate() {
            let extent = u64::try_from(extent.integer()?).map_err(|_| invalid("index overflow"))?;
            axes.push(Value::Axis(Axis {
                owner: 0,
                group: self.groups.len(),
                member,
                extent: extent.into(),
            }));
        }
        self.groups.push(Value::List(axes));
        Ok(())
    }
    pub fn cover(&self, operand: usize) -> EvalResult<()> {
        require(
            self.used[operand].len() == self.operand(operand)?.list()?.len(),
            "axis bindings do not cover operand",
        )
    }
    pub fn output(&mut self, value: Value) -> EvalResult<Vec<u64>> {
        require(
            crate::builtins::distinct(value.list()?, self.budget)?,
            "duplicate output axes",
        )?;
        crate::intrinsics::call("axis_extents", vec![value], self.budget)?.dimensions()
    }
}

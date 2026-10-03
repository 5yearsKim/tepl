use super::super::pattern::TensorInfo;
use super::super::{Op, OpAttrs, OpNode};

/// Input identities with immutable types. Constants carry their own typed
/// payload and are not registered as named inputs.
#[derive(Clone, Debug, Default)]
pub struct TensorBindingTable {
    entries: std::collections::HashMap<String, TensorInfo>,
}

impl TensorBindingTable {
    pub fn register_symbol(
        &mut self,
        name: impl Into<String>,
        info: TensorInfo,
    ) -> Result<(), String> {
        let name = name.into();
        if let Some(previous) = self.entries.get(&name) {
            if previous != &info {
                return Err(format!("conflicting tensor type for input '{name}'"));
            }
            return Ok(());
        }
        self.entries.insert(name, info);
        Ok(())
    }

    pub fn info(&self, node: &OpNode) -> Option<&TensorInfo> {
        match (node.op(), node.attrs()) {
            (Op::Input, OpAttrs::Input { name }) => self.entries.get(name),
            _ => None,
        }
    }
}

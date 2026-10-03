use super::nodes::symbol_name;
use crate::ir::OpNode;
use crate::ir::pattern::TensorInfo;

/// Host input identities with immutable types. Constants carry their own typed
/// payload and are not registered as named inputs.
#[derive(Clone, Debug, Default)]
pub struct TensorBindings {
    entries: std::collections::HashMap<String, TensorInfo>,
}

impl TensorBindings {
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
        self.entries.get(symbol_name(node)?)
    }
}

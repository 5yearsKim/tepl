use crate::ir::dialects::tensor_lang;
use crate::ir::pattern::TensorInfo;
use crate::ir::{Op, OpAttrs, OpNode};
/// Host input identities with immutable types. Register inputs before building
/// the graph; conflicting registrations never overwrite existing metadata.
#[derive(Clone, Debug, Default)]
pub struct TensorBindings {
    entries: std::collections::HashMap<(Op, String), TensorInfo>,
}

impl TensorBindings {
    pub fn register_symbol(
        &mut self,
        name: impl Into<String>,
        info: TensorInfo,
    ) -> Result<(), String> {
        self.register(Op::TensorLang(tensor_lang::Op::Symbol), name.into(), info)
    }

    pub fn register_constant(
        &mut self,
        name: impl Into<String>,
        info: TensorInfo,
    ) -> Result<(), String> {
        self.register(Op::TensorLang(tensor_lang::Op::Constant), name.into(), info)
    }

    fn register(&mut self, op: Op, name: String, info: TensorInfo) -> Result<(), String> {
        let key = (op, name);
        if let Some(previous) = self.entries.get(&key) {
            if previous != &info {
                return Err(format!(
                    "conflicting tensor type for {} '{}'",
                    op.name(),
                    key.1
                ));
            }
            return Ok(());
        }
        self.entries.insert(key, info);
        Ok(())
    }

    pub fn info(&self, node: &OpNode) -> Option<&TensorInfo> {
        let name = match node.attrs() {
            OpAttrs::TensorLang(tensor_lang::OpAttrs::SymbolAttrs { name })
            | OpAttrs::TensorLang(tensor_lang::OpAttrs::ConstantAttrs { name }) => name,
            _ => return None,
        };
        self.entries.get(&(node.op(), name.clone()))
    }
}

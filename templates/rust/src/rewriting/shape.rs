use ::std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub enum ShapePart {
    Dimension(usize),
    Wildcard,
    Sequence(Option<usize>),
}

#[derive(Clone, Debug, Default)]
pub struct MetadataBindings {
    dtypes: HashMap<usize, super::super::DType>,
    dimensions: HashMap<usize, u64>,
    sequences: HashMap<usize, Vec<u64>>,
}

impl MetadataBindings {
    pub fn bind_dtype(&mut self, id: usize, value: super::super::DType) -> Option<()> {
        if self
            .dtypes
            .get(&id)
            .is_some_and(|previous| *previous != value)
        {
            return None;
        }
        self.dtypes.insert(id, value);
        Some(())
    }
    pub fn dtype(&self, id: usize) -> Option<super::super::DType> {
        self.dtypes.get(&id).copied()
    }

    /// Each invocation adds a restriction; repeated IDs must agree across all captures.
    /// The pattern must contain at most one sequence (validated by the compiler or
    /// `TensorConstraints::validate`). Sequences may be empty and appear anywhere.
    /// Discard this binding environment after failure: checks may partially update it.
    pub fn check(&mut self, shape: &[u64], pattern: &[ShapePart]) -> Option<()> {
        let sequence = pattern
            .iter()
            .position(|part| matches!(part, ShapePart::Sequence(_)));
        let fixed = pattern.len() - usize::from(sequence.is_some());
        if shape.len() < fixed || (sequence.is_none() && shape.len() != fixed) {
            return None;
        }
        for (index, part) in pattern.iter().enumerate() {
            match part {
                ShapePart::Dimension(id) => {
                    let axis = if sequence.is_some_and(|split| index > split) {
                        shape.len() - (pattern.len() - index)
                    } else {
                        index
                    };
                    let value = *super::super::builtins::common::index(shape, axis).ok()?;
                    if self
                        .dimensions
                        .get(id)
                        .is_some_and(|previous| *previous != value)
                    {
                        return None;
                    }
                    self.dimensions.insert(*id, value);
                }
                ShapePart::Wildcard => {}
                ShapePart::Sequence(Some(id)) => {
                    let value = super::super::builtins::common::slice(
                        shape,
                        index,
                        shape.len() - (pattern.len() - index - 1),
                    )
                    .ok()?;
                    if self
                        .sequences
                        .get(id)
                        .is_some_and(|previous| previous != &value)
                    {
                        return None;
                    }
                    self.sequences.insert(*id, value);
                }
                ShapePart::Sequence(None) => {}
            }
        }
        Some(())
    }

    pub fn dimension(&self, id: usize) -> Option<u64> {
        self.dimensions.get(&id).copied()
    }
    pub fn sequence(&self, id: usize) -> Option<&[u64]> {
        self.sequences.get(&id).map(Vec::as_slice)
    }
}

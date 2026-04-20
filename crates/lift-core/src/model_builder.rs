// ============================================================================
// model_builder.rs — High-Level Programmatic Model Builder
// ============================================================================
//
// Provides a fluent API for constructing LIFT IR models from Rust code.
// Automatically generates valid `.lif` text output and handles all SSA wiring.
//
// Usage:
//   let lif = ModelBuilder::new("gpt2")
//       .function("forward")
//           .param("x", tensor(1, 768, DataType::FP32))
//           .param("w", tensor_2d(768, 768, DataType::FP32))
//           .op("tensor.matmul", &["x", "w"], "h", tensor(1, 768, DataType::FP32))
//           .op("tensor.relu", &["h"], "out", tensor(1, 768, DataType::FP32))
//           .returns("out")
//           .done()
//       .build_lif();
//
// ============================================================================

use crate::context::Context;
use crate::types::{DataType, Dimension, MemoryLayout};
use crate::attributes::{Attributes, Attribute};
use crate::functions::FunctionData;
use crate::location::Location;
use crate::printer::print_ir;

/// Shape descriptor for tensor types.
#[derive(Debug, Clone)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

/// Describes a value type in the model.
#[derive(Debug, Clone)]
pub enum ModelType {
    Tensor { shape: Vec<usize>, dtype: DataType },
    Qubit,
    Bit,
    Integer { bits: u32 },
}

/// Convenience: create a tensor type descriptor.
pub fn tensor(shape: &[usize], dtype: DataType) -> ModelType {
    ModelType::Tensor { shape: shape.to_vec(), dtype }
}

/// Convenience: create a 2D tensor type descriptor.
pub fn tensor_2d(m: usize, n: usize, dtype: DataType) -> ModelType {
    ModelType::Tensor { shape: vec![m, n], dtype }
}

/// Convenience: create a 3D tensor type descriptor.
pub fn tensor_3d(b: usize, s: usize, d: usize, dtype: DataType) -> ModelType {
    ModelType::Tensor { shape: vec![b, s, d], dtype }
}

/// Convenience: create a 4D tensor type descriptor.
pub fn tensor_4d(b: usize, c: usize, h: usize, w: usize, dtype: DataType) -> ModelType {
    ModelType::Tensor { shape: vec![b, c, h, w], dtype }
}

/// Convenience: create a 1D tensor type descriptor.
pub fn tensor_1d(n: usize, dtype: DataType) -> ModelType {
    ModelType::Tensor { shape: vec![n], dtype }
}

/// Represents a single operation in the builder.
#[derive(Debug, Clone)]
struct OpDef {
    op_name: String,
    inputs: Vec<String>,
    result_name: String,
    result_type: ModelType,
    attrs: Vec<(String, Attribute)>,
}

/// Represents a function being built.
#[derive(Debug)]
pub struct FunctionBuilder {
    name: String,
    params: Vec<(String, ModelType)>,
    ops: Vec<OpDef>,
    return_name: Option<String>,
    dialect: String,
}

impl FunctionBuilder {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            params: Vec::new(),
            ops: Vec::new(),
            return_name: None,
            dialect: "tensor".to_string(),
        }
    }

    /// Add a parameter to the function.
    pub fn param(mut self, name: &str, ty: ModelType) -> Self {
        self.params.push((name.to_string(), ty));
        self
    }

    /// Add an operation.
    pub fn op(mut self, op_name: &str, inputs: &[&str], result: &str, result_type: ModelType) -> Self {
        self.ops.push(OpDef {
            op_name: op_name.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            result_name: result.to_string(),
            result_type,
            attrs: Vec::new(),
        });
        self
    }

    /// Add an operation with attributes.
    pub fn op_with_attrs(
        mut self,
        op_name: &str,
        inputs: &[&str],
        result: &str,
        result_type: ModelType,
        attrs: Vec<(&str, Attribute)>,
    ) -> Self {
        self.ops.push(OpDef {
            op_name: op_name.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            result_name: result.to_string(),
            result_type,
            attrs: attrs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        });
        self
    }

    /// Set the return value.
    pub fn returns(mut self, name: &str) -> Self {
        self.return_name = Some(name.to_string());
        self
    }

    /// Set the dialect prefix.
    pub fn dialect(mut self, d: &str) -> Self {
        self.dialect = d.to_string();
        self
    }
}

/// Main model builder — constructs a complete LIFT module.
#[derive(Debug)]
pub struct ModelBuilder {
    module_name: String,
    functions: Vec<FunctionBuilder>,
    dialect_directive: String,
}

impl ModelBuilder {
    /// Create a new model builder with a module name.
    pub fn new(name: &str) -> Self {
        Self {
            module_name: name.to_string(),
            functions: Vec::new(),
            dialect_directive: "tensor".to_string(),
        }
    }

    /// Set the dialect directive.
    pub fn dialect(mut self, d: &str) -> Self {
        self.dialect_directive = d.to_string();
        self
    }

    /// Add a function definition.
    pub fn function(mut self, name: &str) -> FunctionBuilderHandle {
        let fb = FunctionBuilder::new(name);
        let idx = self.functions.len();
        self.functions.push(fb);
        FunctionBuilderHandle { model: self, func_idx: idx }
    }

    /// Build the model into a LIFT Context and return it.
    pub fn build_context(self) -> Context {
        let mut ctx = Context::new();
        let mod_idx = ctx.create_module(&self.module_name);

        for fb in &self.functions {
            let func_data = self.build_function_data(&mut ctx, fb);
            ctx.add_function_to_module(mod_idx, func_data);
        }

        ctx
    }

    /// Build the model and return the `.lif` source text.
    pub fn build_lif(self) -> String {
        let dialect = self.dialect_directive.clone();
        let ctx = self.build_context();
        let ir = print_ir(&ctx);
        format!("#dialect {}\n\n{}", dialect, ir)
    }

    /// Build and write `.lif` file to disk.
    pub fn write_lif(self, path: &str) -> std::io::Result<()> {
        let lif = self.build_lif();
        std::fs::write(path, lif)
    }

    fn build_function_data(&self, ctx: &mut Context, fb: &FunctionBuilder) -> FunctionData {
        let name_id = ctx.intern_string(&fb.name);

        // Resolve parameter types
        let param_types: Vec<_> = fb.params.iter()
            .map(|(_, ty)| self.resolve_model_type(ctx, ty))
            .collect();

        // Determine return type
        let return_types = if let Some(ref ret_name) = fb.return_name {
            // Find the op that produces the return value
            if let Some(op) = fb.ops.iter().find(|o| o.result_name == *ret_name) {
                vec![self.resolve_model_type(ctx, &op.result_type)]
            } else if let Some((_, ty)) = fb.params.iter().find(|(n, _)| n == ret_name) {
                vec![self.resolve_model_type(ctx, ty)]
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let mut func_data = FunctionData::new(name_id, param_types.clone(), return_types);

        // Create function body
        let region = ctx.create_region();
        let block = ctx.create_block();
        ctx.add_block_to_region(region, block);

        // Create block args for parameters and build name map
        let mut name_map = std::collections::HashMap::new();
        for (i, (pname, _)) in fb.params.iter().enumerate() {
            let val = ctx.create_block_arg(block, param_types[i]);
            name_map.insert(pname.clone(), val);
        }

        // Build operations
        for op_def in &fb.ops {
            let inputs: Vec<_> = op_def.inputs.iter()
                .filter_map(|name| name_map.get(name).copied())
                .collect();

            let result_ty = self.resolve_model_type(ctx, &op_def.result_type);

            let (dialect, _) = op_def.op_name.split_once('.')
                .unwrap_or(("core", &op_def.op_name));

            let mut attrs = Attributes::new();
            for (key, val) in &op_def.attrs {
                attrs.set(key.clone(), val.clone());
            }

            let (op_key, results) = ctx.create_op(
                &op_def.op_name, dialect,
                inputs, vec![result_ty],
                attrs, Location::unknown(),
            );
            ctx.add_op_to_block(block, op_key);

            if !results.is_empty() {
                name_map.insert(op_def.result_name.clone(), results[0]);
            }
        }

        // Add return operation
        if let Some(ref ret_name) = fb.return_name {
            if let Some(&ret_val) = name_map.get(ret_name) {
                let (ret_op, _) = ctx.create_op(
                    "core.return", "core",
                    vec![ret_val], vec![],
                    Attributes::new(), Location::unknown(),
                );
                ctx.add_op_to_block(block, ret_op);
            }
        }

        func_data.body = Some(region);
        func_data
    }

    fn resolve_model_type(&self, ctx: &mut Context, mt: &ModelType) -> crate::types::TypeId {
        match mt {
            ModelType::Tensor { shape, dtype } => {
                let dims: Vec<Dimension> = shape.iter().map(|&d| Dimension::Constant(d)).collect();
                ctx.make_tensor_type(dims, *dtype, MemoryLayout::Contiguous)
            }
            ModelType::Qubit => ctx.make_qubit_type(),
            ModelType::Bit => ctx.make_bit_type(),
            ModelType::Integer { bits } => ctx.make_integer_type(*bits, true),
        }
    }
}

/// Handle that chains function-building methods back to the model builder.
pub struct FunctionBuilderHandle {
    model: ModelBuilder,
    func_idx: usize,
}

impl FunctionBuilderHandle {
    pub fn param(mut self, name: &str, ty: ModelType) -> Self {
        let fb = std::mem::replace(
            &mut self.model.functions[self.func_idx],
            FunctionBuilder::new("__tmp__"),
        );
        self.model.functions[self.func_idx] = fb.param(name, ty);
        self
    }

    pub fn op(mut self, op_name: &str, inputs: &[&str], result: &str, result_type: ModelType) -> Self {
        let fb = std::mem::replace(
            &mut self.model.functions[self.func_idx],
            FunctionBuilder::new("__tmp__"),
        );
        self.model.functions[self.func_idx] = fb.op(op_name, inputs, result, result_type);
        self
    }

    pub fn op_with_attrs(
        mut self,
        op_name: &str,
        inputs: &[&str],
        result: &str,
        result_type: ModelType,
        attrs: Vec<(&str, Attribute)>,
    ) -> Self {
        let fb = std::mem::replace(
            &mut self.model.functions[self.func_idx],
            FunctionBuilder::new("__tmp__"),
        );
        self.model.functions[self.func_idx] = fb.op_with_attrs(op_name, inputs, result, result_type, attrs);
        self
    }

    pub fn returns(mut self, name: &str) -> Self {
        let fb = std::mem::replace(
            &mut self.model.functions[self.func_idx],
            FunctionBuilder::new("__tmp__"),
        );
        self.model.functions[self.func_idx] = fb.returns(name);
        self
    }

    pub fn dialect(mut self, d: &str) -> Self {
        let fb = std::mem::replace(
            &mut self.model.functions[self.func_idx],
            FunctionBuilder::new("__tmp__"),
        );
        self.model.functions[self.func_idx] = fb.dialect(d);
        self
    }

    /// Finish the function and return to the model builder.
    pub fn done(self) -> ModelBuilder {
        self.model
    }
}

/// Generate a `.lith` configuration string.
pub fn build_lith_config(
    backend: &str,
    device: &str,
    precision: &str,
    passes: &[&str],
    max_flops: Option<u64>,
    max_memory: Option<u64>,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("[target]\nbackend = \"{}\"\ndevice = \"{}\"\nprecision = \"{}\"\n\n", backend, device, precision));

    if max_flops.is_some() || max_memory.is_some() {
        out.push_str("[budget]\n");
        if let Some(f) = max_flops { out.push_str(&format!("max_flops = {}\n", f)); }
        if let Some(m) = max_memory { out.push_str(&format!("max_memory_bytes = {}\n", m)); }
        out.push('\n');
    }

    out.push_str("[optimisation]\nlevel = O3\n");
    out.push_str(&format!("passes = {}\n", passes.join(", ")));
    out.push_str("max_iterations = 10\n\n");

    out.push_str("[simulation]\nshape_propagation = true\nflop_counting = true\nmemory_analysis = true\nnoise_simulation = false\n");

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_model() {
        let lif = ModelBuilder::new("test_mlp")
            .function("forward")
                .param("x", tensor(&[1, 784], DataType::FP32))
                .param("w", tensor_2d(784, 256, DataType::FP32))
                .op("tensor.matmul", &["x", "w"], "h", tensor(&[1, 256], DataType::FP32))
                .op("tensor.relu", &["h"], "out", tensor(&[1, 256], DataType::FP32))
                .returns("out")
                .done()
            .build_lif();

        assert!(lif.contains("#dialect tensor"));
        assert!(lif.contains("module @test_mlp"));
        assert!(lif.contains("tensor.matmul"));
        assert!(lif.contains("tensor.relu"));
        assert!(lif.contains("core.return"));
    }

    #[test]
    fn test_build_context() {
        let ctx = ModelBuilder::new("ctx_test")
            .function("f")
                .param("a", tensor(&[4, 4], DataType::FP32))
                .param("b", tensor(&[4, 4], DataType::FP32))
                .op("tensor.add", &["a", "b"], "c", tensor(&[4, 4], DataType::FP32))
                .returns("c")
                .done()
            .build_context();

        assert_eq!(ctx.modules.len(), 1);
        assert_eq!(ctx.modules[0].functions.len(), 1);
        assert!(ctx.ops.len() >= 2); // add + return
    }

    #[test]
    fn test_write_lif() {
        let path = "/tmp/lift_builder_test.lif";
        ModelBuilder::new("write_test")
            .function("f")
                .param("x", tensor(&[8], DataType::FP32))
                .op("tensor.relu", &["x"], "y", tensor(&[8], DataType::FP32))
                .returns("y")
                .done()
            .write_lif(path)
            .unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("#dialect tensor"));
        assert!(content.contains("tensor.relu"));
    }

    #[test]
    fn test_lith_config() {
        let config = build_lith_config(
            "llvm", "h100", "fp16",
            &["canonicalize", "dce", "tensor-fusion"],
            Some(1_000_000_000),
            Some(80_000_000_000),
        );
        assert!(config.contains("backend = \"llvm\""));
        assert!(config.contains("canonicalize, dce, tensor-fusion"));
        assert!(config.contains("max_flops = 1000000000"));
    }
}

/// Comprehensive tests for lift-core: SSA IR, types, verifier, printer, pass manager
use lift_core::*;
use lift_core::context::Context;
use lift_core::types::*;
use lift_core::values::*;
use lift_core::attributes::*;
use lift_core::location::Location;

// ═══════════════════════════════════════════════════
//  TYPE SYSTEM TESTS
// ═══════════════════════════════════════════════════

#[test]
fn test_all_scalar_types() {
    let mut ctx = Context::new();
    let i8t = ctx.make_integer_type(8, true);
    let i16t = ctx.make_integer_type(16, true);
    let i32t = ctx.make_integer_type(32, true);
    let i64t = ctx.make_integer_type(64, true);
    let u8t = ctx.make_integer_type(8, false);
    let f16t = ctx.make_float_type(16);
    let f32t = ctx.make_float_type(32);
    let f64t = ctx.make_float_type(64);
    let bt = ctx.make_bool_type();
    let vt = ctx.make_void_type();
    let it = ctx.make_index_type();

    let types = vec![i8t, i16t, i32t, i64t, u8t, f16t, f32t, f64t, bt, vt, it];
    for i in 0..types.len() {
        for j in (i+1)..types.len() {
            assert_ne!(types[i], types[j], "types {} and {} should differ", i, j);
        }
    }
}

#[test]
fn test_type_interning_deduplication() {
    let mut ctx = Context::new();
    let t1 = ctx.make_float_type(32);
    let t2 = ctx.make_float_type(32);
    let t3 = ctx.make_float_type(64);
    assert_eq!(t1, t2, "same type must intern to same id");
    assert_ne!(t1, t3, "different types must have different ids");
}

#[test]
fn test_tensor_type_shapes() {
    let mut ctx = Context::new();
    let t1 = ctx.make_tensor_type(
        vec![Dimension::Constant(2), Dimension::Constant(3)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let t2 = ctx.make_tensor_type(
        vec![Dimension::Constant(2), Dimension::Constant(3)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let t3 = ctx.make_tensor_type(
        vec![Dimension::Constant(4), Dimension::Constant(5)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    assert_eq!(t1, t2, "identical tensor types must dedup");
    assert_ne!(t1, t3, "different shapes must differ");
}

#[test]
fn test_tensor_type_dtypes() {
    let mut ctx = Context::new();
    let shape = vec![Dimension::Constant(8)];
    let fp32 = ctx.make_tensor_type(shape.clone(), DataType::FP32, MemoryLayout::Contiguous);
    let fp16 = ctx.make_tensor_type(shape.clone(), DataType::FP16, MemoryLayout::Contiguous);
    let bf16 = ctx.make_tensor_type(shape.clone(), DataType::BF16, MemoryLayout::Contiguous);
    let i8t = ctx.make_tensor_type(shape.clone(), DataType::INT8, MemoryLayout::Contiguous);
    assert_ne!(fp32, fp16);
    assert_ne!(fp16, bf16);
    assert_ne!(bf16, i8t);
}

#[test]
fn test_qubit_types() {
    let mut ctx = Context::new();
    let q1 = ctx.make_qubit_type();
    let q2 = ctx.make_qubit_type();
    let pq = ctx.make_physical_qubit_type(0, 100.0, 80.0, 5.0, 0.999);
    assert_eq!(q1, q2, "logical qubits must dedup");
    assert_ne!(q1, pq, "logical vs physical must differ");
}

#[test]
fn test_type_queries() {
    let mut ctx = Context::new();
    let tensor_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(4)], DataType::FP32, MemoryLayout::Contiguous,
    );
    let qubit_ty = ctx.make_qubit_type();
    let bit_ty = ctx.make_bit_type();
    let int_ty = ctx.make_integer_type(32, true);

    assert!(ctx.is_tensor_type(tensor_ty));
    assert!(!ctx.is_tensor_type(qubit_ty));
    assert!(ctx.is_qubit_type(qubit_ty));
    assert!(!ctx.is_qubit_type(tensor_ty));
    assert!(ctx.is_bit_type(bit_ty));
    assert!(!ctx.is_bit_type(int_ty));
}

#[test]
fn test_tensor_info_extraction() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(
        vec![Dimension::Constant(2), Dimension::Constant(3), Dimension::Constant(4)],
        DataType::FP16, MemoryLayout::NCHW,
    );
    let info = ctx.get_tensor_info(ty).unwrap();
    assert_eq!(info.shape.len(), 3);
    assert_eq!(info.shape[0].static_value(), Some(2));
    assert_eq!(info.shape[1].static_value(), Some(3));
    assert_eq!(info.shape[2].static_value(), Some(4));
    assert_eq!(info.dtype, DataType::FP16);
    assert_eq!(info.layout, MemoryLayout::NCHW);
}

#[test]
fn test_datatype_properties() {
    assert_eq!(DataType::FP32.bit_width(), 32);
    assert_eq!(DataType::FP16.bit_width(), 16);
    assert_eq!(DataType::BF16.bit_width(), 16);
    assert_eq!(DataType::INT8.bit_width(), 8);
    assert_eq!(DataType::INT4.bit_width(), 4);
    assert_eq!(DataType::FP64.byte_size(), 8);
    assert_eq!(DataType::FP32.byte_size(), 4);
    assert_eq!(DataType::FP16.byte_size(), 2);
    assert_eq!(DataType::INT8.byte_size(), 1);
    assert!(DataType::FP32.is_float());
    assert!(!DataType::FP32.is_integer());
    assert!(DataType::INT32.is_integer());
    assert!(!DataType::INT32.is_float());
}

// ═══════════════════════════════════════════════════
//  STRING INTERNING TESTS
// ═══════════════════════════════════════════════════

#[test]
fn test_string_interning_correctness() {
    let mut ctx = Context::new();
    let s1 = ctx.intern_string("hello");
    let s2 = ctx.intern_string("hello");
    let s3 = ctx.intern_string("world");
    assert_eq!(s1, s2);
    assert_ne!(s1, s3);
    assert_eq!(ctx.resolve_string(s1), "hello");
    assert_eq!(ctx.resolve_string(s3), "world");
}

#[test]
fn test_string_interning_many() {
    let mut ctx = Context::new();
    let mut ids = Vec::new();
    for i in 0..1000 {
        ids.push(ctx.intern_string(&format!("string_{}", i)));
    }
    for i in 0..1000 {
        assert_eq!(ctx.resolve_string(ids[i]), format!("string_{}", i));
    }
    for i in 0..1000 {
        let re = ctx.intern_string(&format!("string_{}", i));
        assert_eq!(re, ids[i]);
    }
}

// ═══════════════════════════════════════════════════
//  SSA IR CONSTRUCTION TESTS
// ═══════════════════════════════════════════════════

#[test]
fn test_ssa_basic_construction() {
    let mut ctx = Context::new();
    let f32_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(4)], DataType::FP32, MemoryLayout::Contiguous,
    );

    let region = ctx.create_region();
    let block = ctx.create_block();
    ctx.add_block_to_region(region, block);

    let arg0 = ctx.create_block_arg(block, f32_ty);
    let arg1 = ctx.create_block_arg(block, f32_ty);

    let (op, results) = ctx.create_op(
        "tensor.add", "tensor",
        vec![arg0, arg1], vec![f32_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op);

    assert_eq!(results.len(), 1);
    assert_eq!(ctx.get_op(op).unwrap().inputs.len(), 2);
    assert_eq!(ctx.get_block(block).unwrap().args.len(), 2);
    assert_eq!(ctx.get_block(block).unwrap().ops.len(), 1);
}

#[test]
fn test_ssa_chain_of_operations() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(
        vec![Dimension::Constant(2), Dimension::Constant(3)],
        DataType::FP32, MemoryLayout::Contiguous,
    );

    let block = ctx.create_block();
    let x = ctx.create_block_arg(block, ty);

    let (relu_op, relu_res) = ctx.create_op(
        "tensor.relu", "tensor", vec![x], vec![ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, relu_op);

    let (neg_op, neg_res) = ctx.create_op(
        "tensor.neg", "tensor", vec![relu_res[0]], vec![ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, neg_op);

    let (add_op, _add_res) = ctx.create_op(
        "tensor.add", "tensor", vec![x, neg_res[0]], vec![ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, add_op);

    let add = ctx.get_op(add_op).unwrap();
    assert_eq!(add.inputs[0], x);
    assert_eq!(add.inputs[1], neg_res[0]);
    let neg = ctx.get_op(neg_op).unwrap();
    assert_eq!(neg.inputs[0], relu_res[0]);
}

#[test]
fn test_ssa_multiple_results() {
    let mut ctx = Context::new();
    let q_ty = ctx.make_qubit_type();

    let block = ctx.create_block();
    let q0 = ctx.create_block_arg(block, q_ty);
    let q1 = ctx.create_block_arg(block, q_ty);

    let (cx_op, cx_res) = ctx.create_op(
        "quantum.cx", "quantum",
        vec![q0, q1], vec![q_ty, q_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, cx_op);

    assert_eq!(cx_res.len(), 2);
    let op = ctx.get_op(cx_op).unwrap();
    assert_eq!(op.results.len(), 2);
    assert_ne!(cx_res[0], cx_res[1]);
}

#[test]
fn test_module_and_function_structure() {
    let mut ctx = Context::new();
    let mod_idx = ctx.create_module("test_module");

    let f32_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(4)], DataType::FP32, MemoryLayout::Contiguous,
    );

    let region = ctx.create_region();
    let block = ctx.create_block();
    ctx.add_block_to_region(region, block);
    let _arg = ctx.create_block_arg(block, f32_ty);

    let func_name = ctx.intern_string("my_func");
    let func = lift_core::FunctionData {
        name: func_name,
        params: vec![f32_ty],
        returns: vec![f32_ty],
        body: Some(region),
        location: Location::unknown(),
        is_declaration: false,
    };
    ctx.add_function_to_module(mod_idx, func);

    let module = ctx.get_module(mod_idx).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(ctx.resolve_string(module.functions[0].name), "my_func");
}

// ═══════════════════════════════════════════════════
//  ATTRIBUTES TESTS
// ═══════════════════════════════════════════════════

#[test]
fn test_attributes_crud() {
    let mut attrs = Attributes::new();
    assert!(attrs.is_empty());
    attrs.set("value", Attribute::Integer(42));
    attrs.set("name", Attribute::Bool(true));
    assert_eq!(attrs.len(), 2);
    assert_eq!(attrs.get("value"), Some(&Attribute::Integer(42)));
    assert_eq!(attrs.get("name"), Some(&Attribute::Bool(true)));
    assert_eq!(attrs.get("missing"), None);
}

#[test]
fn test_attributes_get_helpers() {
    let mut attrs = Attributes::new();
    attrs.set("int_val", Attribute::Integer(42));
    attrs.set("float_val", Attribute::Float(3.14));
    attrs.set("bool_val", Attribute::Bool(true));
    assert_eq!(attrs.get_integer("int_val"), Some(42));
    assert_eq!(attrs.get_float("float_val"), Some(3.14));
    assert_eq!(attrs.get_bool("bool_val"), Some(true));
    assert_eq!(attrs.get_integer("missing"), None);
}

// ═══════════════════════════════════════════════════
//  VERIFIER TESTS
// ═══════════════════════════════════════════════════

#[test]
fn test_verifier_empty_context() {
    let ctx = Context::new();
    assert!(lift_core::verifier::verify(&ctx).is_ok());
}

#[test]
fn test_verifier_valid_tensor_program() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(
        vec![Dimension::Constant(4)], DataType::FP32, MemoryLayout::Contiguous,
    );
    let block = ctx.create_block();
    let x = ctx.create_block_arg(block, ty);
    let y = ctx.create_block_arg(block, ty);

    let (op, _) = ctx.create_op(
        "tensor.add", "tensor", vec![x, y], vec![ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op);

    assert!(lift_core::verifier::verify(&ctx).is_ok());
}

#[test]
fn test_verifier_qubit_linearity() {
    let mut ctx = Context::new();
    let q_ty = ctx.make_qubit_type();
    let block = ctx.create_block();
    let q = ctx.create_block_arg(block, q_ty);

    let (op1, _) = ctx.create_op(
        "quantum.h", "quantum", vec![q], vec![q_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op1);

    let (op2, _) = ctx.create_op(
        "quantum.x", "quantum", vec![q], vec![q_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op2);

    let result = lift_core::verifier::verify(&ctx);
    assert!(result.is_err(), "must detect qubit linearity violation");
}

// ═══════════════════════════════════════════════════
//  PRINTER TESTS
// ═══════════════════════════════════════════════════

#[test]
fn test_printer_tensor_type_format() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(
        vec![Dimension::Constant(2), Dimension::Constant(3)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let formatted = format!("{}", ctx.resolve_type(ty));
    assert!(formatted.contains("tensor"));
    assert!(formatted.contains("2"));
    assert!(formatted.contains("3"));
    assert!(formatted.contains("f32"));
}

#[test]
fn test_printer_full_program() {
    let mut ctx = Context::new();
    let mod_idx = ctx.create_module("printer_test");
    let ty = ctx.make_tensor_type(
        vec![Dimension::Constant(4)], DataType::FP32, MemoryLayout::Contiguous,
    );
    let region = ctx.create_region();
    let block = ctx.create_block();
    ctx.add_block_to_region(region, block);
    let x = ctx.create_block_arg(block, ty);

    let (relu, _) = ctx.create_op(
        "tensor.relu", "tensor", vec![x], vec![ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, relu);

    let func_name = ctx.intern_string("test_fn");
    let func = lift_core::FunctionData {
        name: func_name, params: vec![ty], returns: vec![ty],
        body: Some(region), location: Location::unknown(), is_declaration: false,
    };
    ctx.add_function_to_module(mod_idx, func);

    let output = lift_core::printer::print_ir(&ctx);
    assert!(output.contains("module @printer_test"));
    assert!(output.contains("func @test_fn"));
    assert!(output.contains("tensor.relu"));
}

// ═══════════════════════════════════════════════════
//  PASS MANAGER TESTS
// ═══════════════════════════════════════════════════

#[test]
fn test_pass_manager_ordering() {
    let mut pm = PassManager::new();
    pm.add_pass(Box::new(lift_opt::Canonicalize));
    pm.add_pass(Box::new(lift_opt::ConstantFolding));
    pm.add_pass(Box::new(lift_opt::DeadCodeElimination));
    assert_eq!(pm.num_passes(), 3);

    let mut ctx = Context::new();
    let results = pm.run_all(&mut ctx);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, "canonicalize");
    assert_eq!(results[1].0, "constant-folding");
    assert_eq!(results[2].0, "dce");
}

#[test]
fn test_analysis_cache() {
    let mut cache = AnalysisCache::new();
    cache.insert("count", 42u64);
    assert_eq!(*cache.get::<u64>("count").unwrap(), 42);
    cache.invalidate(vec!["count"]);
    assert!(cache.get::<u64>("count").is_none());
}

#[test]
fn test_context_snapshot() {
    let mut ctx = Context::new();
    let snap0 = ctx.snapshot();
    assert_eq!(snap0.num_values, 0);
    assert_eq!(snap0.num_ops, 0);

    let ty = ctx.make_float_type(32);
    let block = ctx.create_block();
    let _v = ctx.create_value(ty, None, DefSite::BlockArg { block, arg_index: 0 });
    let snap1 = ctx.snapshot();
    assert_eq!(snap1.num_values, 1);
    assert_eq!(snap1.num_blocks, 1);
}

// ═══════════════════════════════════════════════════
//  ALL DTYPE BYTE SIZES
// ═══════════════════════════════════════════════════

#[test]
fn test_all_dtype_byte_sizes() {
    assert_eq!(DataType::FP64.byte_size(), 8);
    assert_eq!(DataType::FP32.byte_size(), 4);
    assert_eq!(DataType::FP16.byte_size(), 2);
    assert_eq!(DataType::BF16.byte_size(), 2);
    assert_eq!(DataType::FP8E4M3.byte_size(), 1);
    assert_eq!(DataType::FP8E5M2.byte_size(), 1);
    assert_eq!(DataType::INT64.byte_size(), 8);
    assert_eq!(DataType::INT32.byte_size(), 4);
    assert_eq!(DataType::INT16.byte_size(), 2);
    assert_eq!(DataType::INT8.byte_size(), 1);
    assert_eq!(DataType::UINT8.byte_size(), 1);
    assert_eq!(DataType::Bool.byte_size(), 1);
}

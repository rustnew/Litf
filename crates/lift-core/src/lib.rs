//! LIFT Core: SSA-based intermediate representation foundation.
//!
//! This crate provides the core data structures and algorithms for the LIFT
//! compiler framework, including types, values, operations, blocks, regions,
//! functions, modules, attributes, a verifier, an IR printer, a pass manager,
//! and a dialect registry.

pub mod types;
pub mod values;
pub mod operations;
pub mod blocks;
pub mod regions;
pub mod functions;
pub mod module;
pub mod context;
pub mod attributes;
pub mod location;
pub mod interning;
pub mod verifier;
pub mod printer;
pub mod dialect;
pub mod pass;
pub mod model_builder;

pub use context::Context;
pub use types::{TypeId, CoreType, DataType};
pub use values::{ValueKey, ValueData, DefSite};
pub use operations::{OpKey, OperationData};
pub use blocks::{BlockKey, BlockData};
pub use regions::{RegionKey, RegionData};
pub use functions::FunctionData;
pub use module::ModuleData;
pub use attributes::{Attribute, Attributes};
pub use location::Location;
pub use interning::{StringId, StringInterner, TypeInterner};
pub use dialect::{Dialect, DialectRegistry};
pub use verifier::{Verifier, VerifyError};
pub use printer::Printer;
pub use pass::{Pass, PassManager, PassResult, AnalysisCache};
pub use model_builder::{ModelBuilder, tensor, tensor_1d, tensor_2d, tensor_3d, tensor_4d, ModelType, build_lith_config};

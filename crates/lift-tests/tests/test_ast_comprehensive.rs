/// Comprehensive tests for lift-ast: Lexer, Parser, Builder
use lift_ast::*;

// ═══════════════════════════════════════════════════
//  LEXER EDGE CASES
// ═══════════════════════════════════════════════════

#[test]
fn test_lex_empty_input() {
    let mut lexer = Lexer::new("");
    let tokens = lexer.tokenize().to_vec();
    assert_eq!(tokens.len(), 1);
    assert!(matches!(tokens[0].kind, TokenKind::Eof));
}

#[test]
fn test_lex_only_whitespace() {
    let mut lexer = Lexer::new("   \t\t   ");
    let tokens = lexer.tokenize().to_vec();
    assert_eq!(tokens.len(), 1);
}

#[test]
fn test_lex_all_punctuation() {
    let mut lexer = Lexer::new("( ) { } [ ] < > , : ; = . *");
    let tokens = lexer.tokenize().to_vec();
    assert!(matches!(tokens[0].kind, TokenKind::LParen));
    assert!(matches!(tokens[1].kind, TokenKind::RParen));
    assert!(matches!(tokens[2].kind, TokenKind::LBrace));
    assert!(matches!(tokens[3].kind, TokenKind::RBrace));
    assert!(matches!(tokens[4].kind, TokenKind::LBracket));
    assert!(matches!(tokens[5].kind, TokenKind::RBracket));
    assert!(matches!(tokens[6].kind, TokenKind::LAngle));
    assert!(matches!(tokens[7].kind, TokenKind::RAngle));
    assert!(matches!(tokens[8].kind, TokenKind::Comma));
    assert!(matches!(tokens[9].kind, TokenKind::Colon));
    assert!(matches!(tokens[10].kind, TokenKind::Semicolon));
    assert!(matches!(tokens[11].kind, TokenKind::Equal));
    assert!(matches!(tokens[12].kind, TokenKind::Dot));
    assert!(matches!(tokens[13].kind, TokenKind::Star));
}

#[test]
fn test_lex_keywords() {
    let mut lexer = Lexer::new("module func return if else true false tensor qubit bit hamiltonian void index");
    let tokens = lexer.tokenize().to_vec();
    assert!(matches!(tokens[0].kind, TokenKind::Module));
    assert!(matches!(tokens[1].kind, TokenKind::Func));
    assert!(matches!(tokens[2].kind, TokenKind::Return));
    assert!(matches!(tokens[3].kind, TokenKind::If));
    assert!(matches!(tokens[4].kind, TokenKind::Else));
    assert!(matches!(tokens[5].kind, TokenKind::True));
    assert!(matches!(tokens[6].kind, TokenKind::False));
    assert!(matches!(tokens[7].kind, TokenKind::Tensor));
    assert!(matches!(tokens[8].kind, TokenKind::Qubit));
    assert!(matches!(tokens[9].kind, TokenKind::Bit));
    assert!(matches!(tokens[10].kind, TokenKind::Hamiltonian));
    assert!(matches!(tokens[11].kind, TokenKind::Void));
    assert!(matches!(tokens[12].kind, TokenKind::Index));
}

#[test]
fn test_lex_numbers() {
    let mut lexer = Lexer::new("0 42 -1 3.14 -0.5 2.5e-3");
    let tokens = lexer.tokenize().to_vec();
    assert!(matches!(tokens[0].kind, TokenKind::Integer(0)));
    assert!(matches!(tokens[1].kind, TokenKind::Integer(42)));
    assert!(matches!(tokens[2].kind, TokenKind::Integer(-1)));
    assert!(matches!(tokens[3].kind, TokenKind::Float(v) if (v - 3.14).abs() < 1e-10));
    assert!(matches!(tokens[4].kind, TokenKind::Float(v) if (v + 0.5).abs() < 1e-10));
    assert!(matches!(tokens[5].kind, TokenKind::Float(v) if (v - 2.5e-3).abs() < 1e-10));
}

#[test]
fn test_lex_identifiers() {
    let mut lexer = Lexer::new("@main %x %result ^bb0 my_var");
    let tokens = lexer.tokenize().to_vec();
    assert!(matches!(&tokens[0].kind, TokenKind::AtIdent(s) if s == "main"));
    assert!(matches!(&tokens[1].kind, TokenKind::PercentIdent(s) if s == "x"));
    assert!(matches!(&tokens[2].kind, TokenKind::PercentIdent(s) if s == "result"));
    assert!(matches!(&tokens[3].kind, TokenKind::CaretIdent(s) if s == "bb0"));
    assert!(matches!(&tokens[4].kind, TokenKind::Ident(s) if s == "my_var"));
}

#[test]
fn test_lex_string_literal() {
    let mut lexer = Lexer::new("\"tensor.add\"");
    let tokens = lexer.tokenize().to_vec();
    assert!(matches!(&tokens[0].kind, TokenKind::StringLiteral(s) if s == "tensor.add"));
}

#[test]
fn test_lex_arrow() {
    let mut lexer = Lexer::new("-> ->");
    let tokens = lexer.tokenize().to_vec();
    assert!(matches!(tokens[0].kind, TokenKind::Arrow));
    assert!(matches!(tokens[1].kind, TokenKind::Arrow));
}

#[test]
fn test_lex_dialect_directive() {
    let mut lexer = Lexer::new("#dialect tensor");
    let tokens = lexer.tokenize().to_vec();
    assert!(matches!(&tokens[0].kind, TokenKind::HashDialect(s) if s == "tensor"));
}

#[test]
fn test_lex_comment_ignored() {
    let mut lexer = Lexer::new("module // this is ignored\n@test");
    let tokens = lexer.tokenize().to_vec();
    assert!(matches!(tokens[0].kind, TokenKind::Module));
    assert!(matches!(&tokens[1].kind, TokenKind::AtIdent(s) if s == "test"));
}

#[test]
fn test_lex_tensor_dimension_separator() {
    let mut lexer = Lexer::new("tensor < 1 x 784 x f32 >");
    let tokens = lexer.tokenize().to_vec();
    assert!(matches!(tokens[0].kind, TokenKind::Tensor));
    assert!(matches!(tokens[1].kind, TokenKind::LAngle));
    assert!(matches!(tokens[2].kind, TokenKind::Integer(1)));
    assert!(matches!(&tokens[3].kind, TokenKind::Ident(s) if s == "x"));
    assert!(matches!(tokens[4].kind, TokenKind::Integer(784)));
}

#[test]
fn test_lex_compact_tensor_type() {
    let mut lexer = Lexer::new("tensor<1x784xf32>");
    let tokens = lexer.tokenize().to_vec();
    assert!(matches!(tokens[0].kind, TokenKind::Tensor));
    assert!(matches!(tokens[1].kind, TokenKind::LAngle));
    assert!(matches!(tokens[2].kind, TokenKind::Integer(1)));
    assert!(matches!(&tokens[3].kind, TokenKind::Ident(s) if s == "x"));
    assert!(matches!(tokens[4].kind, TokenKind::Integer(784)));
    assert!(matches!(&tokens[5].kind, TokenKind::Ident(s) if s == "x"));
    assert!(matches!(&tokens[6].kind, TokenKind::Ident(s) if s == "f32"));
    assert!(matches!(tokens[7].kind, TokenKind::RAngle));
}

#[test]
fn test_lex_high_dimensional_tensor() {
    let mut lexer = Lexer::new("tensor<2x3x4x5x6xf32>");
    let tokens = lexer.tokenize().to_vec();
    let ints: Vec<i64> = tokens.iter().filter_map(|t| match &t.kind {
        TokenKind::Integer(n) => Some(*n),
        _ => None,
    }).collect();
    assert_eq!(ints, vec![2, 3, 4, 5, 6]);
}

// ═══════════════════════════════════════════════════
//  PARSER TESTS
// ═══════════════════════════════════════════════════

fn parse_program(src: &str) -> Program {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().to_vec();
    assert!(lexer.errors().is_empty(), "lexer errors: {:?}", lexer.errors());
    let mut parser = Parser::new(tokens);
    parser.parse().expect("parse failed")
}

#[test]
fn test_parse_minimal_module() {
    let prog = parse_program("module @empty {}");
    assert_eq!(prog.modules.len(), 1);
    assert_eq!(prog.modules[0].name, "empty");
    assert_eq!(prog.modules[0].functions.len(), 0);
}

#[test]
fn test_parse_func_with_body() {
    let prog = parse_program(r#"
module @m {
    func @f(%x: tensor<4xf32>) -> tensor<4xf32> {
        return %x
    }
}
"#);
    assert_eq!(prog.modules[0].functions.len(), 1);
    assert_eq!(prog.modules[0].functions[0].name, "f");
    assert_eq!(prog.modules[0].functions[0].params.len(), 1);
}

#[test]
fn test_parse_multiple_functions() {
    let prog = parse_program(r#"
module @m {
    func @f1(%x: tensor<4xf32>) -> tensor<4xf32> { return %x }
    func @f2(%a: qubit) -> qubit { return %a }
}
"#);
    assert_eq!(prog.modules[0].functions.len(), 2);
}

#[test]
fn test_parse_tensor_operations() {
    let prog = parse_program(r#"
#dialect tensor
module @t {
    func @forward(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %z = "tensor.add"(%x, %y) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
        return %z
    }
}
"#);
    let func = &prog.modules[0].functions[0];
    assert_eq!(func.params.len(), 2);
    assert!(!func.body.is_empty());
}

#[test]
fn test_parse_quantum_operations() {
    let prog = parse_program(r#"
#dialect quantum
module @q {
    func @ghz(%q0: qubit, %q1: qubit, %q2: qubit) -> (qubit, qubit, qubit) {
        %q3 = "quantum.h"(%q0) : (qubit) -> qubit
        %q4, %q5 = "quantum.cx"(%q3, %q1) : (qubit, qubit) -> (qubit, qubit)
        %q6, %q7 = "quantum.cx"(%q5, %q2) : (qubit, qubit) -> (qubit, qubit)
        return %q4, %q6, %q7
    }
}
"#);
    let func = &prog.modules[0].functions[0];
    assert_eq!(func.params.len(), 3);
    assert_eq!(func.returns.len(), 3);
}

#[test]
fn test_parse_multiple_return_types() {
    let prog = parse_program(r#"
module @m {
    func @split(%x: tensor<8xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
        return %x, %x
    }
}
"#);
    assert_eq!(prog.modules[0].functions[0].returns.len(), 2);
}

#[test]
fn test_parse_various_dtypes() {
    let prog = parse_program(r#"
module @m {
    func @dtypes(%a: tensor<4xf16>, %b: tensor<4xbf16>, %c: tensor<4xi8>, %d: tensor<4xf64>) -> tensor<4xf32> {
        return %a
    }
}
"#);
    assert_eq!(prog.modules[0].functions[0].params.len(), 4);
}

#[test]
fn test_parse_large_shapes() {
    let prog = parse_program(r#"
module @m {
    func @big(%x: tensor<1x1024x4096xf32>) -> tensor<1x1024x4096xf32> {
        return %x
    }
}
"#);
    assert_eq!(prog.modules[0].functions[0].params.len(), 1);
}

// ═══════════════════════════════════════════════════
//  BUILDER INTEGRATION TESTS
// ═══════════════════════════════════════════════════

fn parse_and_build(src: &str) -> lift_core::Context {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().to_vec();
    assert!(lexer.errors().is_empty(), "lexer errors: {:?}", lexer.errors());
    let mut parser = Parser::new(tokens);
    let program = parser.parse().expect("parse failed");
    let mut ctx = lift_core::Context::new();
    let mut builder = IrBuilder::new();
    builder.build_program(&mut ctx, &program).expect("build failed");
    ctx
}

#[test]
fn test_build_and_verify_mlp() {
    let ctx = parse_and_build(r#"
#dialect tensor
module @mlp {
    func @forward(%x: tensor<1x784xf32>, %w: tensor<784x256xf32>, %b: tensor<256xf32>) -> tensor<1x256xf32> {
        %h1 = "tensor.matmul"(%x, %w) : (tensor<1x784xf32>, tensor<784x256xf32>) -> tensor<1x256xf32>
        %h2 = "tensor.add"(%h1, %b) : (tensor<1x256xf32>, tensor<256xf32>) -> tensor<1x256xf32>
        %h3 = "tensor.relu"(%h2) : (tensor<1x256xf32>) -> tensor<1x256xf32>
        return %h3
    }
}
"#);
    assert!(lift_core::verifier::verify(&ctx).is_ok());
    assert_eq!(ctx.modules.len(), 1);
    assert_eq!(ctx.ops.len(), 4);
}

#[test]
fn test_build_and_verify_quantum() {
    let ctx = parse_and_build(r#"
#dialect quantum
module @bell {
    func @create(%q0: qubit, %q1: qubit) -> (qubit, qubit) {
        %q2 = "quantum.h"(%q0) : (qubit) -> qubit
        %q3, %q4 = "quantum.cx"(%q2, %q1) : (qubit, qubit) -> (qubit, qubit)
        return %q3, %q4
    }
}
"#);
    assert!(lift_core::verifier::verify(&ctx).is_ok());
    assert_eq!(ctx.ops.len(), 3);
}

#[test]
fn test_build_print_roundtrip() {
    let ctx = parse_and_build(r#"
#dialect tensor
module @test {
    func @relu(%x: tensor<4xf32>) -> tensor<4xf32> {
        %y = "tensor.relu"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %y
    }
}
"#);
    let printed = lift_core::printer::print_ir(&ctx);
    assert!(printed.contains("module @test"));
    assert!(printed.contains("tensor.relu"));
    assert!(printed.contains("tensor<4xf32>"));
    assert!(printed.contains("core.return"));
}

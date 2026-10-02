grammar Tepl;

// -----------------------------------------------------------------------------
// File structure: imports, operation visibility, and top-level declarations
// -----------------------------------------------------------------------------

// Whitespace, including newlines, is insignificant. A file contains imports,
// dialects, and/or rules (concrete, abstract, or inherited).
// Bindings, inheritance expansion, and typing are validated later.
program
    : (importDecl | useDecl | dialectDecl | ruleDecl)+ EOF
    ;

importDecl
    : IMPORT STRING ';'
    | FROM STRING IMPORT ID (AS ID)? ';'
    | FROM STRING IMPORT '{' ruleImportNames '}' ';'
    ;

ruleImportNames
    : ID (',' ID)*
    ;

useDecl
    : USE ID ('::' '{' opName (',' opName)* '}')? ';'
    ;

// -----------------------------------------------------------------------------
// Shared operation names: used by imports, dialects, and rewrite rules
// -----------------------------------------------------------------------------

opName
    : ID
    | ALIAS
    ;

opRef
    : opName ('.' opName)?
    ;

// -----------------------------------------------------------------------------
// Dialect declarations: operations, operand signatures, and attribute schemas
// -----------------------------------------------------------------------------

dialectDecl
    : DIALECT ID '{' (attrsDecl | opDecl)* '}'
    ;

attrsDecl
    : ATTRS ID '{' attrField* '}'
    ;

opDecl
    : OP opName '(' operandDecls? ')' '->' ID (';' | '{' opProperties? '}')
    ;

operandDecls
    : operandDecl (',' operandDecl)* (',' variadicOperand)?
    | variadicOperand
    ;

operandDecl
    : ID ':' ID
    ;

variadicOperand
    : ID ':' ID ELLIPSIS
    ;

opProperties
    : aliasProperty attrsProperty?
    | attrsProperty aliasProperty?
    ;

aliasProperty
    : ALIAS ':' ID ';'
    ;

attrsProperty
    : ATTRS ':' ID ';'                 # SharedAttrsProperty
    | ATTRS '{' attrField* '}'         # InlineAttrsProperty
    ;

attrField
    : ID ':' attrType attrDefault? ';'
    ;

attrType
    : ID ('[' ']')?
    ;

attrDefault
    : '=' '[' ']'
    ;

// -----------------------------------------------------------------------------
// Rewrite rules: concrete and abstract definitions, parameters, and inheritance
// -----------------------------------------------------------------------------

ruleDecl
    : RULE ID rewriteBody
    | ABSTRACT RULE ID '(' ruleParameters? ')' rewriteBody
    | RULE ID inheritanceClause (';' | inheritedBody)
    ;

rewriteBody
    : '{' shapeDecl* graphExpr ARROW rhsGraphExpr whereBlock? deriveBlock? '}'
    ;

inheritanceClause
    : EXTENDS ID '(' ruleBindings? ')'
    ;

ruleParameters
    : ruleParameter (',' ruleParameter)*
    ;

ruleParameter
    : ID ':' (OP | FN) '<' '(' signatureTypes? ')' '->' signatureType '>'
    ;

signatureTypes
    : signatureType (',' signatureType)*
    ;

signatureType
    : ID
    | SCALAR
    ;

ruleBindings
    : ruleBinding (',' ruleBinding)*
    ;

ruleBinding
    : ID '=' opRef
    ;

// Instances add restrictions to the inherited pattern.
inheritedBody
    : '{' shapeDecl* whereBlock? '}'
    ;

// -----------------------------------------------------------------------------
// Rule declarations: tensor shapes and scalars
// -----------------------------------------------------------------------------

shapeDecl
    : ID ':' dtypeName? '[' shapeElements? ']'           # TensorDecl
    | ID ':' SCALAR                           # ScalarDecl
    ;

dtypeName
    : ID
    ;

// A sequence may appear anywhere, but each shape has at most one sequence.
shapeElements
    : scalarDim (',' scalarDim)* (',' sequenceDim (',' scalarDim)*)?
    | sequenceDim (',' scalarDim)*
    ;

scalarDim
    : ID
    | WILDCARD
    ;

sequenceDim
    : ID? ELLIPSIS
    ;

// -----------------------------------------------------------------------------
// Rule graph expressions: variables, literals, bindings, and operator applications
// -----------------------------------------------------------------------------

// LHS patterns may bind matched values.
graphExpr
    : binding                                # BareBindingGraph
    | ID                                     # VariableGraph
    | ('+' | '-')? (INT | FLOAT) (':' dtypeName)? # NumberGraph
    | '(' binding ')'                        # ParenthesizedBindingGraph
    | '(' GET '[' INT ']' graphExpr ')'       # GetGraph
    | '(' opRef attribute? graphExpr* ')'     # OperatorGraph
    ;

// RHS expressions construct operations from LHS captures; they introduce no bindings.
rhsGraphExpr
    : ID                                     # RhsVariableGraph
    | ('+' | '-')? (INT | FLOAT) (':' dtypeName)? # RhsNumberGraph
    | '(' GET '[' INT ']' rhsGraphExpr ')'    # RhsGetGraph
    | '(' opRef attribute? rhsGraphExpr* ')'  # RhsOperatorGraph
    ;

binding
    : LET ID '=' graphExpr
    ;

attribute
    : '[' attrRef ']'
    ;

attrRef
    : '@' ID
    ;

// -----------------------------------------------------------------------------
// Rule semantics: legality conditions and derived metadata
// -----------------------------------------------------------------------------

whereBlock
    : WHERE '{' (constraintExpr ';')* '}'
    ;

deriveBlock
    : DERIVE '{' (attrRef '=' constraintExpr ';')* '}'
    ;

// -----------------------------------------------------------------------------
// Constraint expressions: shared by where and derive
// -----------------------------------------------------------------------------

// Host calls and constraints use conventional infix syntax. Each level has its
// own rule so precedence remains explicit and independent of graph expressions.
constraintExpr
    : logicalOr
    ;

logicalOr
    : logicalAnd ('||' logicalAnd)*
    ;

logicalAnd
    : equality ('&&' equality)*
    ;

equality
    : comparison (('==' | '!=') comparison)?
    ;

comparison
    : additive (('<' | '<=' | '>' | '>=') additive)?
    ;

additive
    : multiplicative (('+' | '-') multiplicative)*
    ;

multiplicative
    : unary (('*' | '/' | '%') unary)*
    ;

unary
    : ('!' | '+' | '-') unary
    | primary
    ;

primary
    : ID '(' arguments? ')'                  # CallPrimary
    | ID                                     # NamePrimary
    | attrRef                                # AttributePrimary
    | INT                                    # IntegerPrimary
    | FLOAT                                  # FloatPrimary
    | TRUE                                   # TruePrimary
    | FALSE                                  # FalsePrimary
    | '(' constraintExpr ')'                 # GroupPrimary
    ;

arguments
    : constraintExpr (',' constraintExpr)*
    ;

// -----------------------------------------------------------------------------
// Lexer: keywords, punctuation, identifiers, and literals
// -----------------------------------------------------------------------------

RULE: 'rule';
ABSTRACT: 'abstract';
EXTENDS: 'extends';
FN: 'fn';
LET: 'let';
IMPORT: 'import';
FROM: 'from';
AS: 'as';
USE: 'use';
DIALECT: 'dialect';
OP: 'op';
ATTRS: 'attrs';
ALIAS: 'alias';
WHERE: 'where';
DERIVE: 'derive';
SCALAR: 'scalar';
GET: 'get';
TRUE: 'true';
FALSE: 'false';
ARROW: '=>';
ELLIPSIS: '...';
WILDCARD: '_';
ID: [a-zA-Z_] [a-zA-Z_0-9]*;
FLOAT: [0-9]+ '.' [0-9]+;
INT: [0-9]+;
// Keep unsupported numeric suffixes (including exponents) from becoming a
// number followed by an extra graph variable.
INVALID_NUMBER: [0-9]+ ('.' [0-9]*)? [a-zA-Z_] [a-zA-Z_0-9]*;
STRING: '"' (~["\\\r\n] | '\\' ["\\])* '"';

// -----------------------------------------------------------------------------
// Lexer: comments and whitespace
// -----------------------------------------------------------------------------

LINE_COMMENT: '//' ~[\r\n]* -> skip;
BLOCK_COMMENT: '/*' .*? '*/' -> skip;
WS: [ \t\r\n]+ -> skip;

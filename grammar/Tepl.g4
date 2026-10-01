grammar Tepl;

// Whitespace, including newlines, is insignificant. A file contains imports,
// dialects, and/or single-root rules. Bindings and typing are validated later.
program
    : (importDecl | useDecl | dialectDecl | ruleDecl)+ EOF
    ;

importDecl
    : IMPORT STRING ';'
    | FROM STRING IMPORT ID (AS ID)? ';'
    ;

useDecl
    : USE ID ('::' '{' opName (',' opName)* '}')? ';'
    ;

dialectDecl
    : DIALECT ID '{' (attrsDecl | opDecl)* '}'
    ;

attrsDecl
    : ATTRS ID '{' attrField* '}'
    ;

opDecl
    : OP opName '(' operandDecls? ')' '->' ID (';' | '{' opProperty* '}')
    ;

opName
    : ID
    | ALIAS
    ;

opRef
    : opName ('.' opName)?
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

opProperty
    : ALIAS ':' ID ';'                 # AliasProperty
    | ATTRS ':' ID ';'                 # SharedAttrsProperty
    | ATTRS '{' attrField* '}'         # InlineAttrsProperty
    ;

attrField
    : ID ':' ID ('[' ']')? ('=' '[' ']')? ';'
    ;

ruleDecl
    : RULE ID '{' shapeDecl* graphExpr ARROW graphExpr whereBlock? deriveBlock? '}'
    ;

shapeDecl
    : ID ':' '[' shapeElements? ']'           # TensorDecl
    | ID ':' SCALAR                           # ScalarDecl
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

graphExpr
    : binding                                # BareBindingGraph
    | binderRef                              # BinderReferenceGraph
    | ID                                     # VariableGraph
    | '(' binding ')'                        # ParenthesizedBindingGraph
    | '(' GET '[' INT ']' graphExpr ')'       # GetGraph
    | '(' opRef attribute? graphExpr* ')'     # OperatorGraph
    ;

binding
    : binderRef '=' graphExpr
    ;

binderRef
    : '?' ID
    ;

attribute
    : '[' attrRef ']'
    ;

attrRef
    : '@' ID
    ;

whereBlock
    : WHERE '{' (constraintExpr ';')* '}'
    ;

deriveBlock
    : DERIVE '{' (attrRef '=' constraintExpr ';')* '}'
    ;

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
    | binderRef                              # BinderPrimary
    | INT                                    # IntegerPrimary
    | TRUE                                   # TruePrimary
    | FALSE                                  # FalsePrimary
    | '(' constraintExpr ')'                 # GroupPrimary
    ;

arguments
    : constraintExpr (',' constraintExpr)*
    ;

RULE: 'rule';
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
INT: [0-9]+;
STRING: '"' (~["\\\r\n] | '\\' ["\\])* '"';

LINE_COMMENT: '//' ~[\r\n]* -> skip;
BLOCK_COMMENT: '/*' .*? '*/' -> skip;
WS: [ \t\r\n]+ -> skip;

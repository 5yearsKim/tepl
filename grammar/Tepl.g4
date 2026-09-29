grammar Tepl;

// Whitespace, including newlines, is insignificant. A file contains one or more
// single-root rules. Bindings and typing are validated after parsing.
program
    : ruleDecl+ EOF
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
    | '(' ID attribute? graphExpr* ')'        # OperatorGraph
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

LINE_COMMENT: '//' ~[\r\n]* -> skip;
BLOCK_COMMENT: '/*' .*? '*/' -> skip;
WS: [ \t\r\n]+ -> skip;

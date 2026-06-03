// Code-domain knowledge graph schema (kùzu-style).
// Used by code labs (data_shape=code_repo).
//
// Node types capture the structural entities of a codebase + the
// reasoning artifacts a code lab produces (refactor decisions,
// killed approaches, identified bugs).
//
// Edge types capture both structural relationships (CALLS, IMPORTS,
// TESTS) and reasoning relationships (REPLACES, REFUTES, FIXES).

NODE TABLE Module (
    id STRING PRIMARY KEY,
    path STRING,
    language STRING,
    loc INT
);

NODE TABLE Symbol (
    id STRING PRIMARY KEY,
    name STRING,
    qualified_name STRING,
    kind STRING,             // function | class | method | const | type
    signature STRING,
    module_id STRING
);

NODE TABLE Test (
    id STRING PRIMARY KEY,
    symbol_id STRING,
    covers_symbol_id STRING,
    coverage_pct DOUBLE
);

NODE TABLE Refactor (
    id STRING PRIMARY KEY,
    target_symbol_id STRING,
    rationale STRING,
    cycle INT,
    status STRING            // proposed | implemented | reverted
);

NODE TABLE KilledApproach (
    id STRING PRIMARY KEY,
    description STRING,
    reason STRING,
    killed_at_cycle INT
);

NODE TABLE Bug (
    id STRING PRIMARY KEY,
    title STRING,
    severity STRING,         // low | medium | high | critical
    file_path STRING,
    line INT,
    fixed_in_commit STRING   // null if open
);

NODE TABLE Commit (
    sha STRING PRIMARY KEY,
    ts INT,
    author STRING,
    message STRING
);

// ── Edges ──────────────────────────────────────────────────────

REL TABLE CALLS (
    FROM Symbol TO Symbol,
    line INT
);

REL TABLE IMPORTS (
    FROM Module TO Module
);

REL TABLE CONTAINS (
    FROM Module TO Symbol
);

REL TABLE COVERS (
    FROM Test TO Symbol,
    coverage_pct DOUBLE
);

REL TABLE REPLACES (
    FROM Refactor TO Symbol,
    direction STRING         // 'new' or 'old'
);

REL TABLE FIXES (
    FROM Commit TO Bug
);

REL TABLE INTRODUCES_BUG (
    FROM Commit TO Bug
);

REL TABLE CONFLICTS (
    FROM KilledApproach TO Symbol
);

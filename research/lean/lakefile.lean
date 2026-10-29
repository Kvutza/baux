import Lake

open System Lake DSL

package baux_research where version := v!"0.1.0"

require mathlib from git "https://github.com/leanprover-community/mathlib4.git"@"v4.35.0"

require repl from git "https://github.com/leanprover-community/repl.git"@"v4.35.0"

require aeneas from git
  "https://github.com/AeneasVerif/aeneas.git"@"e62dd83d133d9043e7d3628096838b7430f36565"/"backends/lean"

require floatlib from git "https://github.com/lean-dojo/FloatLib.git"@"51e6940eaa4014691be91f663a49ca7239ca0c07"

@[default_target] lean_lib BauxResearch where defaultFacets := #[`lean_lib.leanArts, `lean_lib.static]

lean_exe «baux-research» where root := `Main

target numericalShim pkg : FilePath := do
  let source ← inputFile (pkg.dir / "numerical.c") true
  buildLeanO (pkg.buildDir / "native" / "numerical.o") source #[]
    #["-fPIC", "-DLEAN_EXPORTING"]

lean_exe numerical where
  supportInterpreter := true
  root := `BauxResearch.Semantics.NumericalRuntime
  exeName := "libbaux_numerical.so"
  moreLinkObjs := #[numericalShim]
  moreLinkArgs := #["-shared"]

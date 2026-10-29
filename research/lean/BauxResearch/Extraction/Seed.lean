import BauxResearch.Generated.Seed
import BauxResearch.Generated.Optimizer.Funs

/-!
# Extracted Sobol state-seed correspondence

The upstream ENN 0.4.1 function and the local semantics function are independent
Charon/Aeneas extractions. Production imports the local semantics function
directly, so this theorem checks the integer seed mixer used by the optimizer.
-/

namespace BauxResearch.Extraction.Seed

open Aeneas.Std

theorem seedMatch (base : U64) (restart obs arms : Usize) :
    BauxResearch.Generated.Seed.sobolSeed base restart obs arms =
      BauxResearch.Generated.Optimizer.random.sobol_seed base restart obs arms := by
  rfl

#print axioms seedMatch

end BauxResearch.Extraction.Seed

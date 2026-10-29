import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.Runtime
import FloatLib.Floats.Formats.BinaryInterchange.Operations.Compare.Runtime
import FloatLib.Floats.Formats.BinaryInterchange.Format.Catalog

/-! Executable IEEE arithmetic checker. Correctness is proved in Numerical.lean.
No external native floating-point result is trusted by this predicate. -/

namespace BauxResearch.Semantics.Numerical

open FloatLib.Floats.Formats.BinaryInterchange

abbrev Word := Model FloatFormat.binary64

/-- Interpret every raw word, preserving signed zero and NaN payload bits. -/
def word (bits : UInt64) : Word := Model.ofBits bits.toBitVec

/-- Reference threshold with the production order: square root, then product. -/
def threshold (multiplier variance : Word) : Word :=
  Model.mul multiplier (Model.sqrt variance)

/-- A decidable domain check; no assumption that a variance is well formed. -/
def domain (multiplier variance actual : Word) : Bool :=
  Model.isFinite multiplier && Model.isFinite variance && Model.isFinite actual &&
    (Model.isZero variance || !Model.signBit variance)

/-- Compare actual result bits with the fully rounded reference computation. -/
def checkThreshold (multiplier variance actual : Word) : Bool :=
  domain multiplier variance actual && decide (actual = threshold multiplier variance)

/-- Check arithmetic and the Boolean supplied by the actual optimizer gate. -/
def checkDecision (gain multiplier variance actual : Word) (accepted : Bool) : Bool :=
  checkThreshold multiplier variance actual && Model.isFinite gain &&
    decide (accepted = decide (Model.compare actual gain = some .lt))

/-- All source threshold orders; the third branch adds rounded roots. -/
def sourceScale (kind : UInt8) (left right : Word) : Word :=
  if kind = 0 then Model.sqrt left
  else if kind = 1 then Model.sqrt (Model.add left right)
  else Model.add (Model.sqrt left) (Model.sqrt right)

def sourceDomain (multiplier left right actual : Word) : Bool :=
  domain multiplier left actual && Model.isFinite right &&
    (Model.isZero right || !Model.signBit right)

/-- Check one source expression and reject unknown tags. -/
def sourceCore (kind : UInt8) (multiplier left right actual : Word) : Bool :=
  if kind = 0 then checkThreshold multiplier left actual
  else if kind = 1 then checkThreshold multiplier (Model.add left right) actual
  else if kind = 2 then
    Model.isFinite (sourceScale kind left right) &&
      decide (actual = Model.mul multiplier (sourceScale kind left right))
  else false

def sourceCheck (kind : UInt8) (multiplier left right actual : Word) : Bool :=
  sourceDomain multiplier left right actual && sourceCore kind multiplier left right actual

/-- Require source arithmetic and the actual optimizer comparison to agree. -/
def sourceDecision (kind : UInt8) (gain multiplier left right actual : Word)
    (accepted : Bool) : Bool :=
  sourceCheck kind multiplier left right actual && Model.isFinite gain &&
    decide (accepted = decide (Model.compare actual gain = some .lt))

/-- C-ABI: kind 0 paired, 1 combined variance, 2 sum of roots. -/
@[export baux_numerical]
def exportCheck (kind : UInt8) (gain multiplier left right actual : UInt64)
    (accepted : Bool) : Bool :=
  sourceDecision kind (word gain) (word multiplier) (word left) (word right) (word actual) accepted

end BauxResearch.Semantics.Numerical

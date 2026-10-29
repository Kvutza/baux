import BauxResearch

open BauxResearch

def main : IO Unit := do
  IO.println s!"target.parameters: {currentTarget.parameterCount}"
  IO.println s!"target.context_tokens: {currentTarget.contextTokens}"
  IO.println s!"target.candidates: {currentTarget.candidates}"
  IO.println s!"target.neighbors: {currentTarget.neighbors}"
  IO.println s!"target.objective: {repr currentTarget.objective}"
  IO.println s!"target.sampling: {repr currentTarget.sampling}"
  IO.println s!"target.transaction_target_ms: {currentTarget.targetMs}"
  IO.println s!"target.transaction_ceiling_ms: {currentTarget.ceilingMs}"
  IO.println s!"target.dense_flops: {Forward.Work.roundFlops Forward.Program.current}"
  IO.println s!"artifact.contract_cpu: {Semantics.Artifact.contractOk Semantics.Artifact.defaultMachine}"
  IO.println s!"artifact.contract_gpu: {Semantics.Artifact.contractOk Semantics.Artifact.gpuMachine}"
  IO.println s!"artifact.digest_valid: {Semantics.Artifact.digestValid Semantics.Artifact.pinnedDigest}"
  IO.println s!"artifact.functions_verified: {Semantics.Artifact.pinnedDigest.funcCount}"
  IO.println s!"artifact.audit_ok: {Semantics.Artifact.auditStatus}"

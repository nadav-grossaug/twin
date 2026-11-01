/-
Verified categorical sampler on stdin/stdout — the randomness backbone of the
synthetic generator (scripts/synth.py): every random draw the generator makes
goes through this binary, whose law is the theorem `sampleCellPMF_law`
(P(i) = wᵢ/Σw). Protocol: line 1 = N (number of draws), line 2 =
comma-separated integer weights; output: N lines of sampled indices.
Only this I/O wrapper is unverified.
-/
import EnclaveSeries.Sampler

open SLang PMF EnclaveSeries

def main : IO UInt32 := do
  let stdin ← IO.getStdin
  let l1 ← stdin.getLine
  let some n := l1.trim.toNat?
    | do IO.eprintln "catsample: first line must be the number of draws"; return 1
  let l2 ← stdin.getLine
  let parts := l2.trim.splitOn ","
  let ws := parts.filterMap (fun s => s.trim.toNat?)
  if ws.length ≠ parts.length then
    IO.eprintln "catsample: weights must be comma-separated naturals"
    return 1
  if h : 0 < ws.sum then
    let dist := sampleCellPMF ws h
    let out ← IO.getStdout
    for _ in [0:n] do
      let i ← run dist
      out.putStrLn (toString i)
    return 0
  else
    IO.eprintln "catsample: weights sum to zero"
    return 1

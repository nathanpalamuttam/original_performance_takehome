# VLIW SIMD Kernel Optimization Progress

## Problem Overview

Optimizing a VLIW SIMD kernel for parallel tree traversal with hash computation.

- **Architecture**: VLIW with VLEN=8, slot limits: ALU=12, VALU=6, Load=2, Store=2, Flow=1
- **Problem size**: 256 batch size, 16 rounds, forest height 10
- **Baseline**: 147,734 cycles

## Performance History

| Version                         | Cycles    | Speedup   | Notes                  |
| ------------------------------- | --------- | --------- | ---------------------- |
| Reference baseline              | 147,734   | 1.0x      | Unoptimized            |
| Previous work (from transcript) | 4,470     | 33.0x     | Initial vectorization  |
| Provided new baseline           | 2,803     | 52.7x     | Cross-batch pipelining |
| **Final optimized**             | **2,568** | **57.5x** | Cross-round pipelining |

## Optimizations Applied

### 1. Deferred Const Loading (~19 cycles saved)

**Problem**: The original code used `scratch_const()` which immediately emitted single-instruction const loads via `self.add()`. These happened before `init_body` was built, so they couldn't be packed by the VLIW builder.

**Solution**: Added `scratch_const_deferred()` method that collects const loads into a provided list instead of emitting immediately:

```python
def scratch_const_deferred(self, val, deferred_list, name=None):
    """Like scratch_const but adds to deferred_list instead of self.instrs"""
    if val not in self.const_map:
        addr = self.alloc_scratch(name)
        deferred_list.append(("load", ("const", addr, val)))
        self.const_map[val] = addr
    return self.const_map[val]
```

**Result**: Const loads are now packed 2 per cycle (using Load engine's limit of 2) and can overlap with VALU broadcasts.

### 2. Cross-Round Prolog Pipelining (~216 cycles saved)

**Problem**: For gather rounds (level ≥ 2), the code emitted a "prolog" gather for the first batch at the start of each round. This prolog was not overlapped with any compute, wasting cycles.

**Analysis**:

- 12 gather rounds × 24 cycles prolog = 288 potential cycles of waste
- The last batch of each round (2 vectors) takes ~18 cycles of compute
- These could overlap with the next round's prolog gather

**Solution**: Modified the round loop to:

1. Track which rounds have had their prolog emitted (`prolog_emitted_for_next` dict)
2. When processing the last batch of round N, if round N+1 is a gather round, emit its prolog overlapped with current compute
3. Skip standalone prolog emission if it was already done by the previous round

**Code change** (simplified):

```python
# For last batch, check if we can overlap with next round's prolog
next_vec_i = vec_i + batches
if next_vec_i >= num_vectors and next_level >= 2:
    # Overlap with next round's prolog gather
    next_batches = min(6, num_vectors)
    next_idxs = [v_idx_bank + i * VLEN for i in range(next_batches)]
    next_node_vals = v_node_bufs[1 - buf_sel][:next_batches]
    gather_cycles = build_gather_cycles(next_idxs, next_node_vals)
    prolog_emitted_for_next[round + 1] = True
```

**Result**: Most prolog gathers are now overlapped with the previous round's last batch compute.

## Utilization Analysis

### Before Optimization (2,803 cycles)

- VALU: 74.7% utilization
- Load: 57.0% utilization
- Mixed load+valu instructions: 1,201

### After Optimization (2,568 cycles)

- VALU: 81.5% utilization
- Load: 62.3% utilization
- Mixed load+valu instructions: increased overlap

### VALU Slot Distribution (unchanged)

| Slots | Instructions | Source                    |
| ----- | ------------ | ------------------------- |
| 6     | 1,960        | Full batches (6 vectors)  |
| 4     | 96           | Partial batch hash stages |
| 2     | 200          | Partial batch other ops   |
| 1     | 16           | Init broadcasts           |
| 3     | 1            | Init                      |

## Remaining Gap Analysis

**Current**: 2,568 cycles  
**Theoretical VALU minimum**: 2,094 cycles  
**Gap**: 474 cycles
**Note**: The overall theoretical limit is around **1,100 cycles**, so we are still far from the absolute ceiling.

### Sources of Remaining Gap

1. **Partial batch waste** (~165 cycles)
   - Last batch has only 2 vectors (32 % 6 = 2)
   - Uses only 2/6 or 4/6 VALU slots
   - 200 instructions × 4 wasted slots + 96 instructions × 2 wasted slots = 992 wasted slots
   - 992 / 6 ≈ 165 cycles

2. **Init/final overhead** (~82 cycles)
   - Parameter loading, broadcasts, stores
   - Already optimized with deferred const loading

3. **First gather round prolog** (~24 cycles)
   - Round 2 (first gather round) has no previous round to overlap with
   - Must emit prolog standalone

4. **Other scheduling inefficiencies** (~203 cycles)
   - Dependencies preventing perfect packing
   - Instruction ordering constraints

## Further Optimization Ideas

### Priority Order (Highest → Lowest)

1. **Deeper gather/compute overlap**: Start round N+1 gather earlier inside round N (not just the last batch) to hide more prolog cost.
2. **Load/VALU pairing pass**: Lightweight scheduling to co-pack load slots with VALU-heavy hash sequences and avoid load-only cycles.
3. **Stage fusion across hash steps**: Reduce scratch round-trips between consecutive VALU ops to remove dependency breaks that prevent packing.
4. **Cross-round unrolling (2 rounds)**: Unroll small groups of rounds to enable instruction scheduling across boundaries and cut per-round overhead.
5. **Mask/select reuse**: Precompute common masks and reuse across batches/rounds to reduce flow/select pressure.
6. **Init + round-0 overlap**: Extend init pipelining so more broadcasts/loads overlap with the first round’s compute.

### High Potential

1. **Eliminate partial batch waste**: Process 32 vectors differently
   - Option A: Pad to 36 vectors (6 batches of 6) - but wastes compute
   - Option B: Use batch size 4 for last segment - but less efficient
   - Current approach (6+6+6+6+6+2) is actually optimal for total VALU cycles

### Medium Potential

2. **Level 0/1 within-round pipelining**: These rounds have no gather, but could potentially overlap compute between batches differently

3. **Better init pipelining**: Overlap more of init with first round's work

4. **Aggressive gather/compute overlap**: Pipeline gather for round N+1 earlier within round N (not just the last batch), if dependencies allow, to hide more of the 24-cycle prolog cost.

5. **Reduce load bottlenecks in hash stages**: Reorder hash stages to cluster loads with VALU-heavy sequences, maximizing 2-load slots per cycle and reducing isolated load-only cycles.

6. **Stage fusion for VALU chains**: Combine consecutive VALU ops (e.g., hash stage sequences) with fewer scratch round-trips, reducing dependency breaks that prevent VLIW packing.

7. **Vector-wide mask reuse**: If mask patterns repeat across rounds/batches, precompute and reuse masks to cut down on flow/select overhead.

8. **Unroll across rounds**: Unroll small round groups (e.g., 2 rounds) to enable scheduling across round boundaries and reduce per-round overhead.

### Low Potential

4. **Instruction reordering**: Manual fine-tuning of emission order for better packing

5. **Slot balancing heuristics**: Add a simple reordering pass to prioritize VALU+Load pairing and defer ALU-only ops when a load is available.

6. **Scratch locality tuning**: Reassign scratch addresses so that hot values stay in low indices, potentially enabling cheaper address arithmetic.

## Files

- `/mnt/user-data/outputs/perf_takehome_optimized.py` - Final optimized kernel (2,568 cycles)
- `/home/claude/perf_baseline2.py` - Provided baseline (2,803 cycles)
- `/home/claude/perf_opt1.py` - Working copy of optimizations

## Conclusion

Achieved **57.5x speedup** over the reference baseline (147,734 → 2,568 cycles), improving on the provided 2,803 cycle version by **8.4%** (235 cycles saved).

The main innovations were:

1. Better VLIW packing of initialization code
2. Cross-round software pipelining to overlap compute with gather prologs

The kernel is now within ~22% of the theoretical VALU-bound minimum (2,568 vs 2,094 cycles), with the gap primarily due to unavoidable partial batch inefficiency from the problem size (256 vectors = 32 SIMD groups, which doesn't divide evenly by the optimal batch size of 6).

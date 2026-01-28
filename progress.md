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

**Solution**: Added `scratch_const_deferred()` method that collects const loads into a provided list instead of emitting immediately.

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

**Result**: Most prolog gathers are now overlapped with the previous round's last batch compute.

## Attempted Optimizations (Not Applied)

### Load/VALU Merge Pass (Only 1 cycle saved)

Attempted to merge load-only cycles with following VALU-only cycles in a post-processing pass. Analysis showed that 62 potential merges were blocked by true RAW (Read-After-Write) dependencies - the VALU operations needed the values being loaded. Only 1 merge was safe, making this optimization not worthwhile for the code complexity added.

### Adaptive Batch Sizing (12 cycles WORSE)

Attempted to use batch sizes of (6,6,6,6,8) instead of (6,6,6,6,6,2) to eliminate the partial 2-vector batch. While this eliminates VALU slot waste from partial batches, the 8-vector batch requires more cycles overall and doesn't provide better gather/compute overlap. Reverted.

## Utilization Analysis

### After Optimization (2,568 cycles)

- VALU: 81.5% utilization
- Load: 62.3% utilization
- Mixed load+valu instructions: 1,418

### VALU Slot Distribution

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
   - 992 wasted VALU slots / 6 ≈ 165 cycles

2. **Init/final overhead** (~82 cycles)
   - Parameter loading, broadcasts, stores
   - Already optimized with deferred const loading

3. **First gather round prolog** (~24 cycles)
   - Round 2 (first gather round) has no previous round to overlap with
   - Must emit prolog standalone

4. **Data dependencies** (~203 cycles)
   - Gather epilogs can't merge with following hash because hash needs gathered values
   - This is a fundamental algorithmic constraint

## Additional Improvement Ideas (Prioritized)

1. **Earlier next-round gather kickoff**: Compute `next_idx` sooner within a batch and start the next round's address math/gather before the batch finishes.
2. **Batch interleaving to hide the 2-vector tail**: Interleave the last 2-vector batch of round N with a full batch of round N+1 so VALU stays fuller.
3. **Two-batch gather queue**: Build gather cycles for batch N+1 while hashing batch N, then feed them to `emit_hash_with_gather` to overlap the load-only tail.
4. **ALU address hoisting**: Move address generation (`base + idx`) into cycles that are VALU-heavy but ALU-light to avoid ALU-only cycles in gather.
5. **Hash stage fusion**: Fuse consecutive hash-stage ops to reduce scratch round-trips and remove dependency breaks that block packing.
6. **Pointer increment reuse**: Keep rolling pointers to `v_idx_bank`/`v_val_bank` and increment per batch instead of recomputing offsets every time.
7. **Round-group unrolling**: Unroll 2-3 rounds to let the scheduler pack VALU across round boundaries and overlap gathers more aggressively.

## Files

- `/mnt/user-data/outputs/perf_takehome_optimized.py` - Final optimized kernel (2,568 cycles)
- `/home/claude/perf_opt1.py` - Working copy of optimizations

## Conclusion

Achieved **57.5x speedup** over the reference baseline (147,734 → 2,568 cycles), improving on the provided 2,803 cycle version by **8.4%** (235 cycles saved).

The main innovations were:

1. Better VLIW packing of initialization code through deferred const loading
2. Cross-round software pipelining to overlap compute with gather prologs

Further optimizations face diminishing returns due to:

- True data dependencies preventing instruction merging
- Partial batch inefficiency being fundamental to the 256/8/6 = 32 vector / 6 batch structure
- The kernel already being within ~22% of the theoretical VALU-bound minimum

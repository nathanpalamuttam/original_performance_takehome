"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self, enable_debug: bool = False):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.enable_debug = enable_debug

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots, vliw=False):
        if not self.enable_debug:
            slots = [slot for slot in slots if slot[0] != "debug"]
        if not vliw:
            return [{engine: [slot]} for engine, slot in slots]

        def get_deps(engine, slot):
            writes, reads = set(), set()
            if engine == "alu":
                op, dest, a1, a2 = slot
                writes.add(dest); reads.add(a1); reads.add(a2)
            elif engine == "valu":
                if slot[0] == "vbroadcast":
                    _, dest, src = slot
                    for i in range(VLEN): writes.add(dest + i)
                    reads.add(src)
                elif slot[0] == "multiply_add":
                    _, dest, a1, a2, a3 = slot
                    for i in range(VLEN):
                        writes.add(dest + i); reads.add(a1 + i); reads.add(a2 + i); reads.add(a3 + i)
                else:
                    op, dest, a1, a2 = slot[:4]
                    for i in range(VLEN):
                        writes.add(dest + i); reads.add(a1 + i); reads.add(a2 + i)
            elif engine == "load":
                if slot[0] == "load": _, dest, addr = slot; writes.add(dest); reads.add(addr)
                elif slot[0] == "vload": _, dest, addr = slot; [writes.add(dest + i) for i in range(VLEN)]; reads.add(addr)
                elif slot[0] == "const": _, dest, val = slot; writes.add(dest)
            elif engine == "store":
                if slot[0] == "store": _, addr, src = slot; reads.add(addr); reads.add(src)
                elif slot[0] == "vstore": _, addr, src = slot; reads.add(addr); [reads.add(src + i) for i in range(VLEN)]
            elif engine == "flow":
                if slot[0] == "select": _, dest, cond, a, b = slot; writes.add(dest); reads.update([cond, a, b])
                elif slot[0] == "vselect":
                    _, dest, cond, a, b = slot
                    for i in range(VLEN): writes.add(dest + i); reads.update([cond + i, a + i, b + i])
            elif engine == "debug":
                if slot[0] == "compare": _, loc, key = slot; reads.add(loc)
            return reads, writes

        instrs, current_instr = [], {}
        slot_counts = {name: 0 for name in SLOT_LIMITS}
        written_this_cycle = set()

        for engine, slot in slots:
            reads, writes = get_deps(engine, slot)
            if slot_counts[engine] >= SLOT_LIMITS[engine] or (reads & written_this_cycle):
                if current_instr: instrs.append(current_instr)
                current_instr, slot_counts, written_this_cycle = {}, {n: 0 for n in SLOT_LIMITS}, set()
            if engine not in current_instr: current_instr[engine] = []
            current_instr[engine].append(slot)
            slot_counts[engine] += 1
            written_this_cycle.update(writes)
        if current_instr: instrs.append(current_instr)
        return instrs

    def add(self, engine, slot):
        if engine == "debug" and not self.enable_debug: return
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name: self.scratch[name] = addr; self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_const_deferred(self, val, deferred_list, name=None):
        """Like scratch_const but adds to deferred_list instead of self.instrs"""
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            deferred_list.append(("load", ("const", addr, val)))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel_pipelined(self, forest_height, n_nodes, batch_size, rounds):
        tmp1 = self.alloc_scratch("tmp1")
        for v in ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Collect all const loads in init_body for better packing
        init_body = []
        
        one_const = self.scratch_const_deferred(1, init_body)
        two_const = self.scratch_const_deferred(2, init_body)
        self.add("flow", ("pause",))

        num_vectors = batch_size // VLEN

        v_idx_bank = self.alloc_scratch("v_idx_bank", num_vectors * VLEN)
        v_val_bank = self.alloc_scratch("v_val_bank", num_vectors * VLEN)
        # Use 6 batches for better hash efficiency, double-buffered
        v_node_bufs = [
            [self.alloc_scratch(f"v_node_val0_{i}", VLEN) for i in range(6)],
            [self.alloc_scratch(f"v_node_val1_{i}", VLEN) for i in range(6)],
        ]
        first_tmps = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(6)]
        second_tmps = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(6)]
        base_addr_A = self.alloc_scratch("base_addr_A")
        node_addr_tmp = self.alloc_scratch("node_addr_tmp")
        node_addr_tmp2 = self.alloc_scratch("node_addr_tmp2")
        node_addr_tmp3 = self.alloc_scratch("node_addr_tmp3")
        node_addr_tmp4 = self.alloc_scratch("node_addr_tmp4")
        v_one, v_two, v_n_nodes = [self.alloc_scratch(n, VLEN) for n in ["v_one", "v_two", "v_n_nodes"]]

        init_body.append(("valu", ("vbroadcast", v_one, one_const)))
        init_body.append(("valu", ("vbroadcast", v_two, two_const)))
        init_body.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))

        v_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_val1 = self.alloc_scratch(f"v_hash_c1_{hi}", VLEN)
            v_val3 = self.alloc_scratch(f"v_hash_c3_{hi}", VLEN)
            init_body.append(("valu", ("vbroadcast", v_val1, self.scratch_const_deferred(val1, init_body))))
            init_body.append(("valu", ("vbroadcast", v_val3, self.scratch_const_deferred(val3, init_body))))
            v_hash_consts.extend([v_val1, v_val3])

        v_root, v_node1, v_node2 = [self.alloc_scratch(n, VLEN) for n in ["v_root", "v_node1", "v_node2"]]
        v_node1_minus_node2 = self.alloc_scratch("v_node1_minus_node2", VLEN)

        init_body.append(("load", ("load", tmp1, self.scratch["forest_values_p"])))
        init_body.append(("valu", ("vbroadcast", v_root, tmp1)))
        init_body.append(("alu", ("+", base_addr_A, self.scratch["forest_values_p"], one_const)))
        init_body.append(("load", ("load", tmp1, base_addr_A)))
        init_body.append(("valu", ("vbroadcast", v_node1, tmp1)))
        init_body.append(("alu", ("+", base_addr_A, self.scratch["forest_values_p"], two_const)))
        init_body.append(("load", ("load", tmp1, base_addr_A)))
        init_body.append(("valu", ("vbroadcast", v_node2, tmp1)))
        init_body.append(("valu", ("-", v_node1_minus_node2, v_node1, v_node2)))

        offset_consts = [self.scratch_const_deferred(vec_i * VLEN, init_body) for vec_i in range(num_vectors)]

        for vec_i in range(num_vectors):
            offset_const = offset_consts[vec_i]
            init_body.append(("alu", ("+", base_addr_A, self.scratch["inp_indices_p"], offset_const)))
            init_body.append(("load", ("vload", v_idx_bank + vec_i * VLEN, base_addr_A)))
            init_body.append(("alu", ("+", base_addr_A, self.scratch["inp_values_p"], offset_const)))
            init_body.append(("load", ("vload", v_val_bank + vec_i * VLEN, base_addr_A)))

        self.instrs.extend(self.build(init_body, vliw=True))

        def chunk_slots(slots, limit):
            for i in range(0, len(slots), limit):
                yield slots[i : i + limit]

        def emit_debug_slots(slots):
            if not self.enable_debug:
                return
            limit = SLOT_LIMITS["debug"]
            for i in range(0, len(slots), limit):
                self.instrs.append({"debug": slots[i : i + limit]})

        def debug_idx_val(round_i, offsets, v_idxs, v_vals):
            if not self.enable_debug:
                return
            slots = []
            for b in range(len(v_idxs)):
                base = offsets[b]
                for lane in range(VLEN):
                    slots.append(("compare", v_idxs[b] + lane, (round_i, base + lane, "idx")))
                    slots.append(("compare", v_vals[b] + lane, (round_i, base + lane, "val")))
            emit_debug_slots(slots)

        def debug_node_val(round_i, offsets, v_node_vals):
            if not self.enable_debug:
                return
            slots = []
            for b in range(len(v_node_vals)):
                base = offsets[b]
                for lane in range(VLEN):
                    slots.append(("compare", v_node_vals[b] + lane, (round_i, base + lane, "node_val")))
            emit_debug_slots(slots)

        def debug_hash_stage(round_i, offsets, v_vals, hi):
            if not self.enable_debug:
                return
            slots = []
            for b in range(len(v_vals)):
                base = offsets[b]
                for lane in range(VLEN):
                    slots.append(("compare", v_vals[b] + lane, (round_i, base + lane, "hash_stage", hi)))
            emit_debug_slots(slots)

        def debug_next_idx(round_i, offsets, v_idxs):
            if not self.enable_debug:
                return
            slots = []
            for b in range(len(v_idxs)):
                base = offsets[b]
                for lane in range(VLEN):
                    slots.append(("compare", v_idxs[b] + lane, (round_i, base + lane, "next_idx")))
            emit_debug_slots(slots)

        def debug_wrapped_idx_hashed_val(round_i, offsets, v_idxs, v_vals):
            if not self.enable_debug:
                return
            slots = []
            for b in range(len(v_idxs)):
                base = offsets[b]
                for lane in range(VLEN):
                    slots.append(("compare", v_idxs[b] + lane, (round_i, base + lane, "wrapped_idx")))
                    slots.append(("compare", v_vals[b] + lane, (round_i, base + lane, "hashed_val")))
            emit_debug_slots(slots)

        addr_sets = [(node_addr_tmp, node_addr_tmp2), (node_addr_tmp3, node_addr_tmp4)]

        def build_gather_cycles(v_idxs, v_node_vals):
            pairs = []
            for b in range(len(v_idxs)):
                for pair in range(VLEN // 2):
                    l0 = pair * 2
                    pairs.append((b, l0, l0 + 1))
            if not pairs:
                return []
            cycles = []
            set_idx = 0
            b, l0, l1 = pairs[0]
            addr0, addr1 = addr_sets[set_idx]
            cycles.append({
                "alu": [
                    ("+", addr0, self.scratch["forest_values_p"], v_idxs[b] + l0),
                    ("+", addr1, self.scratch["forest_values_p"], v_idxs[b] + l1),
                ],
            })
            pending_b, pending_l0, pending_l1, pending_a0, pending_a1 = b, l0, l1, addr0, addr1
            set_idx = 1 - set_idx
            for b, l0, l1 in pairs[1:]:
                addr0, addr1 = addr_sets[set_idx]
                cycles.append({
                    "load": [
                        ("load", v_node_vals[pending_b] + pending_l0, pending_a0),
                        ("load", v_node_vals[pending_b] + pending_l1, pending_a1),
                    ],
                    "alu": [
                        ("+", addr0, self.scratch["forest_values_p"], v_idxs[b] + l0),
                        ("+", addr1, self.scratch["forest_values_p"], v_idxs[b] + l1),
                    ],
                })
                pending_b, pending_l0, pending_l1, pending_a0, pending_a1 = b, l0, l1, addr0, addr1
                set_idx = 1 - set_idx
            cycles.append({
                "load": [
                    ("load", v_node_vals[pending_b] + pending_l0, pending_a0),
                    ("load", v_node_vals[pending_b] + pending_l1, pending_a1),
                ],
            })
            return cycles

        def build_level0_cycles(v_node_vals):
            slots = [("*", v_node_vals[b], v_root, v_one) for b in range(len(v_node_vals))]
            return [{"valu": chunk} for chunk in chunk_slots(slots, SLOT_LIMITS["valu"])]

        def build_level1_cycles(v_idxs, v_node_vals, tmps):
            cycles = []
            slots = [("&", tmps[b], v_idxs[b], v_one) for b in range(len(v_node_vals))]
            cycles.extend({"valu": chunk} for chunk in chunk_slots(slots, SLOT_LIMITS["valu"]))
            slots = [("*", tmps[b], v_node1_minus_node2, tmps[b]) for b in range(len(v_node_vals))]
            cycles.extend({"valu": chunk} for chunk in chunk_slots(slots, SLOT_LIMITS["valu"]))
            slots = [("+", v_node_vals[b], v_node2, tmps[b]) for b in range(len(v_node_vals))]
            cycles.extend({"valu": chunk} for chunk in chunk_slots(slots, SLOT_LIMITS["valu"]))
            return cycles

        def emit_hash_with_gather(round_i, offsets, v_idxs, v_vals, v_node_vals, first_tmps_batch, second_tmps_batch, gather_cycles):
            g_idx = 0
            batches = len(v_vals)

            def emit_cycle(cycle):
                nonlocal g_idx
                if g_idx < len(gather_cycles):
                    for eng, slots in gather_cycles[g_idx].items():
                        cycle.setdefault(eng, []).extend(slots)
                    g_idx += 1
                if "valu" in cycle:
                    assert len(cycle["valu"]) <= SLOT_LIMITS["valu"]
                if "load" in cycle:
                    assert len(cycle["load"]) <= SLOT_LIMITS["load"]
                if "alu" in cycle:
                    assert len(cycle["alu"]) <= SLOT_LIMITS["alu"]
                self.instrs.append(cycle)

            xor_slots = [("^", v_vals[b], v_vals[b], v_node_vals[b]) for b in range(batches)]
            for chunk in chunk_slots(xor_slots, SLOT_LIMITS["valu"]):
                emit_cycle({"valu": chunk})

            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                v_c1, v_c3 = v_hash_consts[hi * 2], v_hash_consts[hi * 2 + 1]
                stage_slots = []
                for b in range(batches):
                    stage_slots.append((op1, first_tmps_batch[b], v_vals[b], v_c1))
                    stage_slots.append((op3, second_tmps_batch[b], v_vals[b], v_c3))
                for chunk in chunk_slots(stage_slots, SLOT_LIMITS["valu"]):
                    emit_cycle({"valu": chunk})
                op2_slots = [(op2, v_vals[b], first_tmps_batch[b], second_tmps_batch[b]) for b in range(batches)]
                for chunk in chunk_slots(op2_slots, SLOT_LIMITS["valu"]):
                    emit_cycle({"valu": chunk})
                debug_hash_stage(round_i, offsets, v_vals, hi)

            slots = [("&", first_tmps_batch[b], v_vals[b], v_one) for b in range(batches)]
            for chunk in chunk_slots(slots, SLOT_LIMITS["valu"]):
                emit_cycle({"valu": chunk})
            slots = [("+", second_tmps_batch[b], first_tmps_batch[b], v_one) for b in range(batches)]
            for chunk in chunk_slots(slots, SLOT_LIMITS["valu"]):
                emit_cycle({"valu": chunk})
            slots = [("multiply_add", v_idxs[b], v_idxs[b], v_two, second_tmps_batch[b]) for b in range(batches)]
            for chunk in chunk_slots(slots, SLOT_LIMITS["valu"]):
                emit_cycle({"valu": chunk})
            debug_next_idx(round_i, offsets, v_idxs)

            slots = [("<", first_tmps_batch[b], v_idxs[b], v_n_nodes) for b in range(batches)]
            for chunk in chunk_slots(slots, SLOT_LIMITS["valu"]):
                emit_cycle({"valu": chunk})
            slots = [("*", v_idxs[b], v_idxs[b], first_tmps_batch[b]) for b in range(batches)]
            for chunk in chunk_slots(slots, SLOT_LIMITS["valu"]):
                emit_cycle({"valu": chunk})
            debug_wrapped_idx_hashed_val(round_i, offsets, v_idxs, v_vals)

            while g_idx < len(gather_cycles):
                self.instrs.append(gather_cycles[g_idx])
                g_idx += 1

        prolog_emitted_for_next = {}  # Track which rounds had their prolog emitted by prev round
        
        for round in range(rounds):
            level = round % (forest_height + 1)
            next_level = (round + 1) % (forest_height + 1) if round + 1 < rounds else -1
            vec_i = 0
            buf_sel = 0
            
            # Prolog was emitted by previous round if we recorded it
            prolog_already_done = prolog_emitted_for_next.get(round, False)

            if level >= 2 and num_vectors and not prolog_already_done:
                batches = min(6, num_vectors)
                v_idxs0 = [v_idx_bank + i * VLEN for i in range(batches)]
                v_node_vals0 = v_node_bufs[buf_sel][:batches]
                self.instrs.extend(build_gather_cycles(v_idxs0, v_node_vals0))

            while vec_i < num_vectors:
                batches = min(6, num_vectors - vec_i)
                offsets = [vec_i * VLEN + i * VLEN for i in range(batches)]
                v_idxs = [v_idx_bank + o for o in offsets]
                v_vals = [v_val_bank + o for o in offsets]
                cur_node_vals = v_node_bufs[buf_sel][:batches]
                first_tmps_batch = first_tmps[:batches]
                second_tmps_batch = second_tmps[:batches]

                debug_idx_val(round, offsets, v_idxs, v_vals)

                if level == 0:
                    self.instrs.extend(build_level0_cycles(cur_node_vals))
                    debug_node_val(round, offsets, cur_node_vals)
                    # For last batch, check if we can overlap with next round's prolog
                    next_vec_i = vec_i + batches
                    if next_vec_i >= num_vectors and next_level >= 2:
                        # Overlap with next round's prolog gather
                        next_batches = min(6, num_vectors)
                        next_idxs = [v_idx_bank + i * VLEN for i in range(next_batches)]
                        # Use the OTHER buffer for next round's first batch
                        next_node_vals = v_node_bufs[1 - buf_sel][:next_batches]
                        gather_cycles = build_gather_cycles(next_idxs, next_node_vals)
                        prolog_emitted_for_next[round + 1] = True
                    else:
                        gather_cycles = []
                    emit_hash_with_gather(round, offsets, v_idxs, v_vals, cur_node_vals, first_tmps_batch, second_tmps_batch, gather_cycles)
                elif level == 1:
                    self.instrs.extend(build_level1_cycles(v_idxs, cur_node_vals, first_tmps_batch))
                    debug_node_val(round, offsets, cur_node_vals)
                    # For last batch, check if we can overlap with next round's prolog
                    next_vec_i = vec_i + batches
                    if next_vec_i >= num_vectors and next_level >= 2:
                        next_batches = min(6, num_vectors)
                        next_idxs = [v_idx_bank + i * VLEN for i in range(next_batches)]
                        next_node_vals = v_node_bufs[1 - buf_sel][:next_batches]
                        gather_cycles = build_gather_cycles(next_idxs, next_node_vals)
                        prolog_emitted_for_next[round + 1] = True
                    else:
                        gather_cycles = []
                    emit_hash_with_gather(round, offsets, v_idxs, v_vals, cur_node_vals, first_tmps_batch, second_tmps_batch, gather_cycles)
                else:
                    debug_node_val(round, offsets, cur_node_vals)
                    next_vec_i = vec_i + batches
                    if next_vec_i < num_vectors:
                        next_batches = min(6, num_vectors - next_vec_i)
                        next_offsets = [next_vec_i * VLEN + i * VLEN for i in range(next_batches)]
                        next_idxs = [v_idx_bank + o for o in next_offsets]
                        next_node_vals = v_node_bufs[1 - buf_sel][:next_batches]
                        gather_cycles = build_gather_cycles(next_idxs, next_node_vals)
                    elif next_level >= 2:
                        # Last batch of this round, next round is also gather
                        # Overlap with next round's prolog
                        next_batches = min(6, num_vectors)
                        next_idxs = [v_idx_bank + i * VLEN for i in range(next_batches)]
                        next_node_vals = v_node_bufs[1 - buf_sel][:next_batches]
                        gather_cycles = build_gather_cycles(next_idxs, next_node_vals)
                        prolog_emitted_for_next[round + 1] = True
                    else:
                        gather_cycles = []
                    emit_hash_with_gather(round, offsets, v_idxs, v_vals, cur_node_vals, first_tmps_batch, second_tmps_batch, gather_cycles)

                vec_i += batches
                buf_sel = 1 - buf_sel

        final_body = []
        for vec_i in range(num_vectors):
            offset_const = offset_consts[vec_i]
            final_body.append(("alu", ("+", base_addr_A, self.scratch["inp_indices_p"], offset_const)))
            final_body.append(("store", ("vstore", base_addr_A, v_idx_bank + vec_i * VLEN)))
            final_body.append(("alu", ("+", base_addr_A, self.scratch["inp_values_p"], offset_const)))
            final_body.append(("store", ("vstore", base_addr_A, v_val_bank + vec_i * VLEN)))

        self.instrs.extend(self.build(final_body, vliw=True))
        self.instrs.append({"flow": [("pause",)]})

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        if batch_size % VLEN == 0:
            return self.build_kernel_pipelined(forest_height, n_nodes, batch_size, rounds)

BASELINE = 147734

def do_kernel_test(forest_height, rounds, batch_size, seed=123, trace=False, prints=False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace, trace=trace)
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        assert machine.mem[inp_values_p:inp_values_p + len(inp.values)] == ref_mem[inp_values_p:inp_values_p + len(inp.values)], f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    unittest.main()

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

    def build_kernel_pipelined(self, forest_height, n_nodes, batch_size, rounds):
        tmp1 = self.alloc_scratch("tmp1")
        for v in ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        self.add("flow", ("pause",))

        body = []
        num_vectors = batch_size // VLEN

        v_idx_bank = self.alloc_scratch("v_idx_bank", num_vectors * VLEN)
        v_val_bank = self.alloc_scratch("v_val_bank", num_vectors * VLEN)
        # Use 6 batches for better hash efficiency
        v_node_vals = [self.alloc_scratch(f"v_node_val_{i}", VLEN) for i in range(6)]
        first_tmps = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(6)]
        second_tmps = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(6)]
        base_addr_A = self.alloc_scratch("base_addr_A")
        node_addr_tmp = self.alloc_scratch("node_addr_tmp")
        node_addr_tmp2 = self.alloc_scratch("node_addr_tmp2")
        node_addr_tmp3 = self.alloc_scratch("node_addr_tmp3")
        node_addr_tmp4 = self.alloc_scratch("node_addr_tmp4")
        v_one, v_two, v_n_nodes = [self.alloc_scratch(n, VLEN) for n in ["v_one", "v_two", "v_n_nodes"]]

        body.append(("valu", ("vbroadcast", v_one, one_const)))
        body.append(("valu", ("vbroadcast", v_two, two_const)))
        body.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))

        v_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_val1 = self.alloc_scratch(f"v_hash_c1_{hi}", VLEN)
            v_val3 = self.alloc_scratch(f"v_hash_c3_{hi}", VLEN)
            body.append(("valu", ("vbroadcast", v_val1, self.scratch_const(val1))))
            body.append(("valu", ("vbroadcast", v_val3, self.scratch_const(val3))))
            v_hash_consts.extend([v_val1, v_val3])

        v_root, v_node1, v_node2 = [self.alloc_scratch(n, VLEN) for n in ["v_root", "v_node1", "v_node2"]]
        v_node1_minus_node2 = self.alloc_scratch("v_node1_minus_node2", VLEN)

        body.append(("load", ("load", tmp1, self.scratch["forest_values_p"])))
        body.append(("valu", ("vbroadcast", v_root, tmp1)))
        body.append(("alu", ("+", base_addr_A, self.scratch["forest_values_p"], one_const)))
        body.append(("load", ("load", tmp1, base_addr_A)))
        body.append(("valu", ("vbroadcast", v_node1, tmp1)))
        body.append(("alu", ("+", base_addr_A, self.scratch["forest_values_p"], two_const)))
        body.append(("load", ("load", tmp1, base_addr_A)))
        body.append(("valu", ("vbroadcast", v_node2, tmp1)))
        body.append(("valu", ("-", v_node1_minus_node2, v_node1, v_node2)))

        offset_consts = [self.scratch_const(vec_i * VLEN) for vec_i in range(num_vectors)]

        for vec_i in range(num_vectors):
            offset_const = offset_consts[vec_i]
            body.append(("alu", ("+", base_addr_A, self.scratch["inp_indices_p"], offset_const)))
            body.append(("load", ("vload", v_idx_bank + vec_i * VLEN, base_addr_A)))
            body.append(("alu", ("+", base_addr_A, self.scratch["inp_values_p"], offset_const)))
            body.append(("load", ("vload", v_val_bank + vec_i * VLEN, base_addr_A)))

        for round in range(rounds):
            level = round % (forest_height + 1)
            vec_i = 0
            while vec_i < num_vectors:
                batches = min(6, num_vectors - vec_i)
                
                offsets = [vec_i * VLEN + i * VLEN for i in range(batches)]
                v_idxs = [v_idx_bank + o for o in offsets]
                v_vals = [v_val_bank + o for o in offsets]
                v_node_val_batch = v_node_vals[:batches]
                
                # Debug
                for b in range(batches):
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idxs[b] + lane, (round, offsets[b] + lane, "idx"))))
                        body.append(("debug", ("compare", v_vals[b] + lane, (round, offsets[b] + lane, "val"))))
                
                # Gather node values
                if level == 0:
                    for b in range(batches):
                        body.append(("valu", ("*", v_node_val_batch[b], v_root, v_one)))
                elif level == 1:
                    tmps = first_tmps[:batches]
                    for b in range(batches):
                        body.append(("valu", ("&", tmps[b], v_idxs[b], v_one)))
                    for b in range(batches):
                        body.append(("valu", ("*", tmps[b], v_node1_minus_node2, tmps[b])))
                    for b in range(batches):
                        body.append(("valu", ("+", v_node_val_batch[b], v_node2, tmps[b])))
                else:
                    # Gather from memory
                    addr_tmps = [node_addr_tmp, node_addr_tmp2]
                    for b in range(batches):
                        for pair in range(VLEN // 2):
                            l0, l1 = pair * 2, pair * 2 + 1
                            body.append(("alu", ("+", addr_tmps[0], self.scratch["forest_values_p"], v_idxs[b] + l0)))
                            body.append(("alu", ("+", addr_tmps[1], self.scratch["forest_values_p"], v_idxs[b] + l1)))
                            body.append(("load", ("load", v_node_val_batch[b] + l0, addr_tmps[0])))
                            body.append(("load", ("load", v_node_val_batch[b] + l1, addr_tmps[1])))
                
                for b in range(batches):
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_node_val_batch[b] + lane, (round, offsets[b] + lane, "node_val"))))
                
                # XOR all batches
                for b in range(batches):
                    body.append(("valu", ("^", v_vals[b], v_vals[b], v_node_val_batch[b])))
                
                # Hash - interleave op1 and op3 for all batches
                first_tmp_batch = first_tmps[:batches]
                second_tmp_batch = second_tmps[:batches]
                
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    v_c1, v_c3 = v_hash_consts[hi * 2], v_hash_consts[hi * 2 + 1]
                    # Interleave op1 and op3 emissions
                    for b in range(batches):
                        body.append(("valu", (op1, first_tmp_batch[b], v_vals[b], v_c1)))
                        body.append(("valu", (op3, second_tmp_batch[b], v_vals[b], v_c3)))
                    # Then op2
                    for b in range(batches):
                        body.append(("valu", (op2, v_vals[b], first_tmp_batch[b], second_tmp_batch[b])))
                    
                    for b in range(batches):
                        for lane in range(VLEN):
                            body.append(("debug", ("compare", v_vals[b] + lane, (round, offsets[b] + lane, "hash_stage", hi))))
                
                # Index computation
                for b in range(batches):
                    body.append(("valu", ("&", first_tmp_batch[b], v_vals[b], v_one)))
                for b in range(batches):
                    body.append(("valu", ("+", second_tmp_batch[b], first_tmp_batch[b], v_one)))
                for b in range(batches):
                    body.append(("valu", ("multiply_add", v_idxs[b], v_idxs[b], v_two, second_tmp_batch[b])))
                
                for b in range(batches):
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idxs[b] + lane, (round, offsets[b] + lane, "next_idx"))))
                
                for b in range(batches):
                    body.append(("valu", ("<", first_tmp_batch[b], v_idxs[b], v_n_nodes)))
                for b in range(batches):
                    body.append(("valu", ("*", v_idxs[b], v_idxs[b], first_tmp_batch[b])))
                
                for b in range(batches):
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idxs[b] + lane, (round, offsets[b] + lane, "wrapped_idx"))))
                        body.append(("debug", ("compare", v_vals[b] + lane, (round, offsets[b] + lane, "hashed_val"))))
                
                vec_i += batches

        for vec_i in range(num_vectors):
            offset_const = offset_consts[vec_i]
            body.append(("alu", ("+", base_addr_A, self.scratch["inp_indices_p"], offset_const)))
            body.append(("store", ("vstore", base_addr_A, v_idx_bank + vec_i * VLEN)))
            body.append(("alu", ("+", base_addr_A, self.scratch["inp_values_p"], offset_const)))
            body.append(("store", ("vstore", base_addr_A, v_val_bank + vec_i * VLEN)))

        self.instrs.extend(self.build(body, vliw=True))
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
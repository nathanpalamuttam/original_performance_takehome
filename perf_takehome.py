"""
# Anthropic's Original Performance Engineering Take-home (Release version)
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
        v_node_val_A = self.alloc_scratch("v_node_val_A", VLEN)
        v_node_val_B = self.alloc_scratch("v_node_val_B", VLEN)
        v_tmp1, v_tmp2, v_tmp3, v_tmp4 = [self.alloc_scratch(f"v_tmp{i}", VLEN) for i in range(1, 5)]
        base_addr_A = self.alloc_scratch("base_addr_A")
        node_addr_tmp = self.alloc_scratch("node_addr_tmp")
        node_addr_tmp2 = self.alloc_scratch("node_addr_tmp2")
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

        # Allocate extra node value registers for 4-way processing
        v_node_val_C = self.alloc_scratch("v_node_val_C", VLEN)
        v_node_val_D = self.alloc_scratch("v_node_val_D", VLEN)
        v_tmp5 = self.alloc_scratch("v_tmp5", VLEN)
        v_tmp6 = self.alloc_scratch("v_tmp6", VLEN)
        v_tmp7 = self.alloc_scratch("v_tmp7", VLEN)
        v_tmp8 = self.alloc_scratch("v_tmp8", VLEN)
        node_addr_tmp3 = self.alloc_scratch("node_addr_tmp3")
        node_addr_tmp4 = self.alloc_scratch("node_addr_tmp4")

        for round in range(rounds):
            level = round % (forest_height + 1)
            vec_i = 0
            while vec_i < num_vectors:
                batches = min(4, num_vectors - vec_i)
                
                if batches == 4:
                    # Process 4 batches simultaneously
                    offsets = [vec_i * VLEN + i * VLEN for i in range(4)]
                    v_idxs = [v_idx_bank + o for o in offsets]
                    v_vals = [v_val_bank + o for o in offsets]
                    v_node_vals = [v_node_val_A, v_node_val_B, v_node_val_C, v_node_val_D]
                    
                    # Debug
                    for b in range(4):
                        for lane in range(VLEN):
                            body.append(("debug", ("compare", v_idxs[b] + lane, (round, offsets[b] + lane, "idx"))))
                            body.append(("debug", ("compare", v_vals[b] + lane, (round, offsets[b] + lane, "val"))))
                    
                    # Gather node values
                    if level == 0:
                        for b in range(4):
                            body.append(("valu", ("*", v_node_vals[b], v_root, v_one)))
                    elif level == 1:
                        tmps = [v_tmp1, v_tmp2, v_tmp3, v_tmp4]
                        for b in range(4):
                            body.append(("valu", ("&", tmps[b], v_idxs[b], v_one)))
                        for b in range(4):
                            body.append(("valu", ("*", tmps[b], v_node1_minus_node2, tmps[b])))
                        for b in range(4):
                            body.append(("valu", ("+", v_node_vals[b], v_node2, tmps[b])))
                    else:
                        # Gather for all 4 batches - can do 2 loads per cycle
                        addr_tmps = [node_addr_tmp, node_addr_tmp2, node_addr_tmp3, node_addr_tmp4]
                        for b in range(4):
                            for pair in range(VLEN // 2):
                                l0, l1 = pair * 2, pair * 2 + 1
                                body.append(("alu", ("+", addr_tmps[0], self.scratch["forest_values_p"], v_idxs[b] + l0)))
                                body.append(("alu", ("+", addr_tmps[1], self.scratch["forest_values_p"], v_idxs[b] + l1)))
                                body.append(("load", ("load", v_node_vals[b] + l0, addr_tmps[0])))
                                body.append(("load", ("load", v_node_vals[b] + l1, addr_tmps[1])))
                    
                    for b in range(4):
                        for lane in range(VLEN):
                            body.append(("debug", ("compare", v_node_vals[b] + lane, (round, offsets[b] + lane, "node_val"))))
                    
                    # XOR all 4 batches
                    for b in range(4):
                        body.append(("valu", ("^", v_vals[b], v_vals[b], v_node_vals[b])))
                    
                    # Hash all 4 batches - interleave for better packing
                    # Use 8 temps: v_tmp1-4 for first part, v_tmp5-8 for second part
                    first_tmps = [v_tmp1, v_tmp2, v_tmp3, v_tmp4]
                    second_tmps = [v_tmp5, v_tmp6, v_tmp7, v_tmp8]
                    
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        v_c1, v_c3 = v_hash_consts[hi * 2], v_hash_consts[hi * 2 + 1]
                        # First 4 ops (op1 for all 4 batches)
                        for b in range(4):
                            body.append(("valu", (op1, first_tmps[b], v_vals[b], v_c1)))
                        # Next 4 ops (op3 for all 4 batches)
                        for b in range(4):
                            body.append(("valu", (op3, second_tmps[b], v_vals[b], v_c3)))
                        # Final 4 ops (op2 for all 4 batches)
                        for b in range(4):
                            body.append(("valu", (op2, v_vals[b], first_tmps[b], second_tmps[b])))
                        
                        for b in range(4):
                            for lane in range(VLEN):
                                body.append(("debug", ("compare", v_vals[b] + lane, (round, offsets[b] + lane, "hash_stage", hi))))
                    
                    # Index computation for all 4 batches
                    for b in range(4):
                        body.append(("valu", ("&", first_tmps[b], v_vals[b], v_one)))
                    for b in range(4):
                        body.append(("valu", ("+", second_tmps[b], first_tmps[b], v_one)))
                    for b in range(4):
                        body.append(("valu", ("multiply_add", v_idxs[b], v_idxs[b], v_two, second_tmps[b])))
                    
                    for b in range(4):
                        for lane in range(VLEN):
                            body.append(("debug", ("compare", v_idxs[b] + lane, (round, offsets[b] + lane, "next_idx"))))
                    
                    for b in range(4):
                        body.append(("valu", ("<", first_tmps[b], v_idxs[b], v_n_nodes)))
                    for b in range(4):
                        body.append(("valu", ("*", v_idxs[b], v_idxs[b], first_tmps[b])))
                    
                    for b in range(4):
                        for lane in range(VLEN):
                            body.append(("debug", ("compare", v_idxs[b] + lane, (round, offsets[b] + lane, "wrapped_idx"))))
                            body.append(("debug", ("compare", v_vals[b] + lane, (round, offsets[b] + lane, "hashed_val"))))
                    
                    vec_i += 4
                    
                elif batches >= 2:
                    oA, oB = vec_i * VLEN, (vec_i + 1) * VLEN
                    v_idx_A, v_val_A = v_idx_bank + oA, v_val_bank + oA
                    v_idx_B, v_val_B = v_idx_bank + oB, v_val_bank + oB

                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idx_A + lane, (round, oA + lane, "idx"))))
                        body.append(("debug", ("compare", v_val_A + lane, (round, oA + lane, "val"))))
                        body.append(("debug", ("compare", v_idx_B + lane, (round, oB + lane, "idx"))))
                        body.append(("debug", ("compare", v_val_B + lane, (round, oB + lane, "val"))))

                    if level == 0:
                        body.append(("valu", ("*", v_node_val_A, v_root, v_one)))
                        body.append(("valu", ("*", v_node_val_B, v_root, v_one)))
                    elif level == 1:
                        body.append(("valu", ("&", v_tmp1, v_idx_A, v_one)))
                        body.append(("valu", ("&", v_tmp2, v_idx_B, v_one)))
                        body.append(("valu", ("*", v_tmp1, v_node1_minus_node2, v_tmp1)))
                        body.append(("valu", ("*", v_tmp2, v_node1_minus_node2, v_tmp2)))
                        body.append(("valu", ("+", v_node_val_A, v_node2, v_tmp1)))
                        body.append(("valu", ("+", v_node_val_B, v_node2, v_tmp2)))
                    else:
                        for pair in range(VLEN // 2):
                            l0, l1 = pair * 2, pair * 2 + 1
                            body.append(("alu", ("+", node_addr_tmp, self.scratch["forest_values_p"], v_idx_A + l0)))
                            body.append(("alu", ("+", node_addr_tmp2, self.scratch["forest_values_p"], v_idx_A + l1)))
                            body.append(("load", ("load", v_node_val_A + l0, node_addr_tmp)))
                            body.append(("load", ("load", v_node_val_A + l1, node_addr_tmp2)))
                        for pair in range(VLEN // 2):
                            l0, l1 = pair * 2, pair * 2 + 1
                            body.append(("alu", ("+", node_addr_tmp, self.scratch["forest_values_p"], v_idx_B + l0)))
                            body.append(("alu", ("+", node_addr_tmp2, self.scratch["forest_values_p"], v_idx_B + l1)))
                            body.append(("load", ("load", v_node_val_B + l0, node_addr_tmp)))
                            body.append(("load", ("load", v_node_val_B + l1, node_addr_tmp2)))

                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_node_val_A + lane, (round, oA + lane, "node_val"))))
                        body.append(("debug", ("compare", v_node_val_B + lane, (round, oB + lane, "node_val"))))

                    body.append(("valu", ("^", v_val_A, v_val_A, v_node_val_A)))
                    body.append(("valu", ("^", v_val_B, v_val_B, v_node_val_B)))

                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        v_c1, v_c3 = v_hash_consts[hi * 2], v_hash_consts[hi * 2 + 1]
                        body.append(("valu", (op1, v_tmp1, v_val_A, v_c1)))
                        body.append(("valu", (op3, v_tmp2, v_val_A, v_c3)))
                        body.append(("valu", (op1, v_tmp3, v_val_B, v_c1)))
                        body.append(("valu", (op3, v_tmp4, v_val_B, v_c3)))
                        body.append(("valu", (op2, v_val_A, v_tmp1, v_tmp2)))
                        body.append(("valu", (op2, v_val_B, v_tmp3, v_tmp4)))
                        for lane in range(VLEN):
                            body.append(("debug", ("compare", v_val_A + lane, (round, oA + lane, "hash_stage", hi))))
                            body.append(("debug", ("compare", v_val_B + lane, (round, oB + lane, "hash_stage", hi))))

                    body.append(("valu", ("&", v_tmp1, v_val_A, v_one)))
                    body.append(("valu", ("+", v_tmp3, v_tmp1, v_one)))
                    body.append(("valu", ("multiply_add", v_idx_A, v_idx_A, v_two, v_tmp3)))
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idx_A + lane, (round, oA + lane, "next_idx"))))
                    body.append(("valu", ("<", v_tmp1, v_idx_A, v_n_nodes)))
                    body.append(("valu", ("*", v_idx_A, v_idx_A, v_tmp1)))
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idx_A + lane, (round, oA + lane, "wrapped_idx"))))

                    body.append(("valu", ("&", v_tmp1, v_val_B, v_one)))
                    body.append(("valu", ("+", v_tmp3, v_tmp1, v_one)))
                    body.append(("valu", ("multiply_add", v_idx_B, v_idx_B, v_two, v_tmp3)))
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idx_B + lane, (round, oB + lane, "next_idx"))))
                    body.append(("valu", ("<", v_tmp1, v_idx_B, v_n_nodes)))
                    body.append(("valu", ("*", v_idx_B, v_idx_B, v_tmp1)))
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idx_B + lane, (round, oB + lane, "wrapped_idx"))))

                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_val_A + lane, (round, oA + lane, "hashed_val"))))
                        body.append(("debug", ("compare", v_val_B + lane, (round, oB + lane, "hashed_val"))))
                    vec_i += 2
                else:
                    offset = vec_i * VLEN
                    v_idx, v_val = v_idx_bank + offset, v_val_bank + offset
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idx + lane, (round, offset + lane, "idx"))))
                        body.append(("debug", ("compare", v_val + lane, (round, offset + lane, "val"))))
                    if level == 0:
                        body.append(("valu", ("*", v_node_val_A, v_root, v_one)))
                    elif level == 1:
                        body.append(("valu", ("&", v_tmp1, v_idx, v_one)))
                        body.append(("valu", ("*", v_tmp1, v_node1_minus_node2, v_tmp1)))
                        body.append(("valu", ("+", v_node_val_A, v_node2, v_tmp1)))
                    else:
                        for pair in range(VLEN // 2):
                            l0, l1 = pair * 2, pair * 2 + 1
                            body.append(("alu", ("+", node_addr_tmp, self.scratch["forest_values_p"], v_idx + l0)))
                            body.append(("alu", ("+", node_addr_tmp2, self.scratch["forest_values_p"], v_idx + l1)))
                            body.append(("load", ("load", v_node_val_A + l0, node_addr_tmp)))
                            body.append(("load", ("load", v_node_val_A + l1, node_addr_tmp2)))
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_node_val_A + lane, (round, offset + lane, "node_val"))))
                    body.append(("valu", ("^", v_val, v_val, v_node_val_A)))
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        v_c1, v_c3 = v_hash_consts[hi * 2], v_hash_consts[hi * 2 + 1]
                        body.append(("valu", (op1, v_tmp1, v_val, v_c1)))
                        body.append(("valu", (op3, v_tmp2, v_val, v_c3)))
                        body.append(("valu", (op2, v_val, v_tmp1, v_tmp2)))
                        for lane in range(VLEN):
                            body.append(("debug", ("compare", v_val + lane, (round, offset + lane, "hash_stage", hi))))
                    body.append(("valu", ("&", v_tmp1, v_val, v_one)))
                    body.append(("valu", ("+", v_tmp3, v_tmp1, v_one)))
                    body.append(("valu", ("multiply_add", v_idx, v_idx, v_two, v_tmp3)))
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idx + lane, (round, offset + lane, "next_idx"))))
                    body.append(("valu", ("<", v_tmp1, v_idx, v_n_nodes)))
                    body.append(("valu", ("*", v_idx, v_idx, v_tmp1)))
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_idx + lane, (round, offset + lane, "wrapped_idx"))))
                    for lane in range(VLEN):
                        body.append(("debug", ("compare", v_val + lane, (round, offset + lane, "hashed_val"))))
                    vec_i += 1

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


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}): pass
            assert inp.indices == mem[mem[5]:mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6]:mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
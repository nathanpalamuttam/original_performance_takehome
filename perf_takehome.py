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
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            # Simple slot packing that just uses one slot per instruction bundle
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        # VLIW packing: pack independent operations into same cycle
        # Track which scratch addresses are written/read in each slot
        def get_deps(engine, slot):
            writes = set()
            reads = set()

            if engine == "alu":
                op, dest, a1, a2 = slot
                writes.add(dest)
                reads.add(a1)
                reads.add(a2)
            elif engine == "valu":
                if slot[0] == "vbroadcast":
                    _, dest, src = slot
                    for i in range(VLEN):
                        writes.add(dest + i)
                    reads.add(src)
                else:
                    op, dest, a1, a2 = slot[:4]
                    for i in range(VLEN):
                        writes.add(dest + i)
                        reads.add(a1 + i)
                        reads.add(a2 + i)
            elif engine == "load":
                if slot[0] == "load":
                    _, dest, addr = slot
                    writes.add(dest)
                    reads.add(addr)
                elif slot[0] == "vload":
                    _, dest, addr = slot
                    for i in range(VLEN):
                        writes.add(dest + i)
                    reads.add(addr)
                elif slot[0] == "const":
                    _, dest, val = slot
                    writes.add(dest)
            elif engine == "store":
                if slot[0] == "store":
                    _, addr, src = slot
                    reads.add(addr)
                    reads.add(src)
                elif slot[0] == "vstore":
                    _, addr, src = slot
                    reads.add(addr)
                    for i in range(VLEN):
                        reads.add(src + i)
            elif engine == "flow":
                if slot[0] == "select":
                    _, dest, cond, a, b = slot
                    writes.add(dest)
                    reads.add(cond)
                    reads.add(a)
                    reads.add(b)
                elif slot[0] == "vselect":
                    _, dest, cond, a, b = slot
                    for i in range(VLEN):
                        writes.add(dest + i)
                        reads.add(cond + i)
                        reads.add(a + i)
                        reads.add(b + i)
                elif slot[0] == "pause":
                    pass
            elif engine == "debug":
                if slot[0] == "compare":
                    _, loc, key = slot
                    reads.add(loc)
                elif slot[0] == "vcompare":
                    _, loc, keys = slot
                    for i in range(VLEN):
                        reads.add(loc + i)

            return reads, writes

        # Pack slots into instructions respecting dependencies and slot limits
        # O(n) greedy packing: single pass, pack greedily
        instrs = []
        current_instr = {}
        slot_counts = {name: 0 for name in SLOT_LIMITS}
        written_this_cycle = set()

        for engine, slot in slots:
            reads, writes = get_deps(engine, slot)

            # Check if we need to start a new cycle
            need_new_cycle = (
                slot_counts[engine] >= SLOT_LIMITS[engine] or
                bool(reads & written_this_cycle)
            )

            if need_new_cycle:
                # Flush current instruction and start new one
                if current_instr:
                    instrs.append(current_instr)
                current_instr = {}
                slot_counts = {name: 0 for name in SLOT_LIMITS}
                written_this_cycle = set()

            # Add slot to current instruction
            if engine not in current_instr:
                current_instr[engine] = []
            current_instr[engine].append(slot)
            slot_counts[engine] += 1
            written_this_cycle.update(writes)

        # Don't forget the last instruction
        if current_instr:
            instrs.append(current_instr)

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vectorized(self, val_hash_addr, v_tmp1, v_tmp2, v_hash_consts, round, base_offset):
        """Vectorized hash for VLEN lanes at once. v_hash_consts is pre-allocated."""
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_val1 = v_hash_consts[hi * 2]
            v_val3 = v_hash_consts[hi * 2 + 1]

            slots.append(("valu", (op1, v_tmp1, val_hash_addr, v_val1)))
            slots.append(("valu", (op3, v_tmp2, val_hash_addr, v_val3)))
            slots.append(("valu", (op2, val_hash_addr, v_tmp1, v_tmp2)))

            # Debug each lane
            for lane in range(VLEN):
                slots.append(("debug", ("compare", val_hash_addr + lane, (round, base_offset + lane, "hash_stage", hi))))

        return slots

    def build_kernel_vectorized(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation processing VLEN items at a time.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting vectorized loop"))

        body = []

        # Vector scratch registers
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)

        # Scalar temps for address computation
        base_addr = self.alloc_scratch("base_addr")

        # Broadcast constants to vectors
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        body.append(("valu", ("vbroadcast", v_zero, zero_const)))
        body.append(("valu", ("vbroadcast", v_one, one_const)))
        body.append(("valu", ("vbroadcast", v_two, two_const)))
        body.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))

        # Pre-allocate and initialize hash constant vectors
        v_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            val1_const = self.scratch_const(val1)
            val3_const = self.scratch_const(val3)

            v_val1 = self.alloc_scratch(f"v_hash_c1_{hi}", VLEN)
            v_val3 = self.alloc_scratch(f"v_hash_c3_{hi}", VLEN)

            body.append(("valu", ("vbroadcast", v_val1, val1_const)))
            body.append(("valu", ("vbroadcast", v_val3, val3_const)))

            v_hash_consts.append(v_val1)
            v_hash_consts.append(v_val3)

        num_vectors = batch_size // VLEN

        for round in range(rounds):
            for vec_i in range(num_vectors):
                offset = vec_i * VLEN
                offset_const = self.scratch_const(offset)

                # Load indices: v_idx = vload(inp_indices_p + offset)
                body.append(("alu", ("+", base_addr, self.scratch["inp_indices_p"], offset_const)))
                body.append(("load", ("vload", v_idx, base_addr)))

                # Add debug traces for each lane
                for lane in range(VLEN):
                    body.append(("debug", ("compare", v_idx + lane, (round, offset + lane, "idx"))))

                # Load values: v_val = vload(inp_values_p + offset)
                body.append(("alu", ("+", base_addr, self.scratch["inp_values_p"], offset_const)))
                body.append(("load", ("vload", v_val, base_addr)))

                for lane in range(VLEN):
                    body.append(("debug", ("compare", v_val + lane, (round, offset + lane, "val"))))

                # Load node values - need to do individually since indices differ
                # node_val[i] = mem[forest_values_p + idx[i]]
                for lane in range(VLEN):
                    body.append(("alu", ("+", base_addr, self.scratch["forest_values_p"], v_idx + lane)))
                    body.append(("load", ("load", v_node_val + lane, base_addr)))
                    body.append(("debug", ("compare", v_node_val + lane, (round, offset + lane, "node_val"))))

                # v_val = v_val ^ v_node_val
                body.append(("valu", ("^", v_val, v_val, v_node_val)))

                # Hash all lanes vectorized
                body.extend(self.build_hash_vectorized(v_val, v_tmp1, v_tmp2, v_hash_consts, round, offset))

                for lane in range(VLEN):
                    body.append(("debug", ("compare", v_val + lane, (round, offset + lane, "hashed_val"))))

                # v_tmp1 = v_val % 2
                body.append(("valu", ("%", v_tmp1, v_val, v_two)))
                # v_tmp1 = (v_tmp1 == 0)
                body.append(("valu", ("==", v_tmp1, v_tmp1, v_zero)))
                # v_tmp3 = select(v_tmp1, 1, 2)
                body.append(("flow", ("vselect", v_tmp3, v_tmp1, v_one, v_two)))
                # v_idx = v_idx * 2
                body.append(("valu", ("*", v_idx, v_idx, v_two)))
                # v_idx = v_idx + v_tmp3
                body.append(("valu", ("+", v_idx, v_idx, v_tmp3)))

                for lane in range(VLEN):
                    body.append(("debug", ("compare", v_idx + lane, (round, offset + lane, "next_idx"))))

                # v_tmp1 = (v_idx < n_nodes)
                body.append(("valu", ("<", v_tmp1, v_idx, v_n_nodes)))
                # v_idx = select(v_tmp1, v_idx, 0)
                body.append(("flow", ("vselect", v_idx, v_tmp1, v_idx, v_zero)))

                for lane in range(VLEN):
                    body.append(("debug", ("compare", v_idx + lane, (round, offset + lane, "wrapped_idx"))))

                # Store results
                body.append(("alu", ("+", base_addr, self.scratch["inp_indices_p"], offset_const)))
                body.append(("store", ("vstore", base_addr, v_idx)))

                body.append(("alu", ("+", base_addr, self.scratch["inp_values_p"], offset_const)))
                body.append(("store", ("vstore", base_addr, v_val)))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Main entry point - uses vectorized implementation.
        """
        # Use vectorized implementation if batch_size is divisible by VLEN
        if batch_size % VLEN == 0:
            return self.build_kernel_vectorized(forest_height, n_nodes, batch_size, rounds)
        else:
            return self.build_kernel_scalar(forest_height, n_nodes, batch_size, rounds)

    def build_kernel_scalar(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()

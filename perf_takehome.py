"""
# Anthropic's Original Performance Engineering Take-home (Release version)
Copyright Anthropic PBC 2026.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine, DebugInfo, SLOT_LIMITS, VLEN, N_CORES, SCRATCH_SIZE,
    Machine, Tree, Input, HASH_STAGES, reference_kernel, build_mem_image, reference_kernel2,
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
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            deferred_list.append(("load", ("const", addr, val)))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel_pipelined(self, forest_height, n_nodes, batch_size, rounds):
        batch_group = 6
        tmp1 = self.alloc_scratch("tmp1")
        for v in [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(
            [
                "rounds",
                "n_nodes",
                "batch_size",
                "forest_height",
                "forest_values_p",
                "inp_indices_p",
                "inp_values_p",
            ]
        ):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        init_body = []
        one_const = self.scratch_const_deferred(1, init_body)
        two_const = self.scratch_const_deferred(2, init_body)
        three_const = self.scratch_const_deferred(3, init_body)
        four_const = self.scratch_const_deferred(4, init_body)
        five_const = self.scratch_const_deferred(5, init_body)
        six_const = self.scratch_const_deferred(6, init_body)
        self.add("flow", ("pause",))

        num_vectors = batch_size // VLEN
        v_idx_bank = self.alloc_scratch("v_idx_bank", num_vectors * VLEN)
        v_val_bank = self.alloc_scratch("v_val_bank", num_vectors * VLEN)

        v_node_bufs = [
            [self.alloc_scratch(f"v_node_val0_{i}", VLEN) for i in range(batch_group)],
            [self.alloc_scratch(f"v_node_val1_{i}", VLEN) for i in range(batch_group)],
        ]
        first_tmps = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(batch_group)]
        second_tmps = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(batch_group)]

        base_addr_A = self.alloc_scratch("base_addr_A")
        store_addr0 = self.alloc_scratch("store_addr0")
        store_addr1 = self.alloc_scratch("store_addr1")

        addr_fifo_size = 32
        addr_fifo_base = self.alloc_scratch("addr_fifo", addr_fifo_size)

        v_one, v_two, v_n_nodes, v_three = [
            self.alloc_scratch(n, VLEN) for n in ["v_one", "v_two", "v_n_nodes", "v_three"]
        ]

        init_body.append(("valu", ("vbroadcast", v_one, one_const)))
        init_body.append(("valu", ("vbroadcast", v_two, two_const)))
        init_body.append(("valu", ("vbroadcast", v_three, three_const)))
        init_body.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))

        v_hash_consts = []
        hash_const_addrs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_val1 = self.alloc_scratch(f"v_hash_c1_{hi}", VLEN)
            v_val3 = self.alloc_scratch(f"v_hash_c3_{hi}", VLEN)
            c1_addr = self.scratch_const_deferred(val1, init_body)
            c3_addr = self.scratch_const_deferred(val3, init_body)
            init_body.append(("valu", ("vbroadcast", v_val1, c1_addr)))
            init_body.append(("valu", ("vbroadcast", v_val3, c3_addr)))
            v_hash_consts.extend([v_val1, v_val3])
            hash_const_addrs.append((c1_addr, c3_addr))

        # Preload shallow nodes
        v_root, v_node1, v_node2 = [self.alloc_scratch(n, VLEN) for n in ["v_root", "v_node1", "v_node2"]]
        v_node1_minus_node2 = self.alloc_scratch("v_node1_minus_node2", VLEN)
        v_node3, v_node4, v_node5, v_node6 = [
            self.alloc_scratch(n, VLEN) for n in ["v_node3", "v_node4", "v_node5", "v_node6"]
        ]
        v_node4_minus_node3 = self.alloc_scratch("v_node4_minus_node3", VLEN)
        v_node6_minus_node5 = self.alloc_scratch("v_node6_minus_node5", VLEN)

        init_body.append(("load", ("load", tmp1, self.scratch["forest_values_p"])))
        init_body.append(("valu", ("vbroadcast", v_root, tmp1)))
        for const, node in [
            (one_const, v_node1),
            (two_const, v_node2),
            (three_const, v_node3),
            (four_const, v_node4),
            (five_const, v_node5),
            (six_const, v_node6),
        ]:
            init_body.append(("alu", ("+", base_addr_A, self.scratch["forest_values_p"], const)))
            init_body.append(("load", ("load", tmp1, base_addr_A)))
            init_body.append(("valu", ("vbroadcast", node, tmp1)))
        init_body.append(("valu", ("-", v_node1_minus_node2, v_node1, v_node2)))
        init_body.append(("valu", ("-", v_node4_minus_node3, v_node4, v_node3)))
        init_body.append(("valu", ("-", v_node6_minus_node5, v_node6, v_node5)))

        # Load initial idx/val vectors
        offset_consts = [self.scratch_const_deferred(vec_i * VLEN, init_body) for vec_i in range(num_vectors)]
        for vec_i in range(num_vectors):
            init_body.append(("alu", ("+", base_addr_A, self.scratch["inp_indices_p"], offset_consts[vec_i])))
            init_body.append(("load", ("vload", v_idx_bank + vec_i * VLEN, base_addr_A)))
            init_body.append(("alu", ("+", base_addr_A, self.scratch["inp_values_p"], offset_consts[vec_i])))
            init_body.append(("load", ("vload", v_val_bank + vec_i * VLEN, base_addr_A)))
        self.instrs.extend(self.build(init_body, vliw=True))

        # ---------------- Infrastructure state ----------------
        gather_tasks = []
        addr_head = 0
        addr_count = 0
        addr_inflight = []
        addr_ready = []

        # Track pending loads per buffer (gates group readiness)
        buf_pending = [0, 0]

        # Store pipeline (we will ONLY store values at end; no idx stores)
        store_queue = []
        store_ready = []
        store_inflight = []
        store_enabled = False

        def enqueue_gather(v_idxs, v_nodes, buf_idx):
            if not v_idxs:
                return
            buf_pending[buf_idx] += len(v_idxs) * VLEN
            for b in range(len(v_idxs)):
                for lane in range(VLEN):
                    gather_tasks.append((v_idxs[b] + lane, v_nodes[b] + lane, buf_idx))

        def do_gather_alu(alu_slots):
            nonlocal addr_head, addr_count
            cap = SLOT_LIMITS["alu"] - len(alu_slots)
            while cap > 0 and gather_tasks and addr_count < addr_fifo_size:
                idx_reg, dest_reg, buf_idx = gather_tasks.pop(0)
                fifo_idx = addr_head
                addr_head = (addr_head + 1) % addr_fifo_size
                addr_count += 1
                alu_slots.append(("+", addr_fifo_base + fifo_idx, self.scratch["forest_values_p"], idx_reg))
                addr_inflight.append((fifo_idx, dest_reg, buf_idx))
                cap -= 1

        def do_gather_load(load_slots):
            nonlocal addr_count
            cap = SLOT_LIMITS["load"] - len(load_slots)
            while cap > 0 and addr_ready:
                fifo_idx, dest_reg, buf_idx = addr_ready.pop(0)
                load_slots.append(("load", dest_reg, addr_fifo_base + fifo_idx))
                buf_pending[buf_idx] -= 1
                addr_count -= 1
                cap -= 1

        def do_store(cycle):
            if not store_enabled:
                return
            blocked = {r for r, _ in store_ready}
            free = [r for r in [store_addr0, store_addr1] if r not in blocked]

            if store_ready:
                ss = cycle.setdefault("store", [])
                take = min(SLOT_LIMITS["store"] - len(ss), len(store_ready))
                for _ in range(take):
                    addr_reg, src = store_ready.pop(0)
                    ss.append(("vstore", addr_reg, src))

            if store_queue and free:
                alu = cycle.setdefault("alu", [])
                alu_cap = SLOT_LIMITS["alu"] - len(alu)
                n_addrs = min(len(store_queue), len(free), alu_cap)
                for i in range(n_addrs):
                    vec_i = store_queue.pop(0)
                    alu.append(("+", free[i], self.scratch["inp_values_p"], offset_consts[vec_i]))
                    store_inflight.append((free[i], v_val_bank + vec_i * VLEN))

        def end_cycle():
            store_ready.extend(store_inflight)
            store_inflight.clear()
            if addr_inflight:
                addr_ready.extend(addr_inflight)
                addr_inflight.clear()

        def emit(cycle):
            alu = cycle.setdefault("alu", [])
            load = cycle.setdefault("load", [])
            do_gather_alu(alu)
            do_gather_load(load)
            do_store(cycle)
            if not alu:
                del cycle["alu"]
            if not load:
                del cycle["load"]
            self.instrs.append(cycle)
            end_cycle()

        def chunk(slots, lim):
            for i in range(0, len(slots), lim):
                yield slots[i : i + lim]

        # ---------------- Helpers for levels 0-2 ----------------
        def level_0_2_cycles(v_idxs, v_nodes, level, n):
            cycs = []
            if level == 1:
                cycs.extend({"valu": list(c)} for c in chunk([("&", first_tmps[b], v_idxs[b], v_one) for b in range(n)], 6))
                cycs.extend(
                    {"valu": list(c)}
                    for c in chunk(
                        [("multiply_add", v_nodes[b], first_tmps[b], v_node1_minus_node2, v_node2) for b in range(n)], 6
                    )
                )
            elif level == 2:
                for ops in [
                    [("-", first_tmps[b], v_idxs[b], v_three) for b in range(n)],
                    [("&", second_tmps[b], first_tmps[b], v_two) for b in range(n)],
                    [("&", first_tmps[b], first_tmps[b], v_one) for b in range(n)],
                    [(">>", second_tmps[b], second_tmps[b], v_one) for b in range(n)],
                    [("multiply_add", v_nodes[b], first_tmps[b], v_node4_minus_node3, v_node3) for b in range(n)],
                    [("multiply_add", first_tmps[b], first_tmps[b], v_node6_minus_node5, v_node5) for b in range(n)],
                    [("-", first_tmps[b], first_tmps[b], v_nodes[b]) for b in range(n)],
                    [("multiply_add", v_nodes[b], second_tmps[b], first_tmps[b], v_nodes[b]) for b in range(n)],
                ]:
                    cycs.extend({"valu": list(c)} for c in chunk(ops, 6))
            return cycs

        # State machine: 0=XOR, 1..18=hash (3 ops/stage), 19=idx&1, 20=idx_add, 21=idx_mul, 22=wrap_lt, 23=wrap_mul, 24=done
        def get_op(state, v_val, v_idx, v_node, tmpA, tmpB, idx_mode, wrap):
            if state == 0:
                return ("^", v_val, v_val, v_node), 1
            if 1 <= state <= 18:
                hi, sub = (state - 1) // 3, (state - 1) % 3
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                c1, c3 = v_hash_consts[hi * 2], v_hash_consts[hi * 2 + 1]
                if sub == 0:
                    return (op1, tmpA, v_val, c1), state + 1
                if sub == 1:
                    return (op3, tmpB, v_val, c3), state + 1
                return (op2, v_val, tmpA, tmpB), (1 + (hi + 1) * 3 if hi < 5 else 19)
            if state == 19:
                if idx_mode == "skip":
                    return None, 24
                return ("&", tmpA, v_val, v_one), 20
            if state == 20:
                if idx_mode == "depth0":
                    return ("+", v_idx, tmpA, v_one), (22 if wrap else 24)
                return ("+", tmpB, tmpA, v_one), 21
            if state == 21:
                return ("multiply_add", v_idx, v_idx, v_two, tmpB), (22 if wrap else 24)
            if state == 22:
                return ("<", tmpA, v_idx, v_n_nodes), 23
            if state == 23:
                return ("*", v_idx, v_idx, tmpA), 24
            return None, 24

        # ---------------- Level>=3 scheduler ----------------
        def run_level3plus(idx_mode, wrap):
            nonlocal store_queue

            batches = []
            vi = 0
            buf = 0
            while vi < num_vectors:
                n = min(batch_group, num_vectors - vi)
                for b in range(n):
                    off = vi * VLEN + b * VLEN
                    batches.append(
                        {
                            "vi": vi + b,  # vector index
                            "off": off,
                            "v_idx": v_idx_bank + off,
                            "v_val": v_val_bank + off,
                            "v_node": v_node_bufs[buf][b],
                            "tmpA": first_tmps[b],
                            "tmpB": second_tmps[b],
                            "buf": buf,
                            "state": 0,
                            "ready": False,
                            "val_q": False,
                        }
                    )
                vi += n
                buf = 1 - buf

            if not batches:
                return

            # Prefetch first group
            first_n = min(batch_group, len(batches))
            enqueue_gather(
                [b["v_idx"] for b in batches[:first_n]],
                [b["v_node"] for b in batches[:first_n]],
                batches[0]["buf"],
            )
            queued_groups = {0}
            cur_grp = 0

            while any(b["state"] != 24 for b in batches):
                gs = cur_grp * batch_group
                ge = min((cur_grp + 1) * batch_group, len(batches))

                # Mark whole group ready when its buffer pending hits 0.
                if gs < len(batches) and buf_pending[batches[gs]["buf"]] == 0:
                    for i in range(gs, ge):
                        batches[i]["ready"] = True

                valu_slots = []
                valu_cap = SLOT_LIMITS["valu"]

                for b in batches:
                    if valu_cap <= 0:
                        break
                    if b["state"] == 24:
                        continue
                    if b["state"] == 0 and not b["ready"]:
                        continue

                    op, nxt = get_op(
                        b["state"], b["v_val"], b["v_idx"], b["v_node"], b["tmpA"], b["tmpB"], idx_mode, wrap
                    )
                    old = b["state"]
                    b["state"] = nxt
                    if op is not None:
                        valu_slots.append(op)
                        valu_cap -= 1
                    if store_enabled and (old < 19 <= nxt) and (not b["val_q"]):
                        store_queue.append(b["vi"])
                        b["val_q"] = True

                # Prefetch next group
                ng = cur_grp + 1
                if ng not in queued_groups and ng * batch_group < len(batches):
                    ns = ng * batch_group
                    ne = min((ng + 1) * batch_group, len(batches))
                    enqueue_gather(
                        [batches[i]["v_idx"] for i in range(ns, ne)],
                        [batches[i]["v_node"] for i in range(ns, ne)],
                        batches[ns]["buf"],
                    )
                    queued_groups.add(ng)

                if gs < len(batches) and all(batches[i]["state"] == 24 for i in range(gs, ge)):
                    cur_grp += 1
                emit({"valu": valu_slots} if valu_slots else {})

        # ---------------- Levels 0-2 ----------------
        def run_level_0_2(level, idx_mode, wrap):
            nonlocal store_queue
            vi = 0
            buf = 0
            while vi < num_vectors:
                n = min(batch_group, num_vectors - vi)
                offs = [vi * VLEN + i * VLEN for i in range(n)]
                v_idxs = [v_idx_bank + o for o in offs]
                v_vals = [v_val_bank + o for o in offs]
                v_nodes = [v_root] * n if level == 0 else v_node_bufs[buf][:n]

                if level > 0:
                    for c in level_0_2_cycles(v_idxs, v_nodes, level, n):
                        emit(c)

                # XOR + hash
                for c in chunk([("^", v_vals[b], v_vals[b], v_nodes[b]) for b in range(n)], 6):
                    emit({"valu": list(c)})

                for hi in range(len(HASH_STAGES)):
                    op1, _, op2, op3, _ = HASH_STAGES[hi]
                    c1, c3 = v_hash_consts[hi * 2], v_hash_consts[hi * 2 + 1]
                    ops = []
                    for b in range(n):
                        ops.extend([(op1, first_tmps[b], v_vals[b], c1), (op3, second_tmps[b], v_vals[b], c3)])
                    for c in chunk(ops, 6):
                        emit({"valu": list(c)})
                    for c in chunk([(op2, v_vals[b], first_tmps[b], second_tmps[b]) for b in range(n)], 6):
                        emit({"valu": list(c)})

                # Queue value stores only on final round
                if store_enabled:
                    for b in range(n):
                        store_queue.append(offs[b] // VLEN)

                # idx update (still needed for correctness for next rounds)
                if idx_mode != "skip":
                    for c in chunk([("&", first_tmps[b], v_vals[b], v_one) for b in range(n)], 6):
                        emit({"valu": list(c)})
                    if idx_mode == "depth0":
                        for c in chunk([("+", v_idxs[b], first_tmps[b], v_one) for b in range(n)], 6):
                            emit({"valu": list(c)})
                    else:
                        for c in chunk([("+", second_tmps[b], first_tmps[b], v_one) for b in range(n)], 6):
                            emit({"valu": list(c)})
                        for c in chunk([("multiply_add", v_idxs[b], v_idxs[b], v_two, second_tmps[b]) for b in range(n)], 6):
                            emit({"valu": list(c)})
                    if wrap:
                        for c in chunk([("<", first_tmps[b], v_idxs[b], v_n_nodes) for b in range(n)], 6):
                            emit({"valu": list(c)})
                        for c in chunk([("*", v_idxs[b], v_idxs[b], first_tmps[b]) for b in range(n)], 6):
                            emit({"valu": list(c)})

                vi += n
                buf = 1 - buf

        # ---------------- Main loop ----------------
        for rnd in range(rounds):
            level = rnd % (forest_height + 1)
            last = (rnd == rounds - 1)

            # Only store on the final round; and ONLY store values (no idx stores).
            store_enabled = last

            wrap = (level == forest_height)
            idx_mode = "skip" if last else ("depth0" if level == 0 else "full")

            if level >= 3:
                run_level3plus(idx_mode, wrap)
            else:
                run_level_0_2(level, idx_mode, wrap)

        # ---------------- Drain stores ----------------
        while store_queue or store_ready:
            cycle = {}
            # Add a pause at the end once the last stores can drain in one cycle
            if (not store_queue) and (len(store_ready) <= SLOT_LIMITS["store"]):
                cycle["flow"] = [("pause",)]
            do_store(cycle)
            # still allow gather machinery to run if anything somehow remains
            alu = cycle.setdefault("alu", [])
            load = cycle.setdefault("load", [])
            do_gather_alu(alu)
            do_gather_load(load)
            if not alu:
                del cycle["alu"]
            if not load:
                del cycle["load"]
            self.instrs.append(cycle)
            end_cycle()
            if "flow" in cycle:
                break

        if not (store_queue or store_ready):
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

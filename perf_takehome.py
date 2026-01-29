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
        
        # Allocate header variables
        header_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in header_vars:
            self.alloc_scratch(v, 1)
        
        # FIX: Use deferred building with separate temp registers to enable VLIW packing
        # Instead of reusing tmp1 (which creates dependencies), use separate temps
        header_body = []
        header_temps = [self.alloc_scratch(f"tmp_h{i}") for i in range(len(header_vars))]
        
        # First, emit all const ops (these can be packed)
        for i in range(len(header_vars)):
            header_body.append(("load", ("const", header_temps[i], i)))
        
        # Then emit all load ops (these depend on the consts, but can be packed together)
        for i, v in enumerate(header_vars):
            header_body.append(("load", ("load", self.scratch[v], header_temps[i])))
        
        # Build with VLIW scheduling - this will pack loads efficiently
        self.instrs.extend(self.build(header_body, vliw=True))

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

        addr_fifo_size = 64
        # Size the interleave window for a 2-window pipeline to fit within remaining scratch.
        future_fixed = (
            3  # base_addr_A + store_addr0 + store_addr1
            + addr_fifo_size  # addr_fifo
            + 4 * VLEN  # v_one/v_two/v_n_nodes/v_three
            + 12 * VLEN  # v_hash_consts (6 stages * 2)
            + 3 * VLEN  # v_root/v_node1/v_node2
            + VLEN  # v_node1_minus_node2
            + 4 * VLEN  # v_node3..v_node6
            + VLEN  # v_node4_minus_node3
            + VLEN  # v_node6_minus_node5
            + num_vectors  # offset_consts
        )
        remaining = SCRATCH_SIZE - self.scratch_ptr - future_fixed
        max_window = max(1, remaining // (2 * 3 * VLEN))
        window_vectors = min(num_vectors, max_window)
        bank_stride = window_vectors * VLEN
        v_node_bank = self.alloc_scratch("v_node_bank", 2 * bank_stride)
        tmpA_bank = self.alloc_scratch("v_tmp1_bank", 2 * bank_stride)
        tmpB_bank = self.alloc_scratch("v_tmp2_bank", 2 * bank_stride)

        base_addr_A = self.alloc_scratch("base_addr_A")
        store_addr0 = self.alloc_scratch("store_addr0")
        store_addr1 = self.alloc_scratch("store_addr1")

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
        vec_const_map = {
            v_one: one_const,
            v_two: two_const,
            v_three: three_const,
            v_n_nodes: self.scratch["n_nodes"],
        }

        # Preload shallow nodes
        v_root, v_node1, v_node2 = [self.alloc_scratch(n, VLEN) for n in ["v_root", "v_node1", "v_node2"]]
        v_node1_minus_node2 = self.alloc_scratch("v_node1_minus_node2", VLEN)
        v_node3, v_node4, v_node5, v_node6 = [
            self.alloc_scratch(n, VLEN) for n in ["v_node3", "v_node4", "v_node5", "v_node6"]
        ]
        v_node4_minus_node3 = self.alloc_scratch("v_node4_minus_node3", VLEN)
        v_node6_minus_node5 = self.alloc_scratch("v_node6_minus_node5", VLEN)

        # Allocate second address register and temp for pipelined loading
        base_addr_B = self.alloc_scratch("base_addr_B")
        tmp2 = self.alloc_scratch("tmp2")

        # Load root node
        init_body.append(("load", ("load", tmp1, self.scratch["forest_values_p"])))
        init_body.append(("valu", ("vbroadcast", v_root, tmp1)))
        
        # Load nodes 1-6 with pipelining: process in pairs
        node_pairs = [
            ((one_const, v_node1), (two_const, v_node2)),
            ((three_const, v_node3), (four_const, v_node4)),
            ((five_const, v_node5), (six_const, v_node6)),
        ]
        for (const_a, node_a), (const_b, node_b) in node_pairs:
            # Emit pair of ALUs
            init_body.append(("alu", ("+", base_addr_A, self.scratch["forest_values_p"], const_a)))
            init_body.append(("alu", ("+", base_addr_B, self.scratch["forest_values_p"], const_b)))
            # Emit pair of loads
            init_body.append(("load", ("load", tmp1, base_addr_A)))
            init_body.append(("load", ("load", tmp2, base_addr_B)))
            # Emit pair of broadcasts
            init_body.append(("valu", ("vbroadcast", node_a, tmp1)))
            init_body.append(("valu", ("vbroadcast", node_b, tmp2)))
        
        init_body.append(("valu", ("-", v_node1_minus_node2, v_node1, v_node2)))
        init_body.append(("valu", ("-", v_node4_minus_node3, v_node4, v_node3)))
        init_body.append(("valu", ("-", v_node6_minus_node5, v_node6, v_node5)))

        # Load initial idx/val vectors - restructured for better VLIW pipelining
        # Pattern: (alu A, alu B, load A, load B) repeated allows pipelining
        offset_consts = [self.scratch_const_deferred(vec_i * VLEN, init_body) for vec_i in range(num_vectors)]
        
        # Process idx vectors in pairs for pipelining
        for i in range(0, num_vectors, 2):
            init_body.append(("alu", ("+", base_addr_A, self.scratch["inp_indices_p"], offset_consts[i])))
            if i + 1 < num_vectors:
                init_body.append(("alu", ("+", base_addr_B, self.scratch["inp_indices_p"], offset_consts[i + 1])))
            init_body.append(("load", ("vload", v_idx_bank + i * VLEN, base_addr_A)))
            if i + 1 < num_vectors:
                init_body.append(("load", ("vload", v_idx_bank + (i + 1) * VLEN, base_addr_B)))
        
        # Process val vectors in pairs for pipelining
        for i in range(0, num_vectors, 2):
            init_body.append(("alu", ("+", base_addr_A, self.scratch["inp_values_p"], offset_consts[i])))
            if i + 1 < num_vectors:
                init_body.append(("alu", ("+", base_addr_B, self.scratch["inp_values_p"], offset_consts[i + 1])))
            init_body.append(("load", ("vload", v_val_bank + i * VLEN, base_addr_A)))
            if i + 1 < num_vectors:
                init_body.append(("load", ("vload", v_val_bank + (i + 1) * VLEN, base_addr_B)))
        self.instrs.extend(self.build(init_body, vliw=True))

        # ---------------- Infrastructure state ----------------
        gather_tasks = []
        addr_head = 0
        addr_count = 0
        addr_inflight = []
        addr_ready = []

        # Track pending loads per batch (overlap load/compute within group)
        batch_pending = {}
        batches_ref = {}

        # Store pipeline (we will ONLY store values at end; no idx stores)
        store_queue = []
        store_ready = []
        store_inflight = []
        store_enabled = False

        def enqueue_gather(v_idxs, v_nodes, batch_ids):
            if not v_idxs:
                return
            for b in range(len(v_idxs)):
                batch_id = batch_ids[b]
                batch_pending[batch_id] = VLEN
                for lane in range(VLEN):
                    gather_tasks.append((v_idxs[b] + lane, v_nodes[b] + lane, batch_id))

        def do_gather_alu(alu_slots, max_slots=None):
            nonlocal addr_head, addr_count
            cap = (SLOT_LIMITS["alu"] if max_slots is None else max_slots) - len(alu_slots)
            while cap > 0 and gather_tasks and addr_count < addr_fifo_size:
                idx_reg, dest_reg, batch_id = gather_tasks.pop(0)
                fifo_idx = addr_head
                addr_head = (addr_head + 1) % addr_fifo_size
                addr_count += 1
                alu_slots.append(("+", addr_fifo_base + fifo_idx, self.scratch["forest_values_p"], idx_reg))
                addr_inflight.append((fifo_idx, dest_reg, batch_id))
                cap -= 1

        def do_gather_load(load_slots):
            nonlocal addr_count
            cap = SLOT_LIMITS["load"] - len(load_slots)
            while cap > 0 and addr_ready:
                fifo_idx, dest_reg, batch_id = addr_ready.pop(0)
                load_slots.append(("load", dest_reg, addr_fifo_base + fifo_idx))
                batch_pending[batch_id] -= 1
                if batch_pending[batch_id] == 0 and batch_id in batches_ref:
                    batches_ref[batch_id]["ready"] = True
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

        # State machine: 0=XOR, 1..18=hash (3 ops/stage), 19=idx&1, 20=idx_add, 21=idx_mul, 22=wrap_lt, 23=wrap_mul, 24=done
        def get_op(state, v_val, v_idx, v_node, tmpA, tmpB, idx_mode, wrap):
            if state == 0:
                return ("^", v_val, v_val, v_node), 1
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

        # ---------------- Interleaved scheduler (all levels) ----------------
        def emit_alu_lane_op(op, v_dest, v_src, const_addr):
            # Hash sub-ops that are lane-independent (val op const) can run as 8 scalar ALU ops.
            # This reduces VALU pressure while preserving semantics for op1/op3 in each hash stage.
            return [(op, v_dest + lane, v_src + lane, const_addr) for lane in range(VLEN)]

        def emit_alu_lane_op_mixed(op, v_dest, a1, a2):
            ops = []
            for lane in range(VLEN):
                src1 = vec_const_map.get(a1, a1 + lane)
                src2 = vec_const_map.get(a2, a2 + lane)
                ops.append((op, v_dest + lane, src1, src2))
            return ops

        def get_hash_ops(state, v_val, tmpA, tmpB):
            hi, sub = (state - 1) // 3, (state - 1) % 3
            op1, _, op2, op3, _ = HASH_STAGES[hi]
            c1_vec, c3_vec = v_hash_consts[hi * 2], v_hash_consts[hi * 2 + 1]
            c1_s, c3_s = hash_const_addrs[hi]
            if sub == 0:
                return (op1, tmpA, v_val, c1_vec), emit_alu_lane_op(op1, tmpA, v_val, c1_s), state + 1
            if sub == 1:
                return (op3, tmpB, v_val, c3_vec), emit_alu_lane_op(op3, tmpB, v_val, c3_s), state + 1
            return (op2, v_val, tmpA, tmpB), None, (1 + (hi + 1) * 3 if hi < 5 else 19)

        def get_hash_pair(state, v_val, tmpA, tmpB):
            # Allow op1/op3 in the same cycle when VALU has room (independent reads of v_val).
            hi, sub = (state - 1) // 3, (state - 1) % 3
            if sub != 0:
                return None, None, None
            op1, _, _, op3, _ = HASH_STAGES[hi]
            c1, c3 = v_hash_consts[hi * 2], v_hash_consts[hi * 2 + 1]
            return (op1, tmpA, v_val, c1), (op3, tmpB, v_val, c3), state + 2

        def get_pre_op(level, state, v_idx, v_node, tmpA, tmpB):
            if level == 1:
                if state == -2:
                    return ("&", tmpA, v_idx, v_one), -1
                return ("multiply_add", v_node, tmpA, v_node1_minus_node2, v_node2), 0
            if level == 2:
                if state == -8:
                    return ("-", tmpA, v_idx, v_three), -7
                if state == -7:
                    return ("&", tmpB, tmpA, v_two), -6
                if state == -6:
                    return ("&", tmpA, tmpA, v_one), -5
                if state == -5:
                    return (">>", tmpB, tmpB, v_one), -4
                if state == -4:
                    return ("multiply_add", v_node, tmpA, v_node4_minus_node3, v_node3), -3
                if state == -3:
                    return ("multiply_add", tmpA, tmpA, v_node6_minus_node5, v_node5), -2
                if state == -2:
                    return ("-", tmpA, tmpA, v_node), -1
                return ("multiply_add", v_node, tmpB, tmpA, v_node), 0
            return None, 0

        def start_state(level):
            if level == 1:
                return -2
            if level == 2:
                return -8
            return 0

        def run_interleaved():
            nonlocal store_queue
            nonlocal batch_pending, batches_ref
            nonlocal store_enabled

            store_enabled = True
            next_vi = 0
            next_batch_id = 0
            windows = [None, None]

            def start_round(b):
                b["level"] = b["round"] % (forest_height + 1)
                b["state"] = start_state(b["level"])
                b["val_q"] = False
                if b["level"] >= 3:
                    b["ready"] = False
                    enqueue_gather([b["v_idx"]], [b["v_node"]], [b["id"]])
                else:
                    b["ready"] = True

            def init_window(slot_id):
                nonlocal next_vi, next_batch_id
                if next_vi >= num_vectors:
                    windows[slot_id] = None
                    return
                vecs = list(range(next_vi, min(next_vi + window_vectors, num_vectors)))
                next_vi = vecs[-1] + 1
                base = slot_id * bank_stride
                batches = []
                for slot, vi in enumerate(vecs):
                    off = vi * VLEN
                    buf_off = base + slot * VLEN
                    batch_id = next_batch_id
                    next_batch_id += 1
                    b = {
                        "id": batch_id,
                        "vi": vi,
                        "v_idx": v_idx_bank + off,
                        "v_val": v_val_bank + off,
                        "v_node": v_node_bank + buf_off,
                        "tmpA": tmpA_bank + buf_off,
                        "tmpB": tmpB_bank + buf_off,
                        "round": 0,
                        "level": 0,
                        "state": 0,
                        "ready": True,
                        "val_q": False,
                        "done": False,
                    }
                    batches.append(b)
                    batches_ref[batch_id] = b
                for b in batches:
                    start_round(b)
                windows[slot_id] = {"batches": batches}

            def all_batches():
                bs = []
                for w in windows:
                    if w:
                        bs.extend(w["batches"])
                return bs

            init_window(0)
            init_window(1)

            while any(windows):
                batches = all_batches()
                valu_slots = []
                valu_cap = SLOT_LIMITS["valu"]
                valu_only = []
                offloadable = []

                for b in batches:
                    if b["done"] or b["state"] == 24:
                        continue
                    if b["state"] == 0 and b["level"] >= 3 and (not b["ready"]):
                        continue

                    last_round = (b["round"] == rounds - 1)
                    idx_mode = "skip" if last_round else ("depth0" if b["level"] == 0 else "full")
                    wrap = (b["level"] == forest_height)
                    node_src = v_root if b["level"] == 0 else b["v_node"]

                    if b["state"] < 0:
                        op, nxt = get_pre_op(b["level"], b["state"], b["v_idx"], b["v_node"], b["tmpA"], b["tmpB"])
                        valu_only.append((b, op, nxt, last_round))
                    elif 1 <= b["state"] <= 18:
                        vop, aops, nxt = get_hash_ops(b["state"], b["v_val"], b["tmpA"], b["tmpB"])
                        if aops is not None:
                            offloadable.append((b, vop, aops, nxt, last_round))
                        else:
                            valu_only.append((b, vop, nxt, last_round))
                    else:
                        op, nxt = get_op(
                            b["state"], b["v_val"], b["v_idx"], node_src, b["tmpA"], b["tmpB"], idx_mode, wrap
                        )
                        allow_offload = (
                            (b["state"] >= 19 and op is not None and op[0] != "multiply_add")
                            or (b["state"] == 0 and op is not None and op[0] == "^")
                        )
                        if allow_offload:
                            aops = emit_alu_lane_op_mixed(op[0], op[1], op[2], op[3])
                            offloadable.append((b, op, aops, nxt, last_round))
                        else:
                            valu_only.append((b, op, nxt, last_round))

                scheduled = set()

                def mark_progress(b, old, nxt, last_round):
                    b["state"] = nxt
                    scheduled.add(id(b))
                    if last_round and (old < 19 <= nxt) and (not b["val_q"]):
                        store_queue.append(b["vi"])
                        b["val_q"] = True

                cycle = {}
                alu_slots = cycle.setdefault("alu", [])
                load_slots = cycle.setdefault("load", [])

                reserve_hash = VLEN if offloadable else 0
                do_gather_alu(alu_slots, max_slots=SLOT_LIMITS["alu"] - reserve_hash)
                do_gather_load(load_slots)
                do_store(cycle)
                alu_cap = SLOT_LIMITS["alu"] - len(alu_slots)

                for b, vop, aops, nxt, last_round in offloadable:
                    if alu_cap < VLEN:
                        break
                    old = b["state"]
                    alu_slots.extend(aops)
                    alu_cap -= VLEN
                    mark_progress(b, old, nxt, last_round)

                # Schedule VALU ops (including fallback for hash const ops not offloaded).
                for b, op, nxt, last_round in valu_only:
                    if valu_cap <= 0:
                        break
                    if id(b) in scheduled:
                        continue
                    old = b["state"]
                    if 1 <= old <= 18 and (old - 1) % 3 == 0 and valu_cap >= 2:
                        op1, op3, nxt2 = get_hash_pair(old, b["v_val"], b["tmpA"], b["tmpB"])
                        if op1 is not None and op3 is not None:
                            valu_slots.extend([op1, op3])
                            valu_cap -= 2
                            mark_progress(b, old, nxt2, last_round)
                            continue
                    if op is not None:
                        valu_slots.append(op)
                        valu_cap -= 1
                    mark_progress(b, old, nxt, last_round)

                for b, vop, aops, nxt, last_round in offloadable:
                    if valu_cap <= 0:
                        break
                    if id(b) in scheduled:
                        continue
                    old = b["state"]
                    if 1 <= old <= 18 and (old - 1) % 3 == 0 and valu_cap >= 2:
                        op1, op3, nxt2 = get_hash_pair(old, b["v_val"], b["tmpA"], b["tmpB"])
                        if op1 is not None and op3 is not None:
                            valu_slots.extend([op1, op3])
                            valu_cap -= 2
                            mark_progress(b, old, nxt2, last_round)
                            continue
                    if vop is not None:
                        valu_slots.append(vop)
                        valu_cap -= 1
                    mark_progress(b, old, nxt, last_round)

                if valu_slots:
                    cycle["valu"] = valu_slots
                if not alu_slots:
                    del cycle["alu"]
                if not load_slots:
                    del cycle["load"]
                if "valu" not in cycle and not cycle:
                    cycle = {}
                self.instrs.append(cycle)
                end_cycle()

                for b in batches:
                    if b["done"] or b["state"] != 24:
                        continue
                    if b["round"] == rounds - 1:
                        b["done"] = True
                    else:
                        # Fallback if a batch somehow reached 24 without advance.
                        b["round"] += 1
                        start_round(b)

                for i, w in enumerate(windows):
                    if not w:
                        continue
                    if all(b["done"] for b in w["batches"]):
                        for b in w["batches"]:
                            batches_ref.pop(b["id"], None)
                            batch_pending.pop(b["id"], None)
                        init_window(i)

        # ---------------- Main loop ----------------
        run_interleaved()

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

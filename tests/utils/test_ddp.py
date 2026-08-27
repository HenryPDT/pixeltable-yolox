import os
import tempfile
import unittest
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.data.samplers import DistributedEvalSampler
from yolox.utils.allreduce_norm import average_gradients


def _ddp_worker(rank, world_size, result_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = nn.Linear(4, 2, bias=False)
    ddp_model = DDP(model, device_ids=None)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    grad_accum_steps = 3

    for step in range(7):
        is_boundary = (step + 1) % grad_accum_steps == 0
        sync_cm = ddp_model.no_sync() if not is_boundary else nullcontext()
        with sync_cm:
            out = ddp_model(torch.ones(1, 4))
            out.sum().backward()

        if is_boundary or step == 6:
            if step == 6 and not is_boundary:
                average_gradients(ddp_model)
            optimizer.step()
            optimizer.zero_grad()

    if rank == 0:
        torch.save(ddp_model.module.state_dict(), result_path)

    dist.destroy_process_group()


class TestDDPTraining(unittest.TestCase):
    def test_two_process_cpu_ddp_trailing_flush_completes(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "ddp_state.pt")
            mp.spawn(
                _ddp_worker,
                args=(world_size, tmp_path),
                nprocs=world_size,
                join=True,
            )
            state = torch.load(tmp_path, map_location="cpu", weights_only=False)
            self.assertIn("weight", state)

    def test_distributed_eval_sampler_cpu_world_size_two(self):
        seen = set()
        for rank in range(2):
            sampler = DistributedEvalSampler(range(5), num_replicas=2, rank=rank)
            seen.update(list(sampler))
        self.assertEqual(seen, set(range(5)))

    def test_distributed_eval_sampler_non_divisible_world_size(self):
        seen = set()
        for rank in range(3):
            sampler = DistributedEvalSampler(range(10), num_replicas=3, rank=rank)
            seen.update(list(sampler))
        self.assertEqual(seen, set(range(10)))

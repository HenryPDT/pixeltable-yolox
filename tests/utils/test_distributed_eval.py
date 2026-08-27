import unittest

from yolox.data.samplers import DistributedEvalSampler


class TestDistributedEvalSampler(unittest.TestCase):
    def test_every_sample_evaluated_exactly_once(self):
        dataset_size = 17
        world_size = 4
        seen = set()
        for rank in range(world_size):
            sampler = DistributedEvalSampler(range(dataset_size), num_replicas=world_size, rank=rank)
            indices = list(sampler)
            self.assertEqual(len(indices), len(sampler))
            for idx in indices:
                self.assertNotIn(idx, seen)
                seen.add(idx)
        self.assertEqual(len(seen), dataset_size)

    def test_non_divisible_world_size_has_unequal_shard_lengths(self):
        lengths = []
        for rank in range(3):
            sampler = DistributedEvalSampler(range(10), num_replicas=3, rank=rank)
            lengths.append(len(sampler))
        self.assertEqual(sorted(lengths), [3, 3, 4])

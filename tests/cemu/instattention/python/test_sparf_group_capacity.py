import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cemu_flexgen.sparf_group_capacity import (
    GROUP_IO_BYTES, GROUP_STATUS_BYTES, check_fdm_space, check_filesystem_space,
    grouped_attention_reserve_bytes, plan_group_capacity, sparf_attention_range_bytes,
)


class SparfGroupCapacityTest(unittest.TestCase):
    def test_batched_scratch_matches_device_allocation(self):
        from cemu_flexgen.sparf_group_device import SparfGroupedDevice
        from cemu_flexgen.sparf_group_layout import SparfGroupLayout

        geometry = dict(num_layers=2, batch_size=2, num_kv_heads=4, head_dim=128,
                        element_bytes=2, page_bytes=4096, channels_per_page=8, max_seq_len=531)
        capacity = plan_group_capacity(**geometry, write_batch_pages=256)
        device = SparfGroupedDevice(SparfGroupLayout(128, 8), "/unused/pool", capacity.pool_bytes,
                                    "/unused/fdm", "/unused/program", buffer_streams=capacity.streams,
                                    write_batch_pages=256)
        self.assertEqual(capacity.fdm_required_bytes, device.tail_bytes + GROUP_IO_BYTES + GROUP_STATUS_BYTES)
        self.assertEqual(device.tail_bytes - capacity.fdm_tail_bytes, 256 * 4096)
        for pages in (-1, True, 15, 16385):
            with self.subTest(pages=pages), self.assertRaises(ValueError):
                plan_group_capacity(**geometry, write_batch_pages=pages)

    def test_opt13b_capacity_matches_documented_bounds(self):
        one = plan_group_capacity(
            num_layers=40, batch_size=1, num_kv_heads=40, head_dim=128,
            element_bytes=2, page_bytes=4096, channels_per_page=8,
            max_seq_len=2048,
        )
        four = plan_group_capacity(
            num_layers=40, batch_size=4, num_kv_heads=40, head_dim=128,
            element_bytes=2, page_bytes=4096, channels_per_page=8,
            max_seq_len=2048,
        )
        self.assertEqual(one.streams, 1600)
        self.assertEqual(four.streams, 6400)
        self.assertEqual(one.pages, 614400)
        self.assertEqual(four.pages, 2457600)
        self.assertEqual(one.fdm_tail_bytes, 222822400)
        self.assertEqual(one.fdm_required_bytes, 222887936 + GROUP_IO_BYTES + GROUP_STATUS_BYTES)
        self.assertEqual(four.fdm_tail_bytes, 891289600)
        self.assertEqual(four.fdm_required_bytes, 891355136 + GROUP_IO_BYTES + GROUP_STATUS_BYTES)
        self.assertEqual(four.pool_bytes, 10066329600)

    def test_pool_and_geometry_overflow_is_rejected(self):
        kwargs = dict(
            num_layers=2, batch_size=1, num_kv_heads=2, head_dim=128,
            element_bytes=2, page_bytes=4096, channels_per_page=8,
            max_seq_len=32,
        )
        with self.assertRaisesRegex(ValueError, "pool is too small"):
            plan_group_capacity(**kwargs, pool_bytes=4096)
        with self.assertRaisesRegex(ValueError, "integral"):
            plan_group_capacity(**{**kwargs, "channels_per_page": 7})

    def test_all_storage_ranges_and_attention_allocation_alignment_are_counted(self):
        geometry = dict(batch_size=1, query_heads=4, kv_heads=2, head_dim=8,
                        element_bytes=2, max_seq_len=9, top_r=16, compression_ratio=8)
        sizes = sparf_attention_range_bytes(**geometry)
        self.assertEqual(sizes[1], 4 * 8 * 9 * 2)
        reserve = grouped_attention_reserve_bytes(attention_slots=2, **geometry)
        self.assertEqual(reserve, 2 * 11 * 4096)
        kwargs = dict(num_layers=1, batch_size=1, num_kv_heads=2, head_dim=8,
                      element_bytes=2, page_bytes=4096, channels_per_page=8,
                      max_seq_len=9, extra_fdm_bytes=reserve)
        capacity = plan_group_capacity(**kwargs)
        self.assertEqual(capacity.fdm_required_bytes,
                         capacity.fdm_tail_bytes + 4096 + GROUP_IO_BYTES + GROUP_STATUS_BYTES + reserve)
        with self.assertRaisesRegex(ValueError, "budget is too small"):
            plan_group_capacity(**kwargs, fdm_budget_bytes=capacity.fdm_required_bytes - 1)
        self.assertEqual(plan_group_capacity(**kwargs, fdm_budget_bytes=capacity.fdm_required_bytes), capacity)

    def test_attention_workspace_bound_fails_before_device_allocation(self):
        with self.assertRaisesRegex(ValueError, "workspace bounds"):
            grouped_attention_reserve_bytes(
                attention_slots=2, batch_size=32, query_heads=40, kv_heads=40,
                head_dim=128, element_bytes=2, max_seq_len=2048,
                top_r=16, compression_ratio=8,
            )

    def test_unknown_fdm_capacity_requires_explicit_budget(self):
        usage = SimpleNamespace(total=0, used=0, free=0)
        with patch("cemu_flexgen.sparf_group_capacity.shutil.disk_usage", return_value=usage):
            with self.assertRaisesRegex(ValueError, "SPARF_GROUP_FDM_BUDGET_BYTES"):
                check_fdm_space("/mnt/fdm0", 4096)
            self.assertEqual(check_fdm_space("/mnt/fdm0", 4096, 8192), 8192)
            with self.assertRaisesRegex(ValueError, "insufficient FDM budget"):
                check_fdm_space("/mnt/fdm0", 4096, 4095)

    def test_reported_free_space_and_explicit_budget_both_apply(self):
        usage = SimpleNamespace(total=16384, used=4096, free=12288)
        with patch("cemu_flexgen.sparf_group_capacity.shutil.disk_usage", return_value=usage):
            self.assertEqual(check_fdm_space("/mnt/fdm0", 4096), 12288)
            self.assertEqual(check_fdm_space("/mnt/fdm0", 4096, 8192), 8192)
            with self.assertRaisesRegex(ValueError, "insufficient FDM budget"):
                check_fdm_space("/mnt/fdm0", 13000, 32768)
            with self.assertRaisesRegex(ValueError, "insufficient FDM budget"):
                check_fdm_space("/mnt/fdm0", 9000, 8192)

    def test_known_full_filesystem_is_not_treated_as_unknown(self):
        usage = SimpleNamespace(total=16384, used=16384, free=0)
        with patch("cemu_flexgen.sparf_group_capacity.shutil.disk_usage", return_value=usage):
            with self.assertRaisesRegex(ValueError, "insufficient FDM budget"):
                check_fdm_space("/mnt/fdm0", 4096, 16384)
            with self.assertRaisesRegex(ValueError, "insufficient filesystem space"):
                check_filesystem_space("/mnt/nvme0", 4096)

    def test_invalid_explicit_fdm_budgets_are_rejected(self):
        for budget in (0, -1, True, 4096.0):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                check_fdm_space("/unused", 4096, budget)

    def test_statfs_errors_are_not_masked_by_a_budget(self):
        with patch("cemu_flexgen.sparf_group_capacity.shutil.disk_usage", side_effect=OSError("missing mount")):
            with self.assertRaisesRegex(OSError, "missing mount"):
                check_fdm_space("/unused", 4096, 8192)


if __name__ == "__main__":
    unittest.main()

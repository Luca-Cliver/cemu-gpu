import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cemu_flexgen.sparf_group_capacity import (
    GROUP_IO_BYTES, GROUP_STATUS_BYTES, check_fdm_space, check_filesystem_space,
    grouped_attention_reserve_bytes, plan_group_capacity, sparf_attention_range_bytes,
)


class SparfGroupCapacityTest(unittest.TestCase):
    def launch_profile(self, profile, model="OPT-13B", **overrides):
        repo = Path(__file__).resolve().parents[4]
        environment = {name: value for name, value in os.environ.items()
                       if not name.startswith(("CEMU_", "CUDA_"))}
        environment.update(CEMU_NATIVE_PROFILE=profile, CEMU_NATIVE_DRY_RUN="1",
                           CEMU_NATIVE_CSD_TARGET="cpu", CEMU_NATIVE_KV_PLACEMENT="balanced",
                           CEMU_QEMU_BINARY="/nonexistent/qemu")
        environment.update(overrides)
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "sessions"
            environment["CEMU_NATIVE_TEST_PARENT"] = str(parent)
            result = subprocess.run(["bash", str(repo / "femu-scripts/run-native-opt-test.sh"), model],
                                    env=environment, capture_output=True, text=True, timeout=10)
            self.assertFalse(parent.exists())
        settings = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        return result, settings

    def test_b8_launch_capacity_keeps_nand_and_compute_unchanged(self):
        result, settings = self.launch_profile("b8-p1024-d16")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("input slot is sized by the Guest client", result.stderr)
        self.assertEqual(settings["guest_memory"], "64G")
        self.assertEqual(settings["native_buffer_budget"], str(12 * 1024**3))
        self.assertEqual(settings["native_page_limit"], str(1 << 22))
        self.assertEqual(settings["native_kv_placement"], "balanced")
        self.assertEqual(settings["nvm_backend"], "file")
        path = Path(settings["device_config"])
        self.assertEqual(path.name, "fdm_opt13b_b8_p1024_d16.json")
        config = json.loads(path.read_text())
        previous = json.loads(path.with_name("fdm_opt13b_b4_p1024_d16.json").read_text())
        self.assertEqual(config["namespaces"][1]["size"], "16gb")
        config["namespaces"][1]["size"] = previous["namespaces"][1]["size"]
        self.assertEqual(config, previous)
        capacity = plan_group_capacity(
            num_layers=40, batch_size=8, num_kv_heads=40, head_dim=128,
            element_bytes=2, page_bytes=4096, channels_per_page=8, max_seq_len=1040)
        self.assertEqual(capacity.pages, 2688000)
        self.assertEqual(capacity.pool_bytes, 11010048000)
        self.assertLessEqual(capacity.pages, int(settings["native_page_limit"]))
        self.assertLess(capacity.pool_bytes, 96 * 1024**3)
        reserve = grouped_attention_reserve_bytes(
            attention_slots=2, batch_size=8, query_heads=40, kv_heads=40,
            head_dim=128, element_bytes=2, max_seq_len=1040, top_r=16, compression_ratio=8)
        self.assertLess(8 * 1024**3 + 128 * 1024**2 + reserve, 16 * 1024**3)

    def test_existing_launch_capacities_are_unchanged(self):
        for profile, model, memory, budget, pages in (
                ("smoke", "OPT-125M", "8G", 67108864, 65536),
                ("smoke", "OPT-13B", "64G", 536870912, 65536),
                ("b4-p1024-d16", "OPT-13B", "64G", 6442450944, 2097152)):
            with self.subTest(profile=profile, model=model):
                result, settings = self.launch_profile(profile, model)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(settings["guest_memory"], memory)
                self.assertEqual(int(settings["native_buffer_budget"]), budget)
                self.assertEqual(int(settings["native_page_limit"]), pages)
                self.assertNotIn("multi-data-MR", result.stderr)
                if profile == "b4-p1024-d16":
                    self.assertEqual(Path(settings["device_config"]).name,
                                     "cemu_config_opt13b_sparf_engines2.json")

    def test_capacity_dry_run_validates_selection_and_honors_override(self):
        result, _ = self.launch_profile("b8-p1024-d16", "OPT-125M")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires OPT-13B", result.stderr)
        result, _ = self.launch_profile("invalid")
        self.assertNotEqual(result.returncode, 0)
        result, _ = self.launch_profile("b8-p1024-d16", CEMU_NATIVE_DRY_RUN="invalid")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CEMU_NATIVE_DRY_RUN", result.stderr)
        result, _ = self.launch_profile("b8-p1024-d16", CEMU_NATIVE_CONFIG="/nonexistent/config.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Device config not readable", result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "custom.json"
            config.write_text("{}")
            result, settings = self.launch_profile("b8-p1024-d16", CEMU_NATIVE_CONFIG=str(config))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(settings["device_config"], str(config))

    def test_aggregate_poll_selection(self):
        result, settings = self.launch_profile("smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(settings["fdm_aggregate_mode"], "notify")
        result, settings = self.launch_profile("smoke", CEMU_FDM_AGGREGATE_MODE="poll",
                                               CEMU_FDM_ASYNC_AGGREGATE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(settings["fdm_aggregate_mode"], "poll")
        self.assertEqual(settings["fdm_async_aggregate"], "1")
        for mode in ("poll", "invalid"):
            with self.subTest(mode=mode):
                result, _ = self.launch_profile("smoke", CEMU_FDM_AGGREGATE_MODE=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CEMU_FDM_AGGREGATE_MODE", result.stderr)

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

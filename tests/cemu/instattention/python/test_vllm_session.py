from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from vllm_adapter.session import Config, Session


class VllmSessionCleanupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "session"
        self.config = Config(str(self.directory), "attention.so", "aggregate.so", max_tokens=32)
        self.store = MagicMock()
        self.store.ranges = [SimpleNamespace(path=str(self.directory / name)) for name in
                             ("kv_buffer_meta", "kv_buffer_data", "kv_buffer_data_1")]
        self.store.statistics.return_value = dict(sealed=1, failed=0)
        self.device = MagicMock()
        self.device._config = dict(ranges=[SimpleNamespace(path=str(self.directory / name)) for name in
                                          ("vllm_batch_query", "vllm_batch_output")])
        self.store.__enter__.side_effect = lambda *args: self.enter(self.store, self.store.ranges)
        self.device.__enter__.side_effect = lambda *args: self.enter(self.device, self.device._config["ranges"])
        self.store.__exit__.side_effect = lambda *args: self.assert_files(self.store.ranges)
        self.device.close.side_effect = lambda: self.assert_files(self.device._config["ranges"])
        self.device.__exit__.side_effect = lambda *args: self.device.close()
        self.store_factory = self.patch_factory("cemu_flexgen.fdm_kvbuffer_store.FDM_KVbuffer", self.store)
        self.device_factory = self.patch_factory("cemu_flexgen.cemu_sparf_device.CemuSparfDevice", self.device)

    def patch_factory(self, name, resource):
        patcher = patch(name, return_value=resource)
        factory = patcher.start()
        self.addCleanup(patcher.stop)
        return factory

    def enter(self, resource, ranges):
        for region in ranges:
            Path(region.path).write_bytes(b"owned")
        return resource

    def assert_files(self, ranges):
        for region in ranges:
            self.assertEqual(Path(region.path).read_bytes(), b"owned")

    def session(self):
        return Session(self.config, layers=1, query_heads=2, kv_heads=2, head_dim=128, scale=1.0)

    def test_success_releases_before_unlink_and_allows_next_run(self):
        for iteration in range(2):
            with self.subTest(iteration=iteration):
                session = self.session().open()
                session.close()
                session.close()
                self.assertFalse(self.directory.exists())
                self.assertEqual(session.final_statistics, dict(sealed=1, failed=0))
        self.assertEqual(self.store.drain.call_count, 2)
        self.assertEqual(self.store.__exit__.call_count, 2)
        self.assertEqual(self.device.__exit__.call_count, 2)

    def test_existing_directory_is_not_claimed(self):
        self.directory.mkdir()
        marker = self.directory / "previous-session"
        marker.write_text("keep")
        session = self.session()
        with self.assertRaises(FileExistsError):
            session.open()
        session.close()
        self.assertEqual(marker.read_text(), "keep")
        self.store_factory.assert_not_called()
        self.device_factory.assert_not_called()

    def test_drain_error_preserves_files(self):
        session = self.session().open()
        self.store.drain.side_effect = RuntimeError("drain failed")
        with self.assertRaisesRegex(RuntimeError, "drain failed"):
            session.close()
        self.assert_files(self.store.ranges + self.device._config["ranges"])
        self.store.__exit__.assert_called_once()
        self.device.__exit__.assert_called_once()

    def test_unsealed_drain_preserves_files(self):
        session = self.session().open()
        self.store.statistics.return_value = dict(sealed=0, failed=0)
        with self.assertRaisesRegex(RuntimeError, "did not seal"):
            session.close()
        self.assert_files(self.store.ranges + self.device._config["ranges"])

    def test_release_error_preserves_files(self):
        session = self.session().open()
        self.store.__exit__.side_effect = RuntimeError("release failed")
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            session.close()
        self.assert_files(self.store.ranges + self.device._config["ranges"])

    def test_failed_session_preserves_files(self):
        session = self.session().open()
        session.failure = RuntimeError("append failed")
        session.close()
        self.store.drain.assert_not_called()
        self.assert_files(self.store.ranges + self.device._config["ranges"])

    def test_unrelated_file_is_never_deleted(self):
        session = self.session().open()
        marker = self.directory / "unrelated"
        marker.write_text("keep")
        with self.assertRaises(OSError):
            session.close()
        self.assertEqual(marker.read_text(), "keep")
        self.assertFalse(Path(self.store.ranges[0].path).exists())

    def test_open_error_releases_store_but_preserves_files(self):
        self.device.__enter__.side_effect = RuntimeError("attention open failed")
        session = self.session()
        with self.assertRaisesRegex(RuntimeError, "attention open failed"):
            session.open()
        session.close()
        self.store.__exit__.assert_called_once()
        self.store.drain.assert_not_called()
        self.assert_files(self.store.ranges)


if __name__ == "__main__":
    unittest.main()

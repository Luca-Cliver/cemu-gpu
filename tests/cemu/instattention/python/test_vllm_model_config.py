import unittest
from types import SimpleNamespace

from vllm_adapter.model_config import model_geometry


class ModelGeometryTest(unittest.TestCase):
    def test_opt13b_config_without_kv_head_field(self):
        config = dict(model_type="opt", num_hidden_layers=40,
                      num_attention_heads=40, hidden_size=5120)
        expected = (40, 40, 40, 128, 128**-.5)
        self.assertEqual(model_geometry(config), expected)
        self.assertEqual(model_geometry(SimpleNamespace(**config)), expected)

    def test_llama_gqa_is_preserved(self):
        config = dict(model_type="llama", num_hidden_layers=32,
                      num_attention_heads=32, num_key_value_heads=8, hidden_size=4096)
        self.assertEqual(model_geometry(config), (32, 32, 8, 128, 128**-.5))

    def test_opt125m_is_rejected_before_worker_allocation(self):
        config = dict(model_type="opt", num_hidden_layers=12,
                      num_attention_heads=12, hidden_size=768)
        with self.assertRaisesRegex(ValueError, "head_dim=128, got 64"):
            model_geometry(config)

    def test_unhandled_architecture_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "Llama and OPT"):
            model_geometry(dict(model_type="gpt2"))


if __name__ == "__main__":
    unittest.main()

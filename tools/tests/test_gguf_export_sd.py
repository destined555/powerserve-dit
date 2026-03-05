import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "gguf_export_sd.py"


class GGUFExportSDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("gguf_export_sd", MODULE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load module from {MODULE_PATH}")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_make_sd_convert_cmd(self):
        cmd = self.module.make_sd_convert_cmd(
            Path("/tmp/sd-cli"),
            Path("sd3.5/sd3.5_medium.safetensors"),
            Path("out/ggml/weights.gguf"),
            "q8_0",
        )
        self.assertEqual(cmd[0], "/tmp/sd-cli")
        self.assertEqual(cmd[1:4], ["-M", "convert", "-m"])
        self.assertIn("--type", cmd)
        self.assertIn("q8_0", cmd)
        self.assertIn("out/ggml/weights.gguf", cmd)

    def test_build_model_config_with_all_components(self):
        converted = {
            "weights": "ggml/weights.gguf",
            "clip_l": "ggml/clip_l.gguf",
            "clip_g": "ggml/clip_g.gguf",
            "t5xxl": "ggml/t5xxl.gguf",
            "vae": "ggml/vae.gguf",
        }
        config = self.module.build_model_config("sd3.5-medium", "q8_0", converted)
        self.assertEqual(config["model_id"], "sd3.5-medium")
        self.assertEqual(config["model_arch"], "sd3")
        self.assertEqual(config["components"]["weights"]["path"], "ggml/weights.gguf")
        self.assertEqual(config["components"]["clip_l"]["format"], "gguf")
        self.assertEqual(config["components"]["vae"]["path"], "ggml/vae.gguf")
        self.assertEqual(config["quantization"]["out_type"], "q8_0")

    def test_resolve_model_inputs_requires_all_components(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            model_path = tmp / "sd3.5_medium.safetensors"
            model_path.write_bytes(b"dummy")
            args = SimpleNamespace(
                model_dir=None,
                model_path=model_path,
                clip_l=None,
                clip_g=None,
                t5xxl=None,
                vae=None,
            )
            with self.assertRaises(ValueError):
                self.module._resolve_model_inputs(args)

    def test_is_abi_mismatch_error_true_for_glibcxx(self):
        err = "/workspace/tools/sd_converter/bin/sd-cli: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found"
        self.assertTrue(self.module.is_abi_mismatch_error(err))

    def test_is_abi_mismatch_error_false_for_generic_error(self):
        err = "failed to open model file"
        self.assertFalse(self.module.is_abi_mismatch_error(err))


if __name__ == "__main__":
    unittest.main()

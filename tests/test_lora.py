import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acinotes import naming  # noqa: E402
from acinotes.generate import adapter_variant, thinking_modes  # noqa: E402
from acinotes.train_lora import adapter_variant_name, build_example, lr_factor, make_groups  # noqa: E402


class FakeTok:
    """Tokenitzador de joguina: cada caràcter és un token. Plantilla: SYS|USER|ASSISTANT: nota<end>."""
    eos_token = "</s>"

    def get_vocab(self):
        return {}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        text = "".join(f"[{m['role']}]{m['content']}" for m in msgs if m["role"] != "assistant")
        if add_generation_prompt:
            return text + "[assistant]"
        last = msgs[-1]
        return text + "[assistant]" + last["content"] + "<end>\n"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class TrainingPureTest(unittest.TestCase):
    def test_build_example_masks_prompt_and_covers_note(self):
        e = build_example(FakeTok(), "conversa", "  NOTA CLINICA  ", max_len=100000)
        self.assertEqual(e["comp_len"], len("NOTA CLINICA<end>"))       # la nota (sense espais) + final de torn
        self.assertEqual(len(e["ids"]), e["prompt_len"] + e["comp_len"])
        self.assertEqual(bytes(e["ids"][e["prompt_len"]:]).decode(), "NOTA CLINICA<end>")

    def test_too_long_is_excluded_not_truncated(self):
        self.assertIsNone(build_example(FakeTok(), "x" * 5000, "nota", max_len=100))

    def test_groups_cover_all_examples_once(self):
        g = make_groups(10, 4, seed=1, epoch=0)
        self.assertEqual([len(x) for x in g], [4, 4, 2])
        self.assertEqual(sorted(i for grp in g for i in grp), list(range(10)))
        self.assertNotEqual(g, make_groups(10, 4, seed=1, epoch=1))   # un ordre diferent a cada època

    def test_lr_schedule(self):
        self.assertAlmostEqual(lr_factor(0, 100, 10), 0.1)            # escalfament
        self.assertAlmostEqual(lr_factor(9, 100, 10), 1.0)
        self.assertAlmostEqual(lr_factor(100, 100, 10), 0.1)          # acaba al 10 %

    def test_variant_name(self):
        a = SimpleNamespace(r=16, targets="attn", epochs=3, lr=2e-4, smoke=False)
        self.assertEqual(adapter_variant_name(a, "bf16"), "r16-attn-ep3-lr0.0002-bf16")
        a.smoke = True
        self.assertTrue(adapter_variant_name(a, "bf16").startswith("smoke-"))


class AdapterNamingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.aid = naming.AdapterId.new("Qwen3-4B-Instruct-2507", "r16-attn-ep3-lr0.0002-bf16", "20261118T2130Z")
        self.root = naming.adapter_dir(self.aid, self.tmp)
        for w in ("best", "epoch_2"):
            (self.root / w).mkdir(parents=True)

    def test_stem_roundtrip(self):
        self.assertEqual(self.aid.stem,
                         "lora__qwen3-4b-instruct-2507__r16-attn-ep3-lr0.0002-bf16__20261118T2130Z")
        self.assertEqual(naming.parse_adapter_stem(self.aid.stem), self.aid)
        with self.assertRaises(ValueError):
            naming.parse_adapter_stem("valid__zeroshot__a__b__20261118T2130Z")

    def test_resolve_by_id_defaults_to_best(self):
        ref = naming.resolve_adapter(self.aid.stem, self.tmp)
        self.assertEqual((ref.which, ref.weights), ("best", self.root / "best"))

    def test_resolve_epoch_and_path(self):
        ref = naming.resolve_adapter(f"{self.aid.stem}/epoch_2", self.tmp)
        self.assertEqual((ref.which, ref.aid), ("epoch_2", self.aid))
        ref2 = naming.resolve_adapter(str(self.root / "epoch_2"), self.tmp)
        self.assertEqual(ref2.which, "epoch_2")
        with self.assertRaises(FileNotFoundError):
            naming.resolve_adapter("no-existeix", self.tmp)

    def test_prediction_name_identifies_adapter(self):
        ref = naming.resolve_adapter(f"{self.aid.stem}/epoch_2", self.tmp)
        v = adapter_variant(ref, "nothink-4bit-bf16")
        pid = naming.PredictionId.new("valid", "lora", "qwen3-4b-instruct-2507", v, "20261119T0900Z")
        self.assertEqual(naming.parse_stem(pid.stem), pid)
        self.assertIn("epoch-2", v)
        self.assertIn("20261118t2130z", v)


class ThinkingMessageTest(unittest.TestCase):
    def test_suggests_the_other_model(self):
        with self.assertRaises(ValueError) as c:
            thinking_modes("Qwen/Qwen3-4B-Instruct-2507", "on")
        self.assertIn("Thinking-2507", str(c.exception))
        with self.assertRaises(ValueError) as c:
            thinking_modes("Qwen/Qwen3-4B-Thinking-2507", "off")
        self.assertIn("Instruct-2507", str(c.exception))


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acinotes.generate import model_short_name, sampling_kwargs, split_reasoning, thinking_modes  # noqa: E402
from acinotes.prompts import PROMPT_BUILDERS  # noqa: E402


class GenerateTest(unittest.TestCase):
    def test_split_with_reasoning(self):
        note, reasoning, trunc = split_reasoning("<think>\npenso\n</think>\n\nCHIEF COMPLAINT\nTos", True)
        self.assertEqual((note, reasoning, trunc), ("CHIEF COMPLAINT\nTos", "penso", False))

    def test_thinking_truncated(self):
        note, reasoning, trunc = split_reasoning("<think>\nencara penso", True)
        self.assertEqual((note, trunc), ("", True))
        self.assertIn("encara penso", reasoning)

    def test_no_thinking(self):
        self.assertEqual(split_reasoning("  CHIEF COMPLAINT\nTos ", False), ("CHIEF COMPLAINT\nTos", "", False))

    def test_sampling_defaults(self):
        self.assertFalse(sampling_kwargs(False)["do_sample"])
        s = sampling_kwargs(True)
        self.assertTrue(s["do_sample"])
        self.assertEqual((s["temperature"], s["top_p"], s["top_k"]), (0.6, 0.95, 20))
        self.assertFalse(sampling_kwargs(True, temperature=0)["do_sample"])

    def test_model_short_name(self):
        self.assertEqual(model_short_name("Qwen/Qwen3-32B"), "qwen3-32b")
        self.assertEqual(model_short_name("Qwen/Qwen3-4B-Instruct-2507"), "qwen3-4b-instruct-2507")

    def test_prompt_contains_conversation(self):
        msgs = PROMPT_BUILDERS["zeroshot"]("[doctor] hola")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn("[doctor] hola", msgs[1]["content"])

    def test_thinking_modes(self):
        self.assertEqual(thinking_modes("Qwen/Qwen3-4B-Instruct-2507"), [False])
        self.assertEqual(thinking_modes("Qwen/Qwen3-4B-Thinking-2507"), [True])
        self.assertEqual(thinking_modes("Qwen/Qwen3-32B"), [False, True])
        self.assertEqual(thinking_modes("Qwen/Qwen3-32B", "on"), [True])
        with self.assertRaises(ValueError):
            thinking_modes("Qwen/Qwen3-4B-Instruct-2507", "on")
        with self.assertRaises(ValueError):
            thinking_modes("Qwen/Qwen3-4B-Thinking-2507", "off")


if __name__ == "__main__":
    unittest.main()

import json
import io
import unittest

from llm_policies.openai_policy import build_payload, extract_output_text, read_policy_request


class OpenAIPolicyTests(unittest.TestCase):
    def test_schema_restricts_output_to_allowed_experiments(self):
        request = {
            "allowed_experiments": [
                {"experiment_id": "first"},
                {"experiment_id": "second"},
            ]
        }
        payload = build_payload(request, "test-model")
        enum = payload["text"]["format"]["schema"]["properties"]["experiment_id"]["enum"]
        self.assertEqual(enum, ["first", "second"])
        self.assertFalse(payload["store"])

    def test_extracts_structured_message_text(self):
        expected = {"experiment_id": "first", "reason": "test"}
        response = {
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(expected)}],
            }]
        }
        self.assertEqual(json.loads(extract_output_text(response)), expected)

    def test_accepts_utf8_bom_from_windows_pipe(self):
        expected = {"allowed_experiments": [{"experiment_id": "first"}]}
        raw = io.BytesIO(b"\xef\xbb\xbf" + json.dumps(expected).encode("utf-8"))
        stream = io.TextIOWrapper(raw, encoding="utf-8")
        self.assertEqual(read_policy_request(stream), expected)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from rag import GuideIndex, OllamaClient, RagError, UNKNOWN_ANSWER, answer_question, checked_answer, load_guide


class RagTests(unittest.TestCase):
    def setUp(self):
        self.entries = load_guide(Path(__file__).resolve().parents[1] / "knowledge_base.json")
        self.hits = [{"entry": self.entries[0], "score": .8}]

    def test_guide_contains_unique_places_and_recommendation_metadata(self):
        self.assertEqual(len(self.entries), 16)
        self.assertEqual(len({e["id"] for e in self.entries}), 16)
        self.assertTrue(all(e["category"] and e["tags"] for e in self.entries))

    def test_cosine_retrieval_orders_entries_and_preserves_scores(self):
        index = GuideIndex(self.entries[:3], np.array([[1., 0.], [0., 1.], [.6, .8]]))
        client = Mock()
        client.embed.return_value = np.array([[0., 1.]])
        hits = index.retrieve("question", client, 2)
        self.assertEqual([h["entry"]["id"] for h in hits], ["G02", "G03"])
        self.assertAlmostEqual(hits[1]["score"], .8)

    def test_build_batches_the_whole_guide_once(self):
        client = Mock()
        client.embed.return_value = np.eye(len(self.entries))
        index = GuideIndex.build(self.entries, client)
        client.embed.assert_called_once()
        self.assertEqual(len(client.embed.call_args.args[0]), 16)
        self.assertFalse(index.vectors.flags.writeable)

    def test_category_filter_excludes_other_categories_before_ranking(self):
        index = GuideIndex(self.entries[:3], np.array([[1., 0.], [0., 1.], [.6, .8]]))
        client = Mock()
        client.embed.return_value = np.array([[1., 0.]])
        hits = index.retrieve("question", client, 3, "museum")
        self.assertEqual([hit["entry"]["id"] for hit in hits], ["G02"])

    def test_exact_quote_gets_a_valid_source_id(self):
        draft = {"answerable": True, "sentences": [{"entry_id": "G01", "quote": "Dinh Độc Lập ở 135 Nam Kỳ Khởi Nghĩa."}]}
        self.assertEqual(checked_answer(draft, self.hits), "Dinh Độc Lập ở 135 Nam Kỳ Khởi Nghĩa. [G01]")

    def test_invented_quotes_and_unretrieved_citations_are_rejected(self):
        for item in [{"entry_id": "G01", "quote": "Giá vé là 999.000 đồng."},
                     {"entry_id": "G99", "quote": "Dinh Độc Lập ở 135 Nam Kỳ Khởi Nghĩa."}]:
            self.assertIsNone(checked_answer({"answerable": True, "sentences": [item]}, self.hits))

    def test_low_similarity_refuses_without_calling_generator(self):
        index, client = Mock(), Mock()
        index.retrieve.return_value = [{"entry": self.entries[0], "score": .1}]
        result = answer_question("Thời tiết ngày mai?", index, client)
        self.assertFalse(result["answerable"])
        self.assertEqual(result["answer"], UNKNOWN_ANSWER)
        client.generate.assert_not_called()

    def test_missing_evidence_refuses_even_when_retrieval_is_high(self):
        index, client = Mock(), Mock()
        index.retrieve.return_value = self.hits
        client.generate.return_value = {"answerable": False, "sentences": []}
        self.assertFalse(answer_question("Giá không có trong guide?", index, client)["answerable"])

    def test_empty_question_is_rejected_before_network_access(self):
        index, client = Mock(), Mock()
        with self.assertRaises(RagError): answer_question("   ", index, client)
        index.retrieve.assert_not_called()

    @patch("rag.requests.post")
    def test_embeddings_are_normalized_and_use_required_model(self, post):
        post.return_value.ok = True
        post.return_value.json.return_value = {"embeddings": [[3., 4.]]}
        result = OllamaClient("https://example.org").embed(["hello"])
        np.testing.assert_allclose(result, [[.6, .8]])
        self.assertEqual(post.call_args.kwargs["json"]["model"], "embeddinggemma")

    @patch("rag.requests.post")
    def test_zero_embeddings_and_malformed_payloads_fail_clearly(self, post):
        post.return_value.ok = True
        for payload in [{"embeddings": [[0., 0.]]}, {"embeddings": [[float("nan"), 1.]]}, {}]:
            post.return_value.json.return_value = payload
            with self.assertRaises(RagError): OllamaClient("https://example.org").embed(["hello"])

    @patch("rag.requests.post")
    def test_generation_uses_qwen_and_a_strict_json_schema(self, post):
        post.return_value.ok = True
        post.return_value.json.return_value = {"message": {"content": json.dumps({"answerable": False, "sentences": []})}}
        OllamaClient("https://example.org").generate("price?", self.hits)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"]["type"], "object")


if __name__ == "__main__": unittest.main()

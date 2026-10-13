"""Offline retrieval relevance and public log privacy regressions."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


RAG_PATH = Path(__file__).resolve().parents[1] / "rag.py"


class RAGV41Tests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        knowledge = directory / "knowledge"
        meta = directory / "knowledge_meta"
        knowledge.mkdir()
        meta.mkdir()
        (meta / "source_registry.json").write_text(json.dumps({"sources": [{
            "source_id": "fixture-guide", "title": "随访指南", "year": 2024,
            "base_priority": 95, "population": ["elderly"],
            "population_boost": {"age>=65": 10},
        }]}), encoding="utf-8")
        (knowledge / "guidance.md").write_text(
            "block_id: home-bp\nsource_id: fixture-guide\n"
            "topic: home_bp_monitoring\n# 家庭血压\n"
            "家庭血压测量前安静休息，使用合适的袖带并记录读数。\n\n"
            "block_id: medication\nsource_id: fixture-guide\n"
            "topic: medication_safety\nsafety_layer_candidate: true\n"
            "# 服药安全\n忘记吃药时不要自行多吃一片，请咨询医生。\n",
            encoding="utf-8",
        )
        spec = importlib.util.spec_from_file_location("isolated_rag_v41", RAG_PATH)
        self.rag = importlib.util.module_from_spec(spec)
        # Startup reads only the synthetic knowledge fixtures and never loads
        # an embedding model, environment file or real patient data.
        self.rag.__file__ = str(directory / "rag.py")
        with patch.dict(os.environ, {
            "RAG_DISABLE_EMBEDDING": "1",
            "RAG_CACHE_DIR": str(directory / "cache"),
        }), contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(self.rag)

    def retrieve(self, query, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results = self.rag.retrieve_medical_guidance(query, **kwargs)
        return results, output.getvalue()

    def test_unknown_query_has_no_evidence_despite_metadata_bonuses(self):
        query = "银河星舰曲率发动机"
        self.assertTrue(all(self.rag.lexical_score(query, block["search_text"]) == 0
                            for block in self.rag.knowledge_blocks))
        for kwargs in ({}, {"patient_state": {"age": 72}}, {
            "patient_state": {"age": 72, "medication_status": "missed"},
            "topic_hint": "medication_safety",
        }):
            with self.subTest(kwargs=kwargs):
                results, _ = self.retrieve(query, **kwargs)
                self.assertEqual(results, [])

    def test_real_medical_topic_returns_relevant_evidence_without_zero_score_filler(self):
        for query, expected in [("家庭血压袖带", "home-bp"),
                                ("忘记吃药多吃一片", "medication")]:
            with self.subTest(query=query):
                results, _ = self.retrieve(query, patient_state={"age": 72})
                self.assertEqual([result["block_id"] for result in results], [expected])
                self.assertGreater(results[0]["semantic_score"], 0)
                self.assertEqual(results[0]["source_id"], "fixture-guide")

    def test_empty_query_returns_no_results(self):
        for query in (None, "", "   "):
            with self.subTest(query=query):
                self.assertEqual(self.retrieve(query)[0], [])

    def test_public_logs_contain_only_diagnostics_not_patient_text(self):
        query = "姓名隐私标记张先生，家庭血压袖带怎么选"
        patient_marker = "私密患者状态手机号13800000000"
        results, output = self.retrieve(query, patient_state={
            "age": 72, "name": patient_marker, "red_flags": [patient_marker],
        })
        self.assertTrue(results)
        self.assertNotIn(query, output)
        self.assertNotIn("姓名隐私标记", output)
        self.assertNotIn(patient_marker, output)
        self.assertIn("Selected：", output)
        self.assertIn("fixture-guide", output)

    def test_query_inference_failure_does_not_log_exception_input(self):
        secret = "模型完整输入含患者姓名和电话13812345678"
        self.rag.block_embeddings = np.ones((2, 2), dtype=np.float32)
        with patch.object(self.rag, "_ensure_embedding_index", return_value=True), \
                patch.object(self.rag, "encode_query", side_effect=RuntimeError(secret)):
            results, output = self.retrieve("家庭血压袖带")
        self.assertEqual(results[0]["block_id"], "home-bp")
        self.assertNotIn(secret, output)
        self.assertIn("查询向量生成失败", output)

    def test_document_inference_failure_does_not_log_exception_input(self):
        secret = "模型完整文档输入及患者状态隐私标记"
        self.rag.DISABLE_EMBEDDING = False
        self.rag.embedding_available = True
        with patch.object(self.rag, "load_embedding_model"), \
                patch.object(self.rag, "encode_documents", side_effect=RuntimeError(secret)):
            results, output = self.retrieve("家庭血压袖带")
        self.assertEqual(results[0]["block_id"], "home-bp")
        self.assertNotIn(secret, output)
        self.assertIn("知识块向量生成失败", output)


if __name__ == "__main__":
    unittest.main()

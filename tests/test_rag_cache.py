"""Offline RAG cache regressions; all fixture files live in a temp directory."""

import builtins
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np


RAG_PATH = Path(__file__).resolve().parents[1] / "rag.py"
RESULT_FIELDS = {
    "block_id", "topic", "subtopic", "source_id", "source_title", "source_file",
    "year", "source_pages", "chapter", "population", "authority", "priority",
    "safety_layer_candidate", "semantic_score", "topic_bonus", "population_bonus",
    "priority_bonus", "score", "why_selected", "routed_topics", "text",
}


class FakeEmbeddingModel:
    """Deterministic normalized vectors, without downloads or model weights."""

    constructed = []
    documents = []
    queries = []
    fail_documents = False
    fail_queries = False
    document_delay = 0

    def __init__(self, name):
        self.name = name
        self.constructed.append(name)

    def get_sentence_embedding_dimension(self):
        return 4

    def vector(self, text):
        raw = hashlib.sha256((self.name + text).encode("utf-8")).digest()
        vector = np.asarray([value + 1 for value in raw[:4]], dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def encode_document(self, texts, normalize_embeddings):
        assert normalize_embeddings is True
        self.documents.append(list(texts))
        if self.fail_documents:
            raise RuntimeError("offline document inference failed")
        time.sleep(self.document_delay)
        return np.asarray([self.vector(text) for text in texts])

    def encode_query(self, texts, normalize_embeddings):
        assert normalize_embeddings is True
        self.queries.append(list(texts))
        if self.fail_queries:
            raise RuntimeError("offline query inference failed")
        return np.asarray([self.vector(text) for text in texts])


class RAGCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.knowledge = self.directory / "knowledge"
        self.meta = self.directory / "knowledge_meta"
        self.cache = self.directory / "cache" / "knowledge_embeddings.npz"
        self.knowledge.mkdir()
        self.meta.mkdir()
        self.registry = self.meta / "source_registry.json"
        self.registry.write_text(json.dumps({"sources": [{
            "source_id": "guide", "title": "随访指南", "year": 2024,
            "base_priority": 95, "population": ["elderly"],
            "population_boost": {"age>=65": 10},
        }]}), encoding="utf-8")
        self.markdown = self.knowledge / "guidance.md"
        self.markdown.write_text(
            "block_id: home_bp\nsource_id: guide\ntopic: home_bp_monitoring\n"
            "subtopic: technique\nsource_pages: 1\n"
            "# 家庭血压\n在家测血压应安静休息后测量并记录血压。\n\n"
            "block_id: medication\nsource_id: guide\ntopic: medication_safety\n"
            "safety_layer_candidate: true\n"
            "# 服药安全\n忘记吃降压药时不要自行多吃一片，请咨询医生。\n",
            encoding="utf-8",
        )
        FakeEmbeddingModel.constructed = []
        FakeEmbeddingModel.documents = []
        FakeEmbeddingModel.queries = []
        FakeEmbeddingModel.fail_documents = False
        FakeEmbeddingModel.fail_queries = False
        FakeEmbeddingModel.document_delay = 0
        fake_package = types.ModuleType("sentence_transformers")
        fake_package.SentenceTransformer = FakeEmbeddingModel
        self.stack.enter_context(patch.dict("sys.modules", {
            "sentence_transformers": fake_package,
        }))

    def load_rag(self, model_name="test-model", disabled=False):
        spec = importlib.util.spec_from_file_location("isolated_test_rag", RAG_PATH)
        module = importlib.util.module_from_spec(spec)
        # The loader reads the real source but every derived project path points
        # into the fixture; startup never reads/writes real medical data.
        module.__file__ = str(self.directory / "rag.py")
        with patch.dict(os.environ, {
            "RAG_CACHE_DIR": str(self.cache.parent),
            "RAG_EMBEDDING_MODEL": model_name,
            "RAG_DISABLE_EMBEDDING": "1" if disabled else "0",
        }):
            spec.loader.exec_module(module)
        return module

    def retrieve(self, module):
        results = module.retrieve_medical_guidance(
            "在家怎么测血压", patient_state={"age": 72},
            topic_hint="home_bp_monitoring",
        )
        self.assertTrue(results)
        self.assertEqual(set(results[0]), RESULT_FIELDS)
        self.assertEqual(results[0]["block_id"], "home_bp")
        return results

    def test_import_does_not_import_or_construct_sentence_transformer(self):
        original_import = builtins.__import__

        def reject_heavy_import(name, *args, **kwargs):
            if name == "sentence_transformers" or name.startswith("sentence_transformers."):
                raise AssertionError("heavy model imported during module startup")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_heavy_import):
            module = self.load_rag()
        self.assertEqual(len(module.knowledge_blocks), 2)
        self.assertIsNone(module.embedding_model)
        self.assertIsNone(module.block_embeddings)
        self.assertFalse(self.cache.exists())
        self.assertEqual(FakeEmbeddingModel.constructed, [])

    def test_first_query_builds_cache_and_fresh_start_reuses_it(self):
        first = self.load_rag()
        original = self.retrieve(first)
        self.assertEqual(FakeEmbeddingModel.constructed, ["test-model"])
        self.assertEqual(len(FakeEmbeddingModel.documents), 1)
        self.assertEqual(len(FakeEmbeddingModel.queries), 1)
        self.assertTrue(self.cache.is_file())
        cache_time = self.cache.stat().st_mtime_ns
        with np.load(self.cache, allow_pickle=False) as cache:
            self.assertEqual(cache["embeddings"].shape, (2, 4))
            self.assertEqual(cache["embeddings"].dtype, np.float32)

        restarted = self.load_rag()
        self.assertIsNone(restarted.embedding_model)
        self.assertEqual(restarted.block_embeddings.shape, (2, 4))
        self.assertEqual(FakeEmbeddingModel.constructed, ["test-model"])
        self.assertEqual(self.retrieve(restarted), original)
        self.assertEqual(len(FakeEmbeddingModel.documents), 1)
        self.assertEqual(len(FakeEmbeddingModel.queries), 2)
        self.assertEqual(self.cache.stat().st_mtime_ns, cache_time)
        restarted.rebuild_index()
        self.retrieve(restarted)
        self.assertEqual(len(FakeEmbeddingModel.documents), 1)
        self.assertEqual(self.cache.stat().st_mtime_ns, cache_time)

    def test_touching_unchanged_files_keeps_cache_valid(self):
        self.retrieve(self.load_rag())
        timestamp = self.markdown.stat().st_mtime_ns + 1_000_000_000
        os.utime(self.markdown, ns=(timestamp, timestamp))
        os.utime(self.registry, ns=(timestamp, timestamp))
        self.retrieve(self.load_rag())
        self.assertEqual(len(FakeEmbeddingModel.documents), 1)

    def test_knowledge_content_registry_and_model_changes_invalidate_cache(self):
        self.retrieve(self.load_rag())
        self.markdown.write_text(self.markdown.read_text(encoding="utf-8") +
                                 "补充记录早晚测量时间。\n", encoding="utf-8")
        changed = self.load_rag()
        self.assertIsNone(changed.block_embeddings)
        self.assertEqual(len(FakeEmbeddingModel.documents), 1)
        self.retrieve(changed)
        self.assertEqual(len(FakeEmbeddingModel.documents), 2)

        metadata = json.loads(self.registry.read_text(encoding="utf-8"))
        metadata["sources"][0]["title"] = "更新的随访指南"
        self.registry.write_text(json.dumps(metadata), encoding="utf-8")
        changed = self.load_rag()
        self.assertIsNone(changed.block_embeddings)
        self.assertEqual(changed.knowledge_blocks[0]["source_title"], "更新的随访指南")
        self.retrieve(changed)
        self.assertEqual(len(FakeEmbeddingModel.documents), 3)

        changed = self.load_rag(model_name="different-model")
        self.assertIsNone(changed.block_embeddings)
        self.retrieve(changed)
        self.assertEqual(len(FakeEmbeddingModel.documents), 4)
        self.assertEqual(FakeEmbeddingModel.constructed[-1], "different-model")

    def test_added_and_removed_knowledge_files_invalidate_cache(self):
        self.retrieve(self.load_rag())
        extra = self.knowledge / "legacy.md"
        extra.write_text("普通旧版知识：定期记录家庭血压。", encoding="utf-8")
        added = self.load_rag()
        self.assertIsNone(added.block_embeddings)
        self.assertEqual(len(added.knowledge_blocks), 3)
        self.retrieve(added)
        extra.unlink()
        removed = self.load_rag()
        self.assertIsNone(removed.block_embeddings)
        self.assertEqual(len(removed.knowledge_blocks), 2)
        self.retrieve(removed)
        self.assertEqual(len(FakeEmbeddingModel.documents), 3)

    def test_model_name_change_in_same_process_replaces_old_instance(self):
        module = self.load_rag()
        self.retrieve(module)
        old_model = module.embedding_model
        module.EMBEDDING_MODEL_NAME = "new-runtime-model"
        module.rebuild_index()
        self.assertIsNone(module.embedding_model)
        self.assertIsNone(module.block_embeddings)
        self.retrieve(module)
        self.assertIsNot(module.embedding_model, old_model)
        self.assertEqual(module.embedding_model.name, "new-runtime-model")
        self.assertEqual(len(FakeEmbeddingModel.documents), 2)

    def test_corrupt_or_malformed_cache_is_rebuilt_without_pickle(self):
        module = self.load_rag()
        self.retrieve(module)
        with np.load(self.cache, allow_pickle=False) as cache:
            good_metadata = cache["metadata"].copy()
            good_vectors = cache["embeddings"].copy()
        corruptions = [
            (np.array({"malicious": "object"}, dtype=object), good_vectors),
            (np.array("not-json"), good_vectors),
            (np.array("[]"), good_vectors),
            (good_metadata, good_vectors[:1]),
            (good_metadata, good_vectors.astype(np.float64)),
            (good_metadata, np.full((2, 4), np.nan, dtype=np.float32)),
            (good_metadata, np.zeros((2, 4), dtype=np.float32)),
        ]
        for index, (metadata, vectors) in enumerate(corruptions):
            with self.subTest(corruption=index):
                np.savez(self.cache, metadata=metadata, embeddings=vectors)
                restarted = self.load_rag()
                self.assertIsNone(restarted.block_embeddings)
                self.retrieve(restarted)
                self.assertEqual(len(FakeEmbeddingModel.documents), index + 2)
        self.cache.write_bytes(b"incomplete npz file")
        self.retrieve(self.load_rag())
        self.assertEqual(len(FakeEmbeddingModel.documents), len(corruptions) + 2)

    def test_wrong_cached_dimension_rebuilds_after_lazy_model_load(self):
        self.retrieve(self.load_rag())
        with np.load(self.cache, allow_pickle=False) as cache:
            metadata = json.loads(cache["metadata"].item())
        metadata["embedding_dimension"] = 3
        vectors = np.full((2, 3), 1 / np.sqrt(3), dtype=np.float32)
        np.savez(self.cache, metadata=np.array(json.dumps(metadata)), embeddings=vectors)
        restarted = self.load_rag()
        self.assertEqual(restarted.block_embeddings.shape, (2, 3))
        self.retrieve(restarted)
        self.assertEqual(restarted.block_embeddings.shape, (2, 4))
        self.assertEqual(len(FakeEmbeddingModel.documents), 2)

    def test_cache_write_failure_keeps_working_in_memory_index(self):
        module = self.load_rag()
        with patch.object(module.os, "replace", side_effect=PermissionError("read-only cache")):
            self.retrieve(module)
        self.assertIsNotNone(module.block_embeddings)
        self.assertFalse(self.cache.exists())
        self.assertEqual(list(self.cache.parent.glob(".rag-index-*")), [])
        self.retrieve(module)
        self.assertEqual(len(FakeEmbeddingModel.documents), 1)

    def test_disabled_embeddings_and_missing_model_keep_keyword_fallback(self):
        disabled = self.load_rag(disabled=True)
        self.retrieve(disabled)
        self.assertEqual(FakeEmbeddingModel.constructed, [])
        self.assertFalse(self.cache.exists())
        self.assertTrue(disabled.get_knowledge_status()["fallback_mode"])
        self.assertEqual(disabled.get_knowledge_status()["embedding_error"],
                         "RAG_DISABLE_EMBEDDING=1")
        enabled = self.load_rag()
        original_import = builtins.__import__

        def missing_package(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ModuleNotFoundError("sentence_transformers not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing_package):
            self.retrieve(enabled)
        self.retrieve(enabled)
        self.assertFalse(enabled.embedding_available)
        self.assertIn("not installed", enabled.embedding_error)
        self.assertEqual(FakeEmbeddingModel.constructed, [])

    def test_document_and_query_inference_failure_keep_keyword_fallback(self):
        FakeEmbeddingModel.fail_documents = True
        module = self.load_rag()
        self.retrieve(module)
        self.retrieve(module)
        self.assertFalse(module.embedding_available)
        self.assertFalse(self.cache.exists())
        self.assertEqual(len(FakeEmbeddingModel.documents), 1)
        FakeEmbeddingModel.fail_documents = False
        FakeEmbeddingModel.fail_queries = True
        module = self.load_rag()
        self.retrieve(module)
        self.assertFalse(module.embedding_available)
        self.assertTrue(self.cache.exists())
        self.assertIn("query inference", module.embedding_error)

    def test_explicit_rebuild_recovers_model_load_failure_lazily(self):
        module = self.load_rag()
        with patch.object(FakeEmbeddingModel, "__init__",
                          side_effect=RuntimeError("temporary model load failure")):
            self.retrieve(module)
        self.assertFalse(module.embedding_available)
        self.assertIsNone(module.embedding_model)
        module.rebuild_index()
        self.assertIsNone(module.embedding_model)
        self.assertEqual(FakeEmbeddingModel.constructed, [])
        self.retrieve(module)
        self.assertTrue(module.embedding_available)
        self.assertEqual(FakeEmbeddingModel.constructed, ["test-model"])

    def test_explicit_rebuild_recovers_inference_failure_without_reloading_model(self):
        FakeEmbeddingModel.fail_documents = True
        module = self.load_rag()
        self.retrieve(module)
        original_model = module.embedding_model
        FakeEmbeddingModel.fail_documents = False
        module.rebuild_index()
        self.retrieve(module)
        self.assertIs(module.embedding_model, original_model)
        self.assertEqual(len(FakeEmbeddingModel.constructed), 1)
        self.assertEqual(len(FakeEmbeddingModel.documents), 2)
        FakeEmbeddingModel.fail_queries = True
        self.retrieve(module)
        self.assertFalse(module.embedding_available)
        FakeEmbeddingModel.fail_queries = False
        module.rebuild_index()
        self.retrieve(module)
        self.assertIs(module.embedding_model, original_model)
        self.assertTrue(module.embedding_available)
        self.assertEqual(len(FakeEmbeddingModel.constructed), 1)
        self.assertEqual(len(FakeEmbeddingModel.documents), 2)

    def test_legacy_encode_interface_still_works(self):
        class EncodeOnlyModel:
            def __init__(self, name):
                self.delegate = FakeEmbeddingModel(name)

            def encode(self, texts, normalize_embeddings):
                if len(texts) == 1:
                    return self.delegate.encode_query(texts, normalize_embeddings)
                return self.delegate.encode_document(texts, normalize_embeddings)

        module = self.load_rag()
        module.SentenceTransformer = EncodeOnlyModel
        self.retrieve(module)
        self.assertEqual(len(FakeEmbeddingModel.documents), 1)
        self.assertEqual(len(FakeEmbeddingModel.queries), 1)

    def test_concurrent_first_queries_build_documents_once(self):
        module = self.load_rag()
        FakeEmbeddingModel.document_delay = 0.01
        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(lambda _: self.retrieve(module), range(6)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(len(FakeEmbeddingModel.constructed), 1)
        self.assertEqual(len(FakeEmbeddingModel.documents), 1)
        self.assertEqual(len(FakeEmbeddingModel.queries), 6)

    def test_empty_queries_and_empty_knowledge_do_not_load_model(self):
        module = self.load_rag()
        self.assertEqual(module.retrieve_medical_guidance(""), [])
        self.assertEqual(module.retrieve_medical_guidance("   "), [])
        self.markdown.unlink()
        self.knowledge.rmdir()
        self.assertIsInstance(module._make_cache_metadata(), dict)
        module.rebuild_index()
        self.assertEqual(module.retrieve_medical_guidance("家庭血压"), [])
        self.assertEqual(FakeEmbeddingModel.constructed, [])


if __name__ == "__main__":
    unittest.main()

"""Run real embeddinggemma + qwen3:4b evaluations. Never invent test results."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from config import ROOT, Settings
from rag import CHAT_MODEL, EMBEDDING_MODEL, GuideIndex, OllamaClient, answer_question, load_guide


def evaluate(settings):
    client = OllamaClient(settings.ollama_base_url, settings.ollama_api_token)
    index = GuideIndex.build(load_guide(ROOT / "knowledge_base.json"), client)
    cases = json.loads((ROOT / "evaluation/questions.json").read_text(encoding="utf-8"))
    records = []
    for case in cases:
        result = answer_question(case["question"], index, client, settings.top_k, settings.min_similarity)
        retrieved = [h["entry"]["id"] for h in result["hits"]]
        retrieval_ok = not case["expected_ids"] or bool(set(case["expected_ids"]) & set(retrieved))
        correct = result["answerable"] == case["expected_answerable"]
        if case["expected_answerable"]:
            correct = correct and retrieval_ok and all(term.casefold() in result["answer"].casefold() for term in case["expected_terms"])
        failure = None if correct else ("retrieval" if not retrieval_ok else "generation")
        record = {**case, "retrieved_ids": retrieved, "scores": [round(h["score"], 4) for h in result["hits"]],
                  "answer": result["answer"], "answerable": result["answerable"], "correct": bool(correct),
                  "failure_stage": failure, "reason": result["reason"]}
        records.append(record)
        print(f'{case["id"]}: {"PASS" if correct else "FAIL"} · {retrieved} · {result["reason"]}', flush=True)
    return {"run_at": datetime.now(timezone.utc).isoformat(), "embedding_model": EMBEDDING_MODEL,
            "chat_model": CHAT_MODEL, "top_k": settings.top_k, "min_similarity": settings.min_similarity,
            "passed": sum(r["correct"] for r in records), "total": len(records), "cases": records}


def markdown_table(report):
    rows = ["| ID | Question (English translation) | Expected IDs | Retrieved IDs | Correct? | Failure stage |",
            "|---|---|---|---|---|---|"]
    for record in report["cases"]:
        question=record.get("question_en",record["question"])
        rows.append(f'| {record["id"]} | {question} | {", ".join(record["expected_ids"]) or "None"} | '
                    f'{", ".join(record["retrieved_ids"])} | {"Yes" if record["correct"] else "No"} | {record["failure_stage"] or "—"} |')
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results.json")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    report = evaluate(Settings.load())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if args.update_readme:
        readme = ROOT / "README.md"
        content = readme.read_text(encoding="utf-8")
        start, end = "<!-- TEST_RESULTS_START -->", "<!-- TEST_RESULTS_END -->"
        before, middle = content.split(start, 1)
        _, after = middle.split(end, 1)
        readme.write_text(before+start+"\n"+markdown_table(report)+"\n"+end+after, encoding="utf-8")
    raise SystemExit(0 if report["passed"] == report["total"] else 1)

"""Quick accuracy check. Usage: python evaluate.py your_notes.pdf questions.json

questions.json is a list like:
  {"q": "What is overfitting?", "expect": ["too well", "training data"]}   # pass if ANY keyword appears
  {"q": "Who won the 2018 World Cup?", "expect": "NOT_FOUND"}             # pass if the bot uses the fallback
"""
import json
import sys

from dotenv import load_dotenv

from rag import FALLBACK, StudyMateRAG, get_llm, load_embedder


def main(pdf_path, questions_path):
    load_dotenv()
    rag = StudyMateRAG(load_embedder(), get_llm())
    print(f"Indexed {rag.ingest(pdf_path)} chunks")
    questions = json.load(open(questions_path))
    correct = 0
    for item in questions:
        answer, _ = rag.answer(item["q"])
        if item["expect"] == "NOT_FOUND":
            ok = answer == FALLBACK
        else:
            ok = any(k.lower() in answer.lower() for k in item["expect"])
        correct += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {item['q']}\n        -> {answer[:150]}")
    print(f"\nResult: {correct}/{len(questions)} correct")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

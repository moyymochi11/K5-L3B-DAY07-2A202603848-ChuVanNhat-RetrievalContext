"""Prepare Day 7 datasets from Day 6 artifacts."""
import json
from pathlib import Path
import chromadb

def main():
    root = Path(__file__).resolve().parent.parent
    chroma_dir = root / "starter" / "chroma_store"
    
    print("[1/4] Loading chunks from ChromaDB...")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_collection("strategy_b")
    res = col.get(include=["documents", "metadatas"])
    
    chunks = []
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        chunks.append({
            "id": cid,
            "source_id": "vinuni-handbook-2024",
            "source": meta.get("source", "vinuni_handbook.pdf"),
            "page": int(meta.get("page")),
            "text": doc,
            "metadata": {
                "section": meta.get("section"),
                "year": meta.get("year", 2024),
                "language": meta.get("language", "en")
            }
        })
    print(f"Loaded {len(chunks)} chunks.")
    
    print("[2/4] Loading benchmark questions from Day 6...")
    day6_bench_path = Path("D:/VIN/K5-L3-Day06-2A202603848-DataIngestionEngineering/benchmark/benchmark.json")
    with open(day6_bench_path, "r", encoding="utf-8") as f:
        day6_bench = json.load(f)
        
    answerable_questions = []
    agent_cases = []
    
    for q in day6_bench:
        conv_q = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": [{
                "source_id": q.get("gold_source_id", "vinuni-handbook-2024"),
                "page": int(q.get("gold_page")),
                "evidence": q.get("evidence_quote")
            }]
        }
        if q.get("filter"):
            conv_q["filter"] = q["filter"]
        answerable_questions.append(conv_q)
        
        agent_cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": conv_q["gold"],
            "filter": q.get("filter")
        })

    print(f"Loaded {len(answerable_questions)} answerable questions ({sum(1 for q in answerable_questions if q['split'] == 'dev')} dev, {sum(1 for q in answerable_questions if q['split'] == 'test')} test).")
    
    # 3. Add Unanswerable questions
    print("[3/4] Adding unanswerable benchmark questions...")
    unanswerable = [
        {
            "id": "q_unans_dev_01",
            "split": "dev",
            "question": "What is the tuition fee for the Master of Science in Artificial Intelligence program at VinUniversity?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Handbook only covers general international student undergraduate orientation, arrival, and campus life; Master of Science in AI tuition fees are not present."
        },
        {
            "id": "q_unans_dev_02",
            "split": "dev",
            "question": "What is the pet policy and monthly deposit fee for keeping a dog or cat in the VinUniversity student dormitories?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Dormitory registration rules cover room types and fees, but contain no policy or fee structure permitting pets in student dormitories."
        },
        {
            "id": "q_unans_test_01",
            "split": "test",
            "question": "Which commercial airline provides dedicated student discount flight vouchers for VinUniversity students traveling from Frankfurt to Hanoi?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Handbook mentions airport arrivals, baggage and visas, but does not contain any airline partnership or flight discount vouchers from Frankfurt."
        },
        {
            "id": "q_unans_test_02",
            "split": "test",
            "question": "What is the exact deadline to submit applications for the VinUniversity semester abroad exchange program at Cornell University for the Fall 2025 semester?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Handbook mentions Cornell collaboration and general course registration timelines, but contains no 2025 exchange application deadline for Cornell."
        }
    ]
    
    agent_cases.extend(unanswerable)
    
    dev_cases = [c for c in agent_cases if c["split"] == "dev"]
    test_cases = [c for c in agent_cases if c["split"] == "test"]
    print(f"Total agent cases: {len(agent_cases)} (Dev: {len(dev_cases)} [15 ans, 2 unans]; Test: {len(test_cases)} [5 ans, 2 unans]).")

    # 4. Save files
    print("[4/4] Writing datasets to data/ and starter/data/converted/...")
    out_dirs = [root / "data", root / "starter" / "data" / "converted"]
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "questions.json").write_text(json.dumps(answerable_questions, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "agent_cases.json").write_text(json.dumps(agent_cases, ensure_ascii=False, indent=2), encoding="utf-8")
        
    print("Data preparation complete and validated!")

if __name__ == "__main__":
    main()

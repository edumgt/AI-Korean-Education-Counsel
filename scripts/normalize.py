#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DATA_ROOT 학습데이터를 진로상담 RAG 학습용 JSONL로 정규화한다.

출력:
- documents.jsonl: 검색용 문서 청크 원본
- qas.jsonl      : 평가/샘플 질의응답

지원 스키마:
- 학생기초정보_데이터_*.json (list[dict])
- 상담기록_데이터_*.json (list[dict])
- 전문가_라벨링_데이터_*.json (list[dict])
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def domain_from_path(path: Path, data_root: Path) -> str:
    rel = path.relative_to(data_root)
    parts = rel.parts
    # DATA_ROOT/01.원천데이터/01. 학교급/01. 초등/file.json
    if len(parts) >= 4:
        return " / ".join(parts[1:-1])
    if len(parts) >= 3:
        return parts[-2]
    return "unknown"




def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(as_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [as_text(v).strip() for v in value if as_text(v).strip()]
    t = as_text(value).strip()
    return [t] if t else []

def build_doc(doc_id: str, domain: str, doc_type: str, text: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "domain_name": domain,
        "source": "career-counsel-dataset",
        "source_spec": doc_type,
        "creation_year": None,
        "text": " ".join((text or "").split()),
        "raw": raw,
    }


def extract_student_info(rows: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
    docs = []
    for row in rows:
        sid = str(row.get("student_idx") or "unknown")
        summary_parts = [
            f"학생ID: {sid}",
            f"학교급: {row.get('school_level', '미상')}",
            f"학년: {row.get('grade', '미상')}",
            f"성별: {row.get('gender', '미상')}",
            f"관심사: {', '.join(as_list(row.get('interests')))}",
            f"강점: {', '.join(as_list(row.get('strengths')))}",
            f"희망직업: {', '.join(as_list(row.get('career_aspirations')))}",
        ]
        docs.append(
            build_doc(
                doc_id=f"student_profile::{sid}",
                domain=domain,
                doc_type="student_profile",
                text="\n".join(summary_parts),
                raw=row,
            )
        )
    return docs


def extract_conversation_docs(rows: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
    docs = []
    for item in rows:
        meta = item.get("meta", {})
        sid = str(meta.get("student_idx") or "unknown")
        cid = str(meta.get("counseling_idx") or "0")

        conv_texts: List[str] = []
        for turn in item.get("conversation", []) or []:
            category = turn.get("conv_category") or "기타"
            utterances = turn.get("utterances", []) or []
            conv_texts.append(f"[카테고리] {category}")
            for u in utterances:
                speaker = u.get("speaker_idx", "UNK")
                utt = (u.get("utterance") or "").strip()
                if utt:
                    conv_texts.append(f"{speaker}: {utt}")

        full_text = "\n".join(conv_texts)
        docs.append(
            build_doc(
                doc_id=f"counsel::{sid}::{cid}",
                domain=domain,
                doc_type="counseling_record",
                text=full_text,
                raw=item,
            )
        )
    return docs


def extract_label_docs_and_qas(rows: List[Dict[str, Any]], domain: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    docs = []
    qas = []
    for i, row in enumerate(rows, start=1):
        sid = str(row.get("student_idx") or f"unknown_{i}")
        categories = as_list(row.get("recommended_job_categories"))
        label = as_text(row.get("job_label") or "미상")
        expert_comment = as_text(row.get("expert_comment")).strip()
        summaries = as_list(row.get("counselling_summaries"))

        text = "\n".join(
            [
                f"학생ID: {sid}",
                f"추천직업카테고리: {', '.join(categories)}",
                f"대표직업라벨: {label}",
                f"상담요약: {' | '.join(summaries)}",
                f"전문가코멘트: {expert_comment}",
            ]
        )

        docs.append(
            build_doc(
                doc_id=f"label::{sid}::{i}",
                domain=domain,
                doc_type="expert_label",
                text=text,
                raw=row,
            )
        )

        qas.append(
            {
                "qa_id": f"career_reco::{sid}::{i}",
                "domain_name": domain,
                "q_type": "career_recommendation",
                "question": f"학생 {sid}의 상담 요약을 바탕으로 추천 직업군과 이유를 알려줘.",
                "answer": f"추천 직업군: {', '.join(categories)} / 대표 직업: {label} / 근거: {expert_comment}",
                "raw": row,
            }
        )
    return docs, qas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    docs: List[Dict[str, Any]] = []
    qas: List[Dict[str, Any]] = []
    skipped = 0

    for path in sorted(data_root.rglob("*.json")):
        data = load_json(path)
        if not isinstance(data, list):
            skipped += 1
            continue

        domain = domain_from_path(path, data_root)
        name = path.name

        if "학생기초정보" in name:
            docs.extend(extract_student_info(data, domain))
        elif "상담기록" in name:
            docs.extend(extract_conversation_docs(data, domain))
        elif "전문가_라벨링" in name:
            d, q = extract_label_docs_and_qas(data, domain)
            docs.extend(d)
            qas.extend(q)
        else:
            skipped += 1

    write_jsonl(out_dir / "documents.jsonl", docs)
    write_jsonl(out_dir / "qas.jsonl", qas)

    print(f"[OK] documents: {len(docs)}")
    print(f"[OK] qas      : {len(qas)}")
    print(f"[OK] skipped  : {skipped}")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()

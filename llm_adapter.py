from __future__ import annotations

import json
import os
from typing import Any, Dict

from ollama_client import ollama_generate


def _safe_json_parse(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    return {
        "scores": {"technical": 2, "communication": 2, "depth": 1},
        "overall": 2,
        "feedback": "Evaluation parsing failed.",
    }


class LLMService:
    """
    Providers:
    - question_bank
    - local_model (Flan-T5 only)
    - ollama (Mistral for generation + feedback, Flan-T5 for scoring)
    - openai
    """

    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "mistral")
        self.role_aliases = {
            "frontend": "Frontend Developer",
            "front end": "Frontend Developer",
            "frontend developer": "Frontend Developer",
            "backend": "Backend Developer",
            "backenddeveloper": "Backend Developer",
            "backend developer": "Backend Developer",
            "data scientist": "Data Scientist",
            "datascientist": "Data Scientist",
            "devops": "DevOps Engineer",
            "devops engineer": "DevOps Engineer",
            "product manager": "Product Manager",
            "productmanager": "Product Manager",
            "fullstack": "Fullstack Developer",
            "full stack": "Fullstack Developer",
            "fullstack developer": "Fullstack Developer",
            "qa": "QA Engineer",
            "qa engineer": "QA Engineer",
        }

    def _normalize_role_for_bank(self, role: str) -> str:
        key = (role or "").strip().lower()
        return self.role_aliases.get(key, role)

    def _extract_question_text(self, question_obj: Any) -> str:
        if isinstance(question_obj, str):
            return question_obj.strip()
        if isinstance(question_obj, dict):
            for key in ("question", "main_question", "text"):
                value = question_obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return str(question_obj)
        return ""

    # --------------------------------------------------
    # QUESTION GENERATION
    # --------------------------------------------------
    def generate_question(self, role: str, difficulty: str) -> str:
        prompt = (
            "You are a strict technical interviewer.\n"
            f"Role: {role}\n"
            f"Difficulty: {difficulty}\n"
            "Generate exactly one interview question relevant to this role and difficulty.\n"
            "Return only the question text."
        )

        try:
            generated = ollama_generate(prompt, model=self.model).strip()
            if generated:
                return generated
        except Exception as exc:
            print(f"[LLMService] Ollama question generation failed: {exc}")

        from question_bank import question_bank

        normalized_role = self._normalize_role_for_bank(role)
        question_obj = question_bank.get_question(normalized_role, difficulty)
        question_text = self._extract_question_text(question_obj)
        if question_text:
            return question_text

        fallback_role = role or "this role"
        return f"Tell me about a challenging project you handled as a {fallback_role} and the impact you delivered."

    # --------------------------------------------------
    # FOLLOW-UP GENERATION
    # --------------------------------------------------
    def generate_followup(self, role: str, question: str, answer: str) -> str:
        prompt = (
            "You are a technical interviewer.\n"
            f"Role: {role}\n"
            f"Original Question: {question}\n"
            f"Candidate Answer: {answer}\n"
            "Generate one concise follow-up question that probes depth or clarity.\n"
            "Return only the follow-up question text."
        )
        try:
            followup = ollama_generate(prompt, model=self.model).strip()
            if followup:
                return followup
        except Exception as exc:
            print(f"[LLMService] Ollama follow-up generation failed: {exc}")
        return "Can you elaborate more on your previous answer?"

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------
    def evaluate_answer(self, role: str, question: str, answer: str) -> Dict[str, Any]:
        prompt = (
            "You are an expert technical interviewer and communication coach.\n"
            "Evaluate the candidate answer and return valid JSON only with this exact schema:\n"
            '{"scores":{"technical":0-5,"communication":0-5,"depth":0-5},"overall":0-5,"feedback":"..."}\n'
            "Scoring rules: use integers, be strict, and feedback must be actionable and concise.\n"
            f"Role: {role}\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
        )

        try:
            raw = ollama_generate(prompt, model=self.model)
            result = _safe_json_parse(raw)
            scores = result.get("scores") or {}
            result["scores"] = {
                "technical": int(scores.get("technical", 2)),
                "communication": int(scores.get("communication", 2)),
                "depth": int(scores.get("depth", 1)),
            }
            result["overall"] = int(result.get("overall", 2))
            result["feedback"] = str(result.get("feedback", "No feedback generated.")).strip()
            return result
        except Exception as exc:
            print(f"[LLMService] Ollama evaluation failed: {exc}")

        try:
            from testingmodels import evaluate_answer as flan_evaluate

            result = flan_evaluate(role, question, answer)
            feedback_prompt = (
                "You are an expert interview coach.\n"
                "Write clear, actionable feedback in 2-3 short paragraphs.\n"
                f"Role: {role}\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n"
                f"Scores: {result.get('scores')}, Overall: {result.get('overall')}\n"
                "Return only feedback text."
            )
            result["feedback"] = ollama_generate(feedback_prompt, model=self.model).strip()
            return result
        except Exception as exc:
            print(f"[LLMService] Fallback evaluation failed: {exc}")
            return _safe_json_parse("")

    def generate_feedback(self, role: str, question: str, answer: str) -> str:
        return self.evaluate_answer(role, question, answer).get("feedback", "")

    def save_memory(self) -> None:
        # Placeholder for persistence hooks; keeps existing flow stable.
        return None


llm_service = LLMService()

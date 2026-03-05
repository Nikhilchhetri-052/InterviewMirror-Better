from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from llm_adapter import llm_service

# Import database collections (will be set from app.py)
interview_sessions_collection = None

def set_database_collections(sessions_collection):
    global interview_sessions_collection
    interview_sessions_collection = sessions_collection


@dataclass
class InterviewEntry:
    main_question: str
    main_answer: Optional[str] = None
    main_evaluation: Optional[dict] = None
    followup_question: Optional[str] = None
    followup_answer: Optional[str] = None
    followup_evaluation: Optional[dict] = None


@dataclass
class InterviewSession:
    interview_id: str
    role: str
    difficulty: str
    max_questions: int
    user_id: Optional[str] = None  # Add user_id field
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    current_question: Optional[str] = None
    current_mode: str = "main"  # "main", "followup", "done"
    current_index: int = 0  # number of main questions completed
    active_entry_index: Optional[int] = None
    entries: List[InterviewEntry] = field(default_factory=list)

    def start(self) -> str:
        self.current_question = llm_service.generate_question(self.role, self.difficulty)
        self.current_mode = "main"
        return self.current_question

    def _collect_scores(self) -> List[dict]:
        evaluations: List[dict] = []
        for entry in self.entries:
            if entry.main_evaluation:
                evaluations.append(entry.main_evaluation)
            if entry.followup_evaluation:
                evaluations.append(entry.followup_evaluation)
        return evaluations

    def _average_scores(self, evaluations: List[dict]) -> Dict[str, float]:
        if not evaluations:
            return {"technical": 0.0, "communication": 0.0, "depth": 0.0, "overall": 0.0}

        sums = {"technical": 0.0, "communication": 0.0, "depth": 0.0, "overall": 0.0}
        for ev in evaluations:
            scores = ev.get("scores", {})
            sums["technical"] += float(scores.get("technical", 0))
            sums["communication"] += float(scores.get("communication", 0))
            sums["depth"] += float(scores.get("depth", 0))
            sums["overall"] += float(ev.get("overall", 0))

        count = float(len(evaluations))
        return {k: round(v / count, 2) for k, v in sums.items()}

    def _summary_text(self, averages: Dict[str, float]) -> str:
        strongest = max(
            ("technical", "communication", "depth"),
            key=lambda k: averages.get(k, 0),
        )
        weakest = min(
            ("technical", "communication", "depth"),
            key=lambda k: averages.get(k, 0),
        )
        return (
            "Session complete. "
            f"Strongest area: {strongest}. "
            f"Growth area: {weakest}. "
            "Review feedback to focus your next practice session."
        )

    def summary(self) -> Dict[str, object]:
        evaluations = self._collect_scores()
        averages = self._average_scores(evaluations)
        return {
            "role": self.role,
            "difficulty": self.difficulty,
            "questions_completed": self.current_index,
            "total_questions": self.max_questions,
            "average_scores": averages,
            "summary": self._summary_text(averages),
            "entries": [
                {
                    "main_question": entry.main_question,
                    "main_answer": entry.main_answer,
                    "main_evaluation": entry.main_evaluation,
                    "followup_question": entry.followup_question,
                    "followup_answer": entry.followup_answer,
                    "followup_evaluation": entry.followup_evaluation,
                }
                for entry in self.entries
            ],
        }

    def submit_answer(self, answer: str) -> Dict[str, object]:
        if self.current_mode == "done":
            return {"done": True, "summary": self.summary()}

        if not self.current_question:
            self.start()

        evaluation = llm_service.evaluate_answer(self.role, self.current_question, answer)

        if self.current_mode == "main":
            entry = InterviewEntry(
                main_question=self.current_question,
                main_answer=answer,
                main_evaluation=evaluation,
            )
            self.entries.append(entry)
            self.active_entry_index = len(self.entries) - 1

            followup = llm_service.generate_followup(self.role, self.current_question, answer)
            entry.followup_question = followup
            self.current_question = followup
            self.current_mode = "followup"

            return {
                "evaluation": evaluation,
                "next_question": followup,
                "mode": "followup",
                "question_index": self.current_index + 1,
                "total_questions": self.max_questions,
                "done": False,
            }

        if self.current_mode == "followup":
            if self.active_entry_index is None:
                self.active_entry_index = len(self.entries) - 1
            entry = self.entries[self.active_entry_index]
            entry.followup_answer = answer
            entry.followup_evaluation = evaluation

            self.current_index += 1
            if self.current_index >= self.max_questions:
                self.current_mode = "done"
                llm_service.save_memory()
                return {
                    "evaluation": evaluation,
                    "done": True,
                    "summary": self.summary(),
                    "question_index": self.current_index,
                    "total_questions": self.max_questions,
                    "mode": "done",
                }

            next_question = llm_service.generate_question(self.role, self.difficulty)
            self.current_question = next_question
            self.current_mode = "main"

            return {
                "evaluation": evaluation,
                "next_question": next_question,
                "mode": "main",
                "question_index": self.current_index + 1,
                "total_questions": self.max_questions,
                "done": False,
            }

        return {"done": True, "summary": self.summary()}


class InterviewManager:
    def __init__(self) -> None:
        self.sessions: Dict[str, InterviewSession] = {}

    def start_session(
        self, role: str, difficulty: str = "medium", max_questions: int = 5, user_id: Optional[str] = None
    ) -> InterviewSession:
        interview_id = uuid.uuid4().hex
        session = InterviewSession(
            interview_id=interview_id,
            role=role or "general",
            difficulty=difficulty or "medium",
            max_questions=max_questions or 5,
            user_id=user_id,
        )
        session.start()
        self.sessions[interview_id] = session
        return session

    def get_session(self, interview_id: str) -> Optional[InterviewSession]:
        return self.sessions.get(interview_id)

    def submit_answer(self, interview_id: str, answer: str) -> Dict[str, object]:
        session = self.get_session(interview_id)
        if not session:
            return {"error": "Invalid interview_id", "done": True}
        return session.submit_answer(answer)

    def end_session(self, interview_id: str) -> Dict[str, object]:
        session = self.get_session(interview_id)
        if not session:
            return {"error": "Invalid interview_id", "done": True}
        session.current_mode = "done"
        llm_service.save_memory()
        
        # Save session to database if user_id exists
        if session.user_id and interview_sessions_collection:
            session_data = {
                "interview_id": session.interview_id,
                "user_id": session.user_id,
                "role": session.role,
                "difficulty": session.difficulty,
                "max_questions": session.max_questions,
                "created_at": session.created_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "summary": session.summary()
            }
            interview_sessions_collection.insert_one(session_data)
        
        return {"done": True, "summary": session.summary()}


manager = InterviewManager()

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from bson.objectid import ObjectId

from llm_adapter import llm_service

# Import database collections (will be set from app.py)
interview_sessions_collection = None
job_applications_collection = None

def set_database_collections(sessions_collection, applications_collection=None):
    global interview_sessions_collection
    global job_applications_collection
    interview_sessions_collection = sessions_collection
    job_applications_collection = applications_collection


@dataclass
class InterviewEntry:
    main_question: str
    main_answer: Optional[str] = None
    main_evaluation: Optional[dict] = None
    followup_question: Optional[str] = None
    followup_answer: Optional[str] = None
    followup_evaluation: Optional[dict] = None
    followups: List[Dict[str, Optional[object]]] = field(default_factory=list)


@dataclass
class InterviewSession:
    interview_id: str
    role: str
    difficulty: str
    max_questions: int
    user_id: Optional[str] = None  # Add user_id field
    job_id: Optional[str] = None
    application_id: Optional[str] = None
    custom_questions: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    current_question: Optional[str] = None
    current_mode: str = "main"  # "main", "followup", "done"
    current_index: int = 0  # number of main questions completed
    active_entry_index: Optional[int] = None
    active_followup_index: Optional[int] = None
    followups_remaining: int = 0
    max_followups_per_question: int = 3
    custom_question_index: int = 0
    entries: List[InterviewEntry] = field(default_factory=list)
    persisted: bool = False

    def _next_main_question(self) -> str:
        if self.custom_question_index < len(self.custom_questions):
            question = self.custom_questions[self.custom_question_index]
            self.custom_question_index += 1
            return question
        return llm_service.generate_question(self.role, self.difficulty)

    def start(self) -> str:
        self.current_question = self._next_main_question()
        self.followups_remaining = random.randint(0, self.max_followups_per_question)
        self.active_followup_index = None
        self.current_mode = "main"
        return self.current_question

    def _collect_scores(self) -> List[dict]:
        evaluations: List[dict] = []
        for entry in self.entries:
            if entry.main_evaluation:
                evaluations.append(entry.main_evaluation)
            if entry.followup_evaluation:
                evaluations.append(entry.followup_evaluation)
            for followup in entry.followups:
                evaluation = followup.get("evaluation")
                if evaluation:
                    evaluations.append(evaluation)
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
                    "followups": entry.followups,
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
            self.active_followup_index = None

            if self.followups_remaining > 0:
                followup = llm_service.generate_followup(self.role, self.current_question, answer)
                entry.followup_question = followup
                entry.followups.append({"question": followup, "answer": None, "evaluation": None})
                self.active_followup_index = 0
                self.current_question = followup
                self.current_mode = "followup"

                return {
                    "evaluation": evaluation,
                    "next_question": followup,
                    "mode": "followup",
                    "question_index": self.current_index + 1,
                    "total_questions": self.max_questions,
                    "done": False,
                    "overall_summary": self.summary(),
                }

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

            self.current_question = self._next_main_question()
            self.followups_remaining = random.randint(0, self.max_followups_per_question)
            self.active_followup_index = None
            self.current_mode = "main"

            return {
                "evaluation": evaluation,
                "next_question": self.current_question,
                "mode": "main",
                "question_index": self.current_index + 1,
                "total_questions": self.max_questions,
                "done": False,
                "overall_summary": self.summary(),
            }

        if self.current_mode == "followup":
            if self.active_entry_index is None:
                self.active_entry_index = len(self.entries) - 1
            entry = self.entries[self.active_entry_index]
            if self.active_followup_index is None:
                self.active_followup_index = max(0, len(entry.followups) - 1)

            if entry.followups:
                entry.followups[self.active_followup_index]["answer"] = answer
                entry.followups[self.active_followup_index]["evaluation"] = evaluation

            if entry.followup_answer is None and entry.followup_evaluation is None:
                entry.followup_answer = answer
                entry.followup_evaluation = evaluation

            self.followups_remaining = max(0, self.followups_remaining - 1)
            if self.followups_remaining > 0:
                followup = llm_service.generate_followup(self.role, self.current_question, answer)
                entry.followups.append({"question": followup, "answer": None, "evaluation": None})
                self.active_followup_index = len(entry.followups) - 1
                self.current_question = followup
                self.current_mode = "followup"
                return {
                    "evaluation": evaluation,
                    "next_question": followup,
                    "mode": "followup",
                    "question_index": self.current_index + 1,
                    "total_questions": self.max_questions,
                    "done": False,
                    "overall_summary": self.summary(),
                }

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

            next_question = self._next_main_question()
            self.current_question = next_question
            self.followups_remaining = random.randint(0, self.max_followups_per_question)
            self.active_followup_index = None
            self.current_mode = "main"

            return {
                "evaluation": evaluation,
                "next_question": next_question,
                "mode": "main",
                "question_index": self.current_index + 1,
                "total_questions": self.max_questions,
                "done": False,
                "overall_summary": self.summary(),
            }

        return {"done": True, "summary": self.summary()}


class InterviewManager:
    def __init__(self) -> None:
        self.sessions: Dict[str, InterviewSession] = {}

    def _serialize_entries(self, entries: List[InterviewEntry]) -> List[Dict[str, object]]:
        payload: List[Dict[str, object]] = []
        for entry in entries:
            payload.append(
                {
                    "main_question": entry.main_question,
                    "main_answer": entry.main_answer,
                    "main_evaluation": entry.main_evaluation,
                    "followup_question": entry.followup_question,
                    "followup_answer": entry.followup_answer,
                    "followup_evaluation": entry.followup_evaluation,
                    "followups": entry.followups,
                }
            )
        return payload

    def _deserialize_entries(self, payload: List[Dict[str, object]]) -> List[InterviewEntry]:
        entries: List[InterviewEntry] = []
        for item in payload or []:
            entries.append(
                InterviewEntry(
                    main_question=item.get("main_question", ""),
                    main_answer=item.get("main_answer"),
                    main_evaluation=item.get("main_evaluation"),
                    followup_question=item.get("followup_question"),
                    followup_answer=item.get("followup_answer"),
                    followup_evaluation=item.get("followup_evaluation"),
                    followups=item.get("followups") or [],
                )
            )
        return entries

    def _restore_session_from_doc(self, doc: Dict[str, object]) -> Optional[InterviewSession]:
        interview_id = str(doc.get("interview_id") or "")
        if not interview_id:
            return None

        session = InterviewSession(
            interview_id=interview_id,
            role=str(doc.get("role") or "general"),
            difficulty=str(doc.get("difficulty") or "medium"),
            max_questions=int(doc.get("max_questions") or 5),
            user_id=doc.get("user_id"),
            job_id=doc.get("job_id"),
            application_id=doc.get("application_id"),
            custom_questions=doc.get("custom_questions") or [],
            created_at=str(doc.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )
        session.current_question = doc.get("current_question")
        session.current_mode = str(doc.get("current_mode") or "main")
        session.current_index = int(doc.get("current_index") or 0)
        session.active_entry_index = doc.get("active_entry_index")
        session.active_followup_index = doc.get("active_followup_index")
        session.followups_remaining = int(doc.get("followups_remaining") or 0)
        session.max_followups_per_question = int(doc.get("max_followups_per_question") or 3)
        session.custom_question_index = int(doc.get("custom_question_index") or 0)
        session.entries = self._deserialize_entries(doc.get("entries") or [])
        session.persisted = True

        if not session.current_question:
            session.current_question = session._next_main_question()
            session.current_mode = "main"

        self.sessions[interview_id] = session
        return session

    def resume_session(
        self,
        user_id: str,
        application_id: Optional[str] = None,
        interview_id: Optional[str] = None,
    ) -> Optional[InterviewSession]:
        if not user_id:
            return None

        if interview_id and interview_id in self.sessions:
            return self.sessions.get(interview_id)

        if interview_sessions_collection is None:
            return None

        query: Dict[str, object] = {"user_id": user_id, "status": "in_progress"}
        if interview_id:
            query["interview_id"] = interview_id
        if application_id:
            query["application_id"] = application_id

        doc = interview_sessions_collection.find_one(query, sort=[("last_updated_at", -1)])
        if not doc:
            return None
        return self._restore_session_from_doc(doc)

    def start_session(
        self,
        role: str,
        difficulty: str = "medium",
        max_questions: int = 5,
        user_id: Optional[str] = None,
        job_id: Optional[str] = None,
        application_id: Optional[str] = None,
        custom_questions: Optional[List[str]] = None,
        resume_interview_id: Optional[str] = None,
    ) -> InterviewSession:
        if user_id and (application_id or resume_interview_id):
            resumed = self.resume_session(
                user_id=user_id,
                application_id=application_id,
                interview_id=resume_interview_id,
            )
            if resumed:
                return resumed

        if not max_questions or max_questions <= 0:
            max_questions = 5
        if max_questions > 20:
            max_questions = 20
        interview_id = uuid.uuid4().hex
        session = InterviewSession(
            interview_id=interview_id,
            role=role or "general",
            difficulty=difficulty or "medium",
            max_questions=max_questions or 5,
            user_id=user_id,
            job_id=job_id,
            application_id=application_id,
            custom_questions=custom_questions or [],
        )
        session.start()
        self.sessions[interview_id] = session
        self._persist_session_state(session, status="in_progress")
        return session

    def get_session(self, interview_id: str) -> Optional[InterviewSession]:
        return self.sessions.get(interview_id)

    def submit_answer(self, interview_id: str, answer: str) -> Dict[str, object]:
        session = self.get_session(interview_id)
        if not session:
            return {"error": "Invalid interview_id", "done": True}
        result = session.submit_answer(answer)
        if result.get("done"):
            self._persist_session_state(session, status="completed")
            self.sessions.pop(interview_id, None)
        else:
            self._persist_session_state(session, status="in_progress")
        return result

    def end_session(self, interview_id: str) -> Dict[str, object]:
        session = self.get_session(interview_id)
        if not session:
            return {"error": "Invalid interview_id", "done": True}
        session.current_mode = "done"
        llm_service.save_memory()

        self._persist_session_state(session, status="completed")
        self.sessions.pop(interview_id, None)

        return {"done": True, "summary": session.summary()}

    def _persist_session_state(self, session: InterviewSession, status: str = "in_progress") -> None:
        if not session.user_id or interview_sessions_collection is None:
            return

        completed = status == "completed"
        summary_payload = session.summary()

        session_data = {
            "interview_id": session.interview_id,
            "user_id": session.user_id,
            "job_id": session.job_id,
            "application_id": session.application_id,
            "custom_questions": session.custom_questions,
            "custom_question_index": session.custom_question_index,
            "role": session.role,
            "difficulty": session.difficulty,
            "max_questions": session.max_questions,
            "created_at": session.created_at,
            "current_question": session.current_question,
            "current_mode": session.current_mode,
            "current_index": session.current_index,
            "active_entry_index": session.active_entry_index,
            "active_followup_index": session.active_followup_index,
            "followups_remaining": session.followups_remaining,
            "max_followups_per_question": session.max_followups_per_question,
            "entries": self._serialize_entries(session.entries),
            "status": status,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary_payload,
        }
        if completed:
            session_data["completed_at"] = datetime.now(timezone.utc).isoformat()

        interview_sessions_collection.update_one(
            {"interview_id": session.interview_id},
            {"$set": session_data},
            upsert=True,
        )
        session.persisted = True

        if not session.application_id or job_applications_collection is None:
            return

        try:
            app_object_id = ObjectId(session.application_id)
        except Exception:
            return

        app_update = {
            "interview_status": "completed" if completed else "in_progress",
            "latest_interview_id": session.interview_id,
        }
        if completed:
            app_update["interview_completed_at"] = datetime.now(timezone.utc).isoformat()
        else:
            app_update["interview_started_at"] = datetime.now(timezone.utc).isoformat()

        job_applications_collection.update_one({"_id": app_object_id}, {"$set": app_update})


manager = InterviewManager()

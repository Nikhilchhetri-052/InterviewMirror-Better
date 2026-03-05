import os
import json
import random


class QuestionBank:
    def __init__(self):
        self.base_path = os.path.join(
            os.path.dirname(__file__), "data"
        )

    def _load_role_file(self, role: str):
        """
        Example:
        Role: Backend Developer
        File: BackendDeveloper.json
        """

        filename = role.replace(" ", "") + ".json"
        filepath = os.path.join(self.base_path, filename)

        if not os.path.exists(filepath):
            print("File not found:", filepath)
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_question(self, role: str, difficulty: str):
        data = self._load_role_file(role)
        if not data:
            return None

        questions = data.get(role, [])
        difficulty = difficulty.lower()

        filtered = [
            q for q in questions
            if q.get("difficulty", "").lower() == difficulty
        ]

        if not filtered:
            return None

        return random.choice(filtered)

    def get_followup(self, role: str, parent_question_obj: dict):
        followups = parent_question_obj.get("follow_ups", [])

        if not followups:
            return None

        return random.choice(followups)


question_bank = QuestionBank()
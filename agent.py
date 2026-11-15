import json
import re
import sys
import ollama


class ReasoningAgent:
    def __init__(self, model_name: str = "qwen3:8b", timeout: int = 600):
        self.client = ollama.Client(timeout=timeout)
        self.model_name = self._resolve_model_name(model_name)

    def _resolve_model_name(self, preferred: str) -> str:
        try:
            models_info = self.client.list()
            installed = []
            if hasattr(models_info, "models"):
                installed = [m.model for m in models_info.models if hasattr(m, "model")]
            elif isinstance(models_info, dict):
                installed = [
                    m.get("model") or m.get("name", "")
                    for m in models_info.get("models", [])
                ]
            installed = [m for m in installed if m]
            if not installed or preferred in installed:
                return preferred
            for m in installed:
                if preferred.lower() in m.lower():
                    return m
            for m in installed:
                if "qwen" in m.lower():
                    return m
            return installed[0]
        except Exception:
            return preferred

    def _strip_think_tags(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
        return cleaned.replace("</think>", "").strip()

    def _normalize_output(self, ans: str) -> str:
        if not ans:
            return ""
        ans = self._strip_think_tags(ans)

        persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        english_digits = "0123456789"
        trans_table = str.maketrans(
            persian_digits + arabic_digits, english_digits * 2
        )
        ans = ans.translate(trans_table)

        lines = [line.strip() for line in ans.splitlines() if line.strip()]
        if lines:
            ans = lines[-1]

        ans = ans.strip().strip("`").strip("*").strip('"').strip("'").strip()
        prefixes = [
            "final_answer:",
            "final answer:",
            "verified final answer:",
            "answer:",
            "پاسخ نهایی:",
            "پاسخ:",
            "جواب نهایی:",
            "جواب:",
        ]
        for prefix in prefixes:
            if ans.lower().startswith(prefix.lower()):
                ans = ans[len(prefix) :].strip()

        ans = ans.strip().strip("`").strip("*").strip('"').strip("'").strip()
        ans = re.sub(r"<([^<>]+)>", r"\1", ans)

        if ans.endswith("."):
            ans = ans[:-1].strip()

        return ans.strip()

    def _call_llm(
        self, system_prompt: str, user_prompt: str, json_mode: bool = False
    ) -> str:
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
            },
        }
        if json_mode:
            kwargs["format"] = "json"

        try:
            response = self.client.chat(**kwargs, think=False)
        except TypeError:
            response = self.client.chat(**kwargs)
        except Exception as e:
            print(f"  -> Warning: {e}", file=sys.stderr)
            return ""

        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
        else:
            content = getattr(getattr(response, "message", None), "content", "")

        return self._strip_think_tags(content or "")

    def solve(self, problem_text: str) -> str:
        print("  -> Solving problem (timeout=600s)...", file=sys.stderr)

        # مرحله ۱: تحلیلگر و حل‌کننده اولیه (Planner & Solver)
        planner_solver_system = (
            "You are an expert mathematical and logical reasoning agent. "
            "Break down the problem carefully:\n"
            "1. Identify all variables, entities, and explicit/implicit constraints.\n"
            "2. Identify the EXACT output format requested at the end of the problem (e.g., if the problem is in Persian about days of the week, output the English day name like Monday, Tuesday, Wednesday, Thursday; if it asks for a single letter, output a single uppercase letter).\n"
            "3. Solve the problem step-by-step using rigorous mathematical equations or truth tables.\n"
            "4. Always commit to a concrete valid answer (if multiple valid arrangements exist, pick the first valid one and NEVER leave the answer empty).\n"
            "Write clearly and concisely."
        )
        planner_solver_user = (
            f"Problem:\n{problem_text}\n\nAnalyze constraints and solve step-by-step:"
        )
        draft_solution = self._call_llm(planner_solver_system, planner_solver_user)

        # مرحله ۲: بازبین و منتقد منطقی (Verifier & Critic)
        verifier_system = (
            "You are a strict logical verifier. Your job is to check a proposed solution against EVERY constraint in the original problem.\n"
            "- Substitute the proposed answer back into each condition of the problem.\n"
            "- For Liar/Truth-teller puzzles: carefully test each case.\n"
            "- For Grid/Path planning: verify start point, target point, blocked cells, and minimum Manhattan steps.\n"
            "- For Ordering/Scheduling: verify every relative order constraint and make sure no two entities share the same day/position.\n"
            "- If you find ANY contradiction or arithmetic mistake, re-solve it from scratch and state the corrected final answer.\n"
            "- Make sure the final line strictly matches the requested output format and is NEVER empty."
        )
        verifier_user = (
            f"Original Problem:\n{problem_text}\n\n"
            f"Proposed Solution:\n{draft_solution}\n\n"
            "Verify every single constraint step-by-step. If wrong, fix it. State the verified final answer clearly at the end:"
        )
        verified_solution = self._call_llm(verifier_system, verifier_user)

        # مرحله ۳: قالب‌بند خروجی (Strict JSON Extractor)
        formatter_system = (
            "You are a strict output formatter. Extract ONLY the final answer from the verified reasoning according to the problem's exact formatting instruction.\n"
            "Rules:\n"
            '1. Output valid JSON with a single key: {"final_answer": "..."}\n'
            "2. Do NOT include any explanations, units, angle brackets (< >), or extra words inside final_answer.\n"
            "3. NEVER return an empty string. Always extract the concrete answer from the solution.\n"
            "4. Use English digits (0-9) and English day names (Monday, Tuesday, Wednesday, Thursday) if days of week are asked.\n"
            "5. Copy the exact punctuation/casing requested by the problem options."
        )
        formatter_user = (
            f"Original Problem (check format instructions at the bottom):\n{problem_text}\n\n"
            f"Verified Solution:\n{verified_solution}\n\n"
            'Return JSON {"final_answer": "..."}:'
        )
        json_output = self._call_llm(formatter_system, formatter_user, json_mode=True)

        final_ans = ""
        try:
            parsed = json.loads(json_output)
            if isinstance(parsed, dict):
                final_ans = str(
                    parsed.get("final_answer")
                    or parsed.get("answer")
                    or next(iter(parsed.values()), "")
                ).strip()
        except Exception:
            match = re.search(r'"(?:final_answer|answer)"\s*:\s*"([^"]+)"', json_output)
            if match:
                final_ans = match.group(1)

        cleaned_answer = self._normalize_output(final_ans)
        if not cleaned_answer:
            cleaned_answer = self._normalize_output(verified_solution)
        if not cleaned_answer:
            cleaned_answer = self._normalize_output(draft_solution)
        return cleaned_answer

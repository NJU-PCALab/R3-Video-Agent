"""Keep the selected same-story retrieval payload bounded across both sources."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from lite.completion_memory import recall


class MemoryBoundTests(unittest.TestCase):
    def test_total_advice_is_three_even_with_retained_fallback(self):
        episodes = [{"id": str(i)} for i in range(4)]
        recent = [{"operation": "local/1"}, {"operation": "local/2"}]
        cached = {"episodes": episodes, "recent_in_run_lessons": recent}
        agent = SimpleNamespace(
            s=SimpleNamespace(run=Path("run"), data={"completion_experience_consumption": {"global/memory": cached}}),
            memory=SimpleNamespace(eligible=Mock(side_effect=lambda values, _: values), exclusion=Mock(return_value=None)),
            config={"memory_policy": {"scope": "within_story"}},
        )
        chosen = recall(agent, "global", {}, "global/memory")
        self.assertEqual(chosen["episodes"], episodes[:3])
        self.assertEqual(chosen["recent_in_run_lessons"], [])
        self.assertEqual(cached["episodes"], episodes)
        agent.memory.eligible.side_effect = None
        agent.memory.eligible.return_value = episodes[:2]
        chosen = recall(agent, "global", {}, "global/memory")
        self.assertEqual(chosen["recent_in_run_lessons"], recent[:1])


if __name__ == "__main__":
    unittest.main()

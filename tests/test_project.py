import unittest

from project import build_memory, extract_facts, load_history, parse_budget, recent_window, run


class DemoClient:
    enabled = False


class MemoryTests(unittest.TestCase):
    def test_latest_user_fact_wins(self):
        facts, conflicts = extract_facts(load_history())
        self.assertEqual(facts["职业目标"]["value"], "AI算法研发与大模型应用")
        self.assertEqual(len(conflicts), 1)

    def test_assistant_does_not_override_user(self):
        facts, _ = extract_facts([
            {"id": "1", "role": "user", "text": "设备条件=CPU"},
            {"id": "2", "role": "assistant", "text": "设备条件=GPU"},
        ])
        self.assertEqual(facts["设备条件"]["value"], "CPU")

    def test_budget_is_respected(self):
        for budget in [160, 300, 450, 1200]:
            memory = build_memory(load_history(), "专业 设备条件 最新职业目标", budget)
            self.assertLessEqual(len(memory["context"]), budget)
            window, _ = recent_window(load_history(), budget)
            self.assertLessEqual(len(window), budget)

    def test_outdated_goal_not_reintroduced(self):
        memory = build_memory(load_history(), "职业目标", 2000)
        self.assertNotIn("职业目标=前端开发", memory["context"])
        self.assertIn("AI算法研发与大模型应用", memory["context"])

    def test_budget_validation(self):
        self.assertEqual(parse_budget("问题，预算400"), 400)
        with self.assertRaises(ValueError):
            parse_budget("预算0")

    def test_demo_result(self):
        result = run({"text": "根据设备条件和目标安排学习，预算450"}, DemoClient())
        self.assertGreater(result["metrics"]["结构化保留字段"], 0)
        self.assertLessEqual(result["metrics"]["结构化记忆字符"], 450)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            run({"text": " "}, DemoClient())


if __name__ == "__main__":
    unittest.main()

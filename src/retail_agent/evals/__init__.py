"""Evaluating whether the agent's answers are *right*.

Every other test in this project asserts a path: that a syntax error routes to
repair, that PII never reaches the warehouse, that an exhausted budget degrades
instead of looping. All of those pass while the agent returns a confidently
wrong number, which is the failure mode that actually reached a user here.
"""

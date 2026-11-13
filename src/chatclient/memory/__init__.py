"""Long-term memory: summaries, extracted memories, scene state and fast recall.

Design: recall at chat time is pure code (dates, names, keywords → ranked memories),
so it adds no LLM call to a turn. Building memory (extracting, summarising) happens in
background jobs while the user is idle, through a persistent job queue.
"""

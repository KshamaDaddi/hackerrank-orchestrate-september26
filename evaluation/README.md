# Evaluation artifacts

`code/main.py` writes `evaluation/usage_report.md` after every final run.

The final report must describe the exact model providers/names and token usage for the submitted run. The default configuration uses no LLM calls, so its model token count is zero. If Ollama is enabled, update the generated report with the exact local model/token counts before submission.

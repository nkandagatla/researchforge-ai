# 3-minute demo script

**Before going on stage:** run `python -c "import memory; memory.warmup()"` and run the main question once with `--refresh`, so models are loaded and the timings are representative. Keep **Auto-approve the research plan** OFF in the sidebar. To show the cache live, start from a fresh `.memory/` folder, or use the Refresh button for the first run.

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the title and caption | "Deep-research tools give confident answers you can't check, and they forget everything. This one proves every claim and remembers what it verified." |
| 0:15 | Paste *"How is quantum computing going to change the current AI LLM/models and agentic implementations?"* and click **Run research** | "The Topic Router profiles the question and picks models: cheap for extraction, stronger for synthesis, all validated live against OpenRouter." |
| 0:25 | The **plan approval** appears; edit one subquestion; click **Approve** | "Nothing expensive happens until I approve the plan. That's a LangGraph interrupt." |
| 0:35 | Watch the live agent story (each search shows its exact query and a preview of the first results) | "Parallel and Tavily over real MCP, plus arXiv, all running in parallel. The analyst does hybrid BM25 and dense retrieval, removes near-duplicates, reranks and diversifies with MMR. The supervisor finds a gap and searches again." |
| 1:10 | The answer appears in the chat bubble | "Bottom line first. Every finding is a card with an exact quote verified in its source, linked to that source, and green badges mean the judge models agree." |
| 1:25 | Open **Evidence graph** and **Consensus** | "Claims link to sources. A red edge or row means a judge model disagreed. Contested claims carry the counter-quote." |
| 1:40 | Click **Inject fabricated citation** | "A fake source id and a made-up quote against a real source: both caught by the auditor." |
| 1:55 | Click **Demo: false-premise question** | "Quantum computers have *not* replaced GPUs. The app corrects the premise instead of inventing an answer." |
| 2:15 | Re-run the first question | "Same question: answered instantly from the second brain, for $0." Point at the cache banner, then open the **Library** tab and search "energy". |
| 2:35 | Show the **trust panel**, the **Execution summary**, **Agent workflow** and timeline | "Faithfulness, relevance, citation accuracy, consensus, credibility, and what each agent did, how many tokens it used and what it cost. Overlapping bars are the agents working in parallel." |
| 2:50 | Click **Download report** | "Research you can audit, and that remembers." |

**Fallbacks:**
- If a provider is slow, the timeline shows it, and the run continues with the others.
- If the live run stalls, show the cache path: answer from memory, then the Library tab.

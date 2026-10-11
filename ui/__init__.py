"""Streamlit UI of ResearchForge AI. app.py is the entry point; each module renders one part of the page.

- session.py: cached services (model catalogue, memory) and the background research job of this browser session
- sidebar.py: keys and run settings
- research.py: the Research tab (question, run buttons, plan approval, errors)
- answer.py: the chat-style answer with finding cards
- activity.py: the live agent story and the end-of-run execution summary
- tabs.py: the seven detail tabs under an answer
- charts.py: the Plotly and Recharts figures (pure functions, unit-tested)
- history.py: the History tab (research sessions and the verified claims library)
"""

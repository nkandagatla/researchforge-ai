from conftest import CATALOGUE

from agents import route_models
from models import TopicProfile


def profile(c: int, d: int, recent: bool = False) -> TopicProfile:
    return TopicProfile(domain="x", complexity=c, technical_depth=d, needs_recency=recent, rationale="r")


def test_rules_pick_synthesis_model_by_topic() -> None:
    assert route_models(profile(4, 5), CATALOGUE, "openai/gpt-4o-mini", 3).synth_model == "anthropic/claude-haiku-4.5"
    assert route_models(profile(2, 4), CATALOGUE, "openai/gpt-4o-mini", 3).synth_model == "openai/gpt-4.1-mini"
    assert route_models(profile(2, 2, True), CATALOGUE, "openai/gpt-4o-mini", 3).synth_model == "google/gemini-2.5-flash"
    simple = route_models(profile(1, 1), CATALOGUE, "openai/gpt-4o-mini", 3)
    assert simple.synth_model == simple.extract_model == "openai/gpt-4o-mini"
    assert route_models(None, CATALOGUE, "openai/gpt-4o-mini", 2).synth_model == "openai/gpt-4o-mini"


def test_judges_are_distinct_vendors_and_validated() -> None:
    route = route_models(profile(3, 3), CATALOGUE, "openai/gpt-4o-mini", 3)
    assert route.judges == ["openai/gpt-4o-mini", "google/gemini-2.5-flash", "anthropic/claude-haiku-4.5"]
    assert len(route_models(profile(3, 3), CATALOGUE, "openai/gpt-4o-mini", 2).judges) == 2


def test_unknown_models_are_never_chosen() -> None:
    route = route_models(profile(5, 5), {"openai/gpt-4o-mini": (1e-7, 1e-7)}, "vendor/does-not-exist", 3)
    assert route.extract_model == route.synth_model == "openai/gpt-4o-mini" and route.judges == ["openai/gpt-4o-mini"]
    assert any("not on OpenRouter" in n for n in route.notes)
    offline = route_models(profile(5, 5), {}, "openai/gpt-4o-mini", 3)
    assert offline.judges == ["openai/gpt-4o-mini"] and "unreachable" in offline.notes[0]

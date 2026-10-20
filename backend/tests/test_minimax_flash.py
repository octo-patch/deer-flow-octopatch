"""Exercise MiniMax selections through the factory and real SDK HTTP serialization."""

import json

import httpx
import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from wizard.providers import LLM_PROVIDERS
from wizard.writer import build_minimal_config

from deerflow.config.app_config import AppConfig
from deerflow.config.managed_models import ManagedModel, ManagedModelStore, merge_managed_models
from deerflow.models.factory import create_chat_model

FLASH = "MiniMax-M3.1-Flash"
TOOL = {"type": "function", "function": {"name": "lookup", "description": "Look up a value", "parameters": {"type": "object", "properties": {}}}}


@pytest.fixture
def wire():
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((request, body))
        result = {"id": "chat-test", "object": "chat.completion", "created": 1, "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
        if body["messages"][-1].get("content") == "lookup":
            result["choices"][0] = {
                "index": 0,
                "message": {"role": "assistant", "content": "", "reasoning_content": "Look up the value first.", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                "finish_reason": "tool_calls",
            }
        if body.get("stream"):
            if result["choices"][0]["message"].get("tool_calls"):
                result["choices"][0]["message"]["tool_calls"][0]["index"] = 0
            result["object"] = "chat.completion.chunk"
            result["choices"][0]["delta"] = result["choices"][0].pop("message")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=f"data: {json.dumps(result)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json=result)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        yield client, requests


def profile(source, region, model):
    if source == "managed":
        return ManagedModel(provider="minimax-cn" if region == "cn" else "minimax", model=model, api_key="test-key").runtime_config()
    provider = next(p for p in LLM_PROVIDERS if p.name == ("minimax_cn" if region == "cn" else "minimax"))
    assert provider.models == [FLASH, "MiniMax-M3"]
    assert provider.default_model == FLASH
    content = build_minimal_config(
        provider_use=provider.use,
        model_name=model,
        display_name=provider.display_name,
        api_key_field=provider.api_key_field,
        env_var=provider.env_var,
        extra_model_config=provider.extra_config_for(model),
    )
    config = yaml.safe_load(content)
    config["models"][0]["api_key"] = "test-key"
    return AppConfig.model_validate(config).models[0]


def chat(config, wire, **kwargs):
    client, _ = wire
    app = AppConfig.model_validate({"models": [config], "sandbox": {"use": "test"}})
    return create_chat_model(app_config=app, attach_tracing=False, http_client=client, **kwargs)


@pytest.mark.parametrize("source", ["wizard", "managed"])
@pytest.mark.parametrize("region", ["global", "cn"])
@pytest.mark.parametrize("model", [FLASH, "MiniMax-M3"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_regional_selection_reaches_sdk_with_correct_contract(source, region, model, thinking, stream, wire):
    config = profile(source, region, model)
    assert config.context_window == 1_000_000
    assert config.supports_vision is True
    llm = chat(config, wire, thinking_enabled=thinking)
    messages = [HumanMessage(content="hello", name="user-input")]
    if stream:
        assert list(llm.bind_tools([TOOL]).stream(messages))
    else:
        assert llm.bind_tools([TOOL]).invoke(messages).content == "ok"
    request, body = wire[1][-1]
    assert str(request.url) == f"https://{'api.minimaxi.com' if region == 'cn' else 'api.minimax.io'}/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["model"] == model
    assert body["thinking"] == {"type": "adaptive" if thinking or model == FLASH else "disabled"}
    assert body["reasoning_split"] is True
    assert body["max_tokens"] == (524_288 if model == FLASH else 131_072)
    assert "max_completion_tokens" not in body
    assert "context_window" not in body
    assert "service_tier" not in body
    assert body["tools"] == [TOOL]
    assert "name" not in body["messages"][0]
    if model == FLASH:
        assert body["reasoning_effort"] == "max"
    else:
        assert "reasoning_effort" not in body
    if stream:
        assert body["stream_options"]["include_usage"] is True


@pytest.mark.parametrize("source", ["wizard", "managed"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_flash_effort_is_serialized_at_top_level(source, effort, wire):
    chat(profile(source, "global", FLASH), wire, thinking_enabled=False, reasoning_effort=effort).invoke("hello")
    body = wire[1][-1][1]
    assert body["reasoning_effort"] == effort
    assert body["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize("source", ["wizard", "managed"])
def test_flash_normalizes_unsupported_effort_before_network(source, wire):
    chat(profile(source, "global", FLASH), wire, reasoning_effort="none").invoke("hello")
    assert wire[1][-1][1]["reasoning_effort"] == "max"


def test_media_and_tool_reasoning_history_survive_sdk_serialization(wire):
    media = [
        {"type": "text", "text": "Summarize"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "high"}},
        {"type": "video_url", "video_url": {"url": "mm_file://video", "fps": 1}},
    ]
    history = [
        HumanMessage(content=media),
        AIMessage(content="", additional_kwargs={"reasoning_content": " Check the result.\n"}, tool_calls=[{"id": "call-1", "name": "lookup", "args": {}}]),
        ToolMessage(content="42", tool_call_id="call-1"),
    ]
    chat(profile("managed", "global", FLASH), wire).bind_tools([TOOL]).invoke(history)
    messages = wire[1][-1][1]["messages"]
    assert messages[0]["content"] == media
    assert messages[1]["reasoning_content"] == " Check the result.\n"
    assert messages[1]["tool_calls"][0]["id"] == "call-1"
    assert messages[2]["tool_call_id"] == "call-1"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("region", ["global", "cn"])
def test_sdk_tool_response_replays_reasoning_on_next_turn(stream, region, wire):
    llm = chat(profile("wizard", region, FLASH), wire).bind_tools([TOOL])
    human = HumanMessage(content="lookup")
    if stream:
        chunks = list(llm.stream([human]))
        answer = chunks[0]
        for chunk in chunks[1:]:
            answer += chunk
    else:
        answer = llm.invoke([human])
    assert answer.additional_kwargs["reasoning_content"] == "Look up the value first."
    assert answer.tool_calls[0]["id"] == "call-1"
    llm.invoke([human, answer, ToolMessage(content="42", tool_call_id="call-1")])
    replay = wire[1][-1][1]["messages"][1]
    assert replay["reasoning_content"] == "Look up the value first."
    assert replay["tool_calls"][0]["function"] == {"name": "lookup", "arguments": "{}"}


@pytest.mark.parametrize("model", ["MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M3", "custom-model", FLASH])
def test_saved_custom_models_and_explicit_limits_remain_unchanged(model, tmp_path, monkeypatch, wire):
    monkeypatch.setenv("DEER_FLOW_HOME", str(tmp_path))
    store = ManagedModelStore()
    saved = store.save(
        ManagedModel(provider="openai-compatible", name="saved", model=model, base_url="https://proxy.example.com/v1", api_key="test-key", context_window=32000, max_tokens=321, supports_vision=False),
        expected_revision=None,
    )
    before = store.path.read_bytes()
    app = merge_managed_models(AppConfig.model_validate({"sandbox": {"use": "test"}}))
    config = app.models[0]
    assert config.model == model
    assert config.context_window == 32000
    assert config.supports_vision is False
    chat(config, wire).invoke("hello")
    request, body = wire[1][-1]
    assert request.url.host == "proxy.example.com"
    assert body["model"] == model
    assert body["max_completion_tokens"] == 321
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert store.path.read_bytes() == before
    assert store.list()[0].revision == saved.revision


@pytest.mark.parametrize("model", [FLASH, "MiniMax-M3"])
def test_managed_explicit_limits_override_model_defaults(model):
    config = ManagedModel(provider="minimax", model=model, api_key="test-key", context_window=32000, max_tokens=321, supports_vision=False).runtime_config()
    assert config.context_window == 32000
    assert config.max_tokens == 321
    assert config.supports_vision is False


@pytest.mark.parametrize("region", ["global", "cn"])
@pytest.mark.parametrize("model", ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"])
def test_explicit_legacy_direct_models_still_reach_original_endpoint(region, model, wire):
    config = profile("managed", region, model)
    chat(config, wire).invoke("hello")
    request, body = wire[1][-1]
    assert request.url.host == ("api.minimaxi.com" if region == "cn" else "api.minimax.io")
    assert body["model"] == model
    assert "thinking" not in body
    assert "reasoning_effort" not in body


@pytest.mark.parametrize("thinking", [False, True])
def test_m3_does_not_receive_flash_effort(thinking, wire):
    chat(profile("managed", "global", "MiniMax-M3"), wire, thinking_enabled=thinking, reasoning_effort="max").invoke("hello")
    assert "reasoning_effort" not in wire[1][-1][1]


def test_wizard_overrides_do_not_mutate_shared_config():
    provider = next(p for p in LLM_PROVIDERS if p.name == "minimax")
    first = provider.extra_config_for(FLASH)
    first["reasoning"]["effort"]["values"].clear()
    assert provider.extra_config_for(FLASH)["reasoning"]["effort"]["values"] == ["low", "medium", "high", "xhigh", "max"]
    for model in ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"]:
        config = provider.extra_config_for(model)
        assert config["supports_vision"] is False
        assert "reasoning" not in config

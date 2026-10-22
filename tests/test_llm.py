import pytest

from eurag.llm import AnthropicLLM, LLMError


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Usage:
    input_tokens, output_tokens = 30, 9


class _Resp:
    content, usage = [_Block("hello [x]")], _Usage()


def make_client(with_temperature):
    calls = []

    class Messages:
        if with_temperature:
            def create(self, *, model, max_tokens, system, messages, temperature):
                calls.append(("t", temperature, max_tokens))
                return _Resp()
        else:
            def create(self, *, model, max_tokens, system, messages):
                calls.append(("no-t", max_tokens))
                return _Resp()

    class Client:
        messages = Messages()

    return Client(), calls


def test_temperature_is_sent_only_when_the_sdk_accepts_it():
    client, calls = make_client(True)
    reply = AnthropicLLM(model="m", client=client).complete("s", "u", max_tokens=50)
    assert calls == [("t", 0, 50)] and (reply.text, reply.tokens_in, reply.tokens_out) == ("hello [x]", 30, 9)
    client, calls = make_client(False)
    llm = AnthropicLLM(model="m", client=client)
    assert llm.temperature_zero is False and llm.complete("s", "u").text == "hello [x]" and calls == [("no-t", 600)]


@pytest.mark.parametrize("value", ["", "...", "sk-ant-..."])
def test_missing_or_placeholder_key_is_rejected(monkeypatch, value):
    monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    with pytest.raises(LLMError):
        AnthropicLLM()

from omlx.adapter.output_parser import (
    detect_output_parser,
    install_ernie55_tokenizer_protocol,
)

MARKER_IDS = {
    "<ernie:think>": 10,
    "</ernie:think>": 11,
    "<ernie:tool_call>": 12,
    "</ernie:tool_call>": 13,
}


class _Tokenizer:
    unk_token_id = 0

    def __init__(self):
        self.tool_parser = object()

    def convert_tokens_to_ids(self, text):
        return MARKER_IDS.get(text, 0)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{token_id}>" for token_id in ids)

    def parse_completion(self, ids, *, rendered=None, tools=None, allow_incomplete=False):
        return {"tool_calls": [{"function": {"name": "calc", "arguments": {"x": 2}}}]}


def test_install_protocol_leaves_plain_text_intact():
    from omlx.api.tool_calling import parse_tool_calls

    tokenizer = _Tokenizer()
    assert install_ernie55_tokenizer_protocol(tokenizer)
    assert tokenizer.has_tool_calling is True
    assert tokenizer.tool_parser is None

    assert parse_tool_calls("The answer is 4.", tokenizer) == ("The answer is 4.", None)


def test_detect_parser_streams_reasoning_and_returns_the_tool_call():
    tokenizer = _Tokenizer()
    install_ernie55_tokenizer_protocol(tokenizer)
    factory = detect_output_parser(
        "ERNIE-5.5-Flash-Preview",
        tokenizer,
        {"model_type": "ernie5_5_moe"},
        model_path=None,
    )

    assert factory.kind == "ernie5_5_moe"
    session = factory.create_session_with_tools(tokenizer, [{"name": "calc"}])
    streamed = [
        session.process_token(token_id).stream_text
        for token_id in (10, 5, 11, 12, 20, 13)
    ]
    result = session.finalize()

    assert streamed == [" thinking", "<5>", "</think>", "", "", ""]
    assert result.tool_calls == [{"name": "calc", "arguments": {"x": 2}}]
    assert result.finish_reason == "tool_calls"

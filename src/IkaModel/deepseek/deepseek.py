import json
from typing import Any, Dict, List, Optional

from ..tool_schema import build_provider_tool_payload


def _default_message_history() -> Dict[str, Any]:
    return {
        "system": {"message": "", "tokens": 0},
        "first_input": {"message": "", "tokens": 0},
        "summary": {"message": "", "tokens": 0},
        "messages": {},
    }


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return message.get("content", str(message))
    return str(message)


def _append_deepseek_context(api_messages: List[Dict[str, Any]], message_history: Dict[str, Any]) -> None:
    if message_history["system"]["message"]:
        api_messages.append({"role": "system", "content": message_history["system"]["message"]})
    if message_history["first_input"]["message"]:
        api_messages.append({"role": "user", "content": message_history["first_input"]["message"]})
    if message_history["summary"]["message"]:
        api_messages.append({"role": "assistant", "content": message_history["summary"]["message"]})


def _deepseek_history_messages(message_history: Dict[str, Any]) -> List[Dict[str, Any]]:
    history_messages: List[Dict[str, Any]] = []
    for msg_id in message_history["messages"]:
        msg = message_history["messages"][msg_id]
        msg_type = msg.get("type", "assistant")
        raw = msg.get("message", "")
        if msg_type in {"assistant_with_tools", "tool"}:
            try:
                decoded = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(decoded, dict):
                history_messages.append(decoded)
        else:
            assistant_entry = {"role": "assistant", "content": raw if isinstance(raw, str) else str(raw)}
            reasoning_text = msg.get("reasoning_content")
            if reasoning_text:
                assistant_entry["reasoning_content"] = reasoning_text
            history_messages.append(assistant_entry)
    return history_messages


def _live_messages(messages: List[Dict[str, Any]], first_input_content: str) -> List[Dict[str, Any]]:
    live_messages: List[Dict[str, Any]] = []
    skip_first = bool(first_input_content and messages and _message_content(messages[0]) == first_input_content)
    for index, msg in enumerate(messages):
        if skip_first and index == 0:
            continue
        if isinstance(msg, dict):
            if "role" in msg and "content" in msg:
                live_messages.append(msg)
            elif "content" in msg:
                live_messages.append({"role": "user", "content": msg["content"]})
            else:
                live_messages.append({"role": "user", "content": str(msg)})
        else:
            live_messages.append({"role": "user", "content": str(msg)})
    return live_messages


def _assert_tool_ids_do_not_conflict(
    history_messages: List[Dict[str, Any]], live_messages: List[Dict[str, Any]],
) -> None:
    """Reject divergent records that claim the same provider tool-call id."""
    def records(source: List[Dict[str, Any]]) -> Dict[tuple[str, str], str]:
        found: Dict[tuple[str, str], str] = {}
        for message in source:
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                key = ("result", tool_call_id)
                encoded = json.dumps(message, sort_keys=True, separators=(",", ":"))
                if key in found and found[key] != encoded:
                    raise ValueError(f"conflicting DeepSeek tool result id: {tool_call_id}")
                found[key] = encoded
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    continue
                key = ("call", call_id)
                # Compare the complete assistant record, including the hidden
                # reasoning DeepSeek requires on a thinking continuation.  An
                # identical function call with a missing/different chain is
                # still a conflicting replay and must not be silently merged.
                encoded = json.dumps(message, sort_keys=True, separators=(",", ":"))
                if key in found and found[key] != encoded:
                    raise ValueError(f"conflicting DeepSeek tool call id: {call_id}")
                found[key] = encoded
        return found

    historical = records(history_messages)
    live = records(live_messages)
    for key in historical.keys() & live.keys():
        if historical[key] != live[key]:
            raise ValueError(f"conflicting DeepSeek {key[0]} id: {key[1]}")


def _history_prefix_before_live_overlap(
    history_messages: List[Dict[str, Any]], live_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return only history preceding the current live conversation.

    IkaCore records every tool exchange in both containers.  The live list is
    authoritative for the current chat round because it also carries user
    warnings in their exact position.  Prior stages exist only in history, so
    retain the prefix before the longest history suffix already represented in
    live messages.
    """
    _assert_tool_ids_do_not_conflict(history_messages, live_messages)
    for start in range(len(history_messages)):
        candidate = history_messages[start:]
        live_index = 0
        for historical in candidate:
            while (live_index < len(live_messages)
                   and live_messages[live_index] != historical):
                live_index += 1
            if live_index == len(live_messages):
                break
            live_index += 1
        else:
            return history_messages[:start]
    return history_messages


def _deepseek_max_tokens(model: Any) -> int:
    max_tokens_value = model.max_tokens if model.max_tokens and model.max_tokens > 0 else 4096
    return min(max_tokens_value, 32768)


def deepseek_fill_payload(
    model: Any,
    messages: List[Dict[str, Any]],
    message_history: Optional[Dict[str, Any]] = None,
    agent_tools: Optional[list[Any]] = None,
) -> Dict[str, Any]:
    message_history = message_history or _default_message_history()
    api_messages: List[Dict[str, Any]] = []
    _append_deepseek_context(api_messages, message_history)
    historical = _deepseek_history_messages(message_history)
    live = _live_messages(messages, message_history["first_input"]["message"])
    api_messages.extend(_history_prefix_before_live_overlap(historical, live))
    api_messages.extend(live)

    payload = {
        "model": model.model_id,
        "messages": api_messages,
        "temperature": model.temperature,
        "max_tokens": _deepseek_max_tokens(model),
        "stream": False,
    }

    model_id_lower = (model.model_id or "").lower()
    thinking_enabled = bool(getattr(model, "deepthinking", False)) or "reasoner" in model_id_lower
    if thinking_enabled:
        payload["thinking"] = {"type": "enabled"}
    else:
        payload["thinking"] = {"type": "disabled"}

    agent_tools = model.agent_tools if agent_tools is None else agent_tools

    if agent_tools:
        # DeepSeek defaults to auto tool choice when tools are present, and
        # some V4 routes reject even tool_choice="auto".
        payload["tools"] = build_provider_tool_payload("deepseek", agent_tools).tools

    return payload

"""GAVEL: guarded, proof-carrying selection for tool calls.

GAVEL is deliberately outside the language-model prompt.  It never reads the
benchmark answer graph, labels, task types, or evaluator output.  The only
inputs used here are the same ``messages`` and ``tools`` already supplied to
the stock handler.

The model's first response is the *anchor*.  Additional responses, when used,
are proposals generated from the exact same messages and tool definitions.
Proposals do not win by vote count: after canonical deduplication they must
carry receipts for their intent, arguments, guard, dependencies, authorization
and freshness.  The admissible frontier is selected lexicographically:

    1. cover the most current user obligations;
    2. make the least irreversible commitment;
    3. preserve the anchor where the first two are equal;
    4. make the fewest calls.

When no call is ready but a relevant plan lacks required information, GAVEL
asks the minimum-information clarification that unlocks the most obligations.
No corrective user or system message is appended at any point.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any, Iterable

from wtb.model_handler.api_inference.oai import OpenAIHandler


class _Response:
    """Small OpenAI-response-compatible wrapper used by the WTB harness."""

    def __init__(
        self,
        content: str | None,
        tool_calls: list[dict] | None,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_content: str | None = None,
    ):
        self._payload = {
            "choices": [{
                "message": {
                    "content": content,
                    "tool_calls": tool_calls,
                    "reasoning_content": reasoning_content,
                }
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    def json(self):
        return json.dumps(self._payload, ensure_ascii=False)


@dataclass
class _Generation:
    content: str | None
    tool_calls: list[dict]
    reasoning_content: str | None
    prompt_tokens: int
    completion_tokens: int
    latency: float


@dataclass
class Proposal:
    call: dict
    name: str
    arguments: dict
    canonical: str
    anchor: bool
    clause: int
    entity: str
    risk: int
    strong_intent: bool
    strict_receipts: bool
    missing: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    hard_reasons: tuple[str, ...] = ()
    source: str = "anchor"
    parameter_descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def admissible(self):
        return not self.hard_reasons and not self.missing and (
            self.anchor or self.strict_receipts
        )


_WORD_RE = re.compile(r"[a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_REFERENCE_RE = re.compile(
    r"\b(?:it|its|they|them|those|these|same|former|latter|first|second|third|"
    r"previous|above|earlier|other)\b",
    re.I,
)
_NOVELTY_RE = re.compile(r"\b(?:another|one more|new|different)\b", re.I)
_CONDITION_RE = re.compile(r"\b(?:if|unless|provided that|only if|when)\b", re.I)
_REFRESH_RE = re.compile(
    r"\b(?:again|current|currently|latest|newest|real[ -]?time|recheck|refresh|today|tomorrow|updated)\b",
    re.I,
)
_CLEAR_CHAT_RE = re.compile(
    r"\b(?:fact|explain|explanation|meaning|opinion|think|why|suggestion|advice|use)\b",
    re.I,
)

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please",
    "that", "the", "this", "to", "with", "you", "your", "data", "api",
    "information", "specified", "using", "given", "return", "returns",
}

_ACTION_SYNONYMS = {
    "read": {
        "check", "find", "get", "know", "list", "look", "lookup", "query",
        "retrieve", "search", "show", "tell", "view", "what", "which",
    },
    "generate": {"create", "generate", "make", "produce"},
    "download": {"download", "export", "save"},
    "subscribe": {"subscribe", "notify", "notification"},
    "send": {"email", "message", "notify", "send"},
    "buy": {"buy", "order", "purchase"},
    "book": {"book", "reserve", "schedule"},
    "update": {"change", "edit", "modify", "set", "update"},
    "delete": {"delete", "remove", "erase"},
    "cancel": {"cancel", "revoke"},
    "upload": {"attach", "import", "upload"},
    "create": {"add", "create", "open", "register"},
}
_ACTION_SYNONYMS["send"].update({"forward", "share"})

_READ_PREFIXES = {
    "check", "fetch", "filter", "find", "get", "list", "lookup", "query",
    "read", "retrieve", "search", "show", "view",
}
_ACTION_TO_CLASS = {
    **{x: "read" for x in _READ_PREFIXES},
    "generate": "generate",
    "download": "download",
    "export": "download",
    "subscribe": "subscribe",
    "notify": "send",
    "send": "send",
    "email": "send",
    "buy": "buy",
    "purchase": "buy",
    "order": "buy",
    "book": "book",
    "reserve": "book",
    "schedule": "book",
    "update": "update",
    "modify": "update",
    "edit": "update",
    "set": "update",
    "delete": "delete",
    "remove": "delete",
    "cancel": "cancel",
    "upload": "upload",
    "create": "create",
    "add": "create",
    "register": "create",
}
_RISK = {
    "read": 0,
    "generate": 1,
    "download": 1,
    "create": 2,
    "update": 2,
    "upload": 2,
    "subscribe": 2,
    "send": 2,
    "book": 3,
    "buy": 3,
    "cancel": 3,
    "delete": 3,
}
_STABLE_KEYS = {
    "account", "account_id", "email", "profile_id", "tenant_id", "user",
    "user_id", "username", "workspace_id",
}
_SENSITIVE = {
    "account_number", "address", "bank", "card", "cvv", "email", "password",
    "phone", "pin", "secret", "security_answer", "ssn", "token",
}


def _tokens(text: Any) -> set[str]:
    if text is None:
        return set()
    text = _CAMEL_RE.sub(" ", str(text)).lower()
    return {x for x in _WORD_RE.findall(text) if x not in _STOP}


def _normal(text: Any) -> str:
    return " ".join(_WORD_RE.findall(str(text).lower()))


def _tool_map(tools: list[dict]) -> dict[str, dict]:
    out = {}
    for tool in tools or []:
        fn = tool.get("function", tool)
        name = fn.get("name")
        if name:
            out[name] = fn
    return out


def _split_name(name: str) -> list[str]:
    return _WORD_RE.findall(_CAMEL_RE.sub(" ", name or "").lower())


def _action_class(name: str, description: str = "") -> str:
    words = _split_name(name)
    for word in words:
        if word in _ACTION_TO_CLASS:
            return _ACTION_TO_CLASS[word]
    first_desc = next(iter(_WORD_RE.findall((description or "").lower())), "")
    return _ACTION_TO_CLASS.get(first_desc, "read")


def _risk_of(name: str, description: str = "") -> int:
    return _RISK.get(_action_class(name, description), 1)


def _schema(tool: dict) -> dict:
    return tool.get("parameters", {}) or {}


def _coerce(value: Any, schema: dict) -> Any:
    """Apply only lossless, schema-declared scalar coercions."""
    declared = schema.get("type")
    if declared == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if declared == "integer" and isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
        return int(value)
    if declared == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if declared == "number" and isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    if declared == "boolean" and isinstance(value, str):
        if value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
    if declared == "array" and isinstance(value, list):
        item_schema = schema.get("items", {}) or {}
        return [_coerce(x, item_schema) for x in value]
    if declared == "object" and isinstance(value, dict):
        props = schema.get("properties", {}) or {}
        return {k: _coerce(v, props.get(k, {})) for k, v in value.items()}
    return value


def _valid_type(value: Any, schema: dict) -> bool:
    declared = schema.get("type")
    if declared == "string":
        ok = isinstance(value, str)
    elif declared == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif declared in {"number", "float"}:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif declared == "boolean":
        ok = isinstance(value, bool)
    elif declared == "array":
        ok = isinstance(value, list) and all(_valid_type(x, schema.get("items", {})) for x in value)
    elif declared == "object":
        if not isinstance(value, dict):
            return False
        props = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []
        ok = all(k in value for k in required) and all(
            k in props and _valid_type(v, props[k]) for k, v in value.items()
        )
    else:
        ok = True
    enum = schema.get("enum")
    return ok and (not enum or value in enum)


def _sanitize(arguments: dict, schema: dict) -> tuple[dict, list[str]]:
    props = schema.get("properties", {}) or {}
    errors = []
    cleaned = {}
    for key, value in arguments.items():
        if key not in props:
            # Removing a key which the declared function cannot receive is a
            # semantics-preserving schema repair.  Do not turn a repairable
            # call into a text response.
            continue
        value = _coerce(value, props[key])
        if not _valid_type(value, props[key]):
            errors.append(f"invalid_type:{key}")
        cleaned[key] = value
    ordered = {key: cleaned[key] for key in props if key in cleaned}
    return ordered, errors


def _canonical(name: str, arguments: dict) -> str:
    return f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(',', ':'))}"


def _leaf_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _leaf_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _leaf_values(child)
    else:
        yield value


def _contains_value(text: str, value: Any) -> bool:
    if value is None:
        return "null" in text.lower() or "none" in text.lower()
    if isinstance(value, bool):
        words = {"true", "yes", "enabled", "on"} if value else {"false", "no", "disabled", "off"}
        return bool(_tokens(text) & words)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Word boundaries reject digits embedded in identifiers while still
        # allowing ordinary sentence punctuation immediately after a number.
        return bool(re.search(rf"(?<!\w){re.escape(str(value))}(?!\w)", text, re.I))
    needle = _normal(value)
    return bool(needle) and needle in _normal(text)


def _system_date(messages: list[dict]) -> datetime | None:
    for message in messages:
        if message.get("role") != "system":
            continue
        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", str(message.get("content", "")))
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d")
            except ValueError:
                return None
    return None


def _derived_dates(messages: list[dict], current_text: str) -> set[str]:
    now = _system_date(messages)
    if now is None:
        return set()
    low = current_text.lower()
    out = set()
    if "today" in low:
        out.add(now.strftime("%Y-%m-%d"))
    if "tomorrow" in low:
        out.add((now + timedelta(days=1)).strftime("%Y-%m-%d"))
    if "yesterday" in low:
        out.add((now - timedelta(days=1)).strftime("%Y-%m-%d"))
    if "weekend" in low:
        saturday = now + timedelta(days=(5 - now.weekday()) % 7)
        if saturday.date() == now.date() and now.weekday() == 5:
            saturday = now
        out.update({saturday.strftime("%Y-%m-%d"), (saturday + timedelta(days=1)).strftime("%Y-%m-%d")})
    return out


def _active_user_indices(messages: list[dict]) -> list[int]:
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not users:
        return []
    active = [users[-1]]
    latest = str(messages[users[-1]].get("content", ""))
    # A short answer to a clarification belongs to the same active request.
    if len(users) > 1 and (len(latest.split()) <= 6 or _REFERENCE_RE.search(latest)):
        between = messages[users[-2] + 1:users[-1]]
        if not any(m.get("role") == "tool" for m in between):
            active.insert(0, users[-2])
    return active


def _underspecified_novelty(messages: list[dict]) -> bool:
    """Whether the latest user message asks for a new object but names none."""
    users = [m for m in messages if m.get("role") == "user"]
    if not users:
        return False
    latest = _message_text(users[-1])
    if not _NOVELTY_RE.search(latest) or len(_WORD_RE.findall(latest)) > 12:
        return False
    # Numbers, quoted values, addresses and identifier-like tokens are
    # evidence that the user did specify the new object.
    concrete = re.search(
        r"\d|['\"][^'\"]+['\"]|\b\S+@\S+\b|\b[A-Z]{2,}[A-Z0-9_-]*\b",
        latest,
    )
    return concrete is None


def _message_text(message: dict) -> str:
    content = message.get("content")
    parts = []
    if isinstance(content, str):
        parts.append(content)
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False))
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    return "\n".join(parts)


def _evidence(messages: list[dict]) -> tuple[str, str, str]:
    active_idx = set(_active_user_indices(messages))
    current = "\n".join(_message_text(messages[i]) for i in sorted(active_idx))
    last_active = max(active_idx) if active_idx else len(messages)
    live_tool = "\n".join(
        _message_text(m) for i, m in enumerate(messages)
        if i > last_active and m.get("role") == "tool"
    )
    history = "\n".join(
        _message_text(m) for i, m in enumerate(messages)
        if i not in active_idx and not (i > last_active and m.get("role") == "tool")
    )
    return current, live_tool, history


def _split_clauses(text: str) -> list[str]:
    text = re.sub(r"\b(?:and then|after that|finally|also)\b", ".", text, flags=re.I)
    text = re.sub(
        r",\s*(?:and\s+)?(?=(?:please\s+)?(?:buy|book|cancel|check|create|delete|"
        r"download|find|get|list|look|query|retrieve|search|send|show|subscribe|update)\b)",
        ".",
        text,
        flags=re.I,
    )
    return [x.strip() for x in re.split(r"[.;!?]+", text) if x.strip()] or [text]


def _call_action_tokens(name: str, description: str) -> tuple[str, set[str]]:
    action = _action_class(name, description)
    words = set(_split_name(name)) | _tokens(description)
    action_words = set().union(*_ACTION_SYNONYMS.values()) | set(_ACTION_TO_CLASS)
    return action, {x for x in words if x not in action_words and x not in _STOP}


def _intent_support(name: str, description: str, clause: str, arguments: dict, history: str) -> tuple[bool, int]:
    action, object_tokens = _call_action_tokens(name, description)
    clause_tokens = _tokens(clause)
    action_match = bool(clause_tokens & _ACTION_SYNONYMS.get(action, {action}))
    object_overlap = len(clause_tokens & object_tokens)
    current_value = any(_contains_value(clause, x) for x in _leaf_values(arguments))
    prior_same_tool = name.lower() in history.lower()

    if action == "read":
        strong = action_match and (object_overlap > 0 or current_value or prior_same_tool)
        # A terse follow-up such as "China" can legitimately continue the
        # preceding read operation when the new entity is explicit.
        strong = strong or (current_value and prior_same_tool)
    else:
        # Side effects and constructive actions require an explicit matching
        # verb.  Topic overlap alone is never authorization.
        strong = action_match
    score = 4 * int(action_match) + 2 * object_overlap + int(current_value) + int(prior_same_tool)
    return strong, score


def _best_clause(name: str, description: str, arguments: dict, clauses: list[str], history: str) -> tuple[int, bool]:
    ranked = []
    for i, clause in enumerate(clauses):
        strong, score = _intent_support(name, description, clause, arguments, history)
        ranked.append((strong, score, -i, i))
    strong, _, _, idx = max(ranked)
    return idx, strong


def _argument_source(
    key: str,
    value: Any,
    prop: dict,
    current: str,
    live_tool: str,
    history: str,
    messages: list[dict],
) -> str | None:
    leaves = list(_leaf_values(value))
    if leaves and all(_contains_value(current, x) for x in leaves):
        return "current_user"
    if leaves and all(_contains_value(live_tool, x) for x in leaves):
        return "current_tool_result"
    if isinstance(value, str) and value in _derived_dates(messages, current):
        return "verified_date_transform"
    # ISO-like country codes are accepted as a weak deterministic transform
    # only when the parameter itself is country-related and a country phrase
    # is present in the current request.  The anchor remains preferred.
    desc = f"{key} {prop.get('description', '')}".lower()
    if isinstance(value, str) and re.fullmatch(r"[A-Z]{2}", value) and "country" in desc:
        if re.search(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b", current):
            return "country_code_transform"
    if leaves and all(_contains_value(history, x) for x in leaves):
        if key.lower() in _STABLE_KEYS:
            return "stable_history"
        if _REFERENCE_RE.search(current) or "same" in current.lower():
            return "referenced_history"
    sentinel = object()
    default = prop.get("default", sentinel)
    if default is not sentinel and value == default:
        return "schema_default"
    return None


def _completed_calls(messages: list[dict]) -> set[str]:
    by_id = {}
    completed = set()
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                try:
                    fn = call["function"]
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    by_id[call.get("id")] = _canonical(fn.get("name", ""), args or {})
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id in by_id:
                completed.add(by_id[call_id])
    return completed


def _guard_ready(current: str, live_tool: str, risk: int) -> bool:
    if risk < 2 or not _CONDITION_RE.search(current):
        return True
    # A state-changing conditional action waits until this turn has received
    # a tool result capable of resolving the condition.  We intentionally do
    # not guess the predicate's truth value here; Qwen replans after the result.
    return bool(live_tool.strip())


def _entity_signature(arguments: dict, required: list[str]) -> str:
    picked = {k: arguments[k] for k in required if k in arguments}
    if not picked:
        picked = arguments
    return json.dumps(picked, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _call_dict(call: Any) -> dict:
    if isinstance(call, dict):
        return deepcopy(call)
    if hasattr(call, "model_dump"):
        return call.model_dump()
    raise TypeError("unsupported tool-call object")


def evaluate_call(
    call: Any,
    tools: list[dict],
    messages: list[dict],
    *,
    anchor: bool,
    source: str,
) -> Proposal:
    """Build a proof-carrying proposal from one raw model call."""
    raw_call = _call_dict(call)
    fn = raw_call.get("function", {})
    name = fn.get("name", "")
    tool = _tool_map(tools).get(name)
    hard = []
    if tool is None:
        hard.append("unknown_tool")
        tool = {"name": name, "description": "", "parameters": {}}
    raw_args = fn.get("arguments", {})
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else deepcopy(raw_args or {})
    except (TypeError, json.JSONDecodeError):
        arguments = {}
        hard.append("invalid_json")
    if not isinstance(arguments, dict):
        arguments = {}
        hard.append("arguments_not_object")

    schema = _schema(tool)
    arguments, shape_errors = _sanitize(arguments, schema)
    hard.extend(shape_errors)
    required = list(schema.get("required", []) or [])
    missing = [key for key in required if key not in arguments]
    props = schema.get("properties", {}) or {}

    current, live_tool, history = _evidence(messages)
    clauses = _split_clauses(current)
    clause, strong_intent = _best_clause(
        name, tool.get("description", ""), arguments, clauses, history
    )
    risk = _risk_of(name, tool.get("description", ""))
    if risk >= 1 and not strong_intent:
        # A sampled addition never receives permission from topic overlap.  An
        # anchor is treated more conservatively: reject it only when the user
        # clearly requested conversation rather than an action.  Ambiguity is
        # not enough reason to break a possibly-correct anchor.
        if not anchor or _CLEAR_CHAT_RE.search(current):
            hard.append("missing_intent_receipt")
    if risk >= 2 and not _guard_ready(current, live_tool, risk):
        hard.append("unresolved_guard")

    novelty = _underspecified_novelty(messages)
    unsupported = []
    optional_without_receipt = []
    for key in arguments:
        source_kind = _argument_source(
            key, arguments[key], props.get(key, {}), current, live_tool, history, messages
        )
        if novelty and source_kind in {"referenced_history", "stable_history"} and key.lower() not in _STABLE_KEYS:
            source_kind = None
        if source_kind is None:
            if key in required:
                unsupported.append(key)
            else:
                optional_without_receipt.append(key)

    # "Another/new/different" explicitly revokes inherited content values.
    # A required field supported only by the old object is not merely weak;
    # it is information the user must provide for the new object.
    if novelty and unsupported:
        missing.extend(key for key in unsupported if key not in missing)
        unsupported = []

    canonical = _canonical(name, arguments)
    if canonical in _completed_calls(messages):
        action = _action_class(name, tool.get("description", ""))
        # A read call can be intentionally repeated for fresh state.  Exact
        # repetition is stale only when the user supplied no refresh receipt.
        if action != "read" or not _REFRESH_RE.search(current):
            hard.append("already_completed")

    if not raw_call.get("id"):
        raw_call["id"] = "gavel_" + uuid.uuid4().hex
    raw_call["type"] = "function"
    raw_call["function"] = {
        "name": name,
        "arguments": json.dumps(arguments, ensure_ascii=False),
    }
    descriptions = {k: str(props.get(k, {}).get("description", "")) for k in required}
    strict = (
        strong_intent and not missing and not unsupported
        and not optional_without_receipt and not hard
    )
    return Proposal(
        call=raw_call,
        name=name,
        arguments=arguments,
        canonical=canonical,
        anchor=anchor,
        clause=clause,
        entity=f"{name}:{_entity_signature(arguments, required)}",
        risk=risk,
        strong_intent=strong_intent,
        strict_receipts=strict,
        missing=tuple(missing),
        unsupported=tuple(unsupported),
        hard_reasons=tuple(dict.fromkeys(hard)),
        source=source,
        parameter_descriptions=descriptions,
    )


def _explicit_capacity(text: str) -> int:
    low = text.lower()
    if re.search(r"\b(?:four|4)\b", low):
        return 4
    if re.search(r"\b(?:three|3)\b", low):
        return 3
    if re.search(r"\b(?:both|two|2)\b", low):
        return 2
    return 1


def _compatible(subset: list[Proposal], clauses: list[str]) -> bool:
    per_clause: dict[int, set[str]] = {}
    for proposal in subset:
        entities = per_clause.setdefault(proposal.clause, set())
        if proposal.entity in entities:
            return False
        entities.add(proposal.entity)
    for clause, entities in per_clause.items():
        cap = _explicit_capacity(clauses[clause] if clause < len(clauses) else "")
        if len(entities) > cap:
            return False
    return True


def select_frontier(proposals: list[Proposal], current_text: str) -> list[Proposal]:
    """Exact GAVEL allocation over the usually tiny proposal market."""
    # Proposal multiplicity carries no weight.  Prefer the anchor copy of an
    # identical call so correlated samples cannot outvote it.
    unique: dict[str, Proposal] = {}
    for proposal in proposals:
        previous = unique.get(proposal.canonical)
        if previous is None or (proposal.anchor and not previous.anchor):
            unique[proposal.canonical] = proposal
    eligible = [p for p in unique.values() if p.admissible]
    # Exact enumeration is cheap for the two or three generations used here.
    # Cap pathological server responses before the exponential step.
    eligible = sorted(eligible, key=lambda p: (not p.anchor, p.risk, p.canonical))[:12]
    clauses = _split_clauses(current_text)
    candidate_sets: list[list[Proposal]] = []
    best: list[Proposal] = []
    best_score = (0, 0, 0, 0)
    best_tie = ""
    for size in range(len(eligible) + 1):
        for combo in combinations(eligible, size):
            chosen = list(combo)
            if not _compatible(chosen, clauses):
                continue
            candidate_sets.append(chosen)

    # The original response is a bundle, not a bag of independently ranked
    # calls.  Admit the whole viable bundle as one allocation even when the
    # shallow clause parser under-counts its obligations.  This preserves
    # correct parallel calls while still allowing an equally complete,
    # lower-risk certified alternative to beat the anchor.
    endowed = preserve_and_augment_anchor(proposals, current_text)
    if endowed is not None:
        candidate_sets.append(endowed)

    seen_sets = set()
    for chosen in candidate_sets:
        signature = tuple(sorted(p.canonical for p in chosen))
        if signature in seen_sets:
            continue
        seen_sets.add(signature)
        coverage = len(signature)
        risk = sum(p.risk + int(bool(p.unsupported)) for p in chosen)
        anchor_count = sum(int(p.anchor) for p in chosen)
        tie = "|".join(signature)
        score = (coverage, -risk, anchor_count, -len(chosen))
        if score > best_score or (score == best_score and (not best_tie or tie < best_tie)):
            best, best_score, best_tie = chosen, score, tie
    return sorted(best, key=lambda p: (p.name, p.canonical))


_ANCHOR_VETO_REASONS = {
    "already_completed", "arguments_not_object", "invalid_json",
    "missing_intent_receipt", "unknown_tool", "unresolved_guard",
}


def preserve_and_augment_anchor(proposals: list[Proposal], current_text: str) -> list[Proposal] | None:
    """Keep a mechanically viable anchor bundle intact, then add only proof.

    Parallel calls are an indivisible part of the model's original decision.
    Re-ranking individual anchor calls caused exactly the kind of regression
    GAVEL is meant to prevent.  Extra sampled calls may extend the bundle, but
    only with strict receipts and only where the request explicitly has room
    for another action.
    """
    anchors = [p for p in proposals if p.anchor]
    if not anchors:
        return None
    if any(p.missing or set(p.hard_reasons) & _ANCHOR_VETO_REASONS for p in anchors):
        return None

    chosen = list(anchors)
    seen = {p.canonical for p in chosen}
    clauses = _split_clauses(current_text)
    extras = sorted(
        (p for p in proposals if not p.anchor and p.strict_receipts),
        key=lambda p: (p.risk, p.clause, p.canonical),
    )
    for proposal in extras:
        if proposal.canonical in seen:
            continue
        existing = [p for p in chosen if p.clause == proposal.clause]
        cap = _explicit_capacity(
            clauses[proposal.clause] if proposal.clause < len(clauses) else ""
        )
        if len(existing) >= cap:
            continue
        if proposal.entity in {p.entity for p in existing}:
            continue
        chosen.append(proposal)
        seen.add(proposal.canonical)
    return sorted(chosen, key=lambda p: (p.name, p.canonical))


def _information_cost(field_name: str) -> int:
    low = field_name.lower()
    if any(token in low for token in _SENSITIVE):
        return 4
    if low.endswith("_id") or low in {"id", "latitude", "longitude", "location"}:
        return 2
    return 1


def minimum_information_clarification(blocked: list[Proposal]) -> tuple[list[str], Proposal | None]:
    """Choose the minimum-cost question that unlocks maximum obligations."""
    plans = [p for p in blocked if p.strong_intent and (p.missing or p.unsupported)]
    if not plans:
        return [], None
    best_fields: set[str] = set()
    best_plan = None
    best_score = (0, 0, 0, 0)
    best_tie = ""
    for size in range(1, len(plans) + 1):
        for combo in combinations(plans, size):
            fields = set().union(*(set(p.missing) | set(p.unsupported) for p in combo))
            unlocked = len({p.clause for p in combo})
            cost = sum(_information_cost(x) for x in fields)
            anchor_count = sum(int(p.anchor) for p in combo)
            canonical = "|".join(sorted(fields))
            score = (unlocked, -cost, -len(fields), anchor_count)
            if score > best_score or (score == best_score and (not best_tie or canonical < best_tie)):
                best_score = score
                best_tie = canonical
                best_fields = fields
                best_plan = sorted(combo, key=lambda p: (not p.anchor, p.canonical))[0]
    return sorted(best_fields), best_plan


def _human_field(name: str) -> str:
    return _CAMEL_RE.sub(" ", name).replace("_", " ").strip()


def _clarifying_text(fields: list[str], plan: Proposal | None) -> str:
    labels = []
    for field_name in fields:
        description = plan.parameter_descriptions.get(field_name, "") if plan else ""
        label = _human_field(field_name)
        if description and len(description) <= 80:
            label = description.rstrip(".")
        labels.append(label)
    if len(labels) == 1:
        return f"Could you provide {labels[0]}?"
    return "Could you provide " + ", ".join(labels[:-1]) + f", and {labels[-1]}?"


def _explicit_execution_intent(proposal: Proposal, current_text: str, tools: list[dict]) -> bool:
    """High bar for changing a plain-text anchor into a tool interaction.

    Generic questions such as "what do you think?" can lexically resemble a
    read tool.  They are not sufficient reason to overturn a text response.
    A non-read action must carry its own explicit authorization receipt; a
    read action needs an operational retrieval verb, or a terse entity-valued
    continuation of the same operation.
    """
    tool = _tool_map(tools).get(proposal.name, {})
    action = _action_class(proposal.name, tool.get("description", ""))
    words = _tokens(current_text)
    if action != "read":
        return bool(words & _ACTION_SYNONYMS.get(action, {action}))
    operational = _ACTION_SYNONYMS["read"] - {"know", "tell", "what", "which"}
    if words & operational:
        return True
    return any(_contains_value(current_text, x) for x in _leaf_values(proposal.arguments)) and len(words) <= 8


class GavelHandler(OpenAIHandler):
    """Inference-only, evaluator-free successor to IGAR-v24."""

    request_timeout_seconds = float(os.getenv("WTB_GAVEL_REQUEST_TIMEOUT", "600"))
    total_candidates = max(1, int(os.getenv("WTB_GAVEL_CANDIDATES", "2")))
    proposal_temperature = float(os.getenv("WTB_GAVEL_TEMPERATURE", "0.2"))
    explore_text_anchors = os.getenv("WTB_GAVEL_EXPLORE_TEXT", "1") == "1"

    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.client = self.client.with_options(timeout=self.request_timeout_seconds)

    def _generate(self, messages, tools, *, temperature=None, tool_choice=None) -> _Generation:
        kwargs = {
            "messages": messages,
            "model": self.model_name,
            "temperature": self.temperature if temperature is None else temperature,
            "tools": tools,
        }
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        response, latency = self.generate_with_backoff(**kwargs)
        payload = json.loads(response.json())
        message = payload["choices"][0]["message"]
        usage = payload.get("usage") or {}
        calls = message.get("tool_calls") or []
        return _Generation(
            content=message.get("content"),
            tool_calls=[_call_dict(x) for x in calls],
            reasoning_content=message.get("reasoning_content"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency=latency,
        )

    @staticmethod
    def _sum_usage(generations: list[_Generation]) -> tuple[int, int, float]:
        return (
            sum(x.prompt_tokens for x in generations),
            sum(x.completion_tokens for x in generations),
            sum(x.latency for x in generations),
        )

    def _plain_text(self, messages, tools, generations):
        try:
            text = self._generate(messages, tools, temperature=0.0, tool_choice="none")
            generations.append(text)
            return text.content
        except Exception as exc:
            print(f"[GAVEL] text branch unavailable: {type(exc).__name__}: {exc}", flush=True)
            return None

    def _audit(self, inference_data, anchor, proposals, selected, decision, extra=None):
        record = {
            "id": inference_data.get("test_entry_id"),
            "turn": inference_data.get("task_idx"),
            "anchor": "tools" if anchor.tool_calls else "text",
            "decision": decision,
            "selected": [p.canonical for p in selected],
            "proposals": [{
                "call": p.canonical,
                "anchor": p.anchor,
                "source": p.source,
                "risk": p.risk,
                "strict": p.strict_receipts,
                "missing": list(p.missing),
                "unsupported": list(p.unsupported),
                "rejected": list(p.hard_reasons),
            } for p in proposals],
        }
        if extra:
            record.update(extra)
        print("[GAVEL] " + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    def _request_tool_call(self, inference_data):
        # Do not access inference_data["answer_list"] here.  Keeping this
        # method limited to these two keys is the evaluator firewall.
        messages = inference_data["messages"]
        tools = inference_data["tools"]
        generations = []

        anchor = self._generate(messages, tools)
        generations.append(anchor)
        current, _, _ = _evidence(messages)

        # Plain-text anchors are preserved unless a forced-tool proposal has
        # complete strict receipts.  This is the non-regression default.
        if not anchor.tool_calls:
            proposals = []
            if self.explore_text_anchors:
                try:
                    forced = self._generate(
                        messages, tools, temperature=self.proposal_temperature,
                        tool_choice="required",
                    )
                    generations.append(forced)
                    proposals.extend(
                        evaluate_call(c, tools, messages, anchor=False, source="forced_tool")
                        for c in forced.tool_calls
                    )
                except Exception as exc:
                    print(f"[GAVEL] forced-tool branch unavailable: {type(exc).__name__}: {exc}", flush=True)
            selected = select_frontier(proposals, current)
            selected = [
                p for p in selected
                if p.strict_receipts and _explicit_execution_intent(p, current, tools)
            ]
            if selected:
                pin, pout, latency = self._sum_usage(generations)
                self._audit(inference_data, anchor, proposals, selected, "tool_over_text")
                return _Response(anchor.content, [p.call for p in selected], pin, pout), latency

            clarification_plans = [
                p for p in proposals if _explicit_execution_intent(p, current, tools)
            ]
            fields, plan = minimum_information_clarification(clarification_plans)
            if fields and (not anchor.content or "?" not in anchor.content):
                pin, pout, latency = self._sum_usage(generations)
                self._audit(
                    inference_data, anchor, proposals, [], "clarify_over_text",
                    {"fields": fields},
                )
                return _Response(_clarifying_text(fields, plan), None, pin, pout), latency

            pin, pout, latency = self._sum_usage(generations)
            self._audit(inference_data, anchor, proposals, [], "keep_text_anchor")
            return _Response(anchor.content, None, pin, pout, anchor.reasoning_content), latency

        proposals = [
            evaluate_call(c, tools, messages, anchor=True, source="anchor")
            for c in anchor.tool_calls
        ]

        # Extra proposals see the identical prompt.  Their multiplicity is
        # discarded, so correlated sampling cannot turn repetition into proof.
        for index in range(1, self.total_candidates):
            try:
                extra = self._generate(
                    messages, tools, temperature=self.proposal_temperature
                )
                generations.append(extra)
                proposals.extend(
                    evaluate_call(c, tools, messages, anchor=False, source=f"proposal_{index}")
                    for c in extra.tool_calls
                )
            except Exception as exc:
                print(f"[GAVEL] proposal branch unavailable: {type(exc).__name__}: {exc}", flush=True)

        selected = select_frontier(proposals, current)
        if selected:
            pin, pout, latency = self._sum_usage(generations)
            self._audit(inference_data, anchor, proposals, selected, "execute_frontier")
            return _Response(
                anchor.content,
                [p.call for p in selected],
                pin,
                pout,
                anchor.reasoning_content,
            ), latency

        blocked = [
            p for p in proposals
            if (p.missing or p.unsupported)
            and not (set(p.hard_reasons) - {"already_completed"})
        ]
        fields, plan = minimum_information_clarification(blocked)
        if fields:
            pin, pout, latency = self._sum_usage(generations)
            self._audit(
                inference_data, anchor, proposals, [], "clarify", {"fields": fields}
            )
            return _Response(_clarifying_text(fields, plan), None, pin, pout), latency

        # Hard-invalid, repeated, unauthorized, or unresolved guarded calls
        # are withheld.  Ask the unchanged model for its text branch via the
        # API control rather than by appending an instruction.
        hard_anchor = any(
            set(p.hard_reasons) & _ANCHOR_VETO_REASONS
            for p in proposals if p.anchor
        )
        if hard_anchor:
            content = self._plain_text(messages, tools, generations)
            if content:
                pin, pout, latency = self._sum_usage(generations)
                self._audit(inference_data, anchor, proposals, [], "speak")
                return _Response(content, None, pin, pout), latency

        # Uncertainty is not evidence.  If no strictly better certified move
        # exists, hand the original response back unchanged.
        pin, pout, latency = self._sum_usage(generations)
        self._audit(inference_data, anchor, proposals, [], "fallback_anchor")
        return _Response(
            anchor.content,
            anchor.tool_calls,
            pin,
            pout,
            anchor.reasoning_content,
        ), latency

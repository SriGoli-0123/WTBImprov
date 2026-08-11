"""CONCORD: conservative contract-dominance for tool use.

CONCORD is an inference-time controller that operates outside the model
prompt.  It receives only the ordinary ``messages`` and ``tools`` passed to
the stock WTB handler.  It never reads an answer graph, task label, score, or
evaluator result.

The model's greedy response is an endowed anchor.  A complete, schema-valid
anchor is returned unless there is positive evidence of a defect.  In
particular, failure to prove where a *present* required value came from is not
treated as proof that the value is wrong.  This is the key correction to
GAVEL-v2's excessive clarification policy.

When repair is justified, CONCORD builds a small external contract from the
visible dialogue: ordered references, stable slots, relative years, completed
calls, and still-uncovered request clauses.  Extra generations receive the
exact same messages and tools.  They can replace a text anchor only when two
independent native generations agree on the same complete call bundle.
Optional Boolean commitments are kept only when their meaning is supported by
the active request.  Exact duplicate calls are collapsed.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from wtb.model_handler.api_inference.gavel import (
    GavelHandler,
    Proposal,
    _Generation,
    _NOVELTY_RE,
    _REFERENCE_RE,
    _Response,
    _active_user_indices,
    _action_class,
    _canonical,
    _evidence,
    _explicit_capacity,
    _normal,
    _optional_explicit,
    _schema,
    _split_clauses,
    _split_name,
    _system_date,
    _tokens,
    _tool_map,
    evaluate_call,
    minimum_information_clarification,
    _clarifying_text,
)


_STRUCTURAL_REASONS = {
    "ambiguous_selector",
    "already_completed",
    "arguments_not_object",
    "invalid_json",
    "missing_intent_receipt",
    "unknown_tool",
    "unresolved_guard",
}

_CONTINUATION_RE = re.compile(
    r"\b(?:again|also|another|continue|do it|one more|perform|proceed|recheck|"
    r"repeat|same operation|then)\b",
    re.I,
)

_REFERENCE_CUE_RE = re.compile(
    r"\b(?:beginning|earliest|initial|last|latest|previous|same|first round|"
    r"one of them|one of these|one of those)\b",
    re.I,
)

_CLARIFICATION_RE = re.compile(
    r"\b(?:can|could|would) you (?:please )?(?:confirm|provide|specify|tell me)|"
    r"\bwhat (?:is|are|should)|\bwhich (?:one|item|value)\b",
    re.I,
)

_BROAD_DETAIL_RE = re.compile(
    r"\b(?:all|complete|comprehensive|detail(?:ed)?|everything|full|other information)\b",
    re.I,
)

_EXPLICIT_NEGATION_RE = re.compile(
    r"\b(?:disable|exclude|no|not|without)\b",
    re.I,
)

_GENERIC_PARAMETER_WORDS = {
    "boolean", "default", "enable", "enabled", "flag", "include",
    "included", "optional", "parameter", "true", "value", "whether",
}

# These are ordinary language equivalences, not tool or benchmark names.  The
# controller uses them only to decide whether a Boolean option is semantically
# requested.  Required arguments never depend on this table.
_SEMANTIC_FAMILIES = (
    {"address", "location"},
    {"album", "record"},
    {"artist", "singer"},
    {"bibliography", "bibliographic", "citation", "reference"},
    {"facility", "facilities", "amenity", "amenities", "access", "wheelchair"},
    {"history", "historical"},
    {"image", "images", "photo", "photos", "picture", "pictures"},
    {"incident", "incidents", "alarm", "alarms", "alert", "alerts"},
    {"rating", "ratings", "rated"},
    {"round", "rounded", "rounding"},
    {"share", "shared"},
    {"stat", "stats", "statistic", "statistics"},
    {"sync", "synchronize", "synchronized"},
)
_SEMANTIC_CANON = {
    word: min(family) for family in _SEMANTIC_FAMILIES for word in family
}

_NOVEL_IDENTITY_WORDS = {
    "address", "content", "criteria", "email", "file", "id", "item",
    "name", "query", "text", "title", "url", "word",
}

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _has_reference(text: str) -> bool:
    return bool(_REFERENCE_RE.search(text) or _REFERENCE_CUE_RE.search(text))


@dataclass
class ContractBundle:
    """One native generation and its normalized call proposals."""

    generation: _Generation
    proposals: list[Proposal]
    source: str
    defects: tuple[str, ...]
    signature: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return bool(self.proposals) and not self.defects


def _semantic_terms(text: Any) -> set[str]:
    terms = set()
    for token in _tokens(text):
        token = _SEMANTIC_CANON.get(token, token)
        if token in _GENERIC_PARAMETER_WORDS:
            continue
        # A deliberately small morphological normalizer.  It avoids a new NLP
        # dependency while covering schema keys such as `rounding`/`rounded`.
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        terms.add(_SEMANTIC_CANON.get(token, token))
    return terms


def _decode_json(value: Any) -> Any:
    """Decode the one or two JSON layers used by WTB tool observations."""
    decoded = value
    for _ in range(2):
        if not isinstance(decoded, str):
            break
        try:
            decoded = json.loads(decoded)
        except (TypeError, json.JSONDecodeError):
            break
    return decoded


def _walk_payload(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key),)
            yield child_path, child
            yield from _walk_payload(child, child_path)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_payload(child, path)


def _argument_history(messages: list[dict]) -> list[tuple[str, str, Any, int]]:
    """Collect typed slot values already visible in calls and observations."""
    found: list[tuple[str, str, Any, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for raw_call in message.get("tool_calls") or []:
                function = raw_call.get("function", {})
                arguments = _decode_json(function.get("arguments", {}))
                if not isinstance(arguments, dict):
                    continue
                for path, value in _walk_payload(arguments):
                    if path and not isinstance(value, (dict, list)):
                        found.append((function.get("name", ""), path[-1], value, index))
        elif message.get("role") == "tool":
            payload = _decode_json(message.get("content"))
            for path, value in _walk_payload(payload):
                if path and not isinstance(value, (dict, list)):
                    found.append(("tool_observation", path[-1], value, index))
    return found


def _identity_slot(name: str) -> bool:
    words = set(_split_name(name))
    return bool(words & _NOVEL_IDENTITY_WORDS) or name.lower().endswith("_id")


def _required_fields(proposal: Proposal, tools: list[dict]) -> list[str]:
    tool = _tool_map(tools).get(proposal.name, {})
    return list(_schema(tool).get("required", []) or [])


def _actual_missing(proposal: Proposal, tools: list[dict], current: str) -> list[str]:
    """Distinguish absent structure from merely unproven provenance."""
    missing = [key for key in _required_fields(proposal, tools) if key not in proposal.arguments]
    if _NOVELTY_RE.search(current):
        for key in proposal.missing:
            # GAVEL marks a history-sourced value as missing on "another".
            # Preserve contextual fields such as author/account, but do not
            # silently reuse the identity of the old object.
            if key in proposal.arguments and _identity_slot(key):
                missing.append(key)
    return list(dict.fromkeys(missing))


def _hard_defects(proposal: Proposal) -> list[str]:
    defects = []
    for reason in proposal.hard_reasons:
        if reason in _STRUCTURAL_REASONS or reason.startswith("invalid_type:"):
            defects.append(reason)
    if proposal.risk >= 2 and not proposal.strong_intent:
        defects.append("side_effect_without_explicit_intent")
    return defects


def _boolean_supported(key: str, prop: dict, value: bool, current: str) -> bool:
    if _optional_explicit(key, value, current):
        return True
    # "Everything/other information" licenses additive true capabilities,
    # while an explicit negative phrase licenses the model's false setting.
    # Neither rule invents a value; it only avoids deleting an expressed
    # breadth or exclusion choice from the anchor.
    if value is True and _BROAD_DETAIL_RE.search(current):
        return True
    if value is False and _EXPLICIT_NEGATION_RE.search(current):
        return True
    descriptor = " ".join((key, str(prop.get("description", ""))))
    return bool(_semantic_terms(descriptor) & _semantic_terms(current))


def _same_tool_context(messages: list[dict], tool_name: str) -> str:
    """Recover the user request that established an earlier same-tool goal."""
    last_user = ""
    contexts = []
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            last_user = message["content"]
        if message.get("role") == "assistant" and any(
            call.get("function", {}).get("name") == tool_name
            for call in message.get("tool_calls") or []
        ):
            if last_user:
                contexts.append(last_user)
    return "\n".join(contexts[-2:])


def _raw_arguments(call: Any) -> dict:
    raw = call if isinstance(call, dict) else call.model_dump()
    arguments = _decode_json(raw.get("function", {}).get("arguments", {}))
    return arguments if isinstance(arguments, dict) else {}


def _evaluate_contract_call(
    call: Any,
    tools: list[dict],
    messages: list[dict],
    *,
    anchor: bool,
    source: str,
) -> Proposal:
    """Evaluate a call while retaining an explicitly supported false default."""
    proposal = evaluate_call(
        call, tools, messages, anchor=anchor, source=source
    )
    current, _, _ = _evidence(messages)
    support_context = "\n".join((
        _same_tool_context(messages, proposal.name),
        current,
    )).strip()
    tool = _tool_map(tools).get(proposal.name, {})
    schema = _schema(tool)
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    raw_arguments = _raw_arguments(call)
    restored = False
    for key, value in raw_arguments.items():
        if (
            key not in required
            and key not in proposal.arguments
            and isinstance(value, bool)
            and _boolean_supported(key, properties.get(key, {}), value, support_context)
        ):
            proposal.arguments[key] = value
            restored = True
    if restored:
        proposal.canonical = _canonical(proposal.name, proposal.arguments)
        proposal.call = deepcopy(proposal.call)
        proposal.call["function"]["arguments"] = json.dumps(
            proposal.arguments, ensure_ascii=False
        )
        proposal.elided_optional = tuple(
            key for key in proposal.elided_optional if key not in proposal.arguments
        )
    return _minimize_boolean_commitments(
        proposal, tools, support_context
    )


def _minimize_boolean_commitments(
    proposal: Proposal,
    tools: list[dict],
    current: str,
) -> Proposal:
    """Drop an unsupported optional Boolean without touching required data."""
    tool = _tool_map(tools).get(proposal.name, {})
    schema = _schema(tool)
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    arguments = {}
    elided = list(proposal.elided_optional)
    for key, value in proposal.arguments.items():
        prop = properties.get(key, {})
        if (
            key not in required
            and isinstance(value, bool)
            and not _boolean_supported(key, prop, value, current)
        ):
            elided.append(key)
            continue
        arguments[key] = value
    if arguments == proposal.arguments:
        return proposal
    proposal.arguments = arguments
    proposal.canonical = _canonical(proposal.name, arguments)
    proposal.call = deepcopy(proposal.call)
    proposal.call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
    proposal.elided_optional = tuple(dict.fromkeys(elided))
    return proposal


def _deduplicate(proposals: list[Proposal]) -> list[Proposal]:
    unique = {}
    for proposal in proposals:
        unique.setdefault(proposal.canonical, proposal)
    return list(unique.values())


def _bundle_signature(proposals: list[Proposal], tools: list[dict]) -> tuple[str, ...]:
    """Agreement signature over tool names and required commitments only."""
    signatures = []
    for proposal in proposals:
        required = _required_fields(proposal, tools)
        arguments = {key: proposal.arguments[key] for key in required if key in proposal.arguments}
        signatures.append(_canonical(proposal.name, arguments))
    return tuple(sorted(signatures))


def _bundle(
    generation: _Generation,
    tools: list[dict],
    messages: list[dict],
    *,
    anchor: bool,
    source: str,
) -> ContractBundle:
    current, _, _ = _evidence(messages)
    proposals = [
        _evaluate_contract_call(
            call, tools, messages, anchor=anchor, source=source
        )
        for call in generation.tool_calls
    ]
    proposals = _deduplicate(proposals)
    defects = []
    for proposal in proposals:
        defects.extend(_hard_defects(proposal))
        defects.extend(f"missing:{field}" for field in _actual_missing(proposal, tools, current))
    if not proposals:
        defects.append("no_tool_calls")
    return ContractBundle(
        generation=generation,
        proposals=proposals,
        source=source,
        defects=tuple(dict.fromkeys(defects)),
        signature=_bundle_signature(proposals, tools),
    )


def _ordinal(text: str) -> int | None:
    match = re.search(r"\b(first|second|third|fourth|\d+)(?:st|nd|rd|th)?\b", text, re.I)
    if not match:
        return None
    words = {"first": 1, "second": 2, "third": 3, "fourth": 4}
    return words.get(match.group(1).lower(), int(match.group(1)) if match.group(1).isdigit() else None)


def _slot_candidates(messages: list[dict], field: str) -> list[Any]:
    field_terms = _semantic_terms(field)
    exact = []
    related = []
    for _, key, value, index in _argument_history(messages):
        if isinstance(value, bool) or value is None:
            continue
        if _normal(key) == _normal(field):
            exact.append((index, value))
        elif field_terms and field_terms & _semantic_terms(key):
            related.append((index, value))
    ordered = exact or related
    result = []
    seen = set()
    for _, value in ordered:
        marker = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def _referenced_scalar(messages: list[dict], field: str, current: str) -> Any:
    candidates = _slot_candidates(messages, field)
    if not candidates:
        return None
    low = current.lower()
    ordinal = _ordinal(current)
    if ordinal and ordinal <= len(candidates):
        return candidates[ordinal - 1]
    if re.search(r"\b(?:beginning|earliest|first round|initial)\b", low):
        return candidates[0]
    if re.search(r"\b(?:last|latest|previous|most recent)\b", low):
        return candidates[-1]
    if _REFERENCE_RE.search(current) and len(candidates) == 1:
        return candidates[0]
    return None


def _relative_year(field: str, messages: list[dict], current: str) -> int | None:
    now = _system_date(messages)
    if now is None:
        return None
    match = re.search(
        r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\b",
        current,
        re.I,
    )
    if not match:
        return None
    count = int(match.group(1)) if match.group(1).isdigit() else _NUMBER_WORDS[match.group(1).lower()]
    normalized = _normal(field)
    if "start" in normalized and "year" in normalized:
        return now.year - count
    if "end" in normalized and "year" in normalized:
        return now.year - 1
    return None


def _record_lists(payload: Any) -> list[list[dict]]:
    lists = []
    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            lists.append(payload)
        for item in payload:
            lists.extend(_record_lists(item))
    elif isinstance(payload, dict):
        for value in payload.values():
            lists.extend(_record_lists(value))
    return lists


def _latest_records(messages: list[dict], item_schema: dict) -> list[dict]:
    properties = item_schema.get("properties", {}) or {}
    best: tuple[int, int, list[dict]] | None = None
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        for records in _record_lists(_decode_json(message.get("content"))):
            keys = set().union(*(record.keys() for record in records))
            overlap = 0
            for prop in properties:
                if any(_normal(prop) == _normal(key) or _semantic_terms(prop) & _semantic_terms(key) for key in keys):
                    overlap += 1
            candidate = (overlap, index, records)
            if overlap and (best is None or candidate[:2] > best[:2]):
                best = candidate
    return best[2] if best else []


def _respectively_numbers(current: str) -> list[int | float]:
    prefix = re.split(r"\brespectively\b", current, flags=re.I)[0]
    values = []
    for raw in re.findall(r"(?<![\w'])[-+]?\d+(?:\.\d+)?(?![\w'])", prefix):
        number = float(raw) if "." in raw else int(raw)
        values.append(number)
    return values


def _record_value(record: dict, field: str) -> Any:
    for key, value in record.items():
        if _normal(key) == _normal(field):
            return value
    field_terms = _semantic_terms(field)
    matches = [value for key, value in record.items() if field_terms & _semantic_terms(key)]
    return matches[0] if len(matches) == 1 else None


def _referenced_record_array(messages: list[dict], schema: dict, current: str) -> list[dict] | None:
    item_schema = schema.get("items", {}) or {}
    if item_schema.get("type") != "object":
        return None
    if not re.search(r"\b(?:both|first|last|these|those|two|three|four)\b", current, re.I):
        return None
    records = _latest_records(messages, item_schema)
    count = _explicit_capacity(current)
    if len(records) < count:
        return None
    selected = records[-count:] if re.search(r"\blast\b", current, re.I) else records[:count]
    properties = item_schema.get("properties", {}) or {}
    required = list(item_schema.get("required", []) or [])
    respective = _respectively_numbers(current) if "respectively" in current.lower() else []
    built = []
    for index, record in enumerate(selected):
        item = {}
        for field in properties:
            value = _record_value(record, field)
            if value is None and respective and _normal(field) in {"amount", "count", "quantity", "unit", "units"}:
                if index < len(respective):
                    value = respective[-count + index]
            if value is not None:
                item[field] = value
        if any(field not in item for field in required):
            return None
        built.append(item)
    return built


def _repair_missing(
    proposal: Proposal,
    tools: list[dict],
    messages: list[dict],
    current: str,
) -> Proposal:
    """Fill only values with a unique deterministic dialogue derivation."""
    tool = _tool_map(tools).get(proposal.name, {})
    schema = _schema(tool)
    properties = schema.get("properties", {}) or {}
    arguments = dict(proposal.arguments)
    changed = False
    if _NOVELTY_RE.search(current):
        for field in proposal.missing:
            if field in arguments and _identity_slot(field):
                del arguments[field]
                changed = True
    for field in schema.get("required", []) or []:
        if field in arguments:
            continue
        prop = properties.get(field, {})
        value = None
        if prop.get("type") == "array":
            value = _referenced_record_array(messages, prop, current)
        if value is None:
            value = _relative_year(field, messages, current)
        if (
            value is None
            and (_has_reference(current) or _CONTINUATION_RE.search(current))
            and not (_NOVELTY_RE.search(current) and _identity_slot(field))
        ):
            value = _referenced_scalar(messages, field, current)
        if value is not None:
            arguments[field] = value
            changed = True
    if not changed:
        return proposal
    proposal.arguments = arguments
    proposal.canonical = _canonical(proposal.name, arguments)
    proposal.call = deepcopy(proposal.call)
    proposal.call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
    return proposal


def _completed_names_after_active_user(messages: list[dict]) -> set[str]:
    active = _active_user_indices(messages)
    boundary = max(active) if active else -1
    pending = {}
    completed = set()
    for message in messages[boundary + 1:]:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                pending[call.get("id")] = call.get("function", {}).get("name", "")
        elif message.get("role") == "tool":
            name = pending.get(message.get("tool_call_id"))
            if name:
                completed.add(name)
    return completed


def _intent_matches(current: str, history: str, tools: list[dict]) -> dict[int, set[str]]:
    """Map each explicit request clause to schema-derived candidate tools."""
    clauses = _split_clauses(current)
    result: dict[int, set[str]] = {}
    continuation = bool(_CONTINUATION_RE.search(current) and _has_reference(current))
    for index, clause in enumerate(clauses):
        clause_tokens = _semantic_terms(clause)
        for name, tool in _tool_map(tools).items():
            description = str(tool.get("description", ""))
            action = _action_class(name, description)
            action_words = {
                "read": {"check", "find", "get", "know", "list", "look", "query", "retrieve", "search", "show", "view"},
                "buy": {"buy", "order", "purchase"},
                "book": {"book", "reserve", "schedule"},
                "create": {"add", "create", "register"},
                "generate": {"create", "generate", "make", "produce"},
                "send": {"email", "forward", "message", "send", "share"},
                "update": {"change", "edit", "modify", "set", "update"},
                "delete": {"delete", "erase", "remove"},
                "cancel": {"cancel", "revoke"},
                "download": {"download", "export", "save"},
                "upload": {"attach", "import", "upload"},
                "subscribe": {"notify", "subscribe"},
            }.get(action, {action})
            descriptor = _semantic_terms(" ".join((name, description)))
            object_overlap = clause_tokens & (descriptor - set(action_words))
            action_match = bool(clause_tokens & set(action_words))
            prior_same = name.lower() in history.lower()
            if (action_match and object_overlap) or (continuation and prior_same):
                result.setdefault(index, set()).add(name)
    return result


def residual_obligations(messages: list[dict], tools: list[dict]) -> dict[int, set[str]]:
    """Return explicit tool-capable clauses not completed in the active turn."""
    current, _, history = _evidence(messages)
    matches = _intent_matches(current, history, tools)
    completed = _completed_names_after_active_user(messages)
    residual = {}
    for clause, names in matches.items():
        if not (names & completed):
            residual[clause] = names
    return residual


def _optional_count(bundle: ContractBundle, tools: list[dict]) -> int:
    total = 0
    for proposal in bundle.proposals:
        required = set(_required_fields(proposal, tools))
        total += sum(key not in required for key in proposal.arguments)
    return total


def _covers_residual_contract(
    bundle: ContractBundle,
    messages: list[dict],
    residual: dict[int, set[str]],
) -> bool:
    """Require an override to cover every provable outstanding clause."""
    if not residual:
        return False
    current, _, _ = _evidence(messages)
    clauses = _split_clauses(current)
    names = [proposal.name for proposal in bundle.proposals]
    for clause_index, eligible_names in residual.items():
        clause = clauses[clause_index] if clause_index < len(clauses) else ""
        required_count = _explicit_capacity(clause)
        covered = sum(name in eligible_names for name in names)
        if covered < required_count:
            return False
    return True


def select_dominant_bundle(
    bundles: list[ContractBundle],
    tools: list[dict],
    *,
    text_anchor: bool,
) -> ContractBundle | None:
    """Select by exact agreement and minimum commitment, never plurality."""
    complete = [bundle for bundle in bundles if bundle.complete]
    if not complete:
        return None
    counts = {}
    for bundle in complete:
        counts[bundle.signature] = counts.get(bundle.signature, 0) + 1
    if text_anchor:
        # A text decision can be overturned only by two independently sampled
        # native generations agreeing on every required commitment.
        complete = [bundle for bundle in complete if counts[bundle.signature] >= 2]
        if not complete:
            return None
    return min(
        complete,
        key=lambda bundle: (
            _optional_count(bundle, tools),
            sum(proposal.risk for proposal in bundle.proposals),
            len(bundle.proposals),
            bundle.signature,
        ),
    )


class ConcordHandler(GavelHandler):
    """Session-contract successor to IGAR-v24 and GAVEL-v2."""

    total_candidates = max(1, int(os.getenv("WTB_CONCORD_CANDIDATES", "3")))
    proposal_temperature = float(os.getenv("WTB_CONCORD_TEMPERATURE", "0.2"))
    request_timeout_seconds = float(os.getenv("WTB_CONCORD_REQUEST_TIMEOUT", "600"))

    def _audit_concord(
        self,
        inference_data: dict,
        anchor: _Generation,
        bundles: list[ContractBundle],
        decision: str,
        selected: ContractBundle | None = None,
        residual: dict[int, set[str]] | None = None,
    ) -> None:
        record = {
            "id": inference_data.get("test_entry_id"),
            "turn": inference_data.get("task_idx"),
            "anchor": "tools" if anchor.tool_calls else "text",
            "decision": decision,
            "residual_clauses": sorted((residual or {}).keys()),
            "selected": list(selected.signature) if selected else [],
            "bundles": [
                {
                    "source": bundle.source,
                    "signature": list(bundle.signature),
                    "defects": list(bundle.defects),
                    "elided_optional": sorted({
                        key for proposal in bundle.proposals for key in proposal.elided_optional
                    }),
                }
                for bundle in bundles
            ],
        }
        print("[CONCORD] " + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    def _make_bundle(
        self,
        generation: _Generation,
        tools: list[dict],
        messages: list[dict],
        *,
        anchor: bool,
        source: str,
    ) -> ContractBundle:
        current, _, _ = _evidence(messages)
        bundle = _bundle(generation, tools, messages, anchor=anchor, source=source)
        if not bundle.proposals:
            return bundle
        repaired = [
            _repair_missing(proposal, tools, messages, current)
            for proposal in bundle.proposals
        ]
        repaired = [
            _evaluate_contract_call(
                proposal.call,
                tools,
                messages,
                anchor=(source == "anchor"),
                source=source,
            )
            for proposal in repaired
        ]
        repaired = _deduplicate(repaired)
        defects = []
        for proposal in repaired:
            defects.extend(_hard_defects(proposal))
            defects.extend(
                f"missing:{field}"
                for field in _actual_missing(proposal, tools, current)
            )
        return ContractBundle(
            generation=generation,
            proposals=repaired,
            source=source,
            defects=tuple(dict.fromkeys(defects)),
            signature=_bundle_signature(repaired, tools),
        )

    def _request_tool_call(self, inference_data):
        # Evaluator firewall: this method intentionally reads only these keys.
        messages = inference_data["messages"]
        tools = inference_data["tools"]
        generations: list[_Generation] = []
        bundles: list[ContractBundle] = []

        anchor = self._generate(messages, tools)
        generations.append(anchor)
        residual = residual_obligations(messages, tools)

        anchor_bundle = self._make_bundle(
            anchor, tools, messages, anchor=True, source="anchor"
        )
        if anchor.tool_calls:
            bundles.append(anchor_bundle)
            if anchor_bundle.complete:
                pin, pout, latency = self._sum_usage(generations)
                self._audit_concord(
                    inference_data, anchor, bundles, "preserve_complete_anchor",
                    anchor_bundle, residual,
                )
                return _Response(
                    anchor.content,
                    [proposal.call for proposal in anchor_bundle.proposals],
                    pin,
                    pout,
                    anchor.reasoning_content,
                ), latency

        meaningful_text = bool(str(anchor.content or "").strip()) and not anchor.tool_calls
        should_rescue = bool(anchor.tool_calls) or (meaningful_text and bool(residual))

        if should_rescue:
            for index in range(1, self.total_candidates):
                try:
                    candidate = self._generate(
                        messages, tools, temperature=self.proposal_temperature
                    )
                    generations.append(candidate)
                    if candidate.tool_calls:
                        bundles.append(self._make_bundle(
                            candidate,
                            tools,
                            messages,
                            anchor=False,
                            source=f"proposal_{index}",
                        ))
                except Exception as exc:
                    print(
                        f"[CONCORD] proposal branch unavailable: {type(exc).__name__}: {exc}",
                        flush=True,
                    )

        eligible_bundles = [bundle for bundle in bundles if bundle.source != "anchor"]
        if meaningful_text:
            eligible_bundles = [
                bundle for bundle in eligible_bundles
                if _covers_residual_contract(bundle, messages, residual)
            ]
        selected = select_dominant_bundle(
            eligible_bundles,
            tools,
            text_anchor=meaningful_text,
        )
        if selected is not None:
            pin, pout, latency = self._sum_usage(generations)
            self._audit_concord(
                inference_data, anchor, bundles, "execute_dominant_contract",
                selected, residual,
            )
            return _Response(
                selected.generation.content,
                [proposal.call for proposal in selected.proposals],
                pin,
                pout,
                selected.generation.reasoning_content,
            ), latency

        if meaningful_text:
            pin, pout, latency = self._sum_usage(generations)
            decision = "keep_clarification" if _CLARIFICATION_RE.search(anchor.content or "") else "text_sovereignty"
            self._audit_concord(inference_data, anchor, bundles, decision, residual=residual)
            return _Response(
                anchor.content, None, pin, pout, anchor.reasoning_content
            ), latency

        # A tool anchor can reach here only when it has a positive structural
        # defect and no complete agreeing rescue.  Ask only for fields that are
        # actually absent after deterministic reference resolution.
        current, _, _ = _evidence(messages)
        blocked = []
        for bundle in bundles:
            for proposal in bundle.proposals:
                missing = _actual_missing(proposal, tools, current)
                truly_absent = tuple(field for field in missing if field not in proposal.arguments)
                if truly_absent and not _hard_defects(proposal):
                    proposal.missing = truly_absent
                    proposal.unsupported = ()
                    blocked.append(proposal)
        fields, plan = minimum_information_clarification(blocked)
        if fields:
            pin, pout, latency = self._sum_usage(generations)
            self._audit_concord(inference_data, anchor, bundles, "clarify_true_absence", residual=residual)
            return _Response(_clarifying_text(fields, plan), None, pin, pout), latency

        # If the contract cannot prove a better move, preserve the raw anchor.
        pin, pout, latency = self._sum_usage(generations)
        self._audit_concord(inference_data, anchor, bundles, "fallback_anchor", residual=residual)
        return _Response(
            anchor.content,
            anchor.tool_calls,
            pin,
            pout,
            anchor.reasoning_content,
        ), latency

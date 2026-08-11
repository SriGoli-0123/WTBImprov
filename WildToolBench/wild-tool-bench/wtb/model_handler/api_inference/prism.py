"""PRISM: policy-reset inference over unmodified evidence views.

WildToolBench's paper identifies self-conditioning and attention dilution as
two important causes of failure: a model that just used a tool tends to keep
using tools, while the current request competes with a long conversation for
attention.  PRISM attacks those causes without adding instructions.

Each generation receives only messages that were already present in the
ordinary WTB request and the unchanged tool list.  The three default views are
the full dialogue, the current task, and (when the user refers backwards) the
most relevant earlier exchange.  No message is rewritten and no ledger,
critique, label, answer path, or evaluator result is exposed to the model.

The full-history response remains the anchor.  A tool/text decision can replace
it only when it is stable across two views.  Complete calls are normalized by
CONCORD's reference resolver.  A narrow absence gate asks for genuinely
missing required information instead of accepting a confident invented value.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from wtb.model_handler.api_inference.concord import (
    ConcordHandler,
    ContractBundle,
    _CLARIFICATION_RE,
    _REFERENCE_CUE_RE,
    _Response,
    _actual_missing,
    _clarifying_text,
    _optional_count,
    _required_fields,
)
from wtb.model_handler.api_inference.gavel import (
    Proposal,
    _Generation,
    _REFERENCE_RE,
    _canonical,
    _normal,
    _tokens,
)


_STOP_WORDS = {
    "a", "about", "again", "all", "also", "an", "and", "are", "as",
    "at", "be", "before", "but", "by", "can", "could", "do", "for",
    "from", "get", "give", "help", "how", "i", "in", "information",
    "is", "it", "know", "me", "my", "of", "on", "one", "please",
    "that", "the", "then", "this", "to", "want", "what", "which",
    "with", "would", "you",
}

_BACK_REFERENCE_RE = re.compile(
    r"\b(?:above|before|beginning|earlier|first round|initial|last round|"
    r"mentioned|previous|same|that|those|this|these|they|them|he|she|it|"
    r"former|latter|respectively|another|again)\b",
    re.I,
)

_FIRST_REFERENCE_RE = re.compile(
    r"\b(?:at the beginning|beginning|first round|initial(?:ly)?)\b", re.I
)

_ABSENCE_CUE_RE = re.compile(
    r"\b(?:another|a certain|a specific|can't remember|cannot remember|"
    r"do not remember|don't remember|not sure|one of|somewhere|something else|"
    r"someone|somebody|some item|some product|some word|which one)\b",
    re.I,
)

_CHOICE_FIELD_RE = re.compile(
    r"(?:account_?id|campaign_?name|country|credential|email|file|ip|"
    r"location(?:_?id)?|method|party|password|payment(?:_?method)?|platform|"
    r"price|quantity|recipient_?email|room_?id|sample|secret|title|token|type|"
    r"url|user_?id)$",
    re.I,
)

_EXPLICIT_REPEAT_RE = re.compile(
    r"\b(?:again|do it again|one more time|repeat|same operation|recheck|"
    r"run it again|perform it again)\b",
    re.I,
)

_FOLLOWUP_QUESTION_RE = re.compile(
    r"(?:\b(?:what|which|where|when|who|how)\b|"
    r"\b(?:can|could|would) you\b)[^?]*\?\s*$",
    re.I | re.S,
)


@dataclass
class ViewCandidate:
    """One model decision made from one evidence view."""

    view: str
    messages: list[dict]
    generation: _Generation
    bundle: ContractBundle | None

    @property
    def text(self) -> bool:
        return not self.generation.tool_calls and bool(
            str(self.generation.content or "").strip()
        )

    @property
    def complete_tools(self) -> bool:
        return self.bundle is not None and self.bundle.complete


def _leading_system_messages(messages: list[dict]) -> list[dict]:
    prefix = []
    for message in messages:
        if message.get("role") != "system":
            break
        prefix.append(message)
    return prefix


def _current_user_text(messages: list[dict], task_start: int) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages[task_start:]
        if message.get("role") == "user"
    ).strip()


def _content_terms(value: Any) -> set[str]:
    return {
        token for token in _tokens(value)
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _exchange_end(messages: list[dict], start: int, ceiling: int) -> int:
    """Return the end of a compact, protocol-valid earlier exchange.

    A clarification question is followed through its user reply.  A normal
    assistant answer closes the slice.  Tool calls and their tool messages are
    never separated.
    """

    saw_tool = False
    index = start + 1
    while index < ceiling:
        message = messages[index]
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            saw_tool = True
        elif role == "assistant" and str(message.get("content") or "").strip():
            content = str(message.get("content") or "")
            next_is_user = (
                index + 1 < ceiling and messages[index + 1].get("role") == "user"
            )
            asks_for_reply = bool(
                _CLARIFICATION_RE.search(content)
                or _FOLLOWUP_QUESTION_RE.search(content)
            )
            if next_is_user and asks_for_reply and not saw_tool:
                index += 1
                continue
            return index + 1
        index += 1
    return ceiling


def _focused_exchange(messages: list[dict], task_start: int) -> list[dict]:
    """Retrieve one exact earlier exchange without writing a summary."""

    system_count = len(_leading_system_messages(messages))
    user_indices = [
        index for index in range(system_count, task_start)
        if messages[index].get("role") == "user"
        and not (
            index > system_count
            and messages[index - 1].get("role") == "assistant"
            and not messages[index - 1].get("tool_calls")
            and (
                _CLARIFICATION_RE.search(
                    str(messages[index - 1].get("content") or "")
                )
                or _FOLLOWUP_QUESTION_RE.search(
                    str(messages[index - 1].get("content") or "")
                )
            )
        )
    ]
    if not user_indices:
        return []

    current = _current_user_text(messages, task_start)
    if _FIRST_REFERENCE_RE.search(current):
        chosen = user_indices[0]
    else:
        current_terms = _content_terms(current)
        scored = []
        for rank, index in enumerate(user_indices):
            end = _exchange_end(messages, index, task_start)
            block_terms = _content_terms(messages[index:end])
            overlap = len(current_terms & block_terms)
            # Recency is a tie-break, never a substitute for semantic overlap.
            scored.append((overlap, rank, index))
        chosen = max(scored)[2]

    end = _exchange_end(messages, chosen, task_start)
    return messages[chosen:end]


def build_evidence_views(
    messages: list[dict], task_start: int
) -> tuple[list[dict], list[dict], list[dict], bool]:
    """Build full, reset, and focused views from existing message objects."""

    task_start = max(0, min(task_start, len(messages) - 1))
    systems = _leading_system_messages(messages)
    current = messages[task_start:]
    full = list(messages)
    reset = list(systems) + list(current)
    current_text = _current_user_text(messages, task_start)
    referential = bool(
        _BACK_REFERENCE_RE.search(current_text)
        or _REFERENCE_RE.search(current_text)
        or _REFERENCE_CUE_RE.search(current_text)
    )
    # Always construct the middle-distance view.  Real follow-ups are often
    # elliptical ("China", "Convert to EUR", "Where is the location?") and
    # carry no lexical reference cue.  Agreement, rather than a cue list,
    # decides whether that earlier exchange is relevant.
    focused_history = _focused_exchange(messages, task_start)
    focused = list(systems) + list(focused_history) + list(current)
    return full, reset, focused, referential


def _literal_value(value: Any, messages: list[dict]) -> bool:
    """Whether a scalar/container value is visibly supported by the dialogue."""

    haystack = _normal(json.dumps(messages, ensure_ascii=False))
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, dict):
        return all(_literal_value(child, messages) for child in value.values())
    if isinstance(value, list):
        return all(_literal_value(child, messages) for child in value)
    needle = _normal(str(value))
    if not needle:
        return False
    if needle in haystack:
        return True
    # A user may provide an address in pieces across clarification replies
    # (for example, the first and second halves of an IPv4 address).
    text_value = str(value).strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", text_value):
        return all(_normal(part) in haystack for part in text_value.split("."))
    # ISO dates and years can be mechanically resolved from the supplied date.
    if re.fullmatch(r"\d{4}(?:-\d{2}-\d{2}(?:[T ][^ ]+)?)?", text_value):
        return True
    # Short codes are often deterministic contractions: United States -> US.
    if isinstance(value, str) and len(re.sub(r"\W", "", value)) <= 4:
        letters = re.sub(r"\W", "", value).lower()
        words = re.findall(r"[A-Za-z0-9]+", json.dumps(messages, ensure_ascii=False))
        for word in words:
            cursor = iter(word.lower())
            if letters and all(char in cursor for char in letters):
                return True
    return False


def _required_commitments(bundle: ContractBundle, tools: list[dict]) -> tuple[str, ...]:
    rows = []
    for proposal in bundle.proposals:
        required = _required_fields(proposal, tools)
        arguments = {
            key: proposal.arguments.get(key)
            for key in required
            if key in proposal.arguments
        }
        rows.append(_canonical(proposal.name, arguments))
    return tuple(sorted(rows))


def _toolset_signature(bundle: ContractBundle) -> tuple[str, ...]:
    return tuple(sorted(proposal.name for proposal in bundle.proposals))


def _decision_signature(
    bundle: ContractBundle, tools: list[dict]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Agreement requires the same calls and the same required values."""

    return _toolset_signature(bundle), _required_commitments(bundle, tools)


def _question_fields(
    bundles: list[ContractBundle],
    tools: list[dict],
    messages: list[dict],
    current: str,
) -> tuple[list[str], Proposal | None]:
    """Find required fields that are absent rather than merely uncertain."""

    fields = []
    plan = None
    for bundle in bundles:
        for proposal in bundle.proposals:
            required = set(_required_fields(proposal, tools))
            truly_missing = _actual_missing(proposal, tools, current)
            for field in truly_missing:
                if field not in fields:
                    fields.append(field)
                    plan = plan or proposal
            for field in required & set(proposal.unsupported):
                value = proposal.arguments.get(field)
                choice_like = bool(_CHOICE_FIELD_RE.search(field))
                absence_cue = bool(_ABSENCE_CUE_RE.search(current))
                risky_choice = proposal.risk >= 2 and choice_like
                if (
                    not _literal_value(value, messages)
                    and choice_like
                    and (absence_cue or risky_choice)
                    and field not in fields
                ):
                    fields.append(field)
                    plan = plan or proposal
    return fields, plan


def _repeated_without_request(bundle: ContractBundle, current: str) -> bool:
    if _EXPLICIT_REPEAT_RE.search(current):
        return False
    return any("already_completed" in proposal.hard_reasons for proposal in bundle.proposals)


def _pick_text(candidates: list[ViewCandidate], referential: bool) -> ViewCandidate | None:
    texts = [candidate for candidate in candidates if candidate.text]
    if not texts:
        return None
    preference = ("focused", "full", "reset") if referential else ("reset", "full", "focused")
    for name in preference:
        for candidate in texts:
            if candidate.view.startswith(name):
                return candidate
    return texts[0]


def _pick_tool_bundle(
    candidates: list[ViewCandidate], tools: list[dict]
) -> tuple[ViewCandidate | None, list[ViewCandidate]]:
    viable = [candidate for candidate in candidates if candidate.complete_tools]
    if not viable:
        return None, []
    groups: dict[
        tuple[tuple[str, ...], tuple[str, ...]], list[ViewCandidate]
    ] = {}
    for candidate in viable:
        signature = _decision_signature(candidate.bundle, tools)
        groups.setdefault(signature, []).append(candidate)
    best_group = min(
        groups.values(),
        key=lambda group: (
            -len(group),
            len(group[0].bundle.proposals),
            _decision_signature(group[0].bundle, tools),
        ),
    )
    chosen = min(
        best_group,
        key=lambda candidate: (
            _optional_count(candidate.bundle, tools),
            sum(proposal.risk for proposal in candidate.bundle.proposals),
            candidate.view != "full",
            candidate.bundle.signature,
        ),
    )
    return chosen, best_group


class PrismHandler(ConcordHandler):
    """Inference-only policy reset and focused-recall controller."""

    total_candidates = max(1, int(os.getenv("WTB_PRISM_CANDIDATES", "3")))
    request_timeout_seconds = float(os.getenv("WTB_PRISM_REQUEST_TIMEOUT", "600"))
    shadow_temperatures = tuple(
        float(value.strip())
        for value in os.getenv("WTB_PRISM_TEMPERATURES", "0.15,0.45,0.7,0.9").split(",")
        if value.strip()
    )

    def _candidate_specs(
        self, messages: list[dict], task_start: int
    ) -> tuple[list[tuple[str, list[dict], float | None]], bool]:
        full, reset, focused, referential = build_evidence_views(messages, task_start)
        active_task_has_progress = len(messages) > task_start + 1
        if focused != reset and not active_task_has_progress:
            base = [("full", full, None), ("focused", focused, None), ("reset", reset, None)]
        else:
            # Once the active task has a tool result or clarification reply,
            # its reset slice already carries that evidence.  Two current-task
            # views prevent older sessions from reopening completed actions.
            # The same rule is ordinary self-consistency on a first turn.
            base = [("full", full, None), ("reset", reset, None), ("reset_shadow", reset, None)]

        specs = []
        for index in range(self.total_candidates):
            view, view_messages, _ = base[index % len(base)]
            if index == 0:
                temperature = None
            else:
                temperature = self.shadow_temperatures[
                    min(index - 1, len(self.shadow_temperatures) - 1)
                ] if self.shadow_temperatures else self.temperature
            specs.append((view, deepcopy(view_messages), temperature))
        return specs, referential

    def _audit_prism(
        self,
        inference_data: dict,
        candidates: list[ViewCandidate],
        decision: str,
        selected: ViewCandidate | None = None,
        question_fields: list[str] | None = None,
    ) -> None:
        record = {
            "id": inference_data.get("test_entry_id"),
            "turn": inference_data.get("task_idx"),
            "decision": decision,
            "selected_view": selected.view if selected else None,
            "question_fields": question_fields or [],
            "candidates": [
                {
                    "view": candidate.view,
                    "mode": "text" if candidate.text else "tools" if candidate.generation.tool_calls else "empty",
                    "complete": candidate.complete_tools,
                    "toolset": list(_toolset_signature(candidate.bundle)) if candidate.bundle else [],
                    "defects": list(candidate.bundle.defects) if candidate.bundle else [],
                }
                for candidate in candidates
            ],
        }
        print("[PRISM] " + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    def _emit_text(
        self,
        inference_data: dict,
        candidates: list[ViewCandidate],
        candidate: ViewCandidate,
        decision: str,
    ):
        generations = [item.generation for item in candidates]
        pin, pout, latency = self._sum_usage(generations)
        self._audit_prism(inference_data, candidates, decision, candidate)
        return _Response(
            candidate.generation.content,
            None,
            pin,
            pout,
            candidate.generation.reasoning_content,
        ), latency

    def _emit_tools(
        self,
        inference_data: dict,
        candidates: list[ViewCandidate],
        candidate: ViewCandidate,
        decision: str,
    ):
        generations = [item.generation for item in candidates]
        pin, pout, latency = self._sum_usage(generations)
        self._audit_prism(inference_data, candidates, decision, candidate)
        return _Response(
            candidate.generation.content,
            [proposal.call for proposal in candidate.bundle.proposals],
            pin,
            pout,
            candidate.generation.reasoning_content,
        ), latency

    def _request_tool_call(self, inference_data):
        # Evaluator firewall: only the model-visible dialogue and schemas affect
        # the decision.  The id and turn are read solely by the audit logger.
        messages = inference_data["messages"]
        tools = inference_data["tools"]
        task_start = inference_data.setdefault("_prism_task_start", len(messages) - 1)
        specs, referential = self._candidate_specs(messages, task_start)
        candidates: list[ViewCandidate] = []

        for index, (view, view_messages, temperature) in enumerate(specs):
            try:
                generation = self._generate(
                    view_messages,
                    tools,
                    temperature=temperature,
                )
            except Exception as exc:
                print(
                    f"[PRISM] {view} view unavailable: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            is_anchor = index == 0 and view == "full"
            source = "anchor" if is_anchor else f"{view}_{index}"
            bundle = None
            if generation.tool_calls:
                bundle = self._make_bundle(
                    generation,
                    tools,
                    messages,
                    anchor=is_anchor,
                    source=source,
                )
            candidates.append(ViewCandidate(view, view_messages, generation, bundle))

        if not candidates:
            raise RuntimeError("PRISM could not obtain any model generation")

        anchor = next(
            (candidate for candidate in candidates if candidate.view == "full"),
            candidates[0],
        )
        text_candidates = [candidate for candidate in candidates if candidate.text]
        chosen_tool, tool_group = _pick_tool_bundle(candidates, tools)
        tool_quorum = len(tool_group)
        text_quorum = len(text_candidates)
        current = _current_user_text(messages, task_start)

        # A completed repeat is usually self-conditioning, not a fresh need.
        repeated_candidates = [
            candidate
            for candidate in candidates
            if candidate.bundle is not None
            and _repeated_without_request(candidate.bundle, current)
        ]
        if repeated_candidates and len(repeated_candidates) == len(candidates):
            selected_text = _pick_text(candidates, referential)
            if selected_text is not None:
                return self._emit_text(
                    inference_data, candidates, selected_text, "veto_redundant_call"
                )
            generations = [item.generation for item in candidates]
            pin, pout, latency = self._sum_usage(generations)
            self._audit_prism(inference_data, candidates, "veto_redundant_call")
            return _Response(
                "That exact operation has already been completed.",
                None,
                pin,
                pout,
            ), latency

        # True absence has priority over confidence.  Prefer a question written
        # by one of the unchanged-input views; synthesize only the field label
        # when no text candidate exists.
        if chosen_tool is not None:
            fields, plan = _question_fields(
                [chosen_tool.bundle], tools, messages, current
            )
            if fields:
                selected_text = _pick_text(candidates, referential)
                if selected_text is not None:
                    generations = [item.generation for item in candidates]
                    pin, pout, latency = self._sum_usage(generations)
                    self._audit_prism(
                        inference_data,
                        candidates,
                        "clarify_true_absence",
                        selected_text,
                        fields,
                    )
                    return _Response(
                        selected_text.generation.content,
                        None,
                        pin,
                        pout,
                        selected_text.generation.reasoning_content,
                    ), latency
                generations = [item.generation for item in candidates]
                pin, pout, latency = self._sum_usage(generations)
                self._audit_prism(
                    inference_data, candidates, "clarify_true_absence", question_fields=fields
                )
                return _Response(_clarifying_text(fields, plan), None, pin, pout), latency

        # Two independent evidence views must agree before changing modes.
        if text_quorum >= 2 and text_quorum > tool_quorum:
            selected_text = _pick_text(candidates, referential)
            return self._emit_text(
                inference_data, candidates, selected_text, "text_view_quorum"
            )
        if chosen_tool is not None and tool_quorum >= 2 and tool_quorum > text_quorum:
            return self._emit_tools(
                inference_data, candidates, chosen_tool, "tool_view_quorum"
            )

        # No stable replacement was proved.  Preserve a sound full-history
        # anchor; otherwise fall back to the best available complete decision.
        if anchor.text:
            return self._emit_text(
                inference_data, candidates, anchor, "preserve_text_anchor"
            )
        if anchor.complete_tools:
            return self._emit_tools(
                inference_data, candidates, anchor, "preserve_tool_anchor"
            )
        selected_text = _pick_text(candidates, referential)
        if selected_text is not None:
            return self._emit_text(
                inference_data, candidates, selected_text, "fallback_text"
            )
        if chosen_tool is not None:
            return self._emit_tools(
                inference_data, candidates, chosen_tool, "fallback_complete_tool"
            )

        generations = [item.generation for item in candidates]
        pin, pout, latency = self._sum_usage(generations)
        self._audit_prism(inference_data, candidates, "fallback_raw_anchor", anchor)
        return _Response(
            anchor.generation.content,
            anchor.generation.tool_calls or None,
            pin,
            pout,
            anchor.generation.reasoning_content,
        ), latency

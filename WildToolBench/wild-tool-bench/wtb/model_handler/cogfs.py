"""Counterfactual Obligation Graph with Frontier Synthesis (COG-FS).

COG-FS is a training-free inference controller for messy, multi-turn tool use.
It never receives benchmark labels, answer lists, graphs, scores, or expected
actions.  Its complete runtime interface is the same one available to a
deployed assistant:

    messages + tool schemas

The controller does not add instructions.  It generates an untouched anchor,
then creates subtractive shadow views by removing the latest external event,
neutralising prior tool-call syntax, or masking one clause of the latest user
message.  Changes across those views identify which pieces of the user's
request cause which calls.  Confirmed calls are assembled into the complete
schema-valid frontier that is executable now.

This differs from ordinary self-consistency: the views have different causal
content and there is no vote.  It also differs from the retired CAV decision
rule: counterfactuals may discover and add a suppressed call instead of merely
vetoing the anchor.
"""

from __future__ import annotations

import json
import re
import time

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from wtb.model_handler.cav import CAVController, RuntimeFirewallError


class COGFSController(CAVController):
    """Discover current obligations and synthesize their executable frontier."""

    _STOPWORDS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
        "do", "for", "from", "get", "give", "help", "i", "in", "is", "it",
        "me", "my", "of", "on", "or", "please", "show", "that", "the",
        "this", "to", "tool", "use", "want", "with", "would", "you",
    })
    _CLAUSE_BOUNDARY = re.compile(
        r"(?<=[.!?;])\s+|\n+|,\s*(?=(?:and|but|then|also)\b)|"
        r"\s+(?=(?:and then|also|then|plus)\b)",
        re.IGNORECASE,
    )

    def __init__(self, generate, generate_text=None, max_workers=8, max_spans=3):
        super().__init__(
            generate=generate,
            generate_text=generate_text,
            max_workers=max_workers,
        )
        # max_workers is also the hard model-call budget.  A value of eight
        # means one untouched anchor plus at most seven diagnostic/confirmation
        # calls, all served by one shared model instance.
        self.max_calls = max(1, int(max_workers))
        self.max_spans = max(1, int(max_spans))

    @staticmethod
    def _latest_user_index(messages):
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                return index
        return None

    @classmethod
    def _clause_spans(cls, text, limit=3):
        """Return conservative, lossless character spans for user obligations.

        Boundaries come only from punctuation, newlines, and general
        coordination markers.  No benchmark task type, domain vocabulary, or
        learned intent classifier is involved.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        raw = []
        start = 0
        for boundary in cls._CLAUSE_BOUNDARY.finditer(text):
            end = boundary.start()
            if text[start:end].strip():
                raw.append((start, end))
            start = boundary.end()
        if text[start:].strip():
            raw.append((start, len(text)))
        if not raw:
            raw = [(0, len(text))]

        # Empty and one-token fragments are attached to a neighbour.  This
        # avoids treating discourse particles such as "also" as obligations.
        merged = []
        for span in raw:
            token_count = len(re.findall(r"[A-Za-z0-9]+", text[span[0]:span[1]]))
            if token_count <= 1 and merged:
                merged[-1] = (merged[-1][0], span[1])
            else:
                merged.append(span)
        raw = merged or [(0, len(text))]

        limit = max(1, int(limit))
        if len(raw) > limit:
            raw = raw[:limit - 1] + [(raw[limit - 1][0], raw[-1][1])]
        return raw

    @staticmethod
    def _mask_user_span(runtime, user_index, span):
        view = {
            "messages": deepcopy(runtime["messages"]),
            "tools": deepcopy(runtime["tools"]),
        }
        content = view["messages"][user_index].get("content")
        if not isinstance(content, str):
            return None
        start, end = span
        # Whitespace is subtractive: it preserves message structure and adds no
        # semantic token that could teach the model a policy.
        masked = content[:start] + (" " * max(0, end - start)) + content[end:]
        view["messages"][user_index]["content"] = masked
        return view

    @staticmethod
    def _fingerprint(runtime):
        return json.dumps(runtime, sort_keys=True, ensure_ascii=False)

    @classmethod
    def _call_key(cls, call):
        if not isinstance(call, dict):
            return None
        function = call.get("function", {})
        name = function.get("name")
        arguments = cls._arguments(call)
        if not name or arguments is None:
            return None
        return (
            name,
            json.dumps(arguments, sort_keys=True, ensure_ascii=False),
        )

    @classmethod
    def _call_map(cls, response):
        calls = {}
        if not isinstance(response, dict):
            return calls
        for call in response.get("tool_calls") or []:
            key = cls._call_key(call)
            if key is not None:
                calls.setdefault(key, deepcopy(call))
        return calls

    @classmethod
    def _call_receipts(cls, call):
        receipts = set()
        function = call.get("function", {})
        name = function.get("name")
        arguments = cls._arguments(call)
        if not name or arguments is None:
            return receipts
        for key, value in arguments.items():
            receipts.add((
                name,
                key,
                json.dumps(value, sort_keys=True, ensure_ascii=False),
            ))
        return receipts

    @staticmethod
    def _current_user_message(runtime):
        for message in reversed(runtime["messages"]):
            if message.get("role") == "user":
                return message
        return None

    @classmethod
    def _tokens_for_text(cls, text):
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in cls._STOPWORDS
        }

    @classmethod
    def _tool_descriptor_tokens(cls, tool):
        schema = cls._tool_schema(tool)
        parts = [schema.get("name", ""), schema.get("description", "")]
        parameters = schema.get("parameters", {}) or {}
        for key, value in (parameters.get("properties", {}) or {}).items():
            parts.extend([key, value.get("description", "")])
        return cls._tokens_for_text(" ".join(parts))

    @classmethod
    def _projection_tool(cls, runtime):
        """Select one schema projection only when it has lexical user support.

        Projection is a last-resort discovery path for a text anchor that hides
        a single-tool obligation.  Tools with no required arguments are
        excluded because a one-tool menu could otherwise manufacture a call
        without any value-level evidence.
        """
        user_message = cls._current_user_message(runtime)
        content = user_message.get("content", "") if user_message else ""
        user_tokens = cls._tokens_for_text(content)
        if not user_tokens:
            return None

        ranked = []
        for tool in runtime["tools"]:
            schema = cls._tool_schema(tool)
            required = (
                schema.get("parameters", {}) or {}
            ).get("required", []) or []
            if not schema.get("name") or not required:
                continue
            overlap = len(user_tokens & cls._tool_descriptor_tokens(tool))
            if overlap:
                ranked.append((-overlap, schema["name"], tool))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1]))
        return deepcopy(ranked[0][2])

    @staticmethod
    def _isolated_runtime(runtime, tool):
        return {
            "messages": deepcopy(runtime["messages"]),
            "tools": [deepcopy(tool)],
        }

    def _parallel_generate(self, requests):
        """Generate labelled runtime views concurrently."""
        if not requests:
            return {}
        results = {}
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(requests))
        ) as executor:
            futures = {
                label: executor.submit(self._safe_generate, view)
                for label, view in requests.items()
            }
            for label, future in futures.items():
                response, error = future.result()
                results[label] = {"response": response, "error": error}
        return results

    @classmethod
    def _directly_supported_by_latest(cls, value, runtime):
        message = cls._current_user_message(runtime)
        return bool(message) and cls._directly_supported(value, [message])

    @classmethod
    def _projection_has_required_source(cls, call, tool, runtime):
        arguments = cls._arguments(call)
        schema = cls._tool_schema(tool)
        required = (
            schema.get("parameters", {}) or {}
        ).get("required", []) or []
        if arguments is None or not required:
            return False
        return any(
            key in arguments
            and cls._directly_supported_by_latest(arguments[key], runtime)
            for key in required
        )

    @classmethod
    def _frontier(
        cls,
        candidates,
        runtime,
        causal_receipts,
    ):
        """Clean and assemble all causally supported calls executable now."""
        schemas = {
            schema.get("name"): schema
            for schema in map(cls._tool_schema, runtime["tools"])
            if schema.get("name")
        }
        completed = cls._completed_calls(runtime["messages"])
        kept = []
        kept_keys = set()
        suppressed = []
        optional_drops = []

        for info in candidates:
            call = info["call"]
            function = call.get("function", {})
            name = function.get("name")
            schema = schemas.get(name)
            arguments = cls._arguments(call)
            reasons = []

            if schema is None:
                reasons.append("unknown_tool")
            if arguments is None:
                reasons.append("invalid_arguments_json")
            if reasons:
                suppressed.append({
                    "tool": name,
                    "reasons": reasons,
                    "origins": sorted(info["origins"]),
                })
                continue

            original_canonical = json.dumps(
                arguments, sort_keys=True, ensure_ascii=False
            )
            parameters = schema.get("parameters", {}) or {}
            properties = parameters.get("properties", {}) or {}
            required = set(parameters.get("required", []) or [])
            cleaned = {}

            for key, raw_value in arguments.items():
                if key not in properties:
                    optional_drops.append({"tool": name, "argument": key})
                    continue
                value = cls._coerce_schema_value(raw_value, properties[key])
                if not cls._schema_valid(value, properties[key]):
                    reasons.append(f"schema_invalid:{key}")
                    continue
                raw_receipt = (
                    name,
                    key,
                    json.dumps(raw_value, sort_keys=True, ensure_ascii=False),
                )
                coerced_receipt = (
                    name,
                    key,
                    json.dumps(value, sort_keys=True, ensure_ascii=False),
                )
                direct_support = cls._directly_supported(
                    value, runtime["messages"]
                )
                causal_support = (
                    raw_receipt in causal_receipts
                    or coerced_receipt in causal_receipts
                )
                supported = direct_support or causal_support
                if key in required:
                    if supported:
                        cleaned[key] = value
                    else:
                        reasons.append(f"required_unsupported:{key}")
                elif direct_support or (
                    causal_support
                    and isinstance(value, (bool, int, float, dict, list))
                ):
                    # Action-level causality can prove a derived required value
                    # (for example an ISO date caused by "next weekend"), but
                    # it is too coarse to justify an unmentioned optional
                    # string such as format="JSON".  Non-string intent values
                    # remain eligible because they commonly encode requested
                    # switches/counts that cannot appear verbatim.
                    cleaned[key] = value
                else:
                    optional_drops.append({"tool": name, "argument": key})

            missing = sorted(key for key in required if key not in cleaned)
            reasons.extend(
                f"required_not_executable:{key}" for key in missing
                if f"required_unsupported:{key}" not in reasons
            )

            canonical = json.dumps(
                cleaned, sort_keys=True, ensure_ascii=False
            )
            reauthorized = bool(info["support"])
            if (
                (name, original_canonical) in completed
                or (name, canonical) in completed
            ) and not reauthorized:
                reasons.append("already_completed")

            if reasons:
                suppressed.append({
                    "tool": name,
                    "reasons": reasons,
                    "origins": sorted(info["origins"]),
                    "support": sorted(info["support"]),
                })
                continue

            clean_key = (name, canonical)
            if clean_key in kept_keys:
                continue
            clean_call = deepcopy(call)
            clean_call.setdefault("function", {})["arguments"] = json.dumps(
                cleaned, ensure_ascii=False
            )
            kept.append(clean_call)
            kept_keys.add(clean_key)

        return kept, {
            "suppressed_calls": suppressed,
            "dropped_optional_arguments": optional_drops,
        }

    def decide(self, data):
        """Run causal discovery and emit the supported executable frontier."""
        started = time.monotonic()
        runtime = self.runtime_view(data)
        anchor, anchor_error = self._safe_generate(runtime)
        if anchor is None:
            raise RuntimeError("anchor generation failed: " + str(anchor_error))

        calls_used = 1
        accounting = [anchor]
        log = {
            "method": "counterfactual_obligation_graph_frontier_synthesis",
            "runtime_keys": sorted(runtime),
            "anchor_signature": list(self.action_signature(anchor)),
            "model_call_budget": self.max_calls,
        }

        user_index = self._latest_user_index(runtime["messages"])
        user_content = (
            runtime["messages"][user_index].get("content", "")
            if user_index is not None else ""
        )
        spans = self._clause_spans(user_content, self.max_spans)
        log["obligation_spans"] = [
            {"index": index, "start": start, "end": end}
            for index, (start, end) in enumerate(spans)
        ]

        labelled_views = {}
        prefix_view = self._counterfactual_prefix(runtime)
        neutral_view = self._syntax_neutral(runtime)
        if prefix_view is not None:
            labelled_views["without_latest_event"] = prefix_view
        if neutral_view is not None:
            labelled_views["without_prior_action_syntax"] = neutral_view
        if user_index is not None:
            for index, span in enumerate(spans):
                view = self._mask_user_span(runtime, user_index, span)
                if view is not None:
                    labelled_views[f"without_span_{index}"] = view

        # Deduplicate identical subtractive views (for example, a one-clause
        # latest user message and the without-latest-event view).
        requests = {}
        aliases = {}
        fingerprints = {}
        diagnostic_limit = max(0, self.max_calls - calls_used)
        for label, view in labelled_views.items():
            fingerprint = self._fingerprint(view)
            if fingerprint in fingerprints:
                aliases[label] = fingerprints[fingerprint]
                continue
            if len(requests) >= diagnostic_limit:
                break
            fingerprints[fingerprint] = label
            requests[label] = view

        diagnostics = self._parallel_generate(requests)
        calls_used += len(requests)
        for item in diagnostics.values():
            if item["response"] is not None:
                accounting.append(item["response"])
        for alias, source in aliases.items():
            if source in diagnostics:
                diagnostics[alias] = diagnostics[source]

        responses = {"anchor": anchor}
        for label, item in diagnostics.items():
            if item["response"] is not None:
                responses[label] = item["response"]
        presence = {
            label: set(self._call_map(response))
            for label, response in responses.items()
        }
        calls_by_view = {
            label: self._call_map(response)
            for label, response in responses.items()
        }

        log["counterfactuals"] = {
            label: {
                "signature": list(self.action_signature(item["response"])),
                "error": item["error"],
            }
            for label, item in diagnostics.items()
            if label not in aliases
        }

        # Candidate calls may come from the untouched anchor, syntax-neutral
        # view, or any clause mask.  The prefix is a control, never a source of
        # new actions.
        candidate_sources = [
            label for label in calls_by_view
            if label != "without_latest_event"
        ]
        candidate_calls = {}
        origins = defaultdict(set)
        for label in candidate_sources:
            for key, call in calls_by_view[label].items():
                candidate_calls.setdefault(key, call)
                origins[key].add(label)

        support = defaultdict(set)
        prefix_calls = presence.get("without_latest_event", set())
        for key in candidate_calls:
            if key in presence.get("anchor", set()) and (
                "without_latest_event" in presence
                and key not in prefix_calls
            ):
                support[key].add("latest_event")

            for index, _span in enumerate(spans):
                removed_label = f"without_span_{index}"
                if removed_label not in presence or key in presence[removed_label]:
                    continue
                retaining_labels = [
                    "anchor",
                    "without_prior_action_syntax",
                    *[
                        f"without_span_{other}"
                        for other in range(len(spans))
                        if other != index
                    ],
                ]
                if any(key in presence.get(label, set()) for label in retaining_labels):
                    support[key].add(f"span:{index}")

        anchor_keys = presence.get("anchor", set())
        current_supported = {
            key for key in candidate_calls if support.get(key)
        }

        # A text anchor can conceal a simple single-tool action even when there
        # is no second clause to expose it.  Probe one schema-selected tool in
        # paired full/prefix projections.  The call is admitted only if it is
        # caused by the latest event and at least one required value is present
        # in the latest user message.
        projected_keys = set()
        if (
            not anchor_keys
            and not current_supported
            and prefix_view is not None
            and calls_used + 2 <= self.max_calls
        ):
            projection_tool = self._projection_tool(runtime)
            if projection_tool is not None:
                projection_requests = {
                    "projection_full": self._isolated_runtime(
                        runtime, projection_tool
                    ),
                    "projection_without_latest": self._isolated_runtime(
                        prefix_view, projection_tool
                    ),
                }
                projection_results = self._parallel_generate(
                    projection_requests
                )
                calls_used += len(projection_requests)
                for item in projection_results.values():
                    if item["response"] is not None:
                        accounting.append(item["response"])
                full_map = self._call_map(
                    projection_results["projection_full"]["response"]
                )
                old_keys = set(self._call_map(
                    projection_results[
                        "projection_without_latest"
                    ]["response"]
                ))
                for key, call in full_map.items():
                    if (
                        key not in old_keys
                        and self._projection_has_required_source(
                            call, projection_tool, runtime
                        )
                    ):
                        candidate_calls.setdefault(key, call)
                        origins[key].add("projection_full")
                        support[key].add("latest_event")
                        projected_keys.add(key)
                log["projection"] = {
                    "tool": self._tool_schema(projection_tool).get("name"),
                    "full_signature": list(self.action_signature(
                        projection_results["projection_full"]["response"]
                    )),
                    "without_latest_signature": list(self.action_signature(
                        projection_results[
                            "projection_without_latest"
                        ]["response"]
                    )),
                    "admitted": len(projected_keys),
                }

        # Shadow-only calls need a second kind of evidence: the exact call must
        # survive on the untouched conversation when only its own schema is
        # exposed.  One confirmation can validate several calls to that tool.
        shadow_keys = [
            key for key in candidate_calls
            if key not in anchor_keys
            and key not in projected_keys
            and support.get(key)
        ]
        by_tool = defaultdict(list)
        for key in shadow_keys:
            by_tool[key[0]].append(key)

        reserve_text = 1 if anchor_keys and self.generate_text is not None else 0
        confirmation_budget = max(
            0, self.max_calls - calls_used - reserve_text
        )
        confirmation_requests = {}
        tool_lookup = {
            self._tool_schema(tool).get("name"): tool
            for tool in runtime["tools"]
        }
        ranked_tools = sorted(
            by_tool,
            key=lambda name: (-len(by_tool[name]), name),
        )[:confirmation_budget]
        for name in ranked_tools:
            tool = tool_lookup.get(name)
            if tool is not None:
                confirmation_requests[name] = self._isolated_runtime(
                    runtime, tool
                )

        confirmation_results = self._parallel_generate(confirmation_requests)
        calls_used += len(confirmation_requests)
        confirmed_shadow = set()
        for name, item in confirmation_results.items():
            if item["response"] is not None:
                accounting.append(item["response"])
            confirmed_shadow.update(
                set(self._call_map(item["response"])) & set(by_tool[name])
            )

        diagnostics_succeeded = any(
            item["response"] is not None for item in diagnostics.values()
        )
        eligible = []
        causal_receipts = set()
        for key, call in candidate_calls.items():
            is_anchor = key in anchor_keys
            has_support = bool(support.get(key))
            confirmed = (
                is_anchor
                or key in projected_keys
                or key in confirmed_shadow
            )

            # Fail open only when every diagnostic failed.  Otherwise an
            # anchor call that also appears without the user's latest event has
            # no current obligation and is treated as action momentum.
            if is_anchor and not has_support and not diagnostics_succeeded:
                confirmed = True
            elif not has_support:
                continue
            if not confirmed:
                continue

            if has_support:
                causal_receipts.update(self._call_receipts(call))
            eligible.append({
                "key": key,
                "call": call,
                "origins": origins[key],
                "support": support[key],
            })

        frontier, audit = self._frontier(
            eligible,
            runtime,
            causal_receipts=causal_receipts,
        )
        frontier_keys = {
            self._call_key(call) for call in frontier
            if self._call_key(call) is not None
        }
        log["candidates"] = [
            {
                "tool": key[0],
                "origins": sorted(origins[key]),
                "support": sorted(support[key]),
                "anchor": key in anchor_keys,
                "confirmed": (
                    key in anchor_keys
                    or key in projected_keys
                    or key in confirmed_shadow
                ),
            }
            for key in sorted(candidate_calls)
        ]
        log["audit"] = audit
        log["frontier"] = {
            "call_count": len(frontier),
            "tools": [
                call.get("function", {}).get("name") for call in frontier
            ],
            "covered_obligations": sorted({
                label
                for info in eligible
                if info["key"] in frontier_keys
                for label in info["support"]
            }),
        }

        result = dict(anchor)
        if frontier:
            result["tool_calls"] = frontier
            if not anchor_keys:
                log["decision"] = "recover_tool_frontier_from_text_anchor"
            elif frontier_keys == anchor_keys and not (
                audit["suppressed_calls"]
                or audit["dropped_optional_arguments"]
            ):
                log["decision"] = "preserve_tool_anchor"
            else:
                log["decision"] = "synthesize_executable_frontier"
        elif not anchor_keys:
            log["decision"] = "preserve_text_anchor"
        elif not diagnostics_succeeded:
            # A server failure never turns a usable baseline call into text.
            log["decision"] = "preserve_anchor_diagnostics_inconclusive"
        elif calls_used < self.max_calls:
            text_response, text_error = self._safe_text(runtime)
            calls_used += 1
            log["text_fallback_error"] = text_error
            if text_response is not None:
                accounting.append(text_response)
            if text_response and text_response.get("content"):
                result = dict(text_response)
                result["tool_calls"] = None
                log["decision"] = "replace_unsupported_action_with_model_text"
            else:
                result = dict(anchor)
                log["decision"] = "preserve_anchor_fallback_unavailable"
        else:
            log["decision"] = "preserve_anchor_call_budget_exhausted"

        result["input_token"] = self._tokens(accounting, "input_token")
        result["output_token"] = self._tokens(accounting, "output_token")
        result["latency"] = time.monotonic() - started
        log["model_calls_used"] = calls_used
        result["cogfs_log"] = log
        return result


__all__ = [
    "COGFSController",
    "RuntimeFirewallError",
]

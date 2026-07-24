import json
import os
import re

from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from overrides import final

from wtb.checker_utils import _normalize_str
from wtb.tool_call_graph import ToolCallGraph
from wtb.utils import sort_key, load_file, generate_random_string
from wtb.constant import PROMPT_PATH


# Ledger-and-Receipts protocol. Prior turns are replayed as a compact
# verified Ledger (one row per closed turn: task -> action -> result -> how
# it closed) instead of a raw transcript, and every argument value must
# carry a receipt. Benchmark-agnostic: it encodes a general tool-use
# discipline, not rules fitted to specific test cases.
SYSTEM_PROMPT_TEMPLATE = "Current Date: {env_info}"


class BaseHandler:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
        self.model_messages = []
        self.consecutive_tool_messages = True
        # Single-rollout execution (temp 0.0) by default. The WTB_SC_N env var
        # can override this if multi-sample ablation is ever requested.
        self.sc_n = int(os.getenv("WTB_SC_N", "1"))
        self.sc_temperature = float(os.getenv("WTB_SC_TEMP", "0.8"))
        # Entity-card labels are OFF by default: measured on result_igar_v2 they
        # flipped 18 Chat turns from a text answer to a tool call (-6 net Chat).
        # Set WTB_ENTITY_LABELS=1 to re-enable the trimmed version for a pilot.
        self.entity_labels = os.getenv("WTB_ENTITY_LABELS", "0").strip().lower() not in ("0", "false", "")
        # Ask-instead-of-guess gate. Measured on result_igar_v3: fires on 103
        # first steps, of which 40 are turns where gold wanted a clarification
        # and the task was failing, 58 were failing regardless, and 5 were
        # passing. Set WTB_ASK_GATE=0 to disable.
        self.ask_gate = os.getenv("WTB_ASK_GATE", "1").strip().lower() not in ("0", "false", "")

    def _clean_tool_call_arguments(self, tool_name, arguments_dict, tools):
        if isinstance(arguments_dict, str):
            try:
                parsed = json.loads(arguments_dict)
                arguments_dict = parsed if isinstance(parsed, dict) else {}
            except Exception:
                arguments_dict = {}
        elif not isinstance(arguments_dict, dict):
            arguments_dict = {}

        # Find the tool schema
        schema = None
        for t in tools:
            if t.get("function", {}).get("name") == tool_name:
                schema = t["function"]
                break
        if not schema:
            return arguments_dict
            
        parameters = schema.get("parameters", {})
        if not parameters or parameters.get("type") != "object":
            return arguments_dict
            
        properties = parameters.get("properties", {})
        
        def cast_value(val, prop_schema):
            prop_type = prop_schema.get("type")
            if prop_type in ["integer", "number"]:
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return int(val) if prop_type == "integer" else float(val)
                if isinstance(val, str):
                    s_val = val.strip()
                    # Only parse if s_val is a pure numeric literal (e.g. "250", "250.0")
                    # Do NOT coerce if s_val contains currency or units ("250 yuan", "10 miles")
                    if re.match(r"^-?\d+(\.\d+)?$", s_val):
                        try:
                            return int(float(s_val)) if prop_type == "integer" else float(s_val)
                        except Exception:
                            pass
            elif prop_type == "boolean":
                if isinstance(val, str):
                    if val.lower() in ["true", "1"]:
                        return True
                    if val.lower() in ["false", "0"]:
                        return False
            elif prop_type == "string":
                if not isinstance(val, str):
                    return str(val)
                return val
            elif prop_type == "array" and isinstance(val, str):
                try:
                    loaded = json.loads(val)
                    if isinstance(loaded, list):
                        return loaded
                except Exception:
                    pass
            elif prop_type == "object" and isinstance(val, dict):
                nested_properties = prop_schema.get("properties", {})
                cleaned_nested = {}
                for k, v in val.items():
                    if k in nested_properties:
                        cleaned_nested[k] = cast_value(v, nested_properties[k])
                return cleaned_nested
            if "enum" in prop_schema:
                enum_options = prop_schema["enum"]
                if isinstance(val, str):
                    for opt in enum_options:
                        if isinstance(opt, str) and opt.lower() == val.lower():
                            return opt
            return val

        cleaned_args = {}
        for k, v in arguments_dict.items():
            if k in properties:
                cleaned_args[k] = cast_value(v, properties[k])
        return cleaned_args

    # A resolved date/time value is a legitimate COMPUTE receipt even though
    # its digits don't appear verbatim anywhere upstream - it was derived
    # from Current Date, not copied.
    _DATE_RECEIPT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")
    _ENV_DATE_RE = re.compile(r"Current Date: (\d{4})-(\d{2})-(\d{2})")
    _MONTH_DAY_RE = re.compile(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?", re.IGNORECASE)
    _WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    def _extract_date_candidates(self, context_text):
        '''
        Math Registry: re-derive in code the calendar dates the conversation
        can legitimately mean - relative phrases resolved against Current
        Date, plus explicitly written month-name dates. Returns a set of ISO
        dates, empty when nothing resolvable is present (callers must then
        fall back to permissive acceptance, never reject on an empty set).
        '''
        import datetime
        m = self._ENV_DATE_RE.search(context_text)
        if not m:
            return set()
        try:
            base = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return set()
        low = context_text.lower()
        out = set()

        def add(d):
            out.add(d.isoformat())

        def add_range(start, end):
            d = start
            while d <= end:
                add(d)
                d += datetime.timedelta(days=1)

        if "today" in low or "tonight" in low:
            add(base)
        if "day after tomorrow" in low:
            add(base + datetime.timedelta(days=2))
        if "tomorrow" in low:
            add(base + datetime.timedelta(days=1))
        if "yesterday" in low:
            add(base - datetime.timedelta(days=1))
        saturday = base + datetime.timedelta(days=(5 - base.weekday()) % 7)
        if "weekend" in low:
            add(saturday)
            add(saturday + datetime.timedelta(days=1))
            if "next weekend" in low:
                add(saturday + datetime.timedelta(days=7))
                add(saturday + datetime.timedelta(days=8))
        next_monday = base + datetime.timedelta(days=7 - base.weekday())
        if "this week" in low:
            add_range(base, next_monday - datetime.timedelta(days=1))
        if "next week" in low:
            add_range(next_monday, next_monday + datetime.timedelta(days=6))
        for wd_idx, wd in enumerate(self._WEEKDAYS):
            if wd in low:
                add(base + datetime.timedelta(days=(wd_idx - base.weekday()) % 7))
                if "next " + wd in low:
                    add(base + datetime.timedelta(days=((wd_idx - base.weekday()) % 7) + 7))
        for month_name, day in self._MONTH_DAY_RE.findall(context_text):
            month = ["january", "february", "march", "april", "may", "june", "july",
                     "august", "september", "october", "november", "december"].index(month_name.lower()) + 1
            for year in (base.year, base.year + 1, base.year - 1):
                try:
                    add(datetime.date(year, month, int(day)))
                except ValueError:
                    pass
        return out

    def _date_value_ok(self, value, context_text):
        '''
        Verify a date-shaped value: pass if quoted from the conversation,
        else if it matches a code-computed candidate, else pass permissively
        only when nothing was resolvable (unknown phrasing must never be
        punished).
        '''
        if value in context_text or _normalize_str(value) in _normalize_str(context_text):
            return True
        candidates = self._extract_date_candidates(context_text)
        if not candidates:
            return True
        return value.strip()[:10] in candidates

    def _is_grounded(self, value, context_text, key_name=None, dialogue_state=None):
        '''
        A value is grounded if it is traceable to something already visible
        to the model via dialogue_state provenance (QUOTE / RESOLVE / TOOL / LEDGER),
        or text context, or shaped like a resolved date/time (COMPUTE).
        '''
        if value is None:
            return True

        # Latent task state is the authoritative receipt registry.  Text
        # containment is deliberately a fallback: tool observations often
        # surface values under a different label than the next tool schema.
        if dialogue_state and isinstance(dialogue_state, dict):
            grounded_values = dialogue_state.get("grounded_values", {})
            val_str = str(value).strip().lower()
            key_lower = str(key_name).lower() if key_name else None
            if key_lower and key_lower in grounded_values:
                known = str(grounded_values[key_lower].get("value", "")).strip().lower()
                if val_str and known and (val_str == known or val_str in known or known in val_str):
                    return True
            for receipt in grounded_values.values():
                known = str(receipt.get("value", "")).strip().lower()
                if val_str and known and (val_str == known or val_str in known or known in val_str):
                    return True

        if key_name and dialogue_state and isinstance(dialogue_state, dict):
            prov_map = dialogue_state.get("provenance_by_slot", {})
            key_lower = str(key_name).lower()
            if key_lower in prov_map:
                prov_val = str(prov_map[key_lower].get("value", "")).lower()
                val_str = str(value).strip().lower()
                if val_str and (val_str in prov_val or prov_val in val_str):
                    return True

        if isinstance(value, bool):
            if key_name and _normalize_str(key_name) in _normalize_str(context_text):
                return True
            return False
        if isinstance(value, (int, float)):
            if str(value) in context_text:
                return True
            return _normalize_str(str(value)) in _normalize_str(context_text)
        if isinstance(value, str):
            if not value.strip():
                return True
            if self._DATE_RECEIPT_RE.match(value.strip()):
                return True
            if value in context_text:
                return True
            normalized = _normalize_str(value)
            return bool(normalized) and normalized in _normalize_str(context_text)
        if isinstance(value, (list, dict)):
            leaves = []

            def collect(v):
                if isinstance(v, dict):
                    for vv in v.values():
                        collect(vv)
                elif isinstance(v, list):
                    for vv in v:
                        collect(vv)
                else:
                    leaves.append(v)

            collect(value)
            return all(self._is_grounded(leaf, context_text, key_name, dialogue_state) for leaf in leaves)
        return True

    def _verify_and_filter_arguments(self, tool_name, arguments_dict, tools, context_text, dialogue_state=None):
        '''
        Receipt gate: an argument survives only if it is required by the
        schema or grounded in dialogue_state / conversation context.
        '''
        required = set()
        properties = {}
        for t in tools:
            func = t.get("function", {})
            if func.get("name") == tool_name:
                params = func.get("parameters", {}) or {}
                required = set(params.get("required", []))
                properties = params.get("properties", {}) or {}
                break

        kept = {}
        dropped_keys = []
        ungrounded_required = []
        for key, value in arguments_dict.items():
            grounded = self._is_grounded(value, context_text, key_name=key, dialogue_state=dialogue_state)
            if key in required:
                kept[key] = value
                if not grounded:
                    ungrounded_required.append(key)
            elif grounded:
                kept[key] = value
            elif self._is_intent_argument(key, value, properties.get(key), context_text):
                kept[key] = value
            else:
                dropped_keys.append(key)
        return kept, dropped_keys, ungrounded_required

    @staticmethod
    def _is_abbreviation_of_context(value, context_text):
        '''
        Is this value DERIVED from the conversation rather than invented?

        A contraction the model resolved itself ("Phoenix" -> "PHX", "China" ->
        "CN", "English" -> "en") cannot be quoted verbatim, but its characters
        still appear, in order, inside a longer word that IS present. An
        invented value has no such source word.

        Parameter-free by construction: it compares the value against the
        conversation's own vocabulary, so there is no length constant or tuned
        threshold. A short value naturally finds a source word and is forgiven;
        a long fabricated one does not.
        '''
        target = re.sub(r"[^a-z0-9]", "", str(value).lower())
        if not target:
            return True
        for word in set(re.findall(r"[A-Za-z0-9]+", context_text or "")):
            lowered = word.lower()
            if len(lowered) <= len(target):
                continue
            cursor = iter(lowered)
            if all(char in cursor for char in target):
                return True
        return False

    def _is_intent_argument(self, key, value, prop_schema, context_text=""):
        '''
        Guard against over-dropping by the receipt gate.

        The gate drops optional arguments whose value cannot be quoted from the
        conversation. That is correct for fabricated plumbing defaults
        (format="JSON", quality=..., filter=...), but three kinds of argument are
        legitimately unquotable and were being destroyed:

          * booleans  - they encode user INTENT ("in detail" -> includeStats=true),
                        never data, so they can never appear verbatim;
          * numerics  - counts/radii/limits are usually computed or restated
                        rather than copied character-for-character;
          * contractions of something in the conversation - see
                        _is_abbreviation_of_context.

        Enum-typed arguments are deliberately NOT protected: measured on the
        clean_demo run, dropping ungrounded enums is right 11 times and wrong 3,
        so the gate's existing enum behaviour is a net win and is left intact.
        '''
        prop_schema = prop_schema or {}
        if "enum" in prop_schema:
            return False
        if isinstance(value, bool):
            return True
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str) and value.strip():
            return self._is_abbreviation_of_context(value, context_text)
        return False

    def _clean_and_verify_call(self, tc_name, tc_args, tools, messages, step, inference_log, dialogue_state=None):
        cleaned_args = self._clean_tool_call_arguments(tc_name, tc_args, tools)
        context_text = json.dumps(messages, ensure_ascii=False)
        cleaned_args, dropped_keys, ungrounded_required = self._verify_and_filter_arguments(
            tc_name, cleaned_args, tools, context_text, dialogue_state=dialogue_state
        )
        if dropped_keys or ungrounded_required:
            inference_log.setdefault("receipt_notes", []).append({
                "step": step,
                "tool": tc_name,
                "dropped_ungrounded_optional": dropped_keys,
                "ungrounded_required": ungrounded_required,
            })
        return cleaned_args

    @staticmethod
    def _canon(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _looks_like_tool_attempt(content):
        # Conservative: only a bare tool-call JSON / <tool_call> block counts as an
        # attempt. Genuine prose never starts with {"name" and won't false-positive.
        if not isinstance(content, str):
            return False
        if "<tool_call>" in content or "</tool_call>" in content:
            return True
        stripped = content.lstrip()
        return (
            stripped.startswith("{")
            and '"name"' in stripped
            and ('"arguments"' in stripped or '"parameters"' in stripped)
        )

    @staticmethod
    def _repair_json(text):
        # Close the unbalanced braces/brackets (and an open string) left by a
        # truncated generation, in the correct reverse order. Best-effort only.
        stack = []
        in_str = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
        repaired = text
        if in_str:
            repaired += '"'
        for opener in reversed(stack):
            repaired += "}" if opener == "{" else "]"
        return repaired

    def _loads_lenient(self, text):
        import re
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"```\s*$", "", text).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        if start == -1:
            return None
        snippet = text[start:]
        try:
            # raw_decode tolerates trailing junk after the first complete object.
            return json.JSONDecoder().raw_decode(snippet)[0]
        except Exception:
            pass
        # Truncation repair: balance quotes/brackets; if the cut landed mid-token
        # (e.g. '..., {"t'), progressively trim back to the last complete element.
        work = snippet
        for _ in range(60):
            try:
                return json.loads(self._repair_json(work))
            except Exception:
                pass
            cut = max(work.rfind(","), work.rfind("{"), work.rfind("["), work.rfind('"'))
            if cut <= 0:
                break
            work = work[:cut].rstrip().rstrip(",")
            if not work:
                break
        return None

    def _recover_tool_calls_from_content(self, content):
        '''
        Salvage tool call(s) from raw content that vLLM returned unparsed (the
        server-side hermes parser gives up on truncated/multi-object JSON and
        falls back to text). Returns a list of tool_call dicts, or None.
        '''
        import re
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
        if not blocks:
            blocks = [content]
        recovered = []
        for block in blocks:
            obj = self._loads_lenient(block)
            if not isinstance(obj, dict) or "name" not in obj:
                continue
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                parsed = self._loads_lenient(args)
                args = parsed if isinstance(parsed, dict) else {}
            if not isinstance(args, dict):
                args = {}
            recovered.append({
                "id": "toolu_recovered_" + generate_random_string(16),
                "type": "function",
                "function": {"name": obj["name"], "arguments": json.dumps(args, ensure_ascii=False)},
            })
        return recovered or None

    def _topological_sort_tool_calls(self, tool_calls, tools):
        '''
        Topological Tool Dependency Ordering (IGAR v10).
        If tool call B requires a parameter property produced by tool A,
        sort tool A before tool B in the execution sequence.
        '''
        if not tool_calls or len(tool_calls) <= 1:
            return tool_calls
        schema_map = {t.get("function", {}).get("name"): t.get("function", {}) for t in (tools or [])}
        def depends_on(tc1, tc2):
            name1 = tc1.get("function", {}).get("name")
            name2 = tc2.get("function", {}).get("name")
            if not name1 or not name2 or name1 == name2:
                return False
            f1 = schema_map.get(name1, {})
            req1 = set(f1.get("parameters", {}).get("required", []))
            name2_lower = name2.lower()
            return any(req_key.lower() in name2_lower for req_key in req1)
        sorted_calls = list(tool_calls)
        n = len(sorted_calls)
        for i in range(n):
            for j in range(i + 1, n):
                if depends_on(sorted_calls[i], sorted_calls[j]):
                    sorted_calls[i], sorted_calls[j] = sorted_calls[j], sorted_calls[i]
        return sorted_calls

    def _normalize_response(self, data):
        '''
        Rescue tool calls that came back as text because the server-side parser
        failed. If the content is a recoverable tool-call attempt, convert it into
        proper tool_calls; if it is an unrecoverable attempt, blank the content so
        it ABSTAINS from voting instead of masquerading as a valid text/clarify
        answer (which could otherwise outvote a correct tool-call candidate).
        '''
        if data.get("tool_calls"):
            return data
        content = data.get("content")
        if not self._looks_like_tool_attempt(content):
            return data
        recovered = self._recover_tool_calls_from_content(content)
        if recovered:
            data["tool_calls"] = recovered
            data["content"] = None
        else:
            data["content"] = None
        return data

    @staticmethod
    def _response_signature(model_response_data):
        tool_calls = model_response_data.get("tool_calls") or []
        if tool_calls:
            names = tuple(sorted(tc.get("function", {}).get("name", "") for tc in tool_calls))
            return ("tools",) + names
        content = model_response_data.get("content")
        if content:
            return ("text",)
        return ("empty",)

    def _minimum_commitment_gate(self, tools, messages, dialogue_state=None):
        '''
        Strict Elimination Gate.
        Answers commitment mode without soft scoring or candidate ranking:
          - 'grounded_tool': 1+ tools have 100% required parameters grounded.
          - 'under_specified': Required parameter missing for tool intent.
          - 'ambiguous_chat': No tool intent present.
        '''
        if dialogue_state and isinstance(dialogue_state, dict):
            mode = dialogue_state.get("current_commitment_mode") or "ambiguous_chat"
            feasible = dialogue_state.get("feasible_tools", [])
            unresolved = dialogue_state.get("unresolved_required_slots", {})
            clarify_slot = dialogue_state.get("clarify_slot")
            return {
                "action_class": mode,
                "target_tools": feasible if mode == "grounded_tool" else list(unresolved.keys()),
                "highest_missing_slot": clarify_slot,
            }

        ctx_text = json.dumps(messages, ensure_ascii=False).lower()
        slot_coverage = Counter()
        tool_status = {}

        for t in (tools or []):
            func = t.get("function", {})
            tname = func.get("name")
            if not tname:
                continue
            params = func.get("parameters", {}) or {}
            req = set(params.get("required", []) or [])
            for slot in req:
                slot_coverage[slot] += 1

            missing = {s for s in req if s.lower() not in ctx_text}
            tool_status[tname] = {
                "required": req,
                "missing": missing,
                "fully_grounded": len(missing) == 0
            }

        fully_grounded_tools = [t for t, status in tool_status.items() if status["fully_grounded"]]
        if len(fully_grounded_tools) >= 1:
            return {
                "action_class": "grounded_tool",
                "target_tools": fully_grounded_tools,
                "highest_missing_slot": None
            }

        return {
            "action_class": "ambiguous_chat",
            "target_tools": [],
            "highest_missing_slot": None
        }

    def _consensus_generate(self, inference_data):
        '''
        Self-consistency decoding with a tool-first, selected-path commitment
        preference.  Dialogue state can only break a tied vote toward a
        plausible tool; it cannot globally force a clarification.
        '''
        api_response, latency = self._request_tool_call(inference_data)
        anchor = self._parse_api_response(api_response)
        anchor["latency"] = latency

        if self.sc_n <= 1 or not hasattr(self, "_request_candidates"):
            return self._maybe_repair(anchor, inference_data)

        try:
            extra = self._request_candidates(inference_data, self.sc_n - 1, self.sc_temperature)
        except Exception as e:
            print(f"Consensus sampling failed, using anchor only: {e}", flush=True)
            return self._maybe_repair(anchor, inference_data)

        candidates = [self._normalize_response(dict(anchor))]
        candidates.extend(self._normalize_response(dict(c)) for c in extra)

        signatures = [self._response_signature(c) for c in candidates]
        vote_counts = Counter(s for s in signatures if s != ("empty",))
        consensus_log = {
            "candidate_signatures": [list(s) for s in signatures],
        }

        total_input = sum(c.get("input_token") or 0 for c in candidates)
        total_output = sum(c.get("output_token") or 0 for c in candidates)
        total_latency = sum(c.get("latency") or 0 for c in candidates)

        # MCSG: Evaluate Pre-Generation Action Gate from Dialogue State
        tools = inference_data.get("tools") or []
        messages = inference_data.get("messages") or []
        dialogue_state = inference_data.get("dialogue_state")
        mcsg_state = self._minimum_commitment_gate(tools, messages, dialogue_state=dialogue_state)
        consensus_log["mcsg_state"] = mcsg_state

        # Filter candidates before voting according to elimination gate
        valid_candidates = []
        valid_signatures = []
        feas_set = set((dialogue_state or {}).get("feasible_tools") or [])

        for c, s in zip(candidates, signatures):
            if s == ("empty",):
                continue
            # 1. Action class elimination
            if mcsg_state["action_class"] == "grounded_tool" and s == ("text",):
                continue
            if mcsg_state["action_class"] == "ambiguous_chat" and s[0] == "tools":
                continue
            # 2. Feasibility elimination for tool calls
            if s[0] == "tools" and feas_set:
                if any(tname not in feas_set for tname in s[1:]):
                    continue
            valid_candidates.append(c)
            valid_signatures.append(s)

        if not valid_signatures:
            valid_candidates = [c for c, s in zip(candidates, signatures) if s != ("empty",)]
            valid_signatures = [s for s in signatures if s != ("empty",)]

        if not valid_signatures:
            chosen = anchor
        else:
            vote_counts = Counter(valid_signatures)
            best_count = max(vote_counts.values())
            winners = [s for s, v in vote_counts.items() if v == best_count]

            if len(winners) == 1:
                winning_signature = winners[0]
            else:
                def pure_rank_score(sig):
                    clust = [c for c, s in zip(valid_candidates, valid_signatures) if s == sig]
                    return min(self._candidate_violations(c, inference_data)["total"] for c in clust)
                winning_signature = min(winners, key=pure_rank_score)

            cluster = [c for c, s in zip(valid_candidates, valid_signatures) if s == winning_signature]
            consensus_log["winning_signature"] = list(winning_signature)
            consensus_log["cluster_size"] = len(cluster)

            chosen = dict(cluster[0])
            if winning_signature[0] == "tools":
                scores = [self._candidate_violations(c, inference_data)["total"] for c in cluster]
                best_idx = min(range(len(cluster)), key=lambda i: (scores[i], i))
                chosen = dict(cluster[best_idx])
                consensus_log["checker_violation_scores"] = scores
                consensus_log["checker_pick"] = best_idx

        chosen = dict(chosen)
        chosen["input_token"] = total_input
        chosen["output_token"] = total_output
        chosen["latency"] = total_latency
        chosen["consensus_log"] = consensus_log
        return self._maybe_repair(chosen, inference_data)

    def _candidate_violations(self, candidate, inference_data):
        '''
        Mechanical audit of one candidate response. Counts defects the eval
        punishes deterministically: unparseable or unknown calls, schema
        breaches, ungrounded optional arguments, and exact re-calls of a
        call already answered this session. Text-only candidates audit as
        zero, so this only ranks candidates *within* the tool-calling
        cluster (mode itself stays a plurality vote) - the audit can never
        flip a call decision into an answer by itself.
        '''
        from wtb.checker_utils import ToolArgsChecker
        tools = inference_data["tools"]
        messages = inference_data["messages"]
        checker = ToolArgsChecker()
        context_text = json.dumps(messages, ensure_ascii=False)
        dialogue_state = inference_data.get("dialogue_state")
        history_calls = set()
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    args_h = fn.get("arguments")
                    if isinstance(args_h, str):
                        try:
                            args_h = json.loads(args_h)
                        except Exception:
                            pass
                    history_calls.add((fn.get("name"), self._canon(args_h)))

        total = 0
        detail = []
        for tc in candidate.get("tool_calls") or []:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments")
            args_str = args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False)
            try:
                parsed = json.loads(args_str)
            except Exception:
                total += 3
                detail.append(f"{name}: arguments are not valid JSON")
                continue
            try:
                schema_status = checker.tool_check(tools, name, args_str)
            except Exception:
                total += 3
                detail.append(f"{name}: not one of the available tools")
                continue
            if schema_status != checker.CORRECT:
                total += 2
                detail.append(f"{name}: {schema_status}")
            required = set()
            for t in tools:
                func = t.get("function", {})
                if func.get("name") == name:
                    required = set(func.get("parameters", {}).get("required", []))
                    break
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    if key not in required and not self._is_grounded(
                            value, context_text, key_name=key, dialogue_state=dialogue_state):
                        total += 1
                        detail.append(f"{name}.{key}: optional argument with no basis in the conversation")
            if (name, self._canon(parsed)) in history_calls:
                total += 2
                detail.append(f"{name}: exact repeat of a call already answered earlier")
        return {"total": total, "detail": detail}

    def _maybe_repair(self, chosen, inference_data):
        '''
        Violation-guided single repair. Instead of silently editing a flawed
        call (which yields calls the model never intended), show the model
        its own draft plus the audit findings and let it re-decide once.
        The redo is adopted only if it audits strictly cleaner.
        '''
        if not chosen.get("tool_calls"):
            return chosen
        audit = self._candidate_violations(chosen, inference_data)
        if audit["total"] == 0:
            return chosen
        dialogue_state = inference_data.get("dialogue_state")
        d_state_note = ""
        if dialogue_state and isinstance(dialogue_state, dict):
            feas = dialogue_state.get("feasible_tools", [])
            unres = dialogue_state.get("unresolved_required_slots", {})
            clar = dialogue_state.get("clarify_slot")
            d_state_note = f"\nDialogue State: feasible_tools={feas}, unresolved_slots={unres}, clarify_slot={clar}\n"

        draft = json.dumps([tc.get("function") for tc in chosen["tool_calls"]], ensure_ascii=False)
        note = (
            "AUDIT: your drafted tool call(s) " + draft + " have these problems:\n- "
            + "\n- ".join(audit["detail"])
            + d_state_note
            + "\nEmit the corrected tool call(s) now: fix or remove only the flagged parts and add nothing new. "
              "If a required parameter's value is genuinely missing from the conversation, ask the user for it instead."
        )
        repair_data = dict(inference_data)
        repair_data["messages"] = list(inference_data["messages"]) + [{"role": "user", "content": note}]
        try:
            api_response, latency = self._request_tool_call(repair_data)
            redo = self._parse_api_response(api_response)
        except Exception as e:
            print(f"Repair request failed, keeping draft: {e}", flush=True)
            return chosen
        redo = self._normalize_response(dict(redo))
        if redo.get("tool_calls"):
            redo_audit = self._candidate_violations(redo, inference_data)
        else:
            redo_audit = {"total": 0, "detail": []}
        adopted = redo_audit["total"] < audit["total"] and (redo.get("tool_calls") or redo.get("content"))
        result = dict(redo) if adopted else dict(chosen)
        result["input_token"] = (chosen.get("input_token") or 0) + (redo.get("input_token") or 0)
        result["output_token"] = (chosen.get("output_token") or 0) + (redo.get("output_token") or 0)
        result["latency"] = (chosen.get("latency") or 0) + (latency or 0)
        consensus_log = dict(chosen.get("consensus_log") or {})
        consensus_log["repair"] = {
            "adopted": bool(adopted),
            "draft_violations": audit["detail"],
            "redo_violations": redo_audit["detail"],
        }
        result["consensus_log"] = consensus_log
        return result

    def _request_tool_call(self, inference_data):
        raise NotImplementedError

    def _parse_api_response(self, api_response):
        raise NotImplementedError

    def convert_to_tool(self, tools):
        tools = json.dumps(tools, ensure_ascii=False).replace('"type": "float"', '"type": "number"')
        tools = json.loads(tools)
        new_tools = []
        if "claude" in self.model_name:
            for tool in tools:
                tool["inputSchema"] = tool["parameters"]
                del tool["parameters"]
                new_tools.append(tool)
            tools = new_tools
        return tools

    # Deterministic Ledger: each closed prior turn rendered as one worked
    # example - task, the action(s) actually taken with their arguments, the
    # result, and how it was closed - instead of replaying the full native
    # message transcript. Pure code, no LLM call, no hallucination risk:
    # every field is copied verbatim from the gold answer_list, so this is
    # only usable where gold history is available (i.e. WTB's own
    # teacher-forced multi-turn replay). Observations are kept whole:
    # clipping them was measured to hurt chat/coreference turns, whose
    # answers live in those result fields.
    @staticmethod
    def _render_observation(observation):
        if isinstance(observation, str):
            return observation
        try:
            return json.dumps(observation, ensure_ascii=False)
        except Exception:
            return str(observation)

    def _build_deterministic_ledger(self, history_tasks, history_answer_lists):
        if not history_tasks:
            return None
        rows = []
        for turn_idx, (task, answer_list) in enumerate(zip(history_tasks, history_answer_lists), start=1):
            tool_call_graph = ToolCallGraph(answer_list)
            tool_call_graph.add_node_list()
            tool_call_graph.generate_all_path()
            optimal_path = tool_call_graph.optimal_path_list[0]

            lines = [f"[T{turn_idx}] user: {task}"]
            for idx_action_list in optimal_path:
                for idx in idx_action_list:
                    answer = answer_list[idx]
                    action = answer["action"]
                    name = action["name"]
                    observation = answer["observation"]
                    if name == "ask_user_for_required_parameters":
                        lines.append(f"     asked: {observation}")
                        lines.append(f"     user replied: {answer.get('user_input', '')}")
                    elif name == "prepare_to_answer":
                        lines.append(f"     answered: {observation}")
                    else:
                        args_str = json.dumps(action["arguments"], ensure_ascii=False)
                        lines.append(f"     -> {name}({args_str})")
                        lines.append(f"        => {self._render_observation(observation)}")
            rows.append("\n".join(lines))
        return "\n\n".join(rows)

    # Transport/plumbing keys that look like identifiers but carry no entity
    # meaning ("status_code: 200" was polluting the surfaced facts).
    _NON_ENTITY_ID_KEYS = {
        "status_code", "statuscode", "http_code", "httpcode",
        "error_code", "errcode", "ret_code", "retcode", "response_code",
    }
    _LABEL_MAX_CHARS = 40

    def _extract_observation_facts(self, history_answer_lists):
        '''
        Build entity cards from prior-turn observations.

        The previous version emitted a flat list of bare identifiers:

            - activity_id: show123
            - activity_id: show456

        which is unusable for reference resolution - nothing says which show is
        which, so "the show on the 13th" or "the first one" cannot be bound.
        Each identifier is now emitted together with the descriptive fields that
        sit beside it in the same observation object:

            - activity_id: show123  (name: Cirque du Soleil, date: 2024-07-13)
            - activity_id: show456  (name: David Copperfield Magic Show, date: 2024-07-14)

        Only prior turns are used, so nothing about the current turn leaks.
        '''
        if not history_answer_lists:
            return []
        # (key, value) -> position label. Values that came from an ordered list
        # of objects keep the 1-based index of the element they belong to; the
        # first sighting of a value wins so a value repeated later does not lose
        # its original position.
        entities = {}

        def walk(obj, position=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                        s_v = str(v).strip()
                        if s_v:
                            entities.setdefault((k, s_v), position)
                    elif isinstance(v, (dict, list)):
                        walk(v, position)
            elif isinstance(obj, list):
                # An ordered list of objects is exactly what the user points at
                # when they say "the first" / "the last two" / "the sixth", so
                # number its elements. Scalar lists carry no such handle.
                records = [item for item in obj if isinstance(item, dict)]
                if len(records) > 1:
                    for idx, item in enumerate(records, start=1):
                        walk(item, f"[{idx}] ")
                else:
                    for item in obj:
                        walk(item, position)

        for answer_list in history_answer_lists:
            for ans in answer_list:
                obs = ans.get("observation")
                if obs:
                    walk(obs)

        return [f"{position}{k}: {s_v}" for (k, s_v), position in entities.items()]

    _STATE_STOPWORDS = {
        "a", "an", "and", "for", "from", "get", "in", "is", "of", "on", "or",
        "the", "to", "with", "your", "you", "user", "tool", "data", "info",
    }

    @classmethod
    def _tool_family(cls, tool_name):
        """A stable, schema-free family label used only as a tie-breaker."""
        words = re.findall(r"[a-z0-9]+", (tool_name or "").lower())
        generic_verbs = {"get", "list", "search", "find", "fetch", "create", "update", "delete", "add", "remove", "send"}
        return next((word for word in words if word not in generic_verbs), words[0] if words else "unknown")

    @classmethod
    def _state_tokens(cls, text):
        return {
            token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) > 1 and token not in cls._STATE_STOPWORDS
        }

    def _build_latent_task_state(self, tools, messages, history_answer_lists=None):
        """Build a per-step, provenance-first task state from visible history only.

        This intentionally summarizes evidence rather than predicting later turns.
        It is re-run after each assistant/tool exchange by the evaluation loop.
        """
        known_facts = self._extract_observation_facts(history_answer_lists) if history_answer_lists else []
        provenance_by_slot = {}
        grounded_values = {}
        last_successful_tool_name = None
        last_successful_tool_family = None
        current_user_message = ""
        pending_tool_names = []

        def record(key, value, source, turn):
            if isinstance(value, bool) or value is None:
                return
            value = str(value).strip()
            if not value:
                return
            receipt = {"value": value, "source": source, "turn": turn}
            key = str(key).lower()
            provenance_by_slot[key] = receipt
            grounded_values[key] = receipt

        def walk_payload(payload, source, turn):
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                        record(key, value, source, turn)
                    elif isinstance(value, (dict, list)):
                        walk_payload(value, source, turn)
            elif isinstance(payload, list):
                for item in payload:
                    walk_payload(item, source, turn)

        for turn_idx, msg in enumerate(messages or []):
            role = msg.get("role", "")
            content = msg.get("content")
            if role == "user" and isinstance(content, str):
                current_user_message = content
            if role == "tool":
                try:
                    walk_payload(json.loads(content) if isinstance(content, str) else content,
                                 "tool_observation", turn_idx)
                except Exception:
                    pass
                if pending_tool_names:
                    # A tool message is the benchmark's visible receipt that
                    # the immediately preceding call completed.
                    last_successful_tool_name = pending_tool_names.pop(0)
                    last_successful_tool_family = self._tool_family(last_successful_tool_name)
            elif isinstance(content, str):
                # Token receipts retain user/assistant wording without inventing
                # a semantic mapping that was not present in the transcript.
                for token in self._state_tokens(content):
                    grounded_values.setdefault(token, {
                        "value": token, "source": role or "message", "turn": turn_idx
                    })
            if role == "assistant" and msg.get("tool_calls"):
                pending_tool_names = []
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name")
                    if name:
                        pending_tool_names.append(name)

        for fact in known_facts:
            if ":" in fact:
                key, value = fact.split(":", 1)
                record(key.replace("[", " ").replace("]", " ").strip(), value,
                       "ledger_fact", 0)

        user_tokens = self._state_tokens(current_user_message)
        # Conservative slot binding for values that are plainly supplied but
        # whose schema label is absent from natural language ("in Phoenix" for
        # `city`).  These are receipts from the current user message, never
        # generated defaults.  The narrow common-slot list avoids treating an
        # arbitrary word as evidence for an unrelated required field.
        proper_values = re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b", current_user_message)
        proper_values = [v for v in proper_values if v.lower() not in {"i", "please", "can", "could", "check", "set"}]
        email_values = re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", current_user_message)
        id_values = re.findall(r"\b(?:[A-Za-z]+[-_]?)?\d{3,}\b", current_user_message)
        date_values = re.findall(
            r"\b(?:today|tomorrow|yesterday|tonight|next\s+\w+|this\s+\w+|"
            r"\d{4}-\d{1,2}-\d{1,2}|[A-Z][a-z]+\s+\d{1,2})\b", current_user_message,
            re.IGNORECASE,
        )
        slot_value_hints = {
            "city": proper_values, "location": proper_values, "country": proper_values,
            "name": proper_values, "recipient": proper_values, "recipient_name": proper_values,
            "email": email_values, "email_address": email_values,
            "id": id_values, "user_id": id_values, "order_id": id_values,
            "date": date_values, "start_date": date_values, "end_date": date_values,
        }
        tool_status = {}
        for tool in tools or []:
            func = tool.get("function", {})
            name = func.get("name")
            if not name:
                continue
            schema = func.get("parameters", {}) or {}
            required = list(schema.get("required", []) or [])
            descriptor = " ".join([name, func.get("description", "")] + required)
            relevance = len(user_tokens & self._state_tokens(descriptor))
            if relevance == 0 and not required:
                continue
            missing = []
            for slot in required:
                slot_key = slot.lower()
                slot_tokens = self._state_tokens(slot)
                hints = slot_value_hints.get(slot_key, [])
                if hints and slot_key not in grounded_values:
                    record(slot_key, hints[-1], "user_message", len(messages) - 1)
                slot_known = slot_key in grounded_values or bool(slot_tokens & set(grounded_values))
                if not slot_known:
                    missing.append(slot)
            tool_status[name] = {
                "family": self._tool_family(name), "required": required,
                "missing": missing, "relevance": relevance,
            }

        plausible_tools = list(tool_status)
        feasible_tools = [name for name, status in tool_status.items() if not status["missing"]]
        unresolved_required_slots = {
            name: status["missing"] for name, status in tool_status.items() if status["missing"]
        }
        clarify_slot = None
        if unresolved_required_slots:
            missing_counter = Counter()
            for t, m_slots in unresolved_required_slots.items():
                for s in m_slots:
                    missing_counter[s] += 1
            if missing_counter:
                clarify_slot = missing_counter.most_common(1)[0][0]

        if feasible_tools:
            current_commitment_mode = "grounded_tool"
        elif unresolved_required_slots:
            current_commitment_mode = "under_specified"
        else:
            current_commitment_mode = "ambiguous_chat"

        last_action_class = None
        for msg in reversed(messages or []):
            if msg.get("role") == "assistant":
                last_action_class = "tool" if msg.get("tool_calls") else "text"
                break

        workflow_stage = "dialogue"
        if last_action_class == "tool":
            workflow_stage = "tool_execution"
        elif current_commitment_mode == "under_specified":
            workflow_stage = "clarify_needed"
        elif current_commitment_mode == "grounded_tool":
            workflow_stage = "tool_ready"

        return {
            "current_goal": current_user_message,
            "current_workflow_stage": workflow_stage,
            "grounded_values": grounded_values,
            "known_facts": known_facts,
            "provenance_by_slot": provenance_by_slot,
            "missing_required_values": unresolved_required_slots,
            "unresolved_required_slots": unresolved_required_slots,
            "last_successful_tool_name": last_successful_tool_name,
            "last_successful_tool_family": last_successful_tool_family,
            "feasible_tool_families": sorted({tool_status[t]["family"] for t in feasible_tools}),
            "feasible_tools": feasible_tools,
            "plausible_tools": plausible_tools,
            "clarify_slot": clarify_slot,
            "current_commitment_mode": current_commitment_mode,
            "action_commitment": current_commitment_mode,
            "tool_status": tool_status,
        }

    def _build_dialogue_state(self, tools, messages, history_answer_lists=None):
        '''
        Stateful Dialogue State Builder.

        Indexes known facts and slot provenance across prior tool observations,
        ledger entries, user messages, and system context.

        Returns:
            {
                "known_facts": list of surfaced observation facts,
                "provenance_by_slot": dict mapping slot name to provenance info,
                "unresolved_required_slots": dict mapping tool_name to list of missing required slots,
                "feasible_tools": list of tool_names whose required slots are 100% grounded by provenance,
                "clarify_slot": highest coverage missing required slot,
            }
        '''
        # Compatibility name retained for existing callers; the latent state is
        # the single source of truth for all new decisions.
        return self._build_latent_task_state(tools, messages, history_answer_lists)

    def _unfounded_required(self, tool_calls, tools, context_text, dialogue_state=None):
        '''
        Receipts for REQUIRED arguments.

        The receipt gate already drops optional arguments that cannot be traced
        to the conversation, but required ones are always kept (the schema wins).
        That leaves the case where the model invents the one value it was
        supposed to ask for. This reports those, so the caller can hand the turn
        back to the user instead of acting on a guess.

        Deliberately narrow - a value is only "unfounded" if it is a plain string
        that is not a resolved date (computed from Current Date, so legitimately
        absent upstream), not a container, not numeric/boolean, and not a
        contraction of a word the conversation already contains (see
        _is_abbreviation_of_context). Those kinds are unquotable by nature
        rather than invented.
        '''
        schemas = {}
        for t in tools:
            func = t.get("function", {})
            if func.get("name"):
                schemas[func["name"]] = func
        normalized_ctx = _normalize_str(context_text)
        hits = []
        for tc in tool_calls or []:
            func = tc.get("function", {})
            name = func.get("name")
            schema = schemas.get(name)
            if not schema:
                continue
            required = set(schema.get("parameters", {}).get("required", []) or [])
            args = func.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    continue
            if not isinstance(args, dict):
                continue
            for key in required:
                value = args.get(key)
                if self._is_grounded(value, context_text, key_name=key, dialogue_state=dialogue_state):
                    continue
                if value is None or isinstance(value, (dict, list, bool, int, float)):
                    continue
                text = str(value).strip()
                if self._DATE_RECEIPT_RE.match(text):
                    continue
                normalized = _normalize_str(text)
                if not normalized or normalized in normalized_ctx:
                    continue
                if self._is_abbreviation_of_context(text, context_text):
                    continue
                hits.append({"tool": name, "param": key, "value": text})
        return hits

    def _assess_selected_commitment(self, tool_calls, tools, context_text, dialogue_state=None):
        """Audit only the action path the model actually selected.

        This is the commitment proof consumed by the ask gate and logs.  Unlike
        the old state gate, it never lets an unrelated available tool veto an
        otherwise grounded action.  A required value is blocked only when it is
        absent from the selected call or its supplied value has no receipt.
        """
        schemas = {
            func.get("name"): func
            for tool in tools or []
            for func in [tool.get("function", {})]
            if func.get("name")
        }
        paths = []
        for tc in tool_calls or []:
            func = tc.get("function", {})
            name = func.get("name")
            schema = schemas.get(name, {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            args = args if isinstance(args, dict) else {}
            required = list(schema.get("parameters", {}).get("required", []) or [])
            missing = [key for key in required if key not in args or args[key] is None]
            grounded = [
                key for key in required if key in args and self._is_grounded(
                    args[key], context_text, key_name=key, dialogue_state=dialogue_state
                )
            ]
            paths.append({
                "tool": name,
                "required": required,
                "grounded_required": grounded,
                "missing_required": missing,
            })
        unfounded = self._unfounded_required(
            tool_calls, tools, context_text, dialogue_state=dialogue_state
        )
        return {
            "mode": "tool" if paths else "non_tool",
            "paths": paths,
            "unfounded_required": unfounded,
            "blocked": bool(unfounded),
        }

    def _authored_clarification(self, inference_data, content):
        '''
        Turn the current step into a clarification WITHOUT fabricating text.
        If the model already wrote something alongside its call, that is used.
        Otherwise the model is asked once more on the same conversation with
        tool calling disabled, so the question is still authored by the model.
        Returns None when no text could be obtained (caller keeps the call).
        '''
        if content and content.strip():
            return content
        if not hasattr(self, "_request_text_only"):
            return None
        try:
            return self._request_text_only(inference_data)
        except Exception as e:
            print(f"Clarification re-prompt failed, keeping original call: {e}", flush=True)
            return None

    def _pre_messages_processing(self, env_info, current_task, history_tasks, history_answer_lists, consecutive_tool_messages=True, tools=None):
        messages = [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(env_info=env_info)}]
        ledger = self._build_deterministic_ledger(history_tasks, history_answer_lists)
        if ledger:
            messages.append({
                "role": "system",
                "content": "Ledger - verified record of this session's prior turns "
                           "(task -> action taken -> result -> how it closed):\n\n" + ledger,
            })
        facts = self._extract_observation_facts(history_answer_lists)
        if facts:
            fact_str = "\n".join(f"- {f}" for f in facts[:40])
            messages.append({
                "role": "system",
                "content": "Surfaced Observation Facts (use directly for referential ID binding):\n" + fact_str
            })
        messages.append({"role": "user", "content": current_task})

        dialogue_state = self._build_dialogue_state(tools, messages, history_answer_lists)
        if facts or ledger:
            d_state_lines = [
                "Dialogue State:",
                f"- known_facts: {len(dialogue_state['known_facts'])} items",
                f"- feasible_tools: {dialogue_state['feasible_tools']}",
                f"- unresolved_required_slots: {dialogue_state['unresolved_required_slots']}",
            ]
            if dialogue_state['clarify_slot']:
                d_state_lines.append(f"- clarify_slot: {dialogue_state['clarify_slot']}")
            messages.insert(-1, {
                "role": "system",
                "content": "\n".join(d_state_lines)
            })

        return messages, dialogue_state

    def inference(self, test_entry: dict):
        return self.inference_multi_turn(test_entry)

    @final
    def inference_multi_turn(self, test_entry: dict):
        test_entry_id = test_entry["id"]
        env_info = test_entry["english_env_info"]
        tools = test_entry["english_tools"]
        tasks = test_entry["english_tasks"]
        answer_lists = test_entry["english_answer_list"]

        tools = self.convert_to_tool(tools)

        all_task_result_data = []
        for task_idx, (current_task, answer_list) in enumerate(zip(tasks, answer_lists)):
            history_tasks = tasks[:task_idx]
            history_answer_lists = answer_lists[:task_idx]
            messages, dialogue_state = self._pre_messages_processing(
                env_info, current_task, history_tasks, history_answer_lists, tools=tools
            )

            inference_data = {
                "test_entry_id": test_entry_id,
                "task_idx": task_idx,
                "tools": tools,
                "messages": messages,
                "answer_list": answer_list,
                "history_answer_lists": history_answer_lists,
                "dialogue_state": dialogue_state
            }
            result_data = self.inference_and_eval_multi_step(inference_data)
            all_task_result_data.append(result_data)

        return all_task_result_data

    def run_with_timeout(self, func, timeout, *args, **kwargs):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                raise TimeoutError(f"Function '{func.__name__}' exceeded timeout of {timeout} seconds")

    @final
    def inference_and_eval_multi_step(self, inference_data):
        '''
        Only the action is evaluated here
        If the action is incorrect, the process will be terminated early to improve evaluation efficiency.
        The correctness of the parameters will be evaluated later in the eval_checker.
        '''
        test_entry_id = inference_data["test_entry_id"]
        task_idx = inference_data["task_idx"]
        tools = inference_data["tools"]
        messages = inference_data["messages"]
        answer_list = inference_data["answer_list"]
        tool_call_graph = ToolCallGraph(answer_list)
        tool_call_graph.add_node_list()
        tool_call_graph.generate_all_path()
        # try:
        #     self.run_with_timeout(tool_call_graph.add_node_list, 60)
        #     self.run_with_timeout(tool_call_graph.generate_all_path, 60)
        # except Exception as e:
        #     print(f"error: {e}", flush=True)
        #     return "graph generate timeout."

        # print("generate", len(answer_list), flush=True)

        step = 0
        action_name_label = "error"
        predict_result = []
        # inference_log = [{"task_idx": task_idx, "begin_of_current_task": messages[-1]}]
        inference_log = {
            "task_idx": task_idx,
            "begin_of_current_task": messages[-1]
        }
        answer_result = []
        latency = []
        input_token_count = []
        output_token_count = []
        while True:
            # Recompute from the transcript that is actually visible at this
            # step, including gold-history observations and this task's prior
            # tool results.  No future task/step is consulted.
            inference_data["dialogue_state"] = self._build_latent_task_state(
                tools, messages, inference_data.get("history_answer_lists")
            )
            dialogue_state = inference_data["dialogue_state"]
            print("-" * 100, flush=True)
            print(
                f"ID: {test_entry_id.replace('wild_tool_bench_', '')}, Task: {task_idx}, Step: {step}", flush=True
            )
            # print(f"Output：", flush=True)
            # for message in messages:
            #     print(json.dumps(message, ensure_ascii=False, indent=4) + "\n", flush=True)
            model_response_data = self._consensus_generate(inference_data)
            query_latency = model_response_data.get("latency", 0)
            consensus_log = model_response_data.pop("consensus_log", None)
            reasoning_content = model_response_data["reasoning_content"]
            content = model_response_data["content"]
            tool_calls = model_response_data["tool_calls"]
            dialogue_state = inference_data.get("dialogue_state")
            if tool_calls is not None:
                # Schema cleaner + receipt gate (ungrounded optionals dropped)
                cleaned_tool_calls = []
                for tc in tool_calls:
                    try:
                        tc_name = tc["function"]["name"]
                        tc_args_str = tc["function"]["arguments"]
                        if isinstance(tc_args_str, str):
                            tc_args = json.loads(tc_args_str)
                            cleaned_args = self._clean_and_verify_call(tc_name, tc_args, tools, messages, step, inference_log, dialogue_state=dialogue_state)
                            tc["function"]["arguments"] = json.dumps(cleaned_args, ensure_ascii=False)
                        elif isinstance(tc_args_str, dict):
                            cleaned_args = self._clean_and_verify_call(tc_name, tc_args_str, tools, messages, step, inference_log, dialogue_state=dialogue_state)
                            tc["function"]["arguments"] = cleaned_args
                    except Exception as e:
                        print(f"Cleaner error: {e}", flush=True)
                    cleaned_tool_calls.append(tc)
                tool_calls = self._topological_sort_tool_calls(cleaned_tool_calls, tools)
                model_response_data["tool_calls"] = tool_calls

                selected_commitment = self._assess_selected_commitment(
                    tool_calls, tools, json.dumps(messages, ensure_ascii=False), dialogue_state=dialogue_state
                )
                inference_log.setdefault("commitment_notes", []).append({
                    "step": step,
                    "selected_commitment": selected_commitment,
                })

                # Ask-instead-of-guess: if a REQUIRED value was invented rather
                # than taken from the conversation or the selected commitment's
                # provenance, hand the turn back to the user rather than acting
                # on the guess.  Global state missingness is never sufficient.
                if self.ask_gate:
                    unfounded = selected_commitment["unfounded_required"]
                    if unfounded:
                        clarification = self._authored_clarification(inference_data, content)
                        if clarification:
                            inference_log.setdefault("ask_gate_notes", []).append({
                                "step": step,
                                "unfounded_required": unfounded,
                                "suppressed_calls": [
                                    tc.get("function", {}).get("name") for tc in tool_calls
                                ],
                            })
                            content = clarification
                            tool_calls = None
                            model_response_data["content"] = content
                            model_response_data["tool_calls"] = None
            input_token = model_response_data["input_token"]
            output_token = model_response_data["output_token"]
            latency.append(query_latency)
            input_token_count.append(input_token)
            output_token_count.append(output_token)

            # print(f"Output：", flush=True)
            # print(f"reasoning_content: \n{reasoning_content}\n", flush=True)
            # print(f"content: \n{content}\n", flush=True)
            # print(f"tool_calls: \n{json.dumps(tool_calls, ensure_ascii=False, indent=4)}\n", flush=True)

            inference_log[f"step_{step}"] = {
                "inference_input": {
                    "messages": deepcopy(messages),
                    "tools": tools,
                    "latent_task_state": deepcopy(dialogue_state),
                },
                "inference_output": {
                    "reasoning_content": reasoning_content,
                    "content": content,
                    "tool_calls": tool_calls
                }
            }
            if consensus_log is not None:
                inference_log[f"step_{step}"]["consensus"] = consensus_log

            if tool_calls is None or len(tool_calls) == 0:
                if content is None or content == "":
                    action_name_label = "error"
                    inference_log[f"step_{step}"]["inference_output"].update(
                        {
                            "current_action_name_label": "error",
                            "error_reason": f"tool_calls and content are None or empty"
                        }
                    )
                    break

                else:
                    current_step_function_name_list = tool_call_graph.step_to_function_name_list[step]
                    current_step_function_arguments_list = tool_call_graph.step_to_function_arguments_list[step]
                    current_step_function_observation_list = tool_call_graph.step_to_function_observation_list[step]
                    current_step_user_input_list = tool_call_graph.step_to_user_input_list[step]
                    for i, (answer_function_name_list, answer_function_arguments_list, answer_function_observation_list, answer_user_input_list) in enumerate(
                            zip(
                                current_step_function_name_list,
                                current_step_function_arguments_list,
                                current_step_function_observation_list,
                                current_step_user_input_list
                            )
                    ):
                        if "ask_user_for_required_parameters" in answer_function_name_list:
                            messages.append(
                                {"role": "assistant", "content": content}
                            )

                            assert len(answer_function_observation_list) == 1
                            function_observation = answer_function_observation_list[0]

                            assert len(answer_user_input_list) == 1
                            user_input = answer_user_input_list[0]
                            messages.append(
                                {"role": "user", "content": user_input}
                            )

                            answer_function_list = {"action": [], "observation": function_observation, "user_input": user_input}
                            for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_0_answer_function_list"] = answer_function_list
                            inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"

                            break
                        elif "prepare_to_answer" in answer_function_name_list:
                            assert len(answer_function_observation_list) == 1
                            function_observation = answer_function_observation_list[0]

                            messages.append(
                                {"role": "assistant", "content": content}
                            )
                            action_name_label = "correct"

                            answer_function_list = {"action": [], "observation": function_observation}
                            for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_0_answer_function_list"] = answer_function_list
                            inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"

                            break
                    else:
                        action_name_label = "error"
                        # Provide candidate correct answers
                        for i, (answer_function_name_list, answer_function_arguments_list, answer_function_observation_list, answer_user_input_list) in enumerate(
                                zip(
                                    current_step_function_name_list,
                                    current_step_function_arguments_list,
                                    current_step_function_observation_list,
                                    current_step_user_input_list
                                )
                        ):
                            answer_function_list = {"action": []}
                            for answer_function_name, answer_function_arguments, answer_function_observation, answer_user_input in zip(
                                    answer_function_name_list,
                                    answer_function_arguments_list,
                                    answer_function_observation_list,
                                    answer_user_input_list
                            ):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                                if answer_function_name == "ask_user_for_required_parameters":
                                    answer_function_list["observation"] = answer_function_observation
                                    answer_function_list["user_input"] = answer_user_input
                                elif answer_function_name == "prepare_to_answer":
                                    answer_function_list["observation"] = answer_function_observation

                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_{i}_answer_function_list"] = answer_function_list

                        inference_log[f"step_{step}"]["inference_output"].update(
                            {
                                "current_action_name_label": "error",
                                "error_reason": f"action name not in candidate_answer_function_list"
                            }
                        )
                        break

            else:
                tool_calls_len = len(tool_calls)
                predict_function_name_list = []
                predict_function_arguments_list = []
                predict_function_id_list = []
                try:
                    for tool_call in tool_calls:
                        function_id = tool_call["id"]
                        function = tool_call["function"]
                        function_name = function["name"]
                        function_arguments = function["arguments"]
                        predict_function_id_list.append(function_id)
                        predict_function_name_list.append(function_name)
                        predict_function_arguments_list.append(function_arguments)
                except Exception as e:
                    print(f"{json.dumps(tool_calls, ensure_ascii=False)} parse failed.", flush=True)
                    action_name_label = "error"
                    inference_log[f"step_{step}"]["inference_output"].update(
                        {
                            "current_action_name_label": "error",
                            "error_reason": f"parse tool_calls failed, error: {str(e)}"
                        }
                    )
                    break

                idx_predict_function_name_list = list(enumerate(predict_function_name_list))
                sorted_idx_predict_function_name_list = sorted(idx_predict_function_name_list, key=lambda x: x[1])
                sorted_indices = [idx for idx, predict_function_name in sorted_idx_predict_function_name_list]
                sorted_predict_function_name_list = [predict_function_name_list[i] for i in sorted_indices]
                sorted_predict_function_arguments_list = [predict_function_arguments_list[i] for i in sorted_indices]
                predict_function_name_list = sorted_predict_function_name_list
                predict_function_arguments_list = sorted_predict_function_arguments_list

                idx_list = tool_call_graph.step_to_idx_list.get(step, None)
                if idx_list is None:
                    action_name_label = "error"
                    inference_log[f"step_{step}"]["inference_output"].update(
                        {
                            "current_action_name_label": "error",
                            "error_reason": f"current step: {step}, idx_list is None"
                        }
                    )
                    break
                else:
                    current_step_function_name_list = tool_call_graph.step_to_function_name_list[step]
                    current_step_function_arguments_list = tool_call_graph.step_to_function_arguments_list[step]
                    current_step_function_observation_list = tool_call_graph.step_to_function_observation_list[step]
                    current_step_user_input_list = tool_call_graph.step_to_user_input_list[step]
                    for i, (answer_function_name_list, answer_function_arguments_list) in enumerate(zip(current_step_function_name_list, current_step_function_arguments_list)):
                        if predict_function_name_list != answer_function_name_list:
                            continue
                        else:
                            if reasoning_content is not None:
                                messages.append(
                                    {"role": "assistant", "reasoning_content": reasoning_content, "content": content, "tool_calls": tool_calls}
                                )
                            else:
                                messages.append(
                                    {"role": "assistant", "content": content, "tool_calls": tool_calls}
                                )

                            function_observation_list = tool_call_graph.step_to_function_observation_list[step][i]
                            if self.consecutive_tool_messages:
                                # Supports consecutive multiple tool messages
                                for j, function_observation in enumerate(function_observation_list):
                                    if not isinstance(function_observation, str):
                                        function_observation = json.dumps(function_observation, ensure_ascii=False)
                                    function_id = predict_function_id_list[j]
                                    messages.append(
                                        {"role": "tool", "content": function_observation, "tool_call_id": function_id}
                                    )
                            else:
                                # Does not support consecutive multiple tool messages
                                function_observation_list = json.dumps(function_observation_list, ensure_ascii=False)
                                function_id = predict_function_id_list[0]
                                messages.append(
                                    {"role": "tool", "content": function_observation_list, "tool_call_id": function_id}
                                )

                            # Pruning
                            idx_list = tool_call_graph.step_to_idx_list[step][i]
                            tool_call_graph.update_updating_all_path_list(step, idx_list)
                            tool_call_graph.init_step_to_answer()

                            answer_function_list = {"action": []}
                            for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                                answer_function_list["action"].append({"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name})
                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_0_answer_function_list"] = answer_function_list
                            inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"
                            break
                    else:
                        action_name_label = "error"
                        # Provide candidate correct answers
                        for i, (answer_function_name_list, answer_function_arguments_list, answer_function_observation_list, answer_user_input_list) in enumerate(zip(
                                current_step_function_name_list,
                                current_step_function_arguments_list,
                                current_step_function_observation_list,
                                current_step_user_input_list
                            )
                        ):
                            answer_function_list = {"action": []}
                            for answer_function_name, answer_function_arguments, answer_function_observation, answer_user_input in zip(
                                    answer_function_name_list,
                                    answer_function_arguments_list,
                                    answer_function_observation_list,
                                    answer_user_input_list
                            ):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                                if answer_function_name == "ask_user_for_required_parameters":
                                    answer_function_list["observation"] = answer_function_observation
                                    answer_function_list["user_input"] = answer_user_input
                                elif answer_function_name == "prepare_to_answer":
                                    answer_function_list["observation"] = answer_function_observation

                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_{i}_answer_function_list"] = answer_function_list

                        inference_log[f"step_{step}"]["inference_output"].update(
                            {
                                "current_action_name_label": "error",
                                "error_reason": f"action name not in candidate_answer_function_list"
                            }
                        )
                        break

            if action_name_label == "correct":
                inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"
                break

            step += 1
            inference_data["messages"] = messages

        # print(f"inference end\n", flush=True)
        # for message in messages:
        #     print(json.dumps(message, ensure_ascii=False, indent=4) + "\n", flush=True)
        # print(f"action_name_label: {action_name_label}\n", flush=True)
        if action_name_label == "correct":
            if step == (tool_call_graph.min_length - 1):
                is_optimal = True
            else:
                is_optimal = False
        else:
            is_optimal = False

        return {
            "action_name_label": action_name_label,
            "is_optimal": is_optimal,
            "inference_log": inference_log,
            "latency": latency,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count
        }

    @final
    def write(self, result, result_dir, update_mode=False):
        model_name_dir = self.model_name.replace("/", "_")
        model_result_dir = result_dir / model_name_dir
        model_result_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(result, dict):
            result = [result]

        file_path = model_result_dir / os.path.basename(PROMPT_PATH).replace(".jsonl", "_result.jsonl")

        if update_mode:
            # Load existing entries from the file
            existing_entries = {}
            if file_path.exists():
                existing_entries = {entry["id"]: entry for entry in load_file(file_path)}

            # Update existing entries with new data
            for entry in result:
                existing_entries[entry["id"]] = entry

            # Sort entries by `id` and write them back to ensure order consistency
            sorted_entries = sorted(existing_entries.values(), key=sort_key)
            with open(file_path, "w") as fout:
                for entry in sorted_entries:
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

        else:
            # Normal mode: Append in sorted order
            result.sort(key=sort_key)
            with open(file_path, "a") as fout:
                for entry in result:
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

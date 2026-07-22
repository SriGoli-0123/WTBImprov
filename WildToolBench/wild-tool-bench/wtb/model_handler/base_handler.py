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
            if prop_type == "integer":
                try:
                    return int(float(val))
                except:
                    pass
            elif prop_type == "number":
                try:
                    return float(val)
                except:
                    pass
            elif prop_type == "boolean":
                if isinstance(val, str):
                    if val.lower() in ["true", "1"]:
                        return True
                    if val.lower() in ["false", "0"]:
                        return False
            elif prop_type == "array" and isinstance(val, str):
                try:
                    loaded = json.loads(val)
                    if isinstance(loaded, list):
                        return loaded
                except:
                    pass
            elif prop_type == "object" and isinstance(val, dict):
                # Recursive cleaning for nested objects
                nested_properties = prop_schema.get("properties", {})
                cleaned_nested = {}
                for k, v in val.items():
                    if k in nested_properties:
                        cleaned_nested[k] = cast_value(v, nested_properties[k])
                return cleaned_nested
            # Enum check
            if "enum" in prop_schema:
                enum_options = prop_schema["enum"]
                if isinstance(val, str):
                    # Case-insensitive match
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

    def _is_grounded(self, value, context_text, key_name=None):
        '''
        A value is grounded if it is traceable to something already visible
        to the model: quoted verbatim from the conversation/Ledger (QUOTE /
        RESOLVE), or shaped like a resolved date/time (COMPUTE). Container
        values are grounded only if every leaf value inside them is.
        '''
        if value is None:
            return True
        if isinstance(value, bool):
            # A boolean is only grounded if its key or boolean context is
            # mentioned in the conversation text, or if it is required by schema.
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
            return all(self._is_grounded(leaf, context_text, key_name) for leaf in leaves)
        return True

    def _verify_and_filter_arguments(self, tool_name, arguments_dict, tools, context_text):
        '''
        Receipt gate: an argument survives only if it is required by the
        schema or grounded in the conversation. Required keys are never
        removed (the schema always wins) but are flagged if ungrounded, so
        an ungrounded-required case is visible in the log instead of silently
        passed through. Every ungrounded optional key is dropped - this is
        what eliminates hallucinated defaults (format, quality, limit, ...)
        mechanically instead of relying on the prompt being obeyed.
        '''
        required = set()
        for t in tools:
            func = t.get("function", {})
            if func.get("name") == tool_name:
                required = set(func.get("parameters", {}).get("required", []))
                break

        kept = {}
        dropped_keys = []
        ungrounded_required = []
        for key, value in arguments_dict.items():
            grounded = self._is_grounded(value, context_text, key_name=key)
            if key in required:
                kept[key] = value
                if not grounded:
                    ungrounded_required.append(key)
            elif grounded:
                kept[key] = value
            else:
                dropped_keys.append(key)
        return kept, dropped_keys, ungrounded_required

    def _match_verbatim_prompt_casing(self, arguments_dict, context_text):
        if not isinstance(arguments_dict, dict) or not context_text:
            return arguments_dict
        
        context_lower = context_text.lower()
        cleaned = {}
        for key, val in arguments_dict.items():
            if isinstance(val, str) and val and len(val) >= 2:
                val_lower = val.lower()
                if val not in context_text and val_lower in context_lower:
                    idx = context_lower.find(val_lower)
                    if idx != -1:
                        verbatim_sub = context_text[idx:idx + len(val)]
                        cleaned[key] = verbatim_sub
                    else:
                        cleaned[key] = val
                else:
                    cleaned[key] = val
            elif isinstance(val, list):
                new_list = []
                for item in val:
                    if isinstance(item, str) and item and item not in context_text and item.lower() in context_lower:
                        idx = context_lower.find(item.lower())
                        if idx != -1:
                            new_list.append(context_text[idx:idx + len(item)])
                        else:
                            new_list.append(item)
                    else:
                        new_list.append(item)
                cleaned[key] = new_list
            else:
                cleaned[key] = val
        return cleaned

    def _clean_and_verify_call(self, tc_name, tc_args, tools, messages, step, inference_log):
        cleaned_args = self._clean_tool_call_arguments(tc_name, tc_args, tools)
        context_text = json.dumps(messages, ensure_ascii=False)
        cleaned_args = self._match_verbatim_prompt_casing(cleaned_args, context_text)
        cleaned_args, dropped_keys, ungrounded_required = self._verify_and_filter_arguments(
            tc_name, cleaned_args, tools, context_text
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

    def _consensus_generate(self, inference_data):
        '''
        Self-consistency (consensus) decoding for one step.
        Draws sc_n candidates (1 anchor at the run temperature + sc_n-1 diversity
        samples), then votes hierarchically: response mode -> tool-name multiset
        -> per-argument key/value majority. Returns a single model_response_data
        dict shaped like _parse_api_response output, plus aggregated token counts,
        latency, and a "consensus_log" entry describing the vote.
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

        # Normalize copies for voting (recover parser-dropped tool calls / abstain
        # on unrecoverable ones) while keeping the faithful anchor for fallback.
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

        if not vote_counts:
            chosen = anchor
        else:
            best_count = max(vote_counts.values())
            winners = {s for s, v in vote_counts.items() if v == best_count}
            if signatures[0] in winners:
                # Tie or win including the anchor: trust the low-temperature sample.
                winning_signature = signatures[0]
            else:
                winning_signature = next(s for s in signatures if s in winners)
            cluster = [c for c, s in zip(candidates, signatures) if s == winning_signature]
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
                    if key not in required and not self._is_grounded(value, context_text):
                        total += 1
                        detail.append(f"{name}.{key}: optional argument with no basis in the conversation")
                    elif (isinstance(value, str) and self._DATE_RECEIPT_RE.match(value.strip())
                          and not self._date_value_ok(value, context_text)):
                        # Required args are never stripped, but a date that
                        # matches no computed candidate is a high-precision
                        # signal for selection and repair.
                        total += 2
                        detail.append(f"{name}.{key}: date does not match any date derivable from the conversation")
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
        draft = json.dumps([tc.get("function") for tc in chosen["tool_calls"]], ensure_ascii=False)
        note = (
            "AUDIT: your drafted tool call(s) " + draft + " have these problems:\n- "
            + "\n- ".join(audit["detail"])
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

    def _extract_observation_facts(self, history_answer_lists):
        if not history_answer_lists:
            return []
        facts = []

        def collect(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (str, int, float)) and v:
                        s_v = str(v).strip()
                        if len(s_v) >= 3 and (k.lower().endswith("id") or k.lower().endswith("code") or k.lower().endswith("number")):
                            facts.append(f"{k}: {s_v}")
                    else:
                        collect(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        for answer_list in history_answer_lists:
            for ans in answer_list:
                obs = ans.get("observation")
                if obs:
                    collect(obs)
        return list(dict.fromkeys(facts))

    def _pre_messages_processing(self, env_info, current_task, history_tasks, history_answer_lists, consecutive_tool_messages=True):
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
            fact_str = "\n".join(f"- {f}" for f in facts[:20])
            messages.append({
                "role": "system",
                "content": "Surfaced Observation Facts (use directly for referential ID binding):\n" + fact_str
            })
        messages.append({"role": "user", "content": current_task})
        return messages

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
            messages = self._pre_messages_processing(env_info, current_task, history_tasks, history_answer_lists)

            inference_data = {"test_entry_id": test_entry_id, "task_idx": task_idx, "tools": tools, "messages": messages, "answer_list": answer_list}
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
            if tool_calls is not None:
                # Schema cleaner + receipt gate (ungrounded optionals dropped)
                cleaned_tool_calls = []
                for tc in tool_calls:
                    try:
                        tc_name = tc["function"]["name"]
                        tc_args_str = tc["function"]["arguments"]
                        if isinstance(tc_args_str, str):
                            tc_args = json.loads(tc_args_str)
                            cleaned_args = self._clean_and_verify_call(tc_name, tc_args, tools, messages, step, inference_log)
                            tc["function"]["arguments"] = json.dumps(cleaned_args, ensure_ascii=False)
                        elif isinstance(tc_args_str, dict):
                            cleaned_args = self._clean_and_verify_call(tc_name, tc_args_str, tools, messages, step, inference_log)
                            tc["function"]["arguments"] = cleaned_args
                    except Exception as e:
                        print(f"Cleaner error: {e}", flush=True)
                    cleaned_tool_calls.append(tc)
                tool_calls = cleaned_tool_calls
                model_response_data["tool_calls"] = tool_calls
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
                    "tools": tools
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

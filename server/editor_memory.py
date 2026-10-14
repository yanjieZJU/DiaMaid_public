"""
EditorAgent memory subsystem.

Three tiers:
  Tier 1 — Working snapshot: focus-aware slim of the current canvas
           (computed inline; this module exposes the helpers).
  Tier 2 — Episodic: last K=3 raw modification entries (recent_actions).
  Tier 3 — Semantic / long-term: running_summary + key_decisions
           + entity_journal + open_questions.

Persistence: each session-tree node carries a full `editor_memory` snapshot
(branch-aware via parent-chain walk; new branches inherit their fork point's
memory and evolve independently).

Update policy:
  - Rule-based update every turn (no extra LLM call).
  - LLM-driven `running_summary` compression every COMPRESSION_INTERVAL turns
    when an OpenAI client is supplied.
"""

import os
import sys
import json
import re
from copy import deepcopy
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAT_MODEL= os.environ.get("CHAT_MODEL")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
RECENT_ACTIONS_CAP = 3
FOCUS_AWARE_THRESHOLD = 8        # module count above which focus-aware slim activates
OPEN_QUESTIONS_CAP = 30
ENTITY_JOURNAL_PER_MODULE_CAP = 12
KEY_DECISIONS_RENDER_CAP = 12
RUNNING_SUMMARY_BUDGET_CHARS = 600
COMPRESSION_INTERVAL = 5

# Strong-signal regex for rule-based key_decision extraction.
# Each entry: (compiled pattern, decision type tag).
_DECISION_PATTERNS = [
    (re.compile(r"必须(保留|独立|存在|有)|不能(合并|去掉|删除|移除)|不要(合并|删除|去掉|移除)"), "preserve_module"),
    (re.compile(r"must (keep|preserve|stay|remain)|do not remove|don't remove", re.I), "preserve_module"),
    (re.compile(r"不要再|别再|拒绝|否决|不需要|不想要|不喜欢"), "rejected_direction"),
    (re.compile(r"do not (use|add|introduce)|don't (use|add|introduce)|reject", re.I), "rejected_direction"),
    (re.compile(r"改名为|重命名为|叫做|命名为|应改名"), "naming_convention"),
    (re.compile(r"\brename\b|should be called|name (it|this)", re.I), "naming_convention"),
    (re.compile(r"我(希望|想要|倾向于|偏好|想让)|更倾向"), "preference"),
    (re.compile(r"\bprefer\b|preferably|i'd rather", re.I), "preference"),
]

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _empty_memory() -> dict:
    return {
        "running_summary": "",
        "key_decisions": [],
        "entity_journal": {},
        "open_questions": [],
        "recent_actions": [],
        "turn_index": 0,
        "last_compressed_turn": 0,
    }


def _ensure_shape(mem: Optional[dict]) -> dict:
    base = _empty_memory()
    if not isinstance(mem, dict):
        return base
    for k, v in base.items():
        mem.setdefault(k, v)
    return mem


# ---------------------------------------------------------------------------
# Loading: branch-aware via parent chain walk
# ---------------------------------------------------------------------------

def load_memory(session_id: Optional[str], plan_id: Optional[str]) -> dict:
    """
    Walk the parent chain from `plan_id` upward and return the editor_memory
    snapshot from the closest ancestor (incl. self) that has one. Returns an
    empty memory if none found. Cycle-safe.
    """
    if not session_id or not plan_id:
        return _empty_memory()
    # Lazy import — session_manager has no module-level dep on this file.
    from session_manager import _load_session
    session_data = _load_session(session_id)
    nodes = session_data.get("nodes", {}) if isinstance(session_data, dict) else {}
    cur = plan_id
    seen = set()
    while cur and cur in nodes and cur not in seen:
        seen.add(cur)
        node = nodes[cur]
        mem = node.get("editor_memory")
        if isinstance(mem, dict) and mem:
            return _ensure_shape(deepcopy(mem))
        cur = node.get("parent_id")
    return _empty_memory()


# ---------------------------------------------------------------------------
# Focus computation + focus-aware canvas slim (Tier 1)
# ---------------------------------------------------------------------------

def _module_list(plan: dict) -> list:
    if not isinstance(plan, dict):
        return []
    md = plan.get("module_details", [])
    if isinstance(md, dict):
        md = list(md.values())
    return md if isinstance(md, list) else []


def _ports_neighbors(mod: dict) -> set:
    """Module names directly connected to `mod` via ports (compact str or dict edges)."""
    out = set()
    ports = mod.get("ports") or {}
    for edge in ports.get("incoming", []) or []:
        if isinstance(edge, str):
            head = edge.split("->", 1)[0].split(".", 1)[0].strip()
            if head:
                out.add(head)
        elif isinstance(edge, dict):
            sm = edge.get("source_mod")
            if sm:
                out.add(sm)
    for edge in ports.get("outgoing", []) or []:
        if isinstance(edge, str):
            head = edge.split("->", 1)[0].split(".", 1)[0].strip()
            if head:
                out.add(head)
        elif isinstance(edge, dict):
            tm = edge.get("target_mod")
            if tm:
                out.add(tm)
    return out


def compute_in_focus(intent_analysis: Optional[dict], current_plan: Optional[dict]) -> set:
    """
    Targets from structural_edits + 1-hop neighbors via the ports graph.
    If no targets are extractable (e.g. exploration-only intent), return all
    module names so callers naturally fall back to the full slim view.
    """
    modules = _module_list(current_plan or {})
    name_set = {m.get("name", "") for m in modules if isinstance(m, dict) and m.get("name")}
    targets = set()
    for e in (intent_analysis or {}).get("structural_edits", []) or []:
        t = (e.get("target") or "").strip()
        if not t:
            continue
        if t in name_set:
            targets.add(t)
            continue
        # Loose substring fallback: target='识别' may refer to '果实识别'.
        for n in name_set:
            if t in n or n in t:
                targets.add(n)
    if not targets:
        return name_set  # broad intent → no aggressive cutting
    in_focus = set(targets)
    by_name = {m.get("name", ""): m for m in modules if isinstance(m, dict)}
    for t in list(targets):
        mod = by_name.get(t)
        if mod:
            in_focus.update(_ports_neighbors(mod))
    return (in_focus & name_set) if name_set else in_focus


def slim_plan_focus_aware(plan: dict, in_focus: set, threshold: int = FOCUS_AWARE_THRESHOLD) -> dict:
    """
    Drop `brief_description` and `sub_nodes` for out-of-focus modules; keep
    `ports` so the topology is intact. Below `threshold`, behave like the
    existing slim (all modules retain brief_description + sub_nodes).
    """
    if not isinstance(plan, dict):
        return plan
    modules = _module_list(plan)
    slim = {k: v for k, v in plan.items() if k != "module_details"}
    use_focus = len(modules) > threshold and bool(in_focus) and len(in_focus) < len(modules)
    out = []
    for m in modules:
        if not isinstance(m, dict):
            continue
        name = m.get("name", "")
        is_in = (name in in_focus) if use_focus else True
        if is_in:
            entry = {k: m[k] for k in ("name", "brief_description", "role", "sub_nodes") if k in m}
        else:
            entry = {k: m[k] for k in ("name", "role") if k in m}
        if "ports" in m:
            entry["ports"] = m["ports"]
        out.append(entry)
    slim["module_details"] = out
    return slim


# ---------------------------------------------------------------------------
# Update (rule-based + optional LLM compression)
# ---------------------------------------------------------------------------

def _extract_decisions(user_instruction: str, intent_analysis: Optional[dict], turn: int) -> list:
    """Rule-based key_decision candidates from instruction text + intent rationale."""
    text_blobs = [user_instruction or ""]
    if isinstance(intent_analysis, dict):
        text_blobs.append(intent_analysis.get("rationale") or "")
    candidates = []
    for blob in text_blobs:
        if not blob:
            continue
        for pat, dtype in _DECISION_PATTERNS:
            if pat.search(blob):
                snippet = blob.strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + "…"
                candidates.append({
                    "turn": turn,
                    "type": dtype,
                    "content": snippet,
                    "still_active": True,
                })
                break  # one decision per blob — same blob shouldn't fire multiple types
    return candidates


def _change_log_for_revised_plan(revised_plan: Optional[dict]) -> list:
    if not isinstance(revised_plan, dict):
        return []
    md = revised_plan.get("module_details", [])
    if isinstance(md, dict):
        md = list(md.values())
    log = []
    for m in md:
        if not isinstance(m, dict):
            continue
        ct = m.get("change_type")
        name = m.get("name")
        if name and ct in ("added", "modified", "ports_only"):
            log.append({"name": name, "change_type": ct})
    return log


def update_memory(
    memory: Optional[dict],
    user_instruction: str,
    intent_analysis: Optional[dict],
    modification_intent: Optional[str],
    revised_plan: Optional[dict],
    unhandled_intents: Optional[list],
    *,
    client=None,
    model: Optional[str] = None,
) -> dict:
    """
    Apply rule-based memory updates and return the new snapshot.

    `client` enables the every-N-turns running_summary compression. Passing
    None skips compression but otherwise behaves identically.
    """
    memory = _ensure_shape(deepcopy(memory) if memory else _empty_memory())
    turn = (memory.get("turn_index") or 0) + 1
    memory["turn_index"] = turn

    # Tier 2: recent_actions (sliding window) ---------------------------------
    change_log = _change_log_for_revised_plan(revised_plan)
    action_entry = {
        "turn": turn,
        "user_instruction": (user_instruction or "")[:300],
        "primary_intent": (intent_analysis or {}).get("primary_intent", ""),
        "modification_intent": (modification_intent or "")[:300],
        "change_log": change_log,
    }
    actions = list(memory.get("recent_actions") or [])
    actions.append(action_entry)
    evicted = []
    while len(actions) > RECENT_ACTIONS_CAP:
        evicted.append(actions.pop(0))
    memory["recent_actions"] = actions

    # Tier 3: entity_journal --------------------------------------------------
    journal = dict(memory.get("entity_journal") or {})
    for cl in change_log:
        name = cl["name"]
        entries = list(journal.get(name) or [])
        entries.append({
            "turn": turn,
            "change": cl["change_type"],
            "reason": (modification_intent or "")[:140],
        })
        if len(entries) > ENTITY_JOURNAL_PER_MODULE_CAP:
            entries = entries[-ENTITY_JOURNAL_PER_MODULE_CAP:]
        journal[name] = entries
    memory["entity_journal"] = journal

    # Tier 3: open_questions --------------------------------------------------
    open_qs = list(memory.get("open_questions") or [])
    seen = set(open_qs)
    for u in (unhandled_intents or []):
        if not isinstance(u, dict):
            continue
        kind = u.get("kind", "")
        if kind not in ("questions_unanswered", "exploration_not_surfaced"):
            continue
        for item in u.get("items", []) or []:
            text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            tagged = f"[turn{turn}/{kind}] {text}"
            if tagged not in seen:
                open_qs.append(tagged)
                seen.add(tagged)
    if len(open_qs) > OPEN_QUESTIONS_CAP:
        open_qs = open_qs[-OPEN_QUESTIONS_CAP:]
    memory["open_questions"] = open_qs

    # Tier 3: key_decisions ---------------------------------------------------
    decisions = list(memory.get("key_decisions") or [])
    for cand in _extract_decisions(user_instruction, intent_analysis, turn):
        if not any(d.get("type") == cand["type"] and d.get("content") == cand["content"] for d in decisions):
            decisions.append(cand)
    memory["key_decisions"] = decisions

    # LLM compression (every COMPRESSION_INTERVAL turns) ----------------------
    should_compress = (
        client is not None
        and turn > 0
        and turn % COMPRESSION_INTERVAL == 0
        and (turn - (memory.get("last_compressed_turn") or 0)) >= COMPRESSION_INTERVAL
        and (evicted or memory.get("running_summary"))
    )
    if should_compress:
        try:
            new_summary = compress_summary_with_llm(
                client=client,
                model=model,
                old_summary=memory.get("running_summary") or "",
                evicted_actions=evicted,
                key_decisions=[d for d in decisions if d.get("still_active")],
            )
            if isinstance(new_summary, str) and new_summary.strip():
                memory["running_summary"] = new_summary.strip()[:RUNNING_SUMMARY_BUDGET_CHARS]
                memory["last_compressed_turn"] = turn
        except Exception as exc:
            print(f"[editor_memory] compression failed (non-fatal): {exc}")

    return memory


def compress_summary_with_llm(client, model: Optional[str], old_summary: str, evicted_actions: list, key_decisions: list) -> str:
    """One short LLM call merging prior summary + evicted Tier-2 entries → new summary."""
    model = model or CHAT_MODEL
    actions_brief = "\n".join(
        f"- turn{a.get('turn')}: user='{(a.get('user_instruction') or '')[:120]}' "
        f"-> change='{(a.get('modification_intent') or '')[:120]}'"
        for a in (evicted_actions or [])
    ) or "(none)"
    decisions_brief = "\n".join(
        f"- {d.get('type')}: {d.get('content')}"
        for d in (key_decisions or [])[:10]
    ) or "(none)"
    sys_prompt = (
        "You are a memory compressor for an architecture-editing assistant. "
        "Merge the prior running summary with newly-evicted actions into ONE concise "
        "summary (under ~200 Chinese characters or ~100 English words, primarily Chinese for user-facing readability). "
        "Preserve: the system being designed, key constraints, the user's evolving intent. "
        "Drop: per-turn micro-details. "
        "Return PLAIN TEXT only — no JSON, no markdown."
    )
    user_prompt = (
        f"Prior summary:\n{old_summary or '(empty)'}\n\n"
        f"Newly evicted turns:\n{actions_brief}\n\n"
        f"Established preferences (do NOT lose these):\n{decisions_brief}\n\n"
        f"Output the new running summary now."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def format_for_prompt(memory: Optional[dict], in_focus_modules: Optional[set] = None) -> str:
    """Render the memory tiers as a prompt-ready markdown block."""
    memory = _ensure_shape(memory or {})
    lines = []

    summary = (memory.get("running_summary") or "").strip()
    if summary:
        lines.append("## Long-term Context")
        lines.append(summary)
        lines.append("")

    active_decisions = [d for d in (memory.get("key_decisions") or []) if d.get("still_active")]
    if active_decisions:
        lines.append("## Established Preferences (from prior turns — must respect)")
        for d in active_decisions[-KEY_DECISIONS_RENDER_CAP:]:
            lines.append(f"- [{d.get('type')}, turn{d.get('turn')}] {d.get('content')}")
        lines.append("")

    open_qs = memory.get("open_questions") or []
    if open_qs:
        lines.append("## Open Threads (unresolved from prior turns — address if relevant)")
        for q in open_qs[-8:]:
            lines.append(f"- {q}")
        lines.append("")

    journal = memory.get("entity_journal") or {}
    if journal:
        focus = in_focus_modules if in_focus_modules else set(journal.keys())
        relevant = {k: v for k, v in journal.items() if k in focus}
        if relevant:
            lines.append("## Module Evolution (only modules relevant to this turn)")
            for name, entries in relevant.items():
                trail = "; ".join(
                    f"turn{e.get('turn')}:{e.get('change')}"
                    + (f"({e.get('reason')})" if e.get('reason') else "")
                    for e in entries[-5:]
                )
                lines.append(f"- {name}: {trail}")
            lines.append("")

    recent = memory.get("recent_actions") or []
    if recent:
        lines.append("## Recent Turns (raw, last 3)")
        for a in recent:
            cl = a.get("change_log") or []
            cl_brief = ", ".join(f"{c.get('name')}:{c.get('change_type')}" for c in cl) or "no-op"
            lines.append(
                f"- turn{a.get('turn')} [{a.get('primary_intent') or '?'}]: "
                f"user='{a.get('user_instruction', '')}' "
                f"→ intent='{a.get('modification_intent', '')}' "
                f"changes=[{cl_brief}]"
            )
        lines.append("")

    if not lines:
        return "No prior modifications."
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Bridge from legacy `history_summary` (transition convenience)
# ---------------------------------------------------------------------------

def memory_from_history_summary(history_summary: Optional[list]) -> dict:
    """
    Bootstrap a minimal memory from the legacy `history_summary` list. Used
    when a node has no `editor_memory` field yet (first edit after upgrade).
    Only the K most recent steps survive — older ones would have been
    compressed into running_summary anyway in the new world.
    """
    mem = _empty_memory()
    if not isinstance(history_summary, list):
        return mem
    for i, item in enumerate(history_summary):
        if not isinstance(item, dict):
            continue
        mem["turn_index"] = item.get("step", i)
        mem["recent_actions"].append({
            "turn": item.get("step", i),
            "user_instruction": (item.get("user_instruction") or "")[:300],
            "primary_intent": "structural_edit" if item.get("type") == "Modification" else "initial",
            "modification_intent": (item.get("result_description") or "")[:300],
            "change_log": [],
        })
    while len(mem["recent_actions"]) > RECENT_ACTIONS_CAP:
        mem["recent_actions"].pop(0)
    return mem

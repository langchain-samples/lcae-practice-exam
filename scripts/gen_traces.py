"""Generate traces.json for the LCAE practice exam repo.

Produces three scenarios in one project:
  A. tool_error_rate: get_order_status has a ~50% error rate; other tools are fine.
  B. hallucination_v3_v4: v3 hallucinates in ~37% of traces; v4 in ~12%.
  C. cost_attribution: main agent + research subagent + draft_response subagent,
     research subagent dominates cost.

Every LLM run carries ls_model_name / ls_provider / usage_metadata so LangSmith
can compute cost. Tool errors are attached to the tool run's `error` field.
Root chain outputs and LLM messages carry realistic content so candidates who
click into a run see a full agent conversation, not empty placeholders.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(2026)

PROJECT = "customer-support"
OUT = Path("/Users/sean/Documents/certs/lcae-practice-exam/traces.json")

MODEL = "gpt-5"
PROVIDER = "openai"


def llm_run(rid, parent, name, start_ms, duration_ms, input_tokens, output_tokens, inputs, outputs):
    return {
        "id": rid,
        "parent": parent,
        "name": name,
        "type": "llm",
        "start_ms": start_ms,
        "duration_ms": duration_ms,
        "inputs": inputs,
        "outputs": outputs,
        "model": MODEL,
        "provider": PROVIDER,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def tool_run(rid, parent, name, start_ms, duration_ms, inputs, outputs=None, error=None):
    r = {
        "id": rid,
        "parent": parent,
        "name": name,
        "type": "tool",
        "start_ms": start_ms,
        "duration_ms": duration_ms,
        "inputs": inputs,
    }
    if error:
        r["error"] = error
    else:
        r["outputs"] = outputs or {}
    return r


def chain_run(rid, parent, name, start_ms, duration_ms, inputs, outputs):
    return {
        "id": rid,
        "parent": parent,
        "name": name,
        "type": "chain",
        "start_ms": start_ms,
        "duration_ms": duration_ms,
        "inputs": inputs,
        "outputs": outputs,
    }


def human_msg(text):
    return {"type": "human", "content": text}


def ai_msg(text, tool_calls=None):
    return {"type": "ai", "content": text, "tool_calls": tool_calls or []}


def tool_msg(tool_call_id, name, content):
    return {"type": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}


def llm_output(assistant_msg, prompt_tokens, completion_tokens, finish_reason="stop"):
    return {
        "choices": [{"message": assistant_msg, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------- Scenario A: tool error rate ----------

TOOL_TEMPLATES = {
    "search_kb": {
        "make_inputs": lambda: {"query": random.choice([
            "reset password", "shipping delay", "return policy", "warranty terms", "how to cancel subscription",
        ])},
        "make_outputs": lambda inp: {
            "results": [
                {"title": f"KB article for {inp['query']}", "snippet": "See section 3 of the article.", "score": 0.87},
                {"title": "Related FAQ", "snippet": "This is a related question.", "score": 0.62},
            ]
        },
        "error_message": "SearchTimeoutError: knowledge-base upstream did not respond within 5000ms",
    },
    "get_order_status": {
        "make_inputs": lambda: {"order_id": f"ORD-{random.randint(1000, 9999)}"},
        "make_outputs": lambda inp: {
            "order_id": inp["order_id"],
            "status": random.choice(["shipped", "processing", "delivered", "awaiting_payment"]),
            "eta": "2026-09-04",
            "carrier": "UPS",
        },
        "error_message": "OrderNotFoundError: order not found in downstream OMS",
    },
    "refund_order": {
        "make_inputs": lambda: {
            "order_id": f"ORD-{random.randint(1000, 9999)}",
            "amount_cents": random.randint(500, 20000),
        },
        "make_outputs": lambda inp: {
            "refund_id": f"REF-{random.randint(1000, 9999)}",
            "status": "queued",
            "eta_business_days": 3,
        },
        "error_message": "RefundGatewayError: gateway rejected refund",
    },
    "send_email": {
        "make_inputs": lambda: {
            "to": f"user_{random.randint(1, 500):03d}@example.com",
            "subject": "your ticket update",
            "body_preview": "Thanks for reaching out. Here is the update on your ticket ...",
        },
        "make_outputs": lambda inp: {"message_id": f"msg_{random.randint(10000, 99999)}", "delivered": True},
        "error_message": "SMTPError: relay refused",
    },
    "escalate_to_human": {
        "make_inputs": lambda: {
            "reason": random.choice(["billing_dispute", "safety", "vip_customer"]),
            "priority": random.choice(["normal", "high"]),
        },
        "make_outputs": lambda inp: {
            "queue_id": f"Q-{random.randint(1, 20)}",
            "eta_minutes": random.randint(5, 40),
            "agent": "not_yet_assigned",
        },
        "error_message": "EscalationQueueFullError: no available agents",
    },
}

USER_PROMPTS_SUPPORT = [
    "Where is my order? I placed it last week.",
    "I want a refund for my last purchase, it arrived damaged.",
    "Can you email me the return label for ticket 4471?",
    "Your rep hung up on me, I want to speak to a manager.",
    "What's your return policy on electronics?",
    "My package says delivered but I never got it. Please help.",
    "Can you check the status of order ORD-2201?",
    "I need to cancel my subscription before it renews tomorrow.",
    "The refund I asked for two days ago hasn't arrived yet.",
    "Please transfer me to a human, this is urgent.",
    "How long does a standard refund take?",
    "Send me a confirmation email that my order shipped.",
    "I want to reset my account password.",
    "The shipping tracker hasn't updated in 4 days, is my package lost?",
]

# Target per-tool call counts and error counts across the tool_error_rate
# scenarios. Chosen so the signal is unambiguous in the LangSmith UI:
# get_order_status is clearly the worst (~50%), the rest are healthy or minor.
TOOL_TARGETS = [
    # (tool_name, total_calls, error_count)
    ("get_order_status", 28, 14),   # 50%
    ("escalate_to_human", 9, 3),    # 33%
    ("search_kb", 22, 3),           # ~14%
    ("refund_order", 12, 0),        # 0%
    ("send_email", 14, 0),          # 0%
]

NUM_TOOL_ERROR_TRACES = 40


def build_tool_sequences() -> list[list[tuple[str, bool]]]:
    """Produce a shuffled list of tool calls (with error flags) that hits the
    per-tool target counts and error counts above, then packs them into per-
    trace sequences of 1-3 tool calls each."""
    pool: list[tuple[str, bool]] = []
    for tool_name, total, errors in TOOL_TARGETS:
        pool.extend([(tool_name, True)] * errors)
        pool.extend([(tool_name, False)] * (total - errors))
    random.shuffle(pool)

    sequences: list[list[tuple[str, bool]]] = []
    idx = 0
    while idx < len(pool):
        n_calls = random.choices([1, 2, 3], weights=[2, 3, 1])[0]
        n_calls = min(n_calls, len(pool) - idx)
        sequences.append(pool[idx : idx + n_calls])
        idx += n_calls

    # Pad or trim to NUM_TOOL_ERROR_TRACES traces.
    while len(sequences) < NUM_TOOL_ERROR_TRACES:
        sequences.append([("search_kb", False)])
    sequences = sequences[:NUM_TOOL_ERROR_TRACES]
    return sequences


TOOL_SEQUENCES = build_tool_sequences()


def make_tool_error_trace(idx, tools):
    user_text = USER_PROMPTS_SUPPORT[idx % len(USER_PROMPTS_SUPPORT)]
    user_message = human_msg(user_text)

    plan_tool_calls = []
    tool_inputs = []
    tool_call_ids = []
    for i, (tool_name, _is_error) in enumerate(tools):
        template = TOOL_TEMPLATES[tool_name]
        tc_id = f"call_{idx:03d}_{i}"
        tool_call_ids.append(tc_id)
        inp = template["make_inputs"]()
        tool_inputs.append(inp)
        plan_tool_calls.append({"id": tc_id, "type": "tool_call", "name": tool_name, "args": inp})
    plan_assistant = ai_msg("Looking into that for you.", tool_calls=plan_tool_calls)

    tool_result_messages = []
    tool_outputs = []
    for i, (tool_name, is_error) in enumerate(tools):
        template = TOOL_TEMPLATES[tool_name]
        if is_error:
            tool_outputs.append(None)
            tool_result_messages.append(tool_msg(tool_call_ids[i], tool_name, json.dumps({"error": template["error_message"]})))
        else:
            out = template["make_outputs"](tool_inputs[i])
            tool_outputs.append(out)
            tool_result_messages.append(tool_msg(tool_call_ids[i], tool_name, json.dumps(out)))

    if any(is_error for _, is_error in tools):
        final_text = (
            "I ran into an issue looking that up. Let me get a human agent to take a closer look. "
            "In the meantime, could you double-check the order number you gave me?"
        )
    else:
        final_text = (
            "Here is what I found. Everything on your account looks in order, and I've kicked off "
            "the follow-up you asked about. Let me know if you need anything else."
        )
    final_ai = ai_msg(final_text)

    runs = []
    duration_ms = 800 + 900 * len(tools) + random.randint(200, 800)
    runs.append(chain_run(
        "root", None, "customer_support_agent", 0, duration_ms,
        inputs={"messages": [user_message]},
        outputs={"messages": [user_message, plan_assistant] + tool_result_messages + [final_ai]},
    ))
    runs.append(llm_run(
        "plan", "root", "plan", 100, 700, 850, 60,
        inputs={"messages": [user_message]},
        outputs=llm_output(plan_assistant, 850, 60, finish_reason="tool_calls"),
    ))

    cursor = 850
    for i, (tool_name, is_error) in enumerate(tools):
        template = TOOL_TEMPLATES[tool_name]
        if is_error:
            runs.append(tool_run(
                f"t{i}", "root", tool_name, cursor, 600,
                inputs=tool_inputs[i], error=template["error_message"],
            ))
        else:
            runs.append(tool_run(
                f"t{i}", "root", tool_name, cursor, 500,
                inputs=tool_inputs[i], outputs=tool_outputs[i],
            ))
        cursor += 700

    runs.append(llm_run(
        "synth", "root", "respond", cursor, 900, 1100, 140,
        inputs={"messages": [user_message, plan_assistant] + tool_result_messages},
        outputs=llm_output(final_ai, 1100, 140),
    ))

    return {
        "scenario": "tool_error_rate",
        "metadata": {"user_id": f"user_{200 + idx}"},
        "start_offset_minutes": -random.uniform(0, 1440),
        "runs": runs,
        "feedback": [],
    }


# ---------- Scenario B: v3 vs v4 hallucination ----------

HALLU_PLAN = (
    [("v3", 1)] * 7 + [("v3", 0)] * 13 +   # 20 v3, 35% hallucinate
    [("v4", 1)] * 2 + [("v4", 0)] * 18     # 20 v4, 10% hallucinate
)
random.shuffle(HALLU_PLAN)

HALLU_ITEMS = [
    {
        "question": "What year did LangChain release LangGraph?",
        "kb_snippet": "LangGraph was open-sourced by LangChain in January 2024 as a library for building stateful, multi-actor agent applications.",
        "truthful": "LangChain open-sourced LangGraph in January 2024.",
        "hallucinated": "LangChain released LangGraph in November 2022 as its first product.",
    },
    {
        "question": "Who founded LangChain?",
        "kb_snippet": "LangChain was founded by Harrison Chase in October 2022.",
        "truthful": "LangChain was founded by Harrison Chase in October 2022.",
        "hallucinated": "LangChain was founded by Andrew Ng and Sam Altman in 2019.",
    },
    {
        "question": "What's the maximum context window on GPT-5?",
        "kb_snippet": "GPT-5 supports up to a 400,000-token context window across text and structured inputs.",
        "truthful": "GPT-5 supports a context window of up to 400,000 tokens.",
        "hallucinated": "GPT-5 supports up to 2 million tokens of context, the largest of any model available today.",
    },
    {
        "question": "How much does the LangSmith Plus plan cost?",
        "kb_snippet": "LangSmith Plus is $39 per user per month, billed annually.",
        "truthful": "LangSmith Plus is $39 per user per month, billed annually.",
        "hallucinated": "LangSmith Plus is free for all users up to 10 million traces per month.",
    },
    {
        "question": "When did LangChain announce LangSmith?",
        "kb_snippet": "LangSmith was announced in July 2023 and moved to general availability in February 2024.",
        "truthful": "LangSmith was announced in July 2023 and became generally available in February 2024.",
        "hallucinated": "LangSmith was announced at OpenAI DevDay in November 2023 as a joint product with OpenAI.",
    },
]


def make_hallucination_trace(idx, version, hallucinated):
    item = HALLU_ITEMS[idx % len(HALLU_ITEMS)]
    user_message = human_msg(item["question"])

    plan_tool_call_id = f"call_hallu_{idx:03d}"
    plan_tool_calls = [{
        "id": plan_tool_call_id,
        "type": "tool_call",
        "name": "search_kb",
        "args": {"query": item["question"]},
    }]
    plan_assistant = ai_msg("Let me check the knowledge base.", tool_calls=plan_tool_calls)

    tool_output = {
        "results": [
            {"title": "Company reference", "snippet": item["kb_snippet"], "score": 0.91},
            {"title": "Related overview", "snippet": "See the company timeline for more.", "score": 0.55},
        ]
    }
    tool_result_message = tool_msg(plan_tool_call_id, "search_kb", json.dumps(tool_output))

    final_text = item["hallucinated"] if hallucinated else item["truthful"]
    final_ai = ai_msg(final_text)

    runs = [
        chain_run(
            "root", None, "support_agent", 0, 2400,
            inputs={"messages": [user_message]},
            outputs={"messages": [user_message, plan_assistant, tool_result_message, final_ai]},
        ),
        llm_run(
            "plan", "root", "plan", 100, 800, 720, 55,
            inputs={"messages": [user_message]},
            outputs=llm_output(plan_assistant, 720, 55, finish_reason="tool_calls"),
        ),
        tool_run(
            "kb", "root", "search_kb", 950, 400,
            inputs={"query": item["question"]}, outputs=tool_output,
        ),
        llm_run(
            "synth", "root", "respond", 1400, 900, 950, 130,
            inputs={"messages": [user_message, plan_assistant, tool_result_message]},
            outputs=llm_output(final_ai, 950, 130),
        ),
    ]

    return {
        "scenario": "hallucination_v3_v4",
        "metadata": {
            "version": version,
            "user_id": f"user_{400 + idx}",
        },
        "start_offset_minutes": -random.uniform(0, 1440),
        "runs": runs,
        "feedback": [{"run_id": "root", "key": "hallucination", "score": hallucinated}],
    }


# ---------- Scenario C: cost attribution ----------

# Four accounts. Acme Corp intentionally consumes ~2x the output tokens per
# trace so it stands out in the "output tokens grouped by metadata.account"
# view — that's the signal the practice-exam question relies on.
COST_ACCOUNTS = [
    "Acme Corp",
    "Globex Industries",
    "Umbrella Group",
    "Initech",
]
DOMINANT_ACCOUNT = "Acme Corp"

SHORT_FINDINGS = (
    "Findings for {topic}:\n"
    "- Usage in the last 30 days is up 22% vs the prior period, driven by increased weekly active teams.\n"
    "- Two power users have opened four support tickets in the past week; two are still open.\n"
    "- Contract auto-renews in 47 days. Procurement contact changed in the last 90 days.\n"
    "- The champion has been quiet in Slack for two weeks; risk score on the account elevated.\n"
    "Recommended next action: schedule an exec business review before day 30, loop in Solutions to "
    "de-risk the open tickets, and confirm the new procurement contact ahead of renewal."
)

LONG_FINDINGS = (
    "Findings for {topic}:\n"
    "- Usage in the last 30 days is up 41% vs the prior period. Growth is concentrated in the "
    "engineering org (up 63%) and the RevOps team (up 28%); support and marketing are roughly flat.\n"
    "- Weekly active users climbed from 112 to 189 in the last quarter, with a healthy tail of new users "
    "onboarding in the last 30 days who have already crossed the 3-session mark.\n"
    "- Five power users have opened eleven support tickets this quarter; four remain open. Two of the "
    "open tickets touch data-export functionality and are blocking a compliance workflow they use "
    "monthly. The other two are model-cost related and align with the increased usage we're seeing.\n"
    "- Contract auto-renews in 63 days. The procurement contact turned over in the last 90 days and "
    "the new contact has already asked for a mid-term price review. Sentiment from the primary "
    "sponsor remains positive but the procurement contact appears skeptical of the current pricing.\n"
    "- The champion has been active in Slack, posting screenshots of usage wins and asking about "
    "expansion into an adjacent product line. Risk score on the account is trending down.\n"
    "- Expansion opportunity: the engineering org is asking about the enterprise SSO SKU, and the "
    "RevOps team is asking about the pro analytics SKU. Combined ARR uplift is estimated at $180k "
    "if both close before renewal.\n"
    "Recommended next actions: (1) book an exec business review inside the next 21 days to confirm "
    "renewal appetite and set up a pricing conversation with the new procurement contact; "
    "(2) loop in Solutions to de-risk the open data-export tickets, which are the biggest churn "
    "risk; (3) route the SSO and analytics expansion inquiries to the appropriate PM to accelerate "
    "the conversation; (4) prep a usage-based ROI story tailored to the champion's Slack callouts "
    "so the QBR opens with concrete wins rather than a status update."
)


def make_cost_trace(idx):
    account = COST_ACCOUNTS[idx % len(COST_ACCOUNTS)]
    is_dominant = account == DOMINANT_ACCOUNT
    # Sized so per-account totals hit exactly 5x. Non-research LLM runs
    # contribute 300 output tokens per trace as a fixed baseline (plan=40,
    # draft=200, synth=60); non-dominant accounts total 1100 tokens per
    # trace, so dominant needs 5*1100 - 300 = 5200 tokens on research.
    research_output_tokens = 5200 if is_dominant else 800
    research_findings_text = LONG_FINDINGS if is_dominant else SHORT_FINDINGS
    topic = f"{account} quarterly account briefing"
    user_message = human_msg(f"Give me a briefing on {topic}, including risks and next best action.")

    research_call_id = f"call_research_{idx:03d}"
    draft_call_id = f"call_draft_{idx:03d}"

    plan_assistant = ai_msg("I'll pull the latest signals and draft a briefing.", tool_calls=[
        {"id": research_call_id, "type": "tool_call", "name": "research_subagent", "args": {"topic": topic}},
    ])

    research_findings = research_findings_text.format(topic=topic)
    research_ai = ai_msg(research_findings)
    research_tool_result = tool_msg(
        research_call_id, "research_subagent",
        json.dumps({"findings": research_findings}),
    )

    draft_text = (
        f"Briefing: {topic}\n\n"
        "Health signals are mixed. Usage is up 22% MoM, which is a positive engagement signal, but "
        "the account has two open support tickets from power users, the champion has gone quiet, and "
        "the procurement contact turned over in the last quarter. Contract auto-renews in ~47 days.\n\n"
        "Next best action: book an executive business review inside 30 days, brief Solutions on the "
        "open tickets, and confirm the new procurement contact before the renewal window opens."
    )
    draft_ai = ai_msg(draft_text)
    draft_tool_result = tool_msg(
        draft_call_id, "draft_response_subagent",
        json.dumps({"draft": draft_text}),
    )

    final_ai = ai_msg(draft_text)

    runs = []
    duration_ms = 8000
    runs.append(chain_run(
        "root", None, "customer_support_agent", 0, duration_ms,
        inputs={"messages": [user_message]},
        outputs={"messages": [
            user_message, plan_assistant, research_tool_result, draft_tool_result, final_ai,
        ]},
    ))

    runs.append(llm_run(
        "plan", "root", "plan", 100, 600, 300, 40,
        inputs={"messages": [user_message]},
        outputs=llm_output(plan_assistant, 300, 40, finish_reason="tool_calls"),
    ))

    runs.append(chain_run(
        "research", "root", "research_subagent", 800, 3800,
        inputs={"topic": topic},
        outputs={"findings": research_findings},
    ))
    runs.append(llm_run(
        "research_llm", "research", "research_llm_call", 900, 3600, 3000, research_output_tokens,
        inputs={"messages": [human_msg(f"Analyze recent signals on {topic} and summarize risks and next actions.")]},
        outputs=llm_output(research_ai, 3000, research_output_tokens),
    ))

    runs.append(chain_run(
        "draft", "root", "draft_response_subagent", 4700, 2400,
        inputs={"findings": research_findings},
        outputs={"draft": draft_text},
    ))
    runs.append(llm_run(
        "draft_llm", "draft", "draft_llm_call", 4800, 2200, 800, 200,
        inputs={"messages": [human_msg(f"Turn these findings into a two-paragraph account briefing for a sales rep:\n\n{research_findings}")]},
        outputs=llm_output(draft_ai, 800, 200),
    ))

    runs.append(llm_run(
        "synth", "root", "respond", 7200, 700, 350, 60,
        inputs={"messages": [user_message, plan_assistant, research_tool_result, draft_tool_result]},
        outputs=llm_output(final_ai, 350, 60),
    ))

    return {
        "scenario": "cost_attribution",
        "metadata": {"user_id": f"user_{600 + idx}", "account": account},
        "start_offset_minutes": -random.uniform(0, 1440),
        "runs": runs,
        "feedback": [],
    }


# ---------- assemble ----------

traces = []

for idx, tools in enumerate(TOOL_SEQUENCES):
    traces.append(make_tool_error_trace(idx, tools))

for idx, (version, hallucinated) in enumerate(HALLU_PLAN):
    traces.append(make_hallucination_trace(idx, version, hallucinated))

for idx in range(40):
    traces.append(make_cost_trace(idx))

payload = {
    "project_name": PROJECT,
    "model_default": MODEL,
    "provider_default": PROVIDER,
    "traces": traces,
}

OUT.write_text(json.dumps(payload, indent=2))
print(f"wrote {len(traces)} traces to {OUT}")
print(f"  tool_error_rate: {sum(1 for t in traces if t['scenario'] == 'tool_error_rate')}")
print(f"  hallucination_v3_v4: {sum(1 for t in traces if t['scenario'] == 'hallucination_v3_v4')}")
print(f"  cost_attribution: {sum(1 for t in traces if t['scenario'] == 'cost_attribution')}")

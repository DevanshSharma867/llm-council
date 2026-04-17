"""3-stage (+ debate) LLM Council orchestration."""

import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple
from .groq import query_models_parallel, query_model, query_model_stream
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_conversation_history(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert stored conversation messages into a flat chat history that LLMs understand.
    User turns map directly; assistant turns use the Stage 3 chairman synthesis.
    """
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            stage3 = msg.get("stage3") or {}
            response_text = stage3.get("response", "")
            if response_text:
                history.append({"role": "assistant", "content": response_text})
    return history


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

async def stage1_collect_responses(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.
    Includes prior conversation history so models have context across turns.
    """
    history = conversation_history or []
    messages = history + [{"role": "user", "content": user_query}]

    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    stage1_results = []
    for model, response in responses.items():
        if response is not None:
            stage1_results.append({
                "model": model,
                "response": response.get("content", ""),
            })

    return stage1_results


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymised responses.

    Returns:
        (rankings_list, label_to_model mapping)
    """
    labels = [chr(65 + i) for i in range(len(stage1_results))]

    label_to_model = {
        f"Response {label}": result["model"]
        for label, result in zip(labels, stage1_results)
    }

    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get("content", "")
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed,
            })

    return stage2_results, label_to_model


# ---------------------------------------------------------------------------
# Stage 2.5 — Debate
# ---------------------------------------------------------------------------

async def stage2_5_debate(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    aggregate_rankings: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Stage 2.5: Debate — the top-ranked model defends its answer against peer critiques.

    - Other models critique the top response.
    - The top model rebuts their criticisms.

    Returns None if there aren't enough models to debate.
    """
    if not aggregate_rankings or len(stage1_results) < 2:
        return None

    top_model = aggregate_rankings[0]["model"]
    top_response = next(
        (r["response"] for r in stage1_results if r["model"] == top_model),
        None,
    )
    if not top_response:
        return None

    other_models = [r["model"] for r in stage1_results if r["model"] != top_model]
    if not other_models:
        return None

    # --- Critiques (parallel) ---
    critique_prompt = f"""A peer evaluation has ranked the following response as the best answer to this question:

Question: {user_query}

Top-ranked response:
{top_response}

Your task: provide specific, intellectually honest criticism of this response.
- What does it get wrong or oversimplify?
- What important nuances or counterpoints does it miss?
- Be direct and substantive — not just "it could be more detailed"."""

    critique_tasks = [
        query_model(model, [{"role": "user", "content": critique_prompt}])
        for model in other_models
    ]
    critique_responses = await asyncio.gather(*critique_tasks)

    critiques = []
    for model, response in zip(other_models, critique_responses):
        if response and response.get("content"):
            critiques.append({"model": model, "critique": response["content"]})

    if not critiques:
        logger.warning("No critiques returned in Stage 2.5 — skipping debate")
        return None

    # --- Defense ---
    critiques_text = "\n\n---\n\n".join([
        f"Critic ({c['model'].split('/')[-1]}):\n{c['critique']}"
        for c in critiques
    ])

    defense_prompt = f"""You previously provided this response to a question, and it was ranked #1 by peer evaluation:

Question: {user_query}

Your response:
{top_response}

Your peers have raised these criticisms:

{critiques_text}

Now defend your answer:
- Address each criticism directly.
- Acknowledge any valid points and explain how they could strengthen the answer.
- Hold your ground where you believe you were correct.
- You may refine or expand your original answer in light of the debate."""

    defense_response = await query_model(
        top_model,
        [{"role": "user", "content": defense_prompt}],
    )

    return {
        "top_model": top_model,
        "top_response": top_response,
        "critiques": critiques,
        "defense": {
            "model": top_model,
            "content": defense_response["content"] if defense_response else None,
        },
    }


# ---------------------------------------------------------------------------
# Stage 3 — Synthesis
# ---------------------------------------------------------------------------

def _build_chairman_messages(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    stage2_5_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Build the message list for the Chairman synthesis prompt."""
    stage1_text = "\n\n".join([
        f"Model: {r['model']}\nResponse: {r['response']}"
        for r in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {r['model']}\nRanking: {r['ranking']}"
        for r in stage2_results
    ])

    debate_section = ""
    if stage2_5_result:
        critiques_text = "\n\n".join([
            f"Critic ({c['model'].split('/')[-1]}):\n{c['critique']}"
            for c in stage2_5_result.get("critiques", [])
        ])
        defense = stage2_5_result.get("defense", {})
        defense_content = defense.get("content") or "No defense provided."
        top_model_short = stage2_5_result["top_model"].split("/")[-1]

        debate_section = f"""

STAGE 2.5 — Debate:
The top-ranked response (from {top_model_short}) was challenged by its peers.

Peer criticisms:
{critiques_text}

Defense by {top_model_short}:
{defense_content}
"""

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models answered a question, \
ranked each other's responses, and then engaged in a debate.

Original Question: {user_query}

STAGE 1 — Individual Responses:
{stage1_text}

STAGE 2 — Peer Rankings:
{stage2_text}
{debate_section}
Your task: synthesize all of this into a single, clear, accurate final answer.
- Draw on the strongest insights from each model.
- Take the debate into account — if a criticism was valid and the defense was weak, factor that in.
- Be direct and comprehensive.

Provide the council's final answer:"""

    return [{"role": "user", "content": chairman_prompt}]


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    stage2_5_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stage 3: Chairman synthesizes the final response (non-streaming)."""
    messages = _build_chairman_messages(user_query, stage1_results, stage2_results, stage2_5_result)
    response = await query_model(CHAIRMAN_MODEL, messages)

    if response is None:
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis.",
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get("content", ""),
    }


async def stage3_stream(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    stage2_5_result: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """Stage 3: Stream the Chairman's synthesis token by token."""
    messages = _build_chairman_messages(user_query, stage1_results, stage2_results, stage2_5_result)
    async for chunk in query_model_stream(CHAIRMAN_MODEL, messages):
        yield chunk


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """Parse the FINAL RANKING section from a model's response."""
    import re

    if "FINAL RANKING:" in ranking_text:
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    return re.findall(r'Response [A-Z]', ranking_text)


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Calculate aggregate rankings across all peer evaluations."""
    from collections import defaultdict

    model_positions: Dict[str, List[int]] = defaultdict(list)

    for ranking in stage2_results:
        parsed = parse_ranking_from_text(ranking["ranking"])
        for position, label in enumerate(parsed, start=1):
            if label in label_to_model:
                model_positions[label_to_model[label]].append(position)

    aggregate = [
        {
            "model": model,
            "average_rank": round(sum(positions) / len(positions), 2),
            "rankings_count": len(positions),
        }
        for model, positions in model_positions.items()
        if positions
    ]
    aggregate.sort(key=lambda x: x["average_rank"])
    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """Generate a short title for a conversation based on the first user message."""
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    response = await query_model(
        COUNCIL_MODELS[0],
        [{"role": "user", "content": title_prompt}],
        timeout=30.0,
    )

    if response is None:
        return "New Conversation"

    title = response.get("content", "New Conversation").strip().strip("\"'")
    return title[:47] + "..." if len(title) > 50 else title


# ---------------------------------------------------------------------------
# Full pipeline (used by non-streaming endpoint)
# ---------------------------------------------------------------------------

async def run_full_council(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[List, List, Optional[Dict], Dict, Dict]:
    """
    Run the complete council process (non-streaming).

    Returns:
        (stage1_results, stage2_results, stage2_5_result, stage3_result, metadata)
    """
    stage1_results = await stage1_collect_responses(user_query, conversation_history)

    if not stage1_results:
        logger.error("All council models failed in Stage 1")
        return [], [], None, {
            "model": "error",
            "response": "All models failed to respond. Please try again.",
        }, {}

    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    stage2_5_result = await stage2_5_debate(user_query, stage1_results, aggregate_rankings)

    stage3_result = await stage3_synthesize_final(
        user_query, stage1_results, stage2_results, stage2_5_result
    )

    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
    }

    return stage1_results, stage2_results, stage2_5_result, stage3_result, metadata

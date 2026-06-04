# ============================================================
# utils/cost_tracker.py
# ============================================================
# Lightweight token usage and cost tracker.
# Accumulates per-operation cost entries in session state.
# Does NOT call any LLM or import from Streamlit.
# ============================================================

from config import COST_PER_1K_INPUT, COST_PER_1K_OUTPUT


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculates estimated USD cost for a model call.

    Args:
        model: Model name string (e.g. 'gpt-4o-mini')
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens used.

    Returns:
        Estimated cost in USD.
    """
    input_cost = (
        input_tokens / 1000
    ) * COST_PER_1K_INPUT.get(model, 0.0)

    output_cost = (
        output_tokens / 1000
    ) * COST_PER_1K_OUTPUT.get(model, 0.0)

    return input_cost + output_cost


def add_cost_entry(
    token_log: list,
    operation: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """
    Appends a cost entry to the session token log.
    Mutates the token_log list in place so session
    state is updated automatically.

    Args:
        token_log: st.session_state["session_token_log"]
        operation: Human-readable label (e.g. "extraction")
        model: Model name string.
        input_tokens: Input tokens used.
        output_tokens: Output tokens used.
    """
    cost = calculate_cost(model, input_tokens, output_tokens)
    token_log.append({
        "operation": operation,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    })


def get_session_totals(token_log: list) -> dict:
    """
    Aggregates all cost entries into session totals.

    Args:
        token_log: st.session_state["session_token_log"]

    Returns:
        Dict with total input tokens, output tokens, and cost.
    """
    total_input = sum(e["input_tokens"] for e in token_log)
    total_output = sum(e["output_tokens"] for e in token_log)
    total_cost = sum(e["cost_usd"] for e in token_log)

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": total_cost,
        "total_calls": len(token_log),
    }


def format_cost_display(token_log: list) -> dict:
    """
    Formats session totals for sidebar display.

    Returns:
        Dict with formatted display strings.
    """
    totals = get_session_totals(token_log)
    return {
        "input_tokens": f"{totals['total_input_tokens']:,}",
        "output_tokens": f"{totals['total_output_tokens']:,}",
        "total_cost": f"${totals['total_cost_usd']:.4f}",
        "total_calls": str(totals["total_calls"]),
    }
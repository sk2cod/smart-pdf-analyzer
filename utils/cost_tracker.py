# ============================================================
# utils/cost_tracker.py
# ============================================================
# Lightweight token usage and cost tracker.
# Accumulates per-operation cost entries in session state.
# Does NOT call any LLM or import from Streamlit.
# ============================================================

# ============================================================
# utils/cost_tracker.py
# ============================================================

from config import COST_PER_1K_INPUT, COST_PER_1K_OUTPUT


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
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
    actual_calls: int = 1,
) -> None:
    """
    Appends a cost entry to the session token log.

    Args:
        token_log: st.session_state["session_token_log"]
        operation: Human-readable label
        model: Model name string
        input_tokens: Total input tokens for this operation
        output_tokens: Total output tokens for this operation
        actual_calls: Actual number of API calls made
    """
    cost = calculate_cost(model, input_tokens, output_tokens)
    token_log.append({
        "operation": operation,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "actual_calls": actual_calls,
    })


def get_session_totals(token_log: list) -> dict:
    total_input = sum(e["input_tokens"] for e in token_log)
    total_output = sum(e["output_tokens"] for e in token_log)
    total_cost = sum(e["cost_usd"] for e in token_log)
    total_actual_calls = sum(
        e.get("actual_calls", 1) for e in token_log
    )
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": total_cost,
        "total_actual_calls": total_actual_calls,
        "total_operations": len(token_log),
    }


def format_cost_display(token_log: list) -> dict:
    """
    Formats session totals for sidebar display.
    Shows operations with actual call counts (Option B).
    """
    totals = get_session_totals(token_log)

    # Build operation breakdown
    breakdown_lines = []
    for entry in token_log:
        calls = entry.get("actual_calls", 1)
        if calls > 1:
            breakdown_lines.append(
                f"{entry['operation']} ({calls} calls) "
                f"— ${entry['cost_usd']:.4f}"
            )
        else:
            breakdown_lines.append(
                f"{entry['operation']} "
                f"— ${entry['cost_usd']:.4f}"
            )

    return {
        "input_tokens": f"{totals['total_input_tokens']:,}",
        "output_tokens": f"{totals['total_output_tokens']:,}",
        "total_cost": f"${totals['total_cost_usd']:.4f}",
        "total_operations": str(totals["total_operations"]),
        "total_actual_calls": str(totals["total_actual_calls"]),
        "breakdown": breakdown_lines,
    }
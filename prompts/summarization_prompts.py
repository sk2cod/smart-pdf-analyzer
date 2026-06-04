# ============================================================
# prompts/summarization_prompts.py
# ============================================================
# All prompt templates for Mode 2: Summarization.
# To change output style, add new formats, or adjust
# the synthesis tone — edit this file only.
# Nothing in modes/summarization.py needs to change.
# ============================================================

from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------
# Map Stage Prompt (Cheap Model)
# Compresses individual chunks into dense bullet summaries
# Called once per chunk — must be efficient and focused
# ------------------------------------------------------------

MAP_SYSTEM = """You are a document compression engine.
Your job is to compress a section of a document into dense bullet points.

Rules:
- Preserve: key entities, numerical data, dates, decisions, causal relationships
- Discard: filler sentences, transitional prose, repetition, pleasantries
- Output: bullet points only, no prose paragraphs
- Each bullet must be self-contained and meaningful without the original context
- Maximum 8 bullets per section regardless of content volume
- Never add information not present in the source text"""

MAP_HUMAN = """Compress this document section into key bullet points:

{text}"""

MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", MAP_SYSTEM),
    ("human", MAP_HUMAN),
])

# ------------------------------------------------------------
# Synthesis Stage Prompt (Premium Model)
# Takes all compressed chunk summaries and produces
# the final output in the user-requested format
# ------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are an expert document analyst and writer.
You are synthesizing compressed summaries from multiple sections
of a document into a single coherent final output.

CRITICAL CONSTRAINTS:
1. Your response must not exceed {max_words} words.
   Count your words before responding.
   If your draft exceeds this limit, remove the least important
   points first until the constraint is satisfied.
2. Output format must be exactly: {output_format_instruction}
3. If a focus instruction is provided, prioritise that content.
4. Do not add information not present in the source summaries.
5. Do not repeat the same point in different sections.

{focus_instruction}"""

SYNTHESIS_HUMAN = """Synthesize these compressed document summaries
into a final output following the format and length constraints
specified in your instructions.

Source summaries:
{text}"""

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYNTHESIS_SYSTEM),
    ("human", SYNTHESIS_HUMAN),
])

# ------------------------------------------------------------
# Format Instructions
# Injected into SYNTHESIS_SYSTEM as {output_format_instruction}
# Add new formats here — nothing else needs to change
# ------------------------------------------------------------

FORMAT_INSTRUCTIONS = {
    "bullet_hierarchy": """Structured markdown with:
- A ## heading for each major theme
- Bullet points under each heading
- Sub-bullets for supporting details
- Bold key terms and figures
- No prose paragraphs""",

    "executive_prose": """Executive summary in flowing prose:
- Opening sentence stating the document's purpose
- 3-5 paragraph body covering key themes
- Closing sentence with main conclusion or recommendation
- Professional formal tone throughout
- No bullet points or headers""",

    "mermaid_flowchart": """A valid Mermaid flowchart diagram.
Output the Mermaid code block only — no prose before or after.
Use this exact format:
```mermaid
flowchart TD
    A[Start] --> B[Step]
```
Nodes must have short labels (max 5 words).
Use TD (top-down) direction.
Maximum 12 nodes for readability.""",

    "mermaid_mindmap": """A valid Mermaid mindmap diagram.
Output the Mermaid code block only — no prose before or after.
Use this exact format:
```mermaid
mindmap
  root((Document Title))
    Topic 1
      Detail
    Topic 2
      Detail
```
Maximum 3 levels of nesting.
Maximum 5 branches from root.""",

    "comparison_table": """A markdown comparison table.
Identify the key dimensions being compared in the document.
Format as:
| Dimension | Value / Finding | Significance |
|-----------|----------------|--------------|
| ...       | ...            | ...          |
Include 5-15 rows covering the most important dimensions.
No prose outside the table.""",
}
# ============================================================
# prompts/summarization_prompts.py
# ============================================================

from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------
# Map Stage Prompt (Cheap Model)
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

    "mermaid_flowchart": """STRICT MERMAID FLOWCHART OUTPUT RULES — YOU MUST FOLLOW EXACTLY:

1. Output ONLY raw Mermaid code. No prose, no explanation, no markdown fences.
2. First line MUST be exactly: flowchart TD
3. Every node label MUST be wrapped in square brackets or parentheses: A[Label] or A(Label)
4. Node labels MUST be 4 words or fewer. No colons, no quotes, no special characters.
5. Use only alphanumeric node IDs: A, B, C or A1, B1 etc.
6. Maximum 10 nodes total.
7. Every line must be a valid connection: A --> B or A --> B --> C

CORRECT example output (output exactly like this, nothing else):
flowchart TD
    A[Petition Filed] --> B[Court Hearing]
    B --> C[Arguments Made]
    C --> D[Decision Issued]
    D --> E[Order Enforced]

OUTPUT ONLY THE FLOWCHART CODE. NOTHING ELSE.""",

    "mermaid_mindmap": """STRICT MERMAID MINDMAP OUTPUT RULES — YOU MUST FOLLOW EXACTLY:

1. Output ONLY raw Mermaid code. No prose, no explanation, no markdown fences.
2. First line MUST be exactly: mindmap
3. Second line MUST be: (two spaces)root((Short Title)) — max 3 words in root
4. Each child node is indented with 4 spaces. Each grandchild with 6 spaces.
5. Node text MUST be 4 words or fewer. No colons, no special characters, no brackets.
6. Maximum 5 branches from root. Maximum 3 sub-items per branch.
7. Maximum 3 levels of nesting total.

CORRECT example output (output exactly like this, nothing else):
mindmap
  root((Case Summary))
    Parties
      Petitioner Society
      State Respondent
    Key Issue
      Limitation Period
      Fee Recovery
    Court Decision
      Petition Dismissed
      Guarantee Honored

OUTPUT ONLY THE MINDMAP CODE. NOTHING ELSE.""",

    "comparison_table": """A markdown comparison table.
Identify the key dimensions being compared in the document.
Format as:
| Dimension | Value / Finding | Significance |
|-----------|----------------|--------------|
| ...       | ...            | ...          |
Include 5-15 rows covering the most important dimensions.
No prose outside the table.""",
}

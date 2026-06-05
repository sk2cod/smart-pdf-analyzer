# ============================================================
# utils/mermaid_renderer.py
# ============================================================

import re
import streamlit as st
import streamlit.components.v1 as components


def extract_mermaid_code(raw: str) -> str:
    """
    Extract clean Mermaid code from LLM output.

    Strategy (in order):
    1. Look for ```mermaid ... ``` fenced block — most reliable
    2. Look for bare ``` ... ``` block
    3. Look for a line starting with flowchart/graph/mindmap
       and take everything from there — handles unfenced output
    4. Return empty string if nothing found
    """
    # Strategy 1: fenced ```mermaid block
    fenced = re.search(
        r"```mermaid\s*\n(.*?)```",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1).strip()

    # Strategy 2: generic ``` block
    generic = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
    if generic:
        candidate = generic.group(1).strip()
        first = candidate.splitlines()[0].lower()
        if any(k in first for k in ("flowchart", "graph", "mindmap")):
            return candidate

    # Strategy 3: find diagram keyword line and take from there
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith(("flowchart", "graph td", "graph lr", "mindmap")):
            return "\n".join(lines[i:]).strip()

    return ""


def render_mermaid(mermaid_code: str, height: int = 500) -> None:
    """
    Render a Mermaid diagram live inside Streamlit.
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8" />
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <style>
        body {{
          margin: 0;
          padding: 12px;
          background: transparent;
          font-family: sans-serif;
        }}
        .mermaid-wrapper {{
          width: 100%;
          overflow-x: auto;
        }}
        .mermaid svg {{
          max-width: 100%;
          height: auto;
        }}
        #error-box {{
          display: none;
          background: #fff3cd;
          border: 1px solid #ffc107;
          border-radius: 6px;
          padding: 12px 16px;
          color: #856404;
          font-size: 13px;
          font-family: monospace;
          white-space: pre-wrap;
          margin-top: 8px;
        }}
      </style>
    </head>
    <body>
      <div class="mermaid-wrapper">
        <pre class="mermaid" id="mermaid-diagram">{mermaid_code}</pre>
      </div>
      <div id="error-box"></div>

      <script>
        mermaid.initialize({{
          startOnLoad: false,
          theme: 'default',
          securityLevel: 'loose',
          flowchart: {{ useMaxWidth: true, htmlLabels: true }},
          mindmap:  {{ useMaxWidth: true }},
        }});

        async function renderDiagram() {{
          const el = document.getElementById('mermaid-diagram');
          const code = el.innerText.trim();
          try {{
            const {{ svg }} = await mermaid.render('rendered-diagram', code);
            el.outerHTML = svg;
          }} catch (err) {{
            const errBox = document.getElementById('error-box');
            errBox.style.display = 'block';
            errBox.textContent = 'Render error: ' + err.message;
            el.style.display = 'none';
          }}
        }}

        renderDiagram();
      </script>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=True)


def estimate_height(mermaid_code: str) -> int:
    lines = mermaid_code.strip().splitlines()
    line_count = len(lines)
    first_line = lines[0].lower() if lines else ""

    base = 500 if "mindmap" in first_line else 420
    extra = max(0, (line_count - 10) * 18)
    return min(base + extra, 900)

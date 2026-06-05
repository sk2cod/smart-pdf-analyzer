# ============================================================
# utils/mermaid_renderer.py
# ============================================================
# Renders Mermaid diagram code as a live visual inside
# Streamlit using st.components.v1.html + Mermaid JS CDN.
#
# Usage:
#   from utils.mermaid_renderer import render_mermaid, extract_mermaid_code
#
#   code = extract_mermaid_code(llm_output)   # strips fences
#   render_mermaid(code)                      # renders diagram
# ============================================================

import re
import streamlit as st
import streamlit.components.v1 as components


# ------------------------------------------------------------
# Code Extraction
# Strips markdown fences the LLM may have added.
# Handles: ```mermaid ... ```, ``` ... ```, or bare code.
# ------------------------------------------------------------

def extract_mermaid_code(raw: str) -> str:
    """
    Extract clean Mermaid code from LLM output.

    The LLM sometimes wraps output in markdown fences even when
    the prompt says not to. This strips them reliably.

    Args:
        raw: Raw string from LLM, possibly fenced.

    Returns:
        Clean Mermaid code string ready for rendering.
    """
    # Match ```mermaid ... ``` or ``` ... ```
    fence_pattern = re.compile(
        r"```(?:mermaid)?\s*\n?(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    match = fence_pattern.search(raw)
    if match:
        return match.group(1).strip()

    # No fences — return as-is (already clean)
    return raw.strip()


# ------------------------------------------------------------
# Mermaid Renderer
# Injects Mermaid JS via CDN into an HTML component.
# The diagram replaces the code block entirely (per spec).
# ------------------------------------------------------------

def render_mermaid(mermaid_code: str, height: int = 500) -> None:
    """
    Render a Mermaid diagram live inside Streamlit.

    Injects the Mermaid JS library via CDN into an HTML
    component and renders the diagram inline — no external
    tool or copy-paste required.

    Args:
        mermaid_code: Clean Mermaid syntax string.
        height:       Height in pixels for the HTML component.
                      Flowcharts typically need 400-600px.
                      Mindmaps may need 500-700px.
    """
    escaped = mermaid_code.replace("`", "&#96;").replace("\\", "\\\\")

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
        <div class="mermaid" id="mermaid-diagram">
{mermaid_code}
        </div>
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
            el.innerHTML = svg;
          }} catch (err) {{
            const errBox = document.getElementById('error-box');
            errBox.style.display = 'block';
            errBox.textContent = 'Diagram render error: ' + err.message + '\\n\\nMermaid code:\\n' + code;
            el.innerHTML = '';
          }}
        }}

        renderDiagram();
      </script>
    </body>
    </html>
    """

    components.html(html, height=height, scrolling=True)


# ------------------------------------------------------------
# Auto-height helper
# Estimates a sensible height based on diagram type and
# number of lines in the code, so caller doesn't have to guess.
# ------------------------------------------------------------

def estimate_height(mermaid_code: str) -> int:
    """
    Estimate a sensible component height for the diagram.

    Args:
        mermaid_code: Clean Mermaid code string.

    Returns:
        Height in pixels.
    """
    lines = mermaid_code.strip().splitlines()
    line_count = len(lines)
    first_line = lines[0].lower() if lines else ""

    # Mindmaps tend to be wider/taller than flowcharts
    if "mindmap" in first_line:
        base = 500
    else:
        base = 420

    # Add headroom for larger diagrams
    extra = max(0, (line_count - 10) * 18)
    return min(base + extra, 900)   # cap at 900px

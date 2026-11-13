"""Export a conversation as Markdown, HTML or plain text."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass

FORMATS = ("md", "html", "txt")


@dataclass
class ExportMessage:
    name: str
    text: str
    is_user: bool
    time: str = ""  # "YYYY-MM-DD HH:MM"


def render(fmt: str, title: str, subtitle: str, messages: Sequence[ExportMessage]) -> str:
    if fmt == "md":
        return render_markdown(title, subtitle, messages)
    if fmt == "html":
        return render_html(title, subtitle, messages)
    if fmt == "txt":
        return render_text(title, subtitle, messages)
    raise ValueError(f"unknown export format {fmt!r} (use {', '.join(FORMATS)})")


def render_markdown(title: str, subtitle: str, messages: Sequence[ExportMessage]) -> str:
    lines = [f"# {title}", ""]
    if subtitle:
        lines += [f"*{subtitle}*", ""]
    for m in messages:
        stamp = f" · {m.time}" if m.time else ""
        lines += [f"**{m.name}**{stamp}", "", m.text.strip(), "", "---", ""]
    return "\n".join(lines).rstrip("-\n") + "\n"


def render_text(title: str, subtitle: str, messages: Sequence[ExportMessage]) -> str:
    out = [title, subtitle, ""] if subtitle else [title, ""]
    for m in messages:
        out += [f"{m.name}:", m.text.strip(), ""]
    return "\n".join(out)


_ACTION = re.compile(r"\*([^*\n]+)\*")
_SPEECH = re.compile(r"(&quot;[^&\n]*?&quot;|“[^”\n]*”)")


def _html_body(text: str) -> str:
    """Escape, then style *actions* and "speech"; paragraphs on blank lines."""
    escaped = html.escape(text.strip())
    escaped = _ACTION.sub(r'<em class="action">\1</em>', escaped)
    escaped = _SPEECH.sub(r'<span class="speech">\1</span>', escaped)
    paragraphs = re.split(r"\n\s*\n", escaped)
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip())


HTML_TEMPLATE = """<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg:#fafaf9; --fg:#292524; --dim:#78716c; --accent:#2563eb; --user:#15803d;
           --action:#78716c; --line:#e7e5e4; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#1c1917; --fg:#e7e5e4; --dim:#a8a29e; --accent:#93c5fd; --user:#a7c4a0;
             --action:#a8a29e; --line:#292524; }}
  }}
  body {{ background:var(--bg); color:var(--fg); margin:0;
         font:16px/1.65 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width:42rem; margin:0 auto; padding:2rem 1rem 4rem; }}
  h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
  .subtitle {{ color:var(--dim); margin:0 0 2rem; }}
  .msg {{ border-top:1px solid var(--line); padding:1rem 0; }}
  .name {{ font-weight:600; color:var(--accent); }}
  .msg.user .name {{ color:var(--user); }}
  .time {{ color:var(--dim); font-size:.85rem; margin-left:.5rem; }}
  .action {{ color:var(--action); }}
  .speech {{ font-weight:500; }}
  p {{ margin:.5rem 0; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>
{messages}
</main>
</body>
</html>
"""


def render_html(
    title: str, subtitle: str, messages: Sequence[ExportMessage], lang: str = "en"
) -> str:
    blocks = []
    for m in messages:
        stamp = f'<span class="time">{html.escape(m.time)}</span>' if m.time else ""
        cls = "msg user" if m.is_user else "msg"
        blocks.append(
            f'<div class="{cls}"><div><span class="name">{html.escape(m.name)}</span>{stamp}'
            f"</div>{_html_body(m.text)}</div>"
        )
    return HTML_TEMPLATE.format(
        lang=lang,
        title=html.escape(title),
        subtitle=html.escape(subtitle),
        messages="\n".join(blocks),
    )

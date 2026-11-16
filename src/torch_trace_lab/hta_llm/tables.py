"""Render rows as GitHub-flavoured markdown tables (no ``tabulate`` dependency)."""


def _cell(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if value != int(value) else f"{value:,.0f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render(rows: list[dict], columns: list[str] | None = None,
           headers: dict[str, str] | None = None) -> str:
    if not rows:
        return "_(no data)_"
    columns = columns or list(rows[0])
    headers = headers or {}
    numeric = {c: all(isinstance(r.get(c), (int, float)) or r.get(c) is None for r in rows)
               for c in columns}
    lines = ["| " + " | ".join(headers.get(c, c) for c in columns) + " |",
             "|" + "|".join("---:" if numeric[c] else "---" for c in columns) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_cell(r.get(c)) for c in columns) + " |")
    return "\n".join(lines)

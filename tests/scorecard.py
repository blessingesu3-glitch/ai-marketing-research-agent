"""Score saved research reports offline (no API calls).

Usage:
    python tests/scorecard.py old_report.json new_report.json

Each file is the JSON returned by POST /research (copy it from the browser's
Network tab or from `curl`). The scorecard counts what can be checked from the
text itself; it does not judge whether the research is *right*.
"""
import json
import re
import sys

URL = re.compile(r"https?://\S+")
FIGURE = re.compile(r"(?<![\w.])\d[\d,.]*\s*(%|x|×)?", re.I)
LABEL = re.compile(r"^\[([^\]]+)\]")


def score(report: dict) -> dict:
    findings = report.get("key_findings", [])
    claims = [f for f in findings if not f.startswith("[Not established]")]
    labels = {}
    for line in findings + report.get("opportunities", []) + report.get("recommendations", []):
        m = LABEL.match(line)
        key = m.group(1) if m else "unlabelled"
        labels[key] = labels.get(key, 0) + 1

    cited = [c for c in claims if URL.search(c)]
    with_figures = [c for c in claims if FIGURE.search(c)]
    figures_cited = [c for c in with_figures if URL.search(c)]
    recs = report.get("recommendations", [])
    recs_linked = [r for r in recs if "(Based on finding" in r]

    return {
        "findings (excluding 'Not established')": len(claims),
        "findings with an inline source link": f"{len(cited)}/{len(claims)}",
        "findings containing numbers": len(with_figures),
        "  ...of which have an inline source": f"{len(figures_cited)}/{len(with_figures)}",
        "'Not established' items": sum(1 for f in findings if f.startswith("[Not established]")),
        "recommendations linked to findings": f"{len(recs_linked)}/{len(recs)}",
        "sources listed": len(report.get("sources", [])),
        "labels used": labels,
    }


def main(paths):
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
        print(f"\n== {path}")
        for key, value in score(report).items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])

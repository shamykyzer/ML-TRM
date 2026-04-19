"""Publish docs/wandb_metrics_glossary.md as a wandb Report.

Usage:
    python scripts/publish_wandb_metrics_report.py               # defaults
    python scripts/publish_wandb_metrics_report.py --project TRM-LLM --entity shamykyzer

The report lives at wandb.ai/<entity>/<project>/reports/<slug> and can be
linked from any run in that project. Edit the source glossary file and
re-run to create a new version (wandb reports are versioned).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import wandb_workspaces.reports.v2 as wr  # noqa: E402
from wandb_workspaces.reports.v2.interface import _get_api  # noqa: E402

# wandb's create_project mutation returns 403 Forbidden when the project
# already exists under a non-admin API key. Report.save() unconditionally
# calls create_project before publishing, which blows up the whole save.
# Patch it to tolerate existing projects — we checked via api.projects()
# that the project already exists, so ensuring it is a no-op.
_api_class = _get_api().__class__
_orig_create_project = _api_class.create_project


def _tolerant_create_project(self, project, entity=None):  # noqa: ANN001
    try:
        return _orig_create_project(self, project, entity)
    except Exception:
        # Project already exists; that's fine for our purposes.
        return None


_api_class.create_project = _tolerant_create_project

GLOSSARY_PATH = REPO_ROOT / "docs" / "wandb_metrics_glossary.md"


def _split_into_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown doc into (heading, body) tuples at H1/H2 boundaries.

    Keeps the heading line attached to its body so each section renders as a
    self-contained MarkdownBlock in the report.
    """
    lines = markdown.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        if line.startswith("# ") or line.startswith("## "):
            # New section — flush the previous one
            if current_heading or current_body:
                sections.append((current_heading, current_body))
            current_heading = line
            current_body = []
        else:
            current_body.append(line)

    if current_heading or current_body:
        sections.append((current_heading, current_body))

    return [(h, "\n".join(body).strip()) for h, body in sections]


def _build_blocks(glossary_md: str) -> list:
    """Convert the glossary markdown into a list of wandb Report blocks."""
    sections = _split_into_sections(glossary_md)

    blocks: list = [
        wr.H1(text="W&B Metrics Glossary — ML-TRM"),
        wr.P(
            text=(
                "Auto-generated from docs/wandb_metrics_glossary.md. "
                "Every metric logged by the four trainers is explained below, "
                "grouped by the same prefix wandb uses for panel sections. "
                "Edit the source markdown and re-run "
                "scripts/publish_wandb_metrics_report.py to refresh."
            )
        ),
        wr.HorizontalRule(),
        wr.TableOfContents(),
        wr.HorizontalRule(),
    ]

    for heading, body in sections:
        if not heading and not body:
            continue
        chunk = f"{heading}\n\n{body}".strip() if heading else body
        if chunk:
            blocks.append(wr.MarkdownBlock(text=chunk))

    return blocks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # `or` (not default=) because .env may set these to empty strings which
    # os.getenv's default doesn't treat as missing.
    parser.add_argument(
        "--entity",
        default=os.getenv("TRM_WANDB_ENTITY") or "shamykyzer",
        help="wandb entity (default: $TRM_WANDB_ENTITY or shamykyzer)",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("TRM_WANDB_PROJECT") or "TRM-LLM",
        help="wandb project to publish the report into (default: TRM-LLM)",
    )
    parser.add_argument(
        "--title",
        default="W&B Metrics Glossary — ML-TRM",
        help="Report title (also used as slug suffix)",
    )
    parser.add_argument(
        "--description",
        default=(
            "Reference for every metric logged by trainer_trm, trainer_official, "
            "trainer_llm, and trainer_distill. Explains units, what to watch for, "
            "and how to read overfitting signals."
        ),
    )
    args = parser.parse_args(argv)

    if not GLOSSARY_PATH.is_file():
        print(f"ERROR: glossary not found at {GLOSSARY_PATH}", file=sys.stderr)
        return 1

    glossary = GLOSSARY_PATH.read_text(encoding="utf-8")
    print(f"[publish] loaded {len(glossary)} chars from {GLOSSARY_PATH.name}")

    blocks = _build_blocks(glossary)
    print(f"[publish] built {len(blocks)} report blocks")

    report = wr.Report(
        entity=args.entity,
        project=args.project,
        title=args.title,
        description=args.description,
        blocks=blocks,
    )

    print(f"[publish] saving to {args.entity}/{args.project} ...")
    report.save()

    print()
    print("REPORT PUBLISHED")
    print(f"URL:     {report.url}")
    print(f"Entity:  {args.entity}")
    print(f"Project: {args.project}")
    print(f"Blocks:  {len(blocks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

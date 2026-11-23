"""Compatibility entry point for the supplementary water-quality figures."""

from pathlib import Path
import runpy


FIGURE_SCRIPT = Path(__file__).resolve().parents[1] / (
    "Figure_V1/SFigure/SFigure23_26_WQ_Background/V_wq_background.py"
)


def main():
    runpy.run_path(str(FIGURE_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()

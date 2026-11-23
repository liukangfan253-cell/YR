"""Draw Figures S23–S26 from the supplied water-quality summary tables."""

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PLOT_SOURCE = ROOT / "Code/06_WQ_Background_Analysis/02_WQ_Background_Plot.py"
SUMMARY_FOLDERS = {
    "A02": "02_Longterm_Background",
    "A03": "03_Event_NonEvent",
    "A05": "05_Low_DO",
}
REQUIRED = (
    "A02_Station_5Year_WQ_Statistics.csv",
    "A02_Reach_5Year_WQ_Statistics.csv",
    "A03_Station_Event_NonEvent_WQ.csv",
    "A03_Reach_Matched_WQ_Changes.csv",
    "A05_Station_DO_Group_WQ.csv",
)
FIGURE_NAMES = {
    "Fig_A1_Longterm_WQ_Background": "Figure_S23_Longterm_WQ_Background_M1",
    "Fig_A3_Matched_Event_WQ_Changes": "Figure_S24",
    "Fig_A4_Low_DO_WQ_Associations": "Figure_S25",
    "Fig_A4b_Downstream_Low_DO_WQ": "Figure_S26",
}


def find_input(input_dir, filename):
    """Locate a summary in the supplied flat layout or an archived grouped layout."""
    flat = input_dir / filename
    if flat.is_file():
        return flat
    grouped = input_dir / SUMMARY_FOLDERS[filename[:3]] / filename
    if grouped.is_file():
        return grouped
    raise FileNotFoundError(str(flat))


def draw_figures(inputs, output_dir):
    spec = importlib.util.spec_from_file_location("wq_plot", PLOT_SOURCE)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    plot.FIGURES = output_dir
    # The package uses the plotting module's built-in panel geometry check.
    plot.ALIGNMENT_HELPER = output_dir / "_optional_alignment_helper_not_included.py"
    read_summary = plot.read
    save_figure = plot.save

    def read_packaged_summary(_folder, filename):
        return read_summary(inputs[filename].parent, filename)

    def save_supplementary_figure(fig, name):
        if name in FIGURE_NAMES:
            save_figure(fig, FIGURE_NAMES[name])
        else:
            plot.plt.close(fig)

    plot.read = read_packaged_summary
    plot.save = save_supplementary_figure
    plot.configure()
    plot.longterm(method="M1")
    plot.event_non_event()
    plot.low_do()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "Supplementary_Data/S10")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Result/Distributed_S10_Figures")
    parser.add_argument(
        "--check-inputs", action="store_true",
        help="Check the summary files without importing plotting libraries or writing outputs.",
    )
    args = parser.parse_args()
    inputs = {}
    missing = []
    for filename in REQUIRED:
        try:
            inputs[filename] = find_input(args.input_dir, filename)
        except FileNotFoundError:
            missing.append(filename)
    if missing:
        parser.error("Missing summary inputs: " + ", ".join(missing))
    if args.check_inputs:
        print(json.dumps({
            "Input_Files": len(inputs),
            "Status": "PRESENT",
            "Statistical_Recalculation": False,
            "Figures_Redrawn": False,
        }))
        return

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        parser.error("Choose an output directory outside the summary input directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        parser.error("Choose an empty output directory to preserve existing figures.")
    draw_figures(inputs, output_dir)
    print(f"Figures S23–S26 saved to {output_dir}")


if __name__ == "__main__":
    main()

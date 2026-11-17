"""Populate the workflow slide's event and observation counts from this run."""

from pathlib import Path
from zipfile import ZipFile

from defusedxml.minidom import parseString
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
TEMPLATE = Path(__file__).resolve().parents[1] / 'Assets/Figure1b_template.pptx'
SOURCE = ROOT / 'Result/Observation_only_references/20261111/20261111_实测来源核对.csv'
DESTINATION = ROOT / 'V/Figure_V1/Figure1/b/Images/Figure1b_Workflow.pptx'
SLIDE = 'ppt/slides/slide1.xml'
DRAWING_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
COUNT_LABELS = {
    '1,192': 'Heatwave_Only',
    '1,035': 'Flood_Only',
    '132': 'Direct_Compound',
    '102': 'H5F_Compound',
    '89': 'F5H_Compound',
}


def main():
    events = pd.read_csv(SOURCE, usecols=['Site_ID', 'Event_ID', 'Event_Type', 'Event_Days', 'Event_Observed_Days'])
    if events.empty or events.duplicated(['Site_ID', 'Event_ID']).any():
        raise ValueError('Expected one row per registered event')
    counts = events.Event_Type.value_counts()
    if set(counts.index) != set(COUNT_LABELS.values()):
        raise ValueError('Workflow requires all five event types')
    full = events.Event_Observed_Days.eq(events.Event_Days).sum()
    none = events.Event_Observed_Days.eq(0).sum()
    partial = len(events) - full - none
    labels = {text: str(counts[kind]) for text, kind in COUNT_LABELS.items()}
    labels['(1,484 full, 547 partial, 519 none)'] = f'({full} full, {partial} partial, {none} none)'

    with ZipFile(TEMPLATE) as source:
        slide = parseString(source.read(SLIDE))
        changed = set()
        for node in slide.getElementsByTagNameNS(DRAWING_NS, 't'):
            if node.firstChild is None:
                continue
            text = node.firstChild.data
            if text in labels:
                if text in changed:
                    raise ValueError(f'Count appears more than once in the template: {text}')
                node.firstChild.data = labels[text]
                changed.add(text)
        if changed != set(labels):
            raise ValueError(f'Workflow count text missing: {set(labels) - changed}')

        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(DESTINATION, 'w') as output:
            for member in source.infolist():
                content = slide.toxml(encoding='UTF-8') if member.filename == SLIDE else source.read(member)
                output.writestr(member, content)
    print(f'Saved {DESTINATION} with {len(events):,} event records')


if __name__ == '__main__':
    main()

"""Normalise published validation labels while accepting archived result tables."""
MASKED_METHOD = "Masked_BiLSTM_Attention"
UNMASKED_METHOD = "Unmasked_BiLSTM_Attention"
# Compatibility aliases are retained for historical Result/V tables, not emitted as new labels.
LEGACY_METHOD_LABELS = {
    "Masked_V1": MASKED_METHOD,
    "Unmasked_V1": UNMASKED_METHOD,
}
_GAP_CODES = {"1", "2-3", "1-3", "4-7", "8-14", "15-30", ">30"}


def normalize_validation_labels(frame):
    """Return a copy with descriptive method values and prediction column names."""
    for old, new in LEGACY_METHOD_LABELS.items():
        if old in frame.columns and new in frame.columns:
            raise ValueError("Ambiguous duplicate prediction columns: " + old + " and " + new)
    result = frame.rename(columns=LEGACY_METHOD_LABELS).copy()
    for column in ("Method", "Baseline"):
        if column in result.columns:
            result[column] = result[column].replace(LEGACY_METHOD_LABELS)
    return result


def gap_bin_code(value):
    """Return the interval code from either legacy or explicit-unit bin text."""
    text = str(value).strip().replace("\u2013", "-").replace("\u2014", "-")
    for suffix in (" days", " day", "\u5929"):
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()
            break
    if text == "1.0":
        text = "1"
    if text not in _GAP_CODES:
        raise ValueError("Unknown DO-gap length bin: " + str(value))
    return text


def format_gap_bin(value):
    """Keep CSV bins as English text so spreadsheet apps do not parse dates."""
    code = gap_bin_code(value)
    return code + (" day" if code == "1" else " days")

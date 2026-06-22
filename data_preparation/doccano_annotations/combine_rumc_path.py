import json
from pathlib import Path

basepath = None

filenames = ["annotations_muradije.jsonl", "annotations_joel.jsonl"]

# Load all entries from both files
all_entries = []
seen = set()


def clean_label_offsets(entry):
    """
    Adjusts label offsets to avoid leading/trailing whitespace in annotated spans.
    Adds debug print statements to trace changes.
    """
    text = entry["text"]
    cleaned_labels = []
    adjusted_count = 0

    for start, end, label in entry["label"]:
        original_start, original_end = start, end
        original_span = text[start:end]

        # Shift start forward if it's whitespace
        while start < end and text[start].isspace():
            start += 1
        # Shift end backward if it's whitespace
        while end > start and text[end - 1].isspace():
            end -= 1

        cleaned_span = text[start:end]
        cleaned_labels.append([start, end, label])

        if (original_start != start) or (original_end != end):
            adjusted_count += 1
            print(f"\n🔧 Adjusted label: {label}")
            print(
                f"    Original span ({original_start}, {original_end}): '{original_span}'"
            )
            print(f"    Cleaned  span ({start}, {end}): '{cleaned_span}'")

    entry["label"] = cleaned_labels
    if adjusted_count > 0:
        print(f"✅ Adjusted {adjusted_count} label(s) for UID: {entry.get('uid')}")
    return entry


for fname in filenames:
    with open(basepath / fname, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            # Clean label offsets to avoid leading/trailing whitespace
            data = clean_label_offsets(data)

            # Use a unique identifier. Adjust as needed.
            identifier = data.get("report_id") or hash(data.get("text", ""))

            if identifier not in seen:
                seen.add(identifier)
                all_entries.append(data)


print(len(all_entries), "unique entries loaded.")

# Optionally save combined data
combined_path = basepath / "rumc_pathology_reports_annotated.jsonl"
with open(combined_path, "w", encoding="utf-8") as f:
    for entry in all_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Combined file saved to: {combined_path}")

# Sanity check

# Load combined pathology
combined_pathology_path = basepath / "rumc_pathology_reports_annotated.jsonl"
with open(combined_pathology_path, "r", encoding="utf-8") as f:
    pathology_entries = [json.loads(line) for line in f]

# Check that the uids are unique
uids = [entry["uid"] for entry in pathology_entries]
unique_uids = set(uids)
if len(unique_uids) != len(uids):
    print("Warning: Duplicate uids found in combined pathology data!")
# Print the number of unique entries
print(f"Number of unique entries in combined pathology data: {len(unique_uids)}")

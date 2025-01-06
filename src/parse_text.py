import re

# For parsing txt file
def parse_txt_file(txt_path):
    segments = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\[(\d+\.\d+),(\d+\.\d+)\]\s+\w+\s+\w+\s+(.+)", line)
            if match:
                # print("matched")
                start, end, text = float(match[1]), float(match[2]), match[3]
                segments.append({"start": start, "end": end, "text": text.strip()})
    return segments
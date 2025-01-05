# Function to do the parsing of the transcription file
def parse_transcription_file(transcription_file):
    segments = []
    with open(transcription_file, "r") as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                start, end = map(float, parts[0].strip('[]').split(','))
                speaker_id = parts[1]
                gender = parts[2]
                transcription = parts[3]
                segments.append({
                    'start': start,
                    'end': end,
                    'speaker_id': speaker_id,
                    'gender': gender,
                    'transcription': transcription
                })
    return segments
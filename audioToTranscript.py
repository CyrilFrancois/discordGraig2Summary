import os
import zipfile
import tempfile
import shutil
import re
import datetime
import glob
import sys
from pydub import AudioSegment
import torch
import whisper
import json
from dotenv import load_dotenv

load_dotenv()

def log(msg):
    print(f"[LOG] {msg}")


def get_most_recent_zip():
    zips = glob.glob(os.path.join(os.getenv("AUDIO_DIR"), "*.zip"))
    if not zips:
        return None
    return max(zips, key=os.path.getmtime)


def extract_zip_to_temp(zip_path):
    temp_dir = tempfile.mkdtemp(prefix="audio_extract_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    return temp_dir


def parse_info_file(temp_dir):
    info_path = None
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith("info.txt"):
                info_path = os.path.join(root, f)
                break

    if info_path is None:
        return None, None

    with open(info_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract start time (ISO format in the file)
    start_time_match = re.search(r"Start time:\s*([0-9TZ:\-\.]+)", content)
    if not start_time_match:
        return None, None
    start_dt = datetime.datetime.fromisoformat(start_time_match.group(1).replace("Z", "+00:00"))

    # Extract names (under "Tracks:")
    names = re.findall(r"\n\s*(.+?#0)", content)
    people = []
    for n in names:
        # Remove trailing IDs inside parentheses
        cleaned = re.sub(r"\s*\([0-9]+\)", "", n).strip()
        people.append(cleaned)

    return start_dt, people


def transcribe_audio_file(model, file_path):
    # Whisper accepts many formats; pydub only needed if conversion is required.
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path + ".wav"
    audio.export(wav_path, format="wav")

    #For autodetection of the language: result = model.transcribe(wav_path, verbose=False)
    print(os.getenv("TARGET_LANG"))
    result = model.transcribe(wav_path, verbose=False, language=os.getenv("TARGET_LANG"), task="transcribe")
    os.remove(wav_path)

    # Whisper outputs segments with timestamps; return timeline transcript.
    transcript = []
    for seg in result["segments"]:
        transcript.append((seg["start"], seg["text"].strip()))
    return transcript


def merge_transcripts(start_dt, by_speaker):
    merged = []
    for speaker, segments in by_speaker.items():
        txt_prev=''
        for (offset_sec, text) in segments:
            if(text!=txt_prev and text!="..." and text!=""):
                timestamp = start_dt + datetime.timedelta(seconds=offset_sec)
                merged.append((timestamp, speaker, text))
            txt_prev=text

    merged.sort(key=lambda x: x[0])

    output_lines = []
    prev_line=''
    SPEAKER_MAP = json.loads(os.getenv("SPEAKER_MAP"))
    
    for ts, spk, line in merged:
        formatted_date = ts.strftime("%d/%m/%Y")
        formatted_time = f"{ts.hour:02d}h{ts.minute:02d}m{ts.second:02d}s"
        line = line.strip()
        
        if spk in SPEAKER_MAP:
            spk = SPEAKER_MAP[spk]

        if (line!="..." and line!="" and line!=prev_line):
            #output_lines.append(f"[{formatted_date}] {formatted_time} {spk} : {line}")
            output_lines.append(f"{spk} : {line}")
        prev_line= line

    return "\n".join(output_lines)


def audioToTranscript():
    log("Selecting most recent zip file...")
    zip_path = get_most_recent_zip()
    if not zip_path:
        log("No zip file found in ./audio")
        sys.exit(1)

    log(f"Using zip: {zip_path}")

    log("Extracting zip contents into temp folder...")
    temp_dir = extract_zip_to_temp(zip_path)

    try:
        log("Parsing info.txt...")
        start_dt, people = parse_info_file(temp_dir)
        if not start_dt or not people:
            log("info.txt missing or incomplete")
            sys.exit(1)    

        log(f"Meeting start: {start_dt}")
        log(f"Speakers found: {people}")
        #['Requester:\tcyril0431#0', 'cyril0431#0', 'radblue#0', 'kami1876#0', 'poulpikiller#0', 'klaonis#0']
        #['Requester:\tcyril0431#0', 'Lugh', 'DM', 'Ailouros', 'Lilie', 'Salazar']

        log("Loading Whisper model (local)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("small", device=device)

        log("Scanning audio files...")
        speaker_transcripts = {p: [] for p in people}

        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith(".aac"):
                    audio_file = os.path.join(root, f)

                    # Assign speaker based on name presence in filename
                    matched_speaker = None
                    for p in people:
                        p_clean = p.split("#")[0].lower()
                        if p_clean in f.lower():
                            matched_speaker = p
                            break

                    if matched_speaker is None:
                        log(f"Warning: cannot match {f} with any speaker; skipping")
                        continue

                    log(f"Transcribing: {f} as speaker {matched_speaker}")
                    try:
                        segments = transcribe_audio_file(model, audio_file)
                        speaker_transcripts[matched_speaker].extend(segments)
                    except Exception as e:
                        log(f"Error transcribing {f}: {e}")

        log("Merging transcripts chronologically...")
        final_text = merge_transcripts(start_dt, speaker_transcripts)

        if not os.path.exists(os.getenv("TRANSCRIPTS_DIR")):
            os.makedirs(os.getenv("TRANSCRIPTS_DIR"))

        output_name = start_dt.strftime("%Y%m%d_%H%M%S") + "_transcripts.txt"
        output_path = os.path.join(os.getenv("TRANSCRIPTS_DIR"), output_name)

        log(f"Writing final transcript to {output_path}")
        with open(output_path, "w+", encoding="utf-8") as f:
            f.write(final_text)

        log("Done.")

    finally:
        log("Cleaning up temp folder...")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    audioToTranscript()

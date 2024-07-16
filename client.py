from whisper_live.client import TranscriptionClient
import json

class EnhancedTranscriptionClient(TranscriptionClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_message(self, message):
        data = json.loads(message)
        print(f"Received message: {data}")  # Debug print
        if 'segments' in data:
            for segment in data['segments']:
                print(f"Transcription: {segment['text']}")
        if 'llm_results' in data:
            for result in data['llm_results']:
                print(f"LLM Result: {result}")

client = EnhancedTranscriptionClient(
    "localhost",
    9090,
    lang="en",
    translate=False,
    model="small",
    use_vad=False,
    save_output_recording=True,
    output_recording_filename="./output_recording.wav"
)

client()

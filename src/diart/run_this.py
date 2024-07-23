import logging
import traceback
import diart.operators as dops
import rich
# import rx.operators as ops
from rx import operators as ops
from diart import SpeakerDiarization , SpeakerDiarizationConfig 
from diart.sources import MicrophoneAudioSource, WavFileSimulatedMicrophoneAudioSource
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import nemo.collections.asr as nemo_asr
from diart.models import EmbeddingModel, SegmentationModel
from pyannote.audio import Model
import diart.models as m

processor = Wav2Vec2Processor.from_pretrained("othrif/wav2vec2-large-xlsr-egyptian")
model = Wav2Vec2ForCTC.from_pretrained("othrif/wav2vec2-large-xlsr-egyptian")

import torch
import os
import sys
import numpy as np
# import whisper_timestamped as whisper
from pyannote.core import Segment
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    # Auxiliary function to suppress Whisper logs (it is quite verbose)
    # All credit goes to: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class Wav2Vec2Transcriber:
    def __init__(self, model_name="othrif/wav2vec2-large-xlsr-egyptian"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self._buffer = ""

    def transcribe(self, waveform):
        """Transcribe audio using Wav2Vec2"""
        # Convert waveform to expected format
        waveform = np.array(waveform.data.astype("float32")).reshape(-1)
        #print(f"waveform {waveform} shape {waveform.shape}")
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        #print(f"inputs {inputs.input_values.shape}")
        
        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        #print(f"output {transcription}")
        return {"text": transcription}

    def identify_speakers(self, transcription, diarization, time_shift):
        """Iterate over transcription segments to assign speakers"""
        speaker_captions = []
        # Split the transcription text into segments based on diarization
        segments = transcription.split('.')
        for segment in segments:
            if not segment.strip():
                continue

            # Assign a speaker to the segment based on diarization
            start = time_shift
            end = time_shift + len(segment.split())
            dia = diarization.crop(Segment(start, end))

            speakers = dia.labels()
            num_speakers = len(speakers)
            if num_speakers == 0:
                caption = (-1, segment)
            elif num_speakers == 1:
                spk_id = int(speakers[0].split("speaker")[1])
                caption = (spk_id, segment)
            else:
                max_speaker = int(np.argmax([
                    dia.label_duration(spk) for spk in speakers
                ]))
                caption = (max_speaker, segment)
            speaker_captions.append(caption)

        return speaker_captions

    def __call__(self, diarization, waveform):
        # Step 1: Transcribe
        #print(f"start ASR model")
        transcription = self.transcribe(waveform)
        #print(f"finished transcript")
        # Update transcription buffer
        self._buffer += transcription["text"]
        # The audio may not be the beginning of the conversation
        time_shift = waveform.sliding_window.start
        #print(f"time shifted")
        # Step 2: Assign speakers
        speaker_transcriptions = self.identify_speakers(transcription["text"], diarization, time_shift)
        return speaker_transcriptions

import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow

def splitter(x):
    print("splitter",np.array(np.array(x)))
    return x

def colorize_transcription(transcription):
    """
    Unify a speaker-aware transcription represented as
    a list of `(speaker: int, text: str)` pairs
    into a single text colored by speakers.
    """
    colors = 2 * [
        "bright_red", "bright_blue", "bright_green", "orange3", "deep_pink1",
        "yellow2", "magenta", "cyan", "bright_magenta", "dodger_blue2"
    ]
    result = []
    for speaker, text in transcription:
        if speaker == -1:
            # No speakerfound for this text, use default terminal color
            result.append(text)
        else:
            result.append(f"[{colors[speaker]}]{text}")
    return "\n".join(result)

def print_output(x):
    print()
    return x

########################################




# Suppress whisper-timestamped warnings for a clean output
logging.getLogger("whisper_timestamped").setLevel(logging.ERROR)

#def segmentation_loader():
#    return Model.from_pretrained("pyannote/segmentation-3.0")
#def embedding_loader():
#    return nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
#embedding = EmbeddingModel(embedding_loader)
#segmentation = SegmentationModel(segmentation_loader)

config = SpeakerDiarizationConfig (
    duration=5, # tried to set it to small duration but VAD model couldn't detect activity correctly
    step=1,
    latency="min",
    tau_active=0.5,
    rho_update=0.1,
    delta_new=0.57,
    segmentation=m.SegmentationModel.from_pretrained('pyannote/segmentation-3.0'),
    embedding=m.EmbeddingModel.from_pretrained('nvidia/speakerverification_en_titanet_large'),
    device=torch.device("cuda")

)
dia = SpeakerDiarization (config)
wav_file = "/kaggle/working/diart_modified/src/diart/audio.wav"
source = WavFileSimulatedMicrophoneAudioSource(wav_file,0.1)

asr = Wav2Vec2Transcriber()

transcription_duration = 2
batch_size = int(transcription_duration // config.step)
sample_rate = 16000
window = 5
step_size = 0.5
chunk_size = int(sample_rate * step_size)  
required_length = sample_rate * window

def update_accumulated_data(acc, new_data):
    acc = np.concatenate((acc, new_data))
    if len(acc) > required_length:
        acc = acc[-required_length:]  # Keep only the latest `required_length` samples
    return acc

# Function to accumulate data until it reaches chunk_size
def accumulate_chunks(acc, new_data):
    acc['buffer'] = np.concatenate((acc['buffer'], new_data))
    if len(acc['buffer']) >= chunk_size:
        acc['data'] = acc['buffer'][:chunk_size]
        acc['buffer'] = acc['buffer'][chunk_size:]
    else:
        acc['data'] = np.array([])
    return acc
    
def process_accumulated_data(data):
    # Process the accumulated data (e.g., print its length)
    print(f"Processing data of length: {len(data)}")
    return data

source.stream.pipe(
    ops.scan(accumulate_chunks, seed={'buffer': np.array([]), 'data': np.array([])}),  # Accumulate data until chunk_size
    ops.filter(lambda acc: len(acc['data']) > 0),  # Filter only when a new chunk is ready
    ops.map(lambda acc: acc['data']),  # Extract the new chunk from the accumulator
    ops.scan(lambda acc, x: update_accumulated_data(acc, x), seed=np.array([])),  # Update the accumulated data
    ops.filter(lambda x: len(x) >= required_length),  # Filter only when data length is >= required_length
    ops.map(print_output),
    # ops.map(print2),
    ops.map(dia),
    ops.map(splitter),
    # ops.filter(lambda ann_wav: ann_wav[0].get_timeline().duration() > 0),
    # ops.starmap(asr),
    # ops.map(colorize_transcription),
).subscribe(on_next=rich.print, on_error=lambda _: traceback.print_exc())

print("Listening...")
source.read()

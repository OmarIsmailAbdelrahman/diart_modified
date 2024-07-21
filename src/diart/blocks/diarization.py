from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.diarization import DiarizationErrorRate
from typing_extensions import Literal

from . import base
from .aggregation import DelayedAggregation
from .clustering import OnlineSpeakerClustering
from .embedding import OverlapAwareSpeakerEmbedding
from .segmentation import SpeakerSegmentation
from .utils import Binarize
from .. import models as m

########################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import yaml
import torch
from pyannote.audio import Inference

from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_frame_pred,
    generate_vad_segment_table,
    init_vad_model,
    prepare_manifest,
)
config_path = "/kaggle/working/diart_modified/src/diart/vad_config.yaml"
with open(config_path, 'r') as file:
    cfg = yaml.safe_load(file)

vad_model = init_vad_model("vad_multilingual_marblenet")
vad_model.eval()
vad_model.to(device)

import librosa

def prepare_input_from_array(audio):
    # Extract 64 MFCC features
    temps = audio.to('cpu').numpy()
    print(f"audio {audio.shape}")
    mfcc = np.array([librosa.feature.melspectrogram(y=temp, sr=16000, n_mels=2400, n_fft=400, hop_length=160) for temp in temps])
    print(f"mfcc {mfcc.shape}")
    return torch.from_numpy(mfcc).to('cuda')

def convert_vad_into_timestamp(audio,model_output):
    audio_data = audio.cpu().numpy()
    model_output_np = model_output.detach().cpu().numpy()
    sr = 16000
    samples_per_frame = len(audio_data) // model_output_np.shape[0]

    # Initialize an array to hold the probabilities for each timestamp
    prob_sums = np.zeros(len(audio_data))
    counts = np.zeros(len(audio_data))

    # Assign probabilities to each frame
    for i in range(model_output_np.shape[0]):
        start_idx = i * samples_per_frame
        end_idx = start_idx + samples_per_frame
        prob_speech = model_output_np[i]  # Probability of speech for this frame
        prob_sums[start_idx:end_idx] += prob_speech
        counts[start_idx:end_idx] += 1
    
    # Ensure the last part of the signal is covered
    if end_idx < len(audio_data):
        prob_sums[end_idx:] += model_output_np[-1]
        counts[end_idx:] += 1
    
    # Calculate the mean probability for each timestamp
    probabilities = prob_sums / counts
    
    # The probabilities array now holds the speech probabilities for each timestamp in the original signal
    print(f"Legendary-convert_vad_into_timestamp probabilities {probabilities} probabilities.shape {probabilities.shape} samples_per_frame {samples_per_frame} audio shape {audio.shape}")
    np.save('probabilities.npy', probabilities)
    return probabilities
########################################################################################


class SpeakerDiarizationConfig(base.PipelineConfig):
    def __init__(
        self,
        segmentation: m.SegmentationModel | None = None,
        embedding: m.EmbeddingModel | None = None,
        duration: float = 5,
        step: float = 0.5,
        latency: float | Literal["max", "min"] | None = None,
        tau_active: float = 0.6,
        rho_update: float = 0.3,
        delta_new: float = 1,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
        normalize_embedding_weights: bool = False,
        device: torch.device | None = None,
        sample_rate: int = 16000,
        **kwargs,
    ):
        # Default segmentation model is pyannote/segmentation
        self.segmentation = segmentation or m.SegmentationModel.from_pyannote(
            "pyannote/segmentation"
        )

        # Default embedding model is pyannote/embedding
        self.embedding = embedding or m.EmbeddingModel.from_pyannote(
            "pyannote/embedding"
        )

        self._duration = duration
        self._sample_rate = sample_rate

        # Latency defaults to the step duration
        self._step = step
        self._latency = latency
        if self._latency is None or self._latency == "min":
            self._latency = self._step
        elif self._latency == "max":
            self._latency = self._duration

        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers
        self.normalize_embedding_weights = normalize_embedding_weights
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def step(self) -> float:
        return self._step

    @property
    def latency(self) -> float:
        return self._latency

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class SpeakerDiarization(base.Pipeline):
    def __init__(self, config: SpeakerDiarizationConfig | None = None):
        self._config = SpeakerDiarizationConfig() if config is None else config

        msg = f"Latency should be in the range [{self._config.step}, {self._config.duration}]"
        assert self._config.step <= self._config.latency <= self._config.duration, msg
        print(f"Legendary-SpeakerDiarization- segmentation config {self._config.segmentation} ")
        self.segmentation = SpeakerSegmentation(
            self._config.segmentation, self._config.device
        )
        self.embedding = OverlapAwareSpeakerEmbedding(
            self._config.embedding,
            self._config.gamma,
            self._config.beta,
            norm=1,
            normalize_weights=self._config.normalize_embedding_weights,
            device=self._config.device,
        )
        self.pred_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="hamming",
            cropping_mode="loose",
        )
        self.audio_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="first",
            cropping_mode="center",
        )
        self.binarize = Binarize(self._config.tau_active)

        # Internal state, handle with care
        self.timestamp_shift = 0
        self.clustering = None
        self.chunk_buffer, self.pred_buffer = [], []
        self.reset()

    @staticmethod
    def get_config_class() -> type:
        return SpeakerDiarizationConfig

    @staticmethod
    def suggest_metric() -> BaseMetric:
        return DiarizationErrorRate(collar=0, skip_overlap=False)

    @staticmethod
    def hyper_parameters() -> Sequence[base.HyperParameter]:
        return [base.TauActive, base.RhoUpdate, base.DeltaNew]

    @property
    def config(self) -> SpeakerDiarizationConfig:
        return self._config

    def set_timestamp_shift(self, shift: float):
        self.timestamp_shift = shift

    def reset(self):
        self.set_timestamp_shift(0)
        self.clustering = OnlineSpeakerClustering(
            self.config.tau_active,
            self.config.rho_update,
            self.config.delta_new,
            "cosine",
            self.config.max_speakers,
        )
        self.chunk_buffer, self.pred_buffer = [], []

    def __call__(
        self, waveforms: Sequence[SlidingWindowFeature]
    ) -> Sequence[tuple[Annotation, SlidingWindowFeature]]:
        """Diarize the next audio chunks of an audio stream.

        Parameters
        ----------
        waveforms: Sequence[SlidingWindowFeature]
            A sequence of consecutive audio chunks from an audio stream.

        Returns
        -------
        Sequence[tuple[Annotation, SlidingWindowFeature]]
            Speaker diarization of each chunk alongside their corresponding audio.
        """
        #for wave in waveforms:
        #    print(f"legendary-SpeakerDiarization-__call__ wave size {wave.data.squeeze().shape} for vad shape {wave.data.squeeze().reshape(-1).shape}") 
        batch_size = len(waveforms)
        msg = "Pipeline expected at least 1 input"
        assert batch_size >= 1, msg 
        # Create batch from chunk sequence, shape (batch, samples, channels)
        batch = torch.stack([torch.from_numpy(w.data) for w in waveforms])

        expected_num_samples = int(
            np.rint(self.config.duration * self.config.sample_rate)
        )
        msg = f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"
        assert batch.shape[1] == expected_num_samples, msg

        ############################################################
        signal = batch.reshape(batch_size,-1).to(torch.float).to(device)
        print(f"Legendary signal shape {signal.shape}")
        temp = prepare_input_from_array(signal).permute(0, 2, 1)
        print(f"Legendary temp shape {temp.shape}")[4,501, 2400]
        input_signal , input_signal_length = torch.tensor(temp).reshape(batch_size,temp.shape[0],-1), torch.full((temp.shape[0], temp.shape[1]), temp.size(2)).long()
        #input_signal_length = [x.shape[0] for x in input_signal]
        print(f"legendary-SpeakerDiarization-__call__ processed_signal  {input_signal.shape} processed_signal_length {input_signal_length} shape {input_signal_length.shape}")
        vad_output = vad_model(processed_signal=input_signal,processed_signal_length=input_signal_length)
        
        print(f"legendary-SpeakerDiarization-__call__ VAD vad_output {vad_output} shape {vad_output.shape} ")
        probs = torch.softmax(vad_output, dim=-1)
        pred = probs[:, 0]
        print(f"legendary-SpeakerDiarization-__call__ VAD vad_output probs {probs} shape {probs.shape} pred {pred} shape {pred.shape} ")
        vad_timestamp_results = convert_vad_into_timestamp(signal,pred)
        ############################################################
        
        #segmentations = torch.max(self.segmentation(batch),axis=2)  # shape (batch, frames, speakers)
        segmentations = self.segmentation(batch)
        # embeddings has shape (batch, speakers, emb_dim)
        embeddings = self.embedding(batch, segmentations)
        seg_resolution = waveforms[0].extent.duration / segmentations.shape[1]
        # s = segmentations.numpy()
        #print(f"legendary-SpeakerDiarization-__call__ batch {batch.shape} segmentation unique values {np.unique(s)} segmentation {s.shape} reduce to (batch,time) {np.max(s, axis=2).shape} {np.max(s, axis=2)} number of 1: {np.sum(np.max(s, axis=2),axis=1)} embedding {s.shape}")
        print(f"legendary-SpeakerDiarization-__call__ batch {batch.shape} segmentations {segmentations.shape} embeddings {embeddings.shape}")

        outputs = []
        for wav, seg, emb in zip(waveforms, segmentations, embeddings):
            
            print(f"legendary-SpeakerDiarization-__call__ wav {wav.data.shape} seg {seg.shape} emb {emb.shape}")
            
            # Add timestamps to segmentation
            sw = SlidingWindow(
                start=wav.extent.start,
                duration=seg_resolution,
                step=seg_resolution,
            )
            seg = SlidingWindowFeature(seg.cpu().numpy(), sw)

            # Update clustering state and permute segmentation
            permuted_seg = self.clustering(seg, emb)
            print(f"legendary-SpeakerDiarization-__call__ permuted_seg {permuted_seg}")
            # Update sliding buffer
            self.chunk_buffer.append(wav)
            self.pred_buffer.append(permuted_seg)

            # Aggregate buffer outputs for this time step
            agg_waveform = self.audio_aggregation(self.chunk_buffer)
            agg_prediction = self.pred_aggregation(self.pred_buffer)
            agg_prediction = self.binarize(agg_prediction)
            
            # Shift prediction timestamps if required
            if self.timestamp_shift != 0:
                shifted_agg_prediction = Annotation(agg_prediction.uri)
                for segment, track, speaker in agg_prediction.itertracks(
                    yield_label=True
                ):
                    new_segment = Segment(
                        segment.start + self.timestamp_shift,
                        segment.end + self.timestamp_shift,
                    )
                    shifted_agg_prediction[new_segment, track] = speaker
                agg_prediction = shifted_agg_prediction

            outputs.append((agg_prediction, agg_waveform))

            # Make place for new chunks in buffer if required
            if len(self.chunk_buffer) == self.pred_aggregation.num_overlapping_windows:
                self.chunk_buffer = self.chunk_buffer[1:]
                self.pred_buffer = self.pred_buffer[1:]

        return outputs

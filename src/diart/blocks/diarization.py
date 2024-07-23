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
global_offset = 0
# import yaml
# import torch
# from pyannote.audio import Inference

# from nemo.collections.asr.parts.utils.vad_utils import (
#     generate_overlap_vad_seq,
#     generate_vad_frame_pred,
#     generate_vad_segment_table,
#     init_vad_model,
#     prepare_manifest,
# )
# config_path = "/kaggle/working/change_vad_config.yaml"
# with open(config_path, 'r') as file:
#     cfg = yaml.safe_load(file)

# vad_model = init_vad_model("vad_multilingual_marblenet")
# vad_model.eval()
# vad_model.to(device)

# import librosa

# def prepare_input_from_array(audio):
#     # Extract 64 MFCC features
#     temp = audio.to('cpu').numpy()
#     print(f"audio {audio.shape}")
#     mfcc = librosa.feature.melspectrogram(y=temp, sr=16000, n_mels=2400, n_fft=400, hop_length=160)
#     print(f"mfcc {mfcc.shape}")
#     return torch.from_numpy(mfcc).to('cuda')

# def convert_vad_into_timestamp(audio,model_output):
#     audio_data = audio.cpu().numpy()
#     model_output_np = model_output.detach().cpu().numpy()
#     sr = 16000
#     samples_per_frame = len(audio_data) // model_output_np.shape[0]

#     # Initialize an array to hold the probabilities for each timestamp
#     prob_sums = np.zeros(len(audio_data))
#     counts = np.zeros(len(audio_data))

#     # Assign probabilities to each frame
#     for i in range(model_output_np.shape[0]):
#         start_idx = i * samples_per_frame
#         end_idx = start_idx + samples_per_frame
#         prob_speech = model_output_np[i]  # Probability of speech for this frame
#         prob_sums[start_idx:end_idx] += prob_speech
#         counts[start_idx:end_idx] += 1
    
#     # Ensure the last part of the signal is covered
#     if end_idx < len(audio_data):
#         prob_sums[end_idx:] += model_output_np[-1]
#         counts[end_idx:] += 1
    
#     # Calculate the mean probability for each timestamp
#     probabilities = prob_sums / counts
    
#     # The probabilities array now holds the speech probabilities for each timestamp in the original signal
#     print(f"Legendary-convert_vad_into_timestamp probabilities {probabilities} probabilities.shape {probabilities.shape} samples_per_frame {samples_per_frame} audio shape {audio.shape}")
#     np.save('probabilities.npy', probabilities)
#     return probabilities
from nemo.collections.asr.models import EncDecSpeakerLabelModel
# VAD model
from pyannote.audio import Model
import soundfile as sf
from pyannote.audio.pipelines import VoiceActivityDetection

model = Model.from_pretrained("pyannote/segmentation")
pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)

def get_vad_timestamps(audio):
    start = []
    end = []
    # save audio here    
    vad = pipeline({"waveform": audio.reshape(1,-1).to(torch.float), "sample_rate": 16000})
    for segment in vad._tracks:
        start.append(segment.start)
        end.append(segment.end)
    return start,end

import math
# Segmentation Module
def segment_audio(audio, start_times, end_times, sample_rate=16000):
    segments = []
    for start, end in zip(start_times, end_times):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segments.append((audio[start_sample:end_sample], start, end))
    return segments

def get_subsegments(offset: float, window: float, shift: float, duration: float):
    subsegments: List[Tuple[float, float]] = []
    start = offset
    slice_end = start + duration
    base = math.ceil((duration - window) / shift)
    slices = 1 if base < 0 else base + 1
    for slice_id in range(slices):
        end = start + window
        if end > slice_end:
            end = slice_end
        subsegments.append((start, end - start))
        start = offset + (slice_id + 1) * shift
    return subsegments

def create_subsegments_from_segments(segments, global_offset, sample_rate=16000, window=0.63, shift=0.08):
    all_subsegments = []
    all_subsegments_starts = []
    all_subsegments_end = []
    for segment, seg_start_time, seg_end_time in segments:
        duration = len(segment) / sample_rate
        subsegments = get_subsegments(0, window, shift, duration)
        subsegment_samples = [(int(start * sample_rate), int((start + length) * sample_rate)) for start, length in subsegments]
        
        for start_sample, end_sample in subsegment_samples:
            subsegment = segment[start_sample:end_sample]
            subsegment_start_time = seg_start_time + (start_sample / sample_rate) + global_offset
            subsegment_end_time = seg_start_time + (end_sample / sample_rate) + global_offset
            all_subsegments.append(subsegment)
            all_subsegments_starts.append(subsegment_start_time)
            all_subsegments_end.append(subsegment_end_time)
            print(f"Subsegment start: {subsegment_start_time}, end: {subsegment_end_time}")
    
    return all_subsegments, all_subsegments_starts, all_subsegments_end

# Embedding Module "Titanet"
speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
def get_embeddings(subsegments):
    embeddings = []
    for segment in subsegments:        
        embedding = speaker_model.get_embedding(segment).detach().cpu().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

# Clustering Module
from nemo.collections.asr.parts.utils.online_clustering import NemoOnlineSpeakerClustering
clustering_model = NemoOnlineSpeakerClustering(
            max_num_speakers=8,
            max_rp_threshold=0.1,
            history_buffer_size=100,
            current_buffer_size=100,
            cuda=device,
        )

# from pyannote.audio.pipelines.clustering import AgglomerativeClustering
# clustering_Agglomerative = AgglomerativeClustering().instantiate(
#     {
#         "method": "centroid",
#         "min_cluster_size": 12,
#         "threshold": 0.7045654963945799,
#     }
# )


# clustering_model.forward_infer(curr_emb=curr_emb, base_segment_indexes=base_segment_indexes, frame_index=frame_index, cuda=cuda)
from nemo.collections.asr.parts.utils.longform_clustering import LongFormSpeakerClustering
speaker_clustering = LongFormSpeakerClustering(cuda=device)

########################################################################################
# Clustring 

def calculate_affinity_matrix(embeddings):
    # Normalize embeddings
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Compute cosine similarity matrix
    affinity_matrix = cosine_similarity(norm_embeddings)
    return affinity_matrix

def apply_threshold(affinity_matrix, threshold=0.5):
    # Apply threshold to affinity matrix
    affinity_matrix[affinity_matrix < threshold] = 0
    return affinity_matrix

def nmesc(affinity_matrix, max_num_speakers=8, threshold=0.2):
    # Apply threshold to affinity matrix
    thresholded_matrix = apply_threshold(affinity_matrix, threshold)
    
    # Apply clustering to the affinity matrix
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Ensure this is set to None
        affinity='precomputed', 
        linkage='average', 
        distance_threshold=1 - threshold  # Set your desired threshold here
    ).fit(1 - thresholded_matrix)
    
    # Estimate the number of distinct clusters (speakers)
    est_num_speakers = len(set(clustering.labels_))
    print(f"number of speaker estimate {est_num_speakers}")
    return min(est_num_speakers, max_num_speakers)
    
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from typing import List

class LoLClusteringAlgorithm:
    def __init__(self, max_points_per_cluster=10, recluster_condition_batches=5):
        self.max_points_per_cluster = max_points_per_cluster
        self.recluster_condition_batches = recluster_condition_batches
        self.embeddings = []  # Store all embeddings
        self.clusters = []  # List of clusters, each cluster is a list of points
        self.cluster_centroids = []  # List of centroids for each cluster
        self.batch_count = 5  # Count batches received, set to 5 to start the first number of clusters
        self.n_clusters = 2  # Initialize with a default value

    def add_embeddings(self, embeddings_batch):
        # Convert to numpy array if not already
        embeddings_batch = np.array(embeddings_batch)
        
        # Add new batch of embeddings to storage
        self.embeddings.extend(embeddings_batch)
        self.batch_count += 1
        
        # Check if we should trigger re-clustering
        if self.batch_count >= self.recluster_condition_batches:
            self._recluster()
            self.batch_count = 0
        
        # Start K-means Cluster
        return self.cluster()

    def cluster(self):
        # Convert embeddings list to numpy array
        all_embeddings = np.vstack(self.embeddings)
        print(f"all_embeddings {all_embeddings.shape}")
        
        kmeans = KMeans(n_clusters=self.n_clusters)
        labels = kmeans.fit_predict(all_embeddings)
        print(f"clustering labels {labels}")
        
        # Clear current clusters and centroids
        self.clusters = [[] for _ in range(self.n_clusters)]
        self.cluster_centroids = kmeans.cluster_centers_
        
        # Assign embeddings to clusters based on labels
        for idx, label in enumerate(labels):
            self.clusters[label].append(all_embeddings[idx])
        
        # Combine clusters if they exceed max points per cluster
        for cluster_idx in range(len(self.clusters)):
            while len(self.clusters[cluster_idx]) > self.max_points_per_cluster:
                self._merge_cluster(cluster_idx)
        return labels
                
    def _recluster(self):
        # Convert embeddings list to numpy array
        all_embeddings = np.vstack(self.embeddings)
        
        self.n_clusters = nmesc(calculate_affinity_matrix(all_embeddings))
        if self.n_clusters < 2:
            self.n_clusters = 2  # Ensure at least 2 clusters

    def _merge_cluster(self, cluster_idx):
        cluster = self.clusters[cluster_idx]
        
        # Calculate pairwise similarities within the cluster
        similarity_matrix = cosine_similarity(np.vstack(cluster))
        
        # Find the pair with the highest similarity
        np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
        max_sim_indices = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
        
        # Combine the points with the highest similarity
        point1_idx, point2_idx = max_sim_indices
        combined_point = (cluster[point1_idx] + cluster[point2_idx]) / 2
        
        # Remove the original points and add the combined point
        cluster.pop(max(point1_idx, point2_idx))
        cluster.pop(min(point1_idx, point2_idx))
        cluster.append(combined_point)
        
        # Update the cluster centroid
        self.cluster_centroids[cluster_idx] = np.mean(np.vstack(cluster), axis=0)
        # print(f"self.cluster_centroids {np.array(self.cluster_centroids).shape}")
        
    def predict_cluster(self, new_embedding):
        # Ensure new_embedding is a numpy array
        new_embedding = np.array(new_embedding)
        
        # Calculate cosine similarity with each cluster centroid
        similarities = cosine_similarity(new_embedding, self.cluster_centroids)
        
        # Find the index of the nearest cluster for each embedding in the batch
        nearest_cluster_indices = np.argmax(similarities, axis=1)
        
        return nearest_cluster_indices
        
    def get_clusters(self):
        return self.clusters

    def get_cluster_centroids(self):
        return self.cluster_centroids
    
lol_cluster = LoLClusteringAlgorithm(max_points_per_cluster=20, recluster_condition_batches=1)
# clustering_algorithm.add_embeddings(batch_embeddings)
# predicted_cluster = clustering_algorithm.predict_cluster(new_embedding)
# print(f"Predicted cluster: {predicted_cluster}")

##########################################################################################

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
        self.global_offset = 0
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
        # signal = batch.reshape(-1).to(torch.float).to(device)
        # print(f"Legendary signal shape {signal.shape}")
        # temp = prepare_input_from_array(signal)
        # print(f"Legendary temp shape {temp.shape}")
        # input_signal  = torch.tensor(temp).reshape(1,-1,temp.shape[0])
        # input_signal_length = torch.tensor([temp.shape[1] for i in range(temp.shape[0])]).long()
        # #input_signal_length = [x.shape[0] for x in input_signal]
        # print(f"legendary-SpeakerDiarization-__call__ input_signal  {input_signal.shape} processed_signal_length {input_signal_length} shape {input_signal_length.shape}")
        # vad_output = vad_model(processed_signal=input_signal,processed_signal_length=input_signal_length)
        
        # print(f"legendary-SpeakerDiarization-__call__ VAD vad_output {vad_output} shape {vad_output.shape} ")
        # probs = torch.softmax(vad_output, dim=-1)
        # pred = probs[:, 0]
        # print(f"legendary-SpeakerDiarization-__call__ VAD vad_output probs {probs} shape {probs.shape} pred {pred} shape {pred.shape} ")
        # vad_timestamp_results = convert_vad_into_timestamp(signal,pred)
        
        start_timestamps,end_timestamps = get_vad_timestamps(batch.reshape(-1))
        segments = segment_audio(batch.reshape(-1), start_timestamps, end_timestamps, sample_rate=16000)
        print(f"Legendary number of segments created from batch {len(segments)} segment sizes {[len(segment[0]) for segment in segments] } from batch size {batch.reshape(-1).shape}")
        subsegments,subseg_start, subseg_ends = create_subsegments_from_segments(segments, self.global_offset, sample_rate=16000, window=0.63, shift=0.08)
        print(f"Legendary number of subSegments created from batch {len(subsegments)} segment sizes {[len(segment) for segment in subsegments] } global offset {self.global_offset}")
        
        emd_tita_net = torch.tensor(get_embeddings(subsegments))
        index_vector = torch.arange(emd_tita_net.shape[0])
        print(f"Legendary emd_tita_net {emd_tita_net.shape} index vector {index_vector.shape}")
        
        # clustering_model.forward_infer(curr_emb=emd_tita_net, cuda=cuda)
        print(f"lol if this wroked first time {clustering_model.forward_infer(curr_emb=emd_tita_net,base_segment_indexes = index_vector)}")
        # clusters = clustering_Agglomerative.cluster(embeddings=emd_tita_net, min_clusters=2, max_clusters=3, num_clusters=len(emd_tita_net))
        # print(f"Legendary clusters pyaanote {clusters}")
        self.global_offset += 0.5 # step size
        
        #lol_cluster.add_embeddings(emd_tita_net)
        #predicted_cluster = lol_cluster.predict_cluster(emd_tita_net)
        #print(f"Predicted cluster: {predicted_cluster}")
        tempo = speaker_clustering.forward_infer(
            embeddings_in_scales=emd_tita_net,
            timestamps_in_scales=torch.tensor([[start,end]for start,end in zip(subseg_start, subseg_ends)]),
            multiscale_segment_counts= torch.tensor([emd_tita_net.shape[0]]),
            multiscale_weights=torch.tensor([1]),
            oracle_num_speakers=-1,
            max_num_speakers=8,
            max_rp_threshold=0.1,
            sparse_search_volume=50,
            chunk_cluster_count=50,
            embeddings_per_chunk=10000,
        )
        print(f"if it reached here, lol man, just lol {tempo}")
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

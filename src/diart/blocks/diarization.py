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


# Embedding Module "Titanet"
speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
def get_embeddings(subsegments):
    embeddings = []
    for segment in subsegments:        
        embedding = speaker_model.get_embedding(segment).detach().cpu().numpy()
        embeddings.append(embedding)
    if len(embeddings) > 1 :
        return np.vstack(embeddings)
    return np.array(embeddings)

# Clustering Module
from nemo.collections.asr.parts.utils.online_clustering import NemoOnlineSpeakerClustering
clustering_model = NemoOnlineSpeakerClustering(
            max_num_speakers=8,
            max_rp_threshold=0.1,
            history_buffer_size=100,
            current_buffer_size=100,
            cuda=device,
        )


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

def majority_voting_labeling(clustering_prediction, embedding_arr):
  # Extract intervals with their labels
  intervals = [(label, (entry[3], entry[4])) for label, entry in zip(clustering_prediction, embedding_arr)]
  
  # Step 1: Extract all unique time points
  time_points = set()
  for label, (start, end) in intervals:
      time_points.update([start, end])
  time_points = sorted(time_points)
  
  # Step 2: Create a list for time intervals with their corresponding labels
  time_labels = []
  for i in range(len(time_points) - 1):
      current_interval = (time_points[i], time_points[i + 1])
      current_labels = [label for label, (start, end) in intervals if start < current_interval[1] and end > current_interval[0]]
      if current_labels:
          most_common_label = Counter(current_labels).most_common(1)[0][0]
          time_labels.append((most_common_label, current_interval))
  
  # Step 3: Combine consecutive intervals with the same label
  result_intervals = []
  if time_labels:
      current_label, current_start = time_labels[0][0], time_labels[0][1][0]
      for label, (start, end) in time_labels:
          if label != current_label:
              result_intervals.append((current_label, (current_start, start)))
              current_label, current_start = label, start
      result_intervals.append((current_label, (current_start, time_labels[-1][1][1])))
  
  return result_intervals

##########################################################################################
# Graph for clustering 
import networkx as nx
from collections import defaultdict

class EmbeddingGraph:
    def __init__(self):
        self.graph = nx.Graph()
    
    def add_embedding(self, embedding_id):
        if not self.graph.has_node(embedding_id):
            self.graph.add_node(embedding_id)

    def update_edges_for_label(self, embedding_ids):
        for i in range(len(embedding_ids)):
            for j in range(i + 1, len(embedding_ids)):
                print(f"update edges between {embedding_ids[i]} {embedding_ids[j]}")
                self.update_edge(embedding_ids[i], embedding_ids[j])

    def update_edge(self, embedding_id1, embedding_id2):
        if self.graph.has_edge(embedding_id1, embedding_id2):
            self.graph[embedding_id1][embedding_id2]['weight'] += 1
        else:
            self.graph.add_edge(embedding_id1, embedding_id2, weight=1)

    def add_embeddings_with_predictions(self, embedding_arr, clustering_prediction):
        # Group embeddings by their predicted labels
        clustering_prediction = [int(label) for label in clustering_prediction]
        unique_labels = set(clustering_prediction)
        label_to_embeddings = {label: [] for label in unique_labels}

        for embedding_id, label in zip(embedding_arr, clustering_prediction):
            self.add_embedding(embedding_id)
            print(f"int(label) {int(label)}, embedding_id {embedding_id}")
            label_to_embeddings[int(label)].append(embedding_id)
            print(f"label_to_embeddings[label] {label_to_embeddings[label]}, label {label}")

        # Update edges within each label group
        for embedding_ids in label_to_embeddings.values():
            print(f"connected embeddings {embedding_ids}")
            self.update_edges_for_label(embedding_ids)
    
    def filter_graph(self, threshold):
        filtered_graph = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            if data['weight'] >= threshold:
                filtered_graph.add_edge(u, v, weight=data['weight'])
        
        # Adding isolated nodes (nodes without edges) to the filtered graph
        for node in self.graph.nodes():
            if node not in filtered_graph:
                filtered_graph.add_node(node)
        
        return filtered_graph



###########################################################################################
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
        self.embedding_arr = [] # added
        self.seen_times = []  # added
        self.global_offset = 0 # added
        self.clustering_results = []
        self.embedding_graph = EmbeddingGraph()

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
        
    def inside_interval(self,new_interval):
        start,end = new_interval
        for interval in self.seen_times:
            interval_start,interval_ends = interval[0], interval[1]
            if start > interval_start and end < interval_ends:
                return True
        return False
        
    def create_subsegments_from_segments(self,segments, sample_rate=16000, window=0.63, shift=0.08):
        all_subsegments = []
        for segment, seg_start_time, seg_end_time in segments:
            duration = len(segment) / sample_rate
            subsegments = get_subsegments(0, window, shift, duration)
            subsegment_samples = [(int(start * sample_rate), int((start + length) * sample_rate)) for start, length in subsegments]
            
            for start_sample, end_sample in subsegment_samples:
                subsegment_start_time = seg_start_time + (start_sample / sample_rate) + self.global_offset
                subsegment_end_time = seg_start_time + (end_sample / sample_rate) + self.global_offset
                if self.inside_interval([subsegment_start_time, subsegment_end_time]):
                    continue 
                subsegment = segment[start_sample:end_sample]
                all_subsegments.append((subsegment,subsegment_start_time,subsegment_end_time))
                print(f"Subsegment start: {subsegment_start_time}, end: {subsegment_end_time}")
        
        return all_subsegments
        
                    
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
        batch_size = len(waveforms)
        batch = torch.tensor(waveforms)
      
        ############################################################
        # Detect segments that contain activatiy
        start_timestamps,end_timestamps = get_vad_timestamps(batch.reshape(-1))
        print(f"VAD number of intervals {len(start_timestamps)} of batch size {batch.reshape(-1).shape}")
        
        # subsegment them on window 0.63 with shift 0.08
        subsegments = []
        segments = segment_audio(batch.reshape(-1), start_timestamps, end_timestamps, sample_rate=16000)
        subsegments += self.create_subsegments_from_segments(segments, sample_rate=16000, window=2, shift=1)
        subsegments += self.create_subsegments_from_segments(segments, sample_rate=16000, window=1, shift=0.5)
        subsegments += self.create_subsegments_from_segments(segments, sample_rate=16000, window=0.5, shift=0.25)
        subsegments += self.create_subsegments_from_segments(segments, sample_rate=16000, window=0.25, shift=0.125)

        print(f"Legendary number of segments {len(segments)} segment sizes {[len(segment[0]) for segment in segments] } from batch size {batch.reshape(-1).shape}")
        print(f"Legendary number of sub segments created {len(subsegments)} global offset {self.global_offset} increase by step {self._config.step}")

        # Calculate the Embedding
        emd_tita_net = torch.tensor(get_embeddings([subsegment[0] for subsegment in subsegments]))
        emd_tita_net = emd_tita_net.reshape(-1,192)
        print(f"Legendary emd_tita_net {emd_tita_net.shape}")

        # creating tuple containing embedding subsegment start end time
        unique_subsegments = []
        for i in range(emd_tita_net.shape[0]):
            temp_segments,temp_start, temp_end = subsegments[i]
            if not self.inside_interval([temp_start,temp_end]): # if the segment is alread created don't add
                print("add sub-segments intervals:",temp_start,temp_end)
                unique_subsegments.append((len(self.embedding_arr)+i,emd_tita_net[i],temp_segments,temp_start, temp_end))
        
        # adding the intervals to stop from creating redundent segments
        for i, j in zip([x+self.global_offset for x in start_timestamps], [x+self.global_offset for x in end_timestamps]): # this is adding the offset 
            print(i,j)
            if not self.inside_interval([i, j]):
                print("added intervals",[i, j])
                self.seen_times.append([i, j])

        self.embedding_arr = self.embedding_arr + unique_subsegments # concatonate to global array
        print(f"global number of embedding {len(self.embedding_arr)}")
        
        self.global_offset += self._config.step # step size
        print(f" emd_tita_net shape {emd_tita_net.shape}")
        clustering_prediction = speaker_clustering.forward_infer(
            embeddings_in_scales=torch.stack([x[1] for x in self.embedding_arr]).to(torch.float),
            timestamps_in_scales=torch.tensor(np.array([[x[3],x[4]] for x in self.embedding_arr])),
            multiscale_segment_counts= torch.tensor([len(self.embedding_arr)]),
            multiscale_weights=torch.tensor([1]),
            oracle_num_speakers=-1,
            max_num_speakers=8,
            max_rp_threshold=0.1,
            sparse_search_volume=50,
            chunk_cluster_count=30,
            embeddings_per_chunk=10000,
        )
      # Increment Graph
        self.embedding_graph.add_embeddings_with_predictions([x[0] for x in self.embedding_arr], clustering_prediction)
        filtered_graph = self.embedding_graph.filter_graph(threshold = 0)
        print(f"Legendary filtered_graph {filtered_graph}")
      
        print(f"if it reached here, lol man, just lol {clustering_prediction} shape {len(clustering_prediction)}")
        self.clustering_results = clustering_prediction
        # self.intervals_predictions = majority_voting_labeling()
        # return clustering_prediction,self.embedding_arr,batch.reshape(-1)
        return waveforms

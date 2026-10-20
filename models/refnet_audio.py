"""
AudioRefNet: models/refnet.py with the language branch replaced by the acoustic one.

    point cloud (N x 10)                     audio (16 kHz)
          |                                        |
          v                                        v
    PointNet++ -> voting -> DETR            frozen wav2vec2/WavLM [offline cache]
      or GroupFree                                 |
          |                                  AudioModule: prosody FiLM
          |                                  -> Sinkhorn slot induction -> K role slots
          +------------> proposals ----------------+
                              |
                              v
                    AudioMatchModule: SPOT fusion
                    (bidirectional slots <-> proposals,
                     OT routing, RSK geometry)
                              |
                              v
                     referred-object scores

The detector half is byte-for-byte the construction in RefNet -- same GroupFree/VoteNet
choice, same config_transformer, same pretrained-weight plumbing -- so a detector
checkpoint trained for the text model loads straight into the audio one and the two are
comparable under a frozen detector.

The model is audio-only, and audio-only in the strong sense: no transcript, no ASR pass and
no text supervision, at training time or inference time. models/lang_module.py is never
constructed here. The text branch of this repository is a baseline to compare against.

That is a deliberately harder setting than the published alternatives. 3DVG-AudioSet's
audio is synthesised *from* ScanRefer text, so the transcript is sitting right there and
distilling from it is free -- which is exactly why it is worth refusing. A result obtained
with a text teacher does not transfer to speech that was never text, and the task exists
for speech that was never text.
"""

import torch
import torch.nn as nn

from lib.audio_config import AUDIO
from models.audio_match_module import AudioMatchModule
from models.audio_module import AudioModule
from models.backbone_module import Pointnet2Backbone
from models.group_free.GF_detector import GroupFreeDetector
from models.proposal_module import ProposalModule
from models.voting_module import VotingModule


class AudioRefNet(nn.Module):
    def __init__(self, args, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, hidden_size=256, dataset_config=None, audio_dim=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert mean_size_arr.shape[0] == self.num_size_cluster
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_bidir = use_bidir
        self.no_reference = no_reference
        self.dataset_config = dataset_config
        self.args = args

        # ---------------------------------------------------------- detector (unchanged)
        if args.detector == "GF":
            self.detector = GroupFreeDetector(
                num_class=num_class,
                num_heading_bin=num_heading_bin,
                num_size_cluster=num_size_cluster,
                mean_size_arr=mean_size_arr,
                input_feature_dim=input_feature_dim,
                width=args.width,
                bn_momentum=args.bn_momentum,
                sync_bn=True if args.syncbn else False,
                num_proposal=args.num_target,
                sampling=args.sampling,
                dropout=args.transformer_dropout,
                activation=args.transformer_activation,
                nhead=args.nhead,
                num_decoder_layers=args.num_decoder_layers,
                dim_feedforward=args.dim_feedforward,
                self_position_embedding=args.self_position_embedding,
                cross_position_embedding=args.cross_position_embedding,
                size_cls_agnostic=True if args.size_cls_agnostic else False)
        else:
            self.backbone_net = Pointnet2Backbone(args=args,
                                                  input_feature_dim=self.input_feature_dim)
            self.vgen = VotingModule(self.vote_factor, 256)
            config_transformer = {
                'mask': 'no_mask',
                'weighted_input': True,
                'transformer_type': 'myAdd_20;deformable',
                'deformable_type': 'myAdd',
                'position_embedding': 'none',
                'input_dim': 0,
                'enc_layers': 0,
                'dec_layers': 2,
                'dim_feedforward': 2048,
                'hidden_dim': 288,
                'dropout': 0.1,
                'nheads': 8,
                'pre_norm': False,
            }
            self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                           mean_size_arr, num_proposal, sampling,
                                           config_transformer=config_transformer,
                                           dataset_config=dataset_config)

        # ---------------------------------------------------------- acoustic branch
        if not no_reference:
            self.audio = AudioModule(num_text_classes=num_class,
                                     in_dim=audio_dim or AUDIO.ENCODER_DIM)
            self.match = AudioMatchModule(args, num_proposals=num_proposal, det_channel=288)


    # ------------------------------------------------------------------ forward

    def forward(self, data_dict):
        if self.args.detector == "GF":
            data_dict = self.detector(data_dict)
        else:
            data_dict = self.backbone_net(data_dict)

            xyz = data_dict["fp2_xyz"]
            features = data_dict["fp2_features"]
            data_dict["seed_inds"] = data_dict["fp2_inds"]
            data_dict["seed_xyz"] = xyz
            data_dict["seed_features"] = features

            xyz, features = self.vgen(xyz, features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            data_dict["vote_xyz"] = xyz
            data_dict["vote_features"] = features

            data_dict = self.proposal(xyz, features, data_dict)

        if self.no_reference:
            return data_dict

        data_dict = self.audio(data_dict)
        data_dict = self.match(data_dict)
        return data_dict


class CachedAudioRefNet(nn.Module):
    """Acoustic branch + fusion only. Signature matches AudioRefNet so callers can swap them.

    The detector's output arrives in ``data_dict`` from the scene cache
    (lib/scene_cache.py), so no detector is built and none runs. This is the model every
    fusion experiment trains: both of its inputs -- proposals and encoder frames -- are
    read from disk.
    """

    def __init__(self, args, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, hidden_size=256, dataset_config=None, audio_dim=None):
        super().__init__()

        if no_reference:
            raise ValueError(
                "CachedAudioRefNet trains the acoustic and fusion branches; "
                "no_reference=True leaves it with nothing to train. Use AudioRefNet for "
                "detection-only runs.")

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.no_reference = no_reference
        self.dataset_config = dataset_config
        self.args = args

        self.audio = AudioModule(num_text_classes=num_class,
                                 in_dim=audio_dim or AUDIO.ENCODER_DIM)
        # det_channel=288 is the width of the cached detr_features.
        self.match = AudioMatchModule(args, num_proposals=num_proposal, det_channel=288)

    def forward(self, data_dict):
        """AudioRefNet.forward without the detection branch; data_dict carries the cache."""
        data_dict = self.audio(data_dict)
        data_dict = self.match(data_dict)
        return data_dict


def build_audio_model(**kwargs):
    """AudioRefNet, or the cached-feature variant under AUDIO.USE_CACHED_SCENES."""
    model = (CachedAudioRefNet if AUDIO.USE_CACHED_SCENES else AudioRefNet)(**kwargs)
    if not model.no_reference:
        n_audio = sum(p.numel() for p in model.audio.parameters() if p.requires_grad)
        n_match = sum(p.numel() for p in model.match.parameters() if p.requires_grad)
        print(f"[3D-VGA] audio branch {n_audio/1e6:.2f}M params, "
              f"fusion head {n_match/1e6:.2f}M params, fusion='{AUDIO.FUSION}'")
    return model

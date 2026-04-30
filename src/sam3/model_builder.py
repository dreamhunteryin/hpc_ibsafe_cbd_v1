from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import CXBlock, SimpleFuser, SimpleMaskDownSampler, SimpleMaskEncoder
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.transformer import RoPEAttention


def _setup_tf32() -> None:
    if not torch.cuda.is_available():
        return
    device_props = torch.cuda.get_device_properties(0)
    if device_props.major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution: Optional[int] = None) -> PositionEmbeddingSine:
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode: Optional[str] = None) -> ViT:
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_vit_neck(
    position_encoding: PositionEmbeddingSine,
    vit_backbone: ViT,
    add_sam2_neck: bool = False,
) -> Sam3DualViTDetNeck:
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=add_sam2_neck,
    )


def _create_transformer_encoder() -> TransformerEncoderFusion:
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )
    return TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


def _create_transformer_decoder() -> TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )
    return TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )


def _create_dot_product_scoring() -> DotProductScoring:
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode: Optional[str] = None) -> UniversalSegmentationHead:
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )
    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0.0,
        embed_dim=256,
    )
    return UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )


def _create_geometry_encoder() -> SequenceGeometryEncoder:
    geo_pos_enc = _create_position_encoding()
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )
    return SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_sam3_model(backbone, transformer, input_geometry_encoder, segmentation_head, dot_prod_scoring, eval_mode):
    matcher = None
    if not eval_mode:
        from sam3.train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )

    return Sam3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False,
        inst_interactive_predictor=None,
        matcher=matcher,
    )


def _strip_prefix_if_present(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _extract_tensor_state_dict(ckpt: dict) -> dict[str, torch.Tensor]:
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    tensor_state = {
        key: value
        for key, value in ckpt.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensor_state:
        raise ValueError("Checkpoint does not contain a tensor state dict")
    return tensor_state


def _extract_model_state_dict(ckpt: dict) -> dict[str, torch.Tensor]:
    tensor_state = _extract_tensor_state_dict(ckpt)

    detector_state = {
        key[len("module.detector.") :]: value
        for key, value in tensor_state.items()
        if key.startswith("module.detector.")
    }
    if detector_state:
        return detector_state

    detector_state = {
        key[len("detector.") :]: value
        for key, value in tensor_state.items()
        if key.startswith("detector.")
    }
    if detector_state:
        return detector_state

    return _strip_prefix_if_present(tensor_state, "module.")


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    with open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu")
    state_dict = _extract_model_state_dict(ckpt)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(
            f"Loaded {checkpoint_path} with partial key coverage:\n"
            f"missing_keys={missing_keys}\nunexpected_keys={unexpected_keys}"
        )


def _setup_device_and_mode(model: nn.Module, device: str, eval_mode: bool) -> nn.Module:
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def download_ckpt_from_hf() -> str:
    try:
        return hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", local_files_only=True)
    except Exception:
        return hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")


def _create_tracker_maskmem_backbone() -> SimpleMaskEncoder:
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
        interpol_size=[1152, 1152],
    )
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    fuser = SimpleFuser(layer=cx_block, num_layers=2)
    return SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def _create_tracker_transformer() -> TransformerWrapper:
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )
    return TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )


def _create_tracker_backbone(compile_mode: Optional[str] = None) -> SAM3VLBackbone:
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(compile_mode=compile_mode)
    vision_encoder = _create_vit_neck(
        position_encoding,
        vit_backbone,
        add_sam2_neck=True,
    )
    return SAM3VLBackbone(visual=vision_encoder, text=None, scalp=1)


def _load_tracker_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    with_backbone: bool = True,
) -> None:
    with open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu")
    tensor_state = _extract_tensor_state_dict(ckpt)

    tracker_state = {}
    tracker_prefixes = ("module.tracker.", "tracker.")
    for key, value in tensor_state.items():
        for prefix in tracker_prefixes:
            if key.startswith(prefix):
                tracker_state[key[len(prefix) :]] = value
                break

    if with_backbone:
        detector_backbone_prefixes = (
            "module.detector.backbone.",
            "detector.backbone.",
        )
        for key, value in tensor_state.items():
            for prefix in detector_backbone_prefixes:
                if key.startswith(prefix):
                    rest = key[len(prefix) :]
                    if rest.startswith("vision_backbone."):
                        tracker_state[f"backbone.{rest}"] = value
                    break

    missing_keys, unexpected_keys = model.load_state_dict(tracker_state, strict=False)
    if missing_keys or unexpected_keys:
        print(
            f"Loaded tracker checkpoint {checkpoint_path} with partial key coverage:\n"
            f"missing_keys={missing_keys}\nunexpected_keys={unexpected_keys}"
        )


def build_sam3_image_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_mode: bool = True,
    compile: bool = False,
    bpe_path: Optional[str] = None,
    load_from_HF: bool = True,
    enable_segmentation: bool = True,
) -> Sam3Image:
    package_dir = Path(__file__).resolve().parent
    if bpe_path is None:
        bpe_path = str(package_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz")

    compile_mode = "default" if compile else None
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(compile_mode=compile_mode)
    vision_encoder = _create_vit_neck(position_encoding, vit_backbone)
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(visual=vision_encoder, text=text_encoder, scalp=1)
    transformer = TransformerWrapper(
        encoder=_create_transformer_encoder(),
        decoder=_create_transformer_decoder(),
        d_model=256,
    )
    segmentation_head = _create_segmentation_head(compile_mode=compile_mode) if enable_segmentation else None
    geometry_encoder = _create_geometry_encoder()
    dot_product_scoring = _create_dot_product_scoring()

    model = _create_sam3_model(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=geometry_encoder,
        segmentation_head=segmentation_head,
        dot_prod_scoring=dot_product_scoring,
        eval_mode=eval_mode,
    )

    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    return _setup_device_and_mode(model, device=device, eval_mode=eval_mode)


def build_sam3_tracker_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_mode: bool = True,
    compile: bool = False,
    load_from_HF: bool = True,
    with_backbone: bool = True,
    apply_temporal_disambiguation: bool = True,
) -> Sam3TrackerPredictor:
    compile_mode = "default" if compile else None
    backbone = _create_tracker_backbone(compile_mode=compile_mode) if with_backbone else None
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=_create_tracker_transformer(),
        maskmem_backbone=_create_tracker_maskmem_backbone(),
        multimask_output_in_sam=True,
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        always_start_from_first_ann_frame=False,
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )

    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        _load_tracker_checkpoint(model, checkpoint_path, with_backbone=with_backbone)

    return _setup_device_and_mode(model, device=device, eval_mode=eval_mode)

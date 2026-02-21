#!/usr/bin/env python3
"""
InfiniteTalk Batch Video Generator

Batch generates lip-sync videos using InfiniteTalk workflow on ComfyUI.
Requires ComfyUI running at http://127.0.0.1:8188 with ComfyUI-WanVideoWrapper installed.

Usage:
    python run_infinitetalk_batch.py
"""
import json, time, wave, os, sys, argparse
import urllib.request, urllib.error


API = "http://127.0.0.1:8188"

# ── Configuration ──────────────────────────────────────────────
# Each video entry: reference frame image, face close-up image, output resolution
VIDEOS = [
    {"name": "bainian4",   "frame": "bainian4_frame.png",   "face": "face_bainian4.png",   "w": 464, "h": 832},
    {"name": "bainian1_1", "frame": "bainian1_1_frame.png", "face": "face_bainian1.png", "w": 464, "h": 832},
]

# Audio files — num_frames will be auto-calculated from duration
AUDIO_DIR = "."  # relative to ComfyUI/input/
AUDIO_FILES = [
    "ref_cat14.wav",
    "ref_cat19_20.wav",
    "ref_custom_newyear1_v1.wav",
    "ref_custom2_v1.wav",
]


def get_num_frames(audio_path, fps=25.0):
    """Calculate num_frames from audio duration: duration * fps + 1"""
    with wave.open(audio_path, 'r') as w:
        duration = w.getnframes() / w.getframerate()
    return int(duration * fps) + 1


def make_workflow(v, audio_file, num_frames):
    """Build the InfiniteTalk ComfyUI workflow dict."""
    prefix = audio_file.replace(".wav", "")
    return {
        "1": {"class_type": "WanVideoVAELoader", "inputs": {
            "model_name": "Wan2_1_VAE_bf16.safetensors", "precision": "bf16"}},
        "2": {"class_type": "WanVideoTextEncodeCached", "inputs": {
            "model_name": "umt5-xxl-enc-bf16.safetensors", "precision": "bf16",
            "positive_prompt": "a person is talking naturally",
            "negative_prompt": "bad quality, blurry, distorted face, static, subtitles, worst quality",
            "quantization": "disabled", "use_disk_cache": True, "device": "gpu"}},
        "3": {"class_type": "CLIPVisionLoader", "inputs": {
            "clip_name": "clip_vision_h.safetensors"}},
        "4": {"class_type": "Wav2VecModelLoader", "inputs": {
            "model": "wav2vec2-chinese-base_fp16.safetensors",
            "base_precision": "fp16", "load_device": "main_device"}},
        "5": {"class_type": "MultiTalkModelLoader", "inputs": {
            "model": "WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf"}},
        "6": {"class_type": "LoadAudio", "inputs": {"audio": audio_file}},
        "7": {"class_type": "LoadImage", "inputs": {"image": v["frame"]}},
        "8": {"class_type": "MultiTalkWav2VecEmbeds", "inputs": {
            "wav2vec_model": ["4", 0], "audio_1": ["6", 0],
            "normalize_loudness": True, "num_frames": num_frames,
            "fps": 25.0, "audio_scale": 1.5, "audio_cfg_scale": 1.0,
            "multi_audio_type": "para"}},
        "17": {"class_type": "LoadImage", "inputs": {"image": v["face"]}},
        "9": {"class_type": "WanVideoClipVisionEncode", "inputs": {
            "clip_vision": ["3", 0], "image_1": ["7", 0],
            "image_2": ["17", 0],
            "strength_1": 1.0, "strength_2": 1.0, "crop": "center",
            "combine_embeds": "concat", "force_offload": True,
            "tiles": 4, "ratio": 0.5}},
        "10": {"class_type": "WanVideoImageToVideoMultiTalk", "inputs": {
            "vae": ["1", 0], "start_image": ["7", 0],
            "width": v["w"], "height": v["h"],
            "frame_window_size": 81, "motion_frame": 9,
            "force_offload": False, "colormatch": "disabled",
            "tiled_vae": False, "clip_embeds": ["9", 0],
            "mode": "infinitetalk"}},
        "11": {"class_type": "WanVideoLoraSelect", "inputs": {
            "lora": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "strength": 1.0, "merge_loras": False}},
        "12": {"class_type": "WanVideoBlockSwap", "inputs": {
            "blocks_to_swap": 20, "offload_img_emb": False,
            "offload_txt_emb": False, "use_non_blocking": True,
            "vace_blocks_to_swap": 0, "prefetch_blocks": 1,
            "block_swap_debug": False}},
        "13": {"class_type": "WanVideoModelLoader", "inputs": {
            "model": "WanVideo/I2V/wan2.1-i2v-14b-480p-Q8_0.gguf",
            "base_precision": "bf16", "quantization": "disabled",
            "load_device": "offload_device", "attention_mode": "sageattn",
            "lora": ["11", 0], "block_swap_args": ["12", 0],
            "multitalk_model": ["5", 0]}},
        "14": {"class_type": "WanVideoSampler", "inputs": {
            "model": ["13", 0], "text_embeds": ["2", 0],
            "image_embeds": ["10", 0], "multitalk_embeds": ["8", 0],
            "steps": 4, "cfg": 1.0, "shift": 11.0, "seed": 2,
            "scheduler": "unipc", "force_offload": True,
            "riflex_freq_index": 0}},
        "15": {"class_type": "WanVideoDecode", "inputs": {
            "vae": ["1", 0], "samples": ["14", 0],
            "enable_vae_tiling": False, "tile_x": 272, "tile_y": 272,
            "tile_stride_x": 144, "tile_stride_y": 128,
            "normalization": "default"}},
        "16": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["15", 0], "audio": ["6", 0],
            "frame_rate": 25, "loop_count": 0,
            "filename_prefix": f"IT_{v['name']}_{prefix}",
            "format": "video/h264-mp4",
            "pingpong": False, "save_output": True}},
    }


def submit(workflow):
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(f"{API}/prompt", data=data,
                                headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read()).get("prompt_id")


def wait(pid, timeout=7200):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{API}/history/{pid}")
            h = json.loads(resp.read())
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("completed") or h[pid].get("outputs"):
                    return h[pid]
                if st.get("status_str") == "error":
                    print(f"  ERROR: {st}")
                    return None
        except:
            pass
        time.sleep(10)
    print("  TIMEOUT!")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniteTalk batch generator")
    parser.add_argument("--comfyui-input", default="input",
                        help="Path to ComfyUI input directory (for audio duration calc)")
    args = parser.parse_args()

    total = len(VIDEOS) * len(AUDIO_FILES)
    done = 0

    for video in VIDEOS:
        for audio_file in AUDIO_FILES:
            audio_path = os.path.join(args.comfyui_input, audio_file)
            num_frames = get_num_frames(audio_path)

            done += 1
            print(f"\n[{done}/{total}] {video['name']} + {audio_file} "
                  f"({video['w']}x{video['h']}, {num_frames} frames)")

            wf = make_workflow(video, audio_file, num_frames)
            pid = submit(wf)
            print(f"  Submitted: {pid}")

            result = wait(pid)
            if result:
                for nid, out in result.get("outputs", {}).items():
                    if "gifs" in out:
                        for g in out["gifs"]:
                            print(f"  Output: {g.get('filename')}")
                print("  Done!")
            else:
                print("  FAILED!")

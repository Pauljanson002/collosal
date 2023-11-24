import torch
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload(gpu_id=0)

prompt = "A runner runs with rhytmic leg strides and synchronized arm swing propelling them forward"

video_frames = pipe(
    prompt=prompt,
    negative_prompt="Ugly , bad , low quality video",
    num_inference_steps=40,
    height=320,
    width=576,
    num_frames=24,
).frames

video_path = export_to_video(video_frames, output_video_path=f"zeroscope.mp4")


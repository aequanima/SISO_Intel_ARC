import torch
from diffusers.utils.torch_utils import randn_tensor

from third_party.renoise_inversion.src.schedulers.ddim_scheduler import MyDDIMScheduler
from third_party.renoise_inversion.src.schedulers.euler_scheduler import (
    MyEulerAncestralDiscreteScheduler,
)
from third_party.renoise_inversion.src.pipes.sdxl_inversion_pipeline import (
    SDXLDDIMPipeline,
)
from third_party.renoise_inversion.src.pipes.sd_inversion_pipeline import SDDDIMPipeline
from third_party.renoise_inversion.src.config import RunConfig
from third_party.renoise_inversion.src.eunms import Model_Type, Scheduler_Type
from third_party.renoise_inversion.src.utils.enums_utils import (
    model_type_to_size,
)
from local_pipelines.pipeline_stable_diffusion_xl_img2img_with_grads import (
    StableDiffusionXLImg2ImgPipelineWithGrads,
)


def create_noise_list(model_type, length, generator, device: str = None):
    """Creates a list of noise tensors on the specified or auto-detected device."""

    # Determine target device if not provided
    if device is None:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            target_device = "xpu"
        elif torch.cuda.is_available():
            target_device = "cuda" # Note: Doesn't specify index like "cuda:0"
        else:
            target_device = "cpu"
        print(f"[create_noise_list] Auto-detected device: {target_device}")
    else:
        target_device = device
        print(f"[create_noise_list] Using provided device: {target_device}")

    img_size = model_type_to_size(model_type)
    VQAE_SCALE = 8
    latents_size = (1, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
    
    noise_list = []
    for _ in range(length):
        noise = randn_tensor(
            latents_size,
            dtype=torch.float32,
            device=torch.device(target_device), # Use determined device
            generator=generator,
        )
        noise_list.append(noise)
        
    return noise_list


def get_renoise_inversion_pipes(
    vae,
    text_encoder_one,
    text_encoder_two,
    unet,
    model_name="stabilityai/sdxl-turbo",
    device: str = None, # Changed default from "cuda"
):
    """Initializes and returns ReNoise inversion and inference pipelines."""

    # Determine target device if not provided
    if device is None:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            target_device = "xpu"
        elif torch.cuda.is_available():
            target_device = "cuda"
        else:
            target_device = "cpu"
        print(f"[get_renoise_inversion_pipes] Auto-detected device: {target_device}")
    else:
        target_device = device
        print(f"[get_renoise_inversion_pipes] Using provided device: {target_device}")

    print(f"[get_renoise_inversion_pipes] Loading inference pipeline ({model_name}) to {target_device}...")
    # Load inference pipeline and move to target device
    pipe_inference = StableDiffusionXLImg2ImgPipelineWithGrads.from_pretrained(
        model_name,
        vae=vae,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        unet=unet,
        use_safetensors=True,
        safety_checker=None,
    ).to(target_device)

    print(f"[get_renoise_inversion_pipes] Creating inversion pipeline components...")
    # Create inversion pipeline using components from inference pipeline (already on target_device)
    # Note: SDXLDDIMPipeline might need specific imports or adjustments
    # Assuming SDXLDDIMPipeline exists and accepts components dictionary
    try:
        # This assumes SDXLDDIMPipeline is defined elsewhere and works like this
        pipe_inversion = SDXLDDIMPipeline(**pipe_inference.components)
        # Explicitly move the inversion pipeline object too, although components might be sufficient
        pipe_inversion.to(target_device)
    except NameError:
        print("[get_renoise_inversion_pipes] WARNING: SDXLDDIMPipeline not found. Inversion pipeline creation skipped/failed.")
        pipe_inversion = None # Handle case where it's not defined

    print(f"[get_renoise_inversion_pipes] Setting schedulers...")
    # Set schedulers (assuming MyEulerAncestralDiscreteScheduler is defined elsewhere)
    try:
        pipe_inference.scheduler = MyEulerAncestralDiscreteScheduler.from_config(
            pipe_inference.scheduler.config
        )
        if pipe_inversion is not None:
            pipe_inversion.scheduler = MyEulerAncestralDiscreteScheduler.from_config(
                pipe_inversion.scheduler.config # Use its own config if it exists
            )
    except NameError:
         print("[get_renoise_inversion_pipes] WARNING: MyEulerAncestralDiscreteScheduler not found. Schedulers not set.")

    print("[get_renoise_inversion_pipes] Initialization complete.")
    return pipe_inversion, pipe_inference


def get_inversion_config(inversion_max_step=0.75, model_name="stabilityai/sdxl-turbo"):
    if "sdxl" in model_name:
        model_type = Model_Type.SDXL_Turbo
        scheduler_type = Scheduler_Type.EULER
        perform_noise_correction = True
        noise_regularization_lambda_ac = 20.0
    else:
        model_type = Model_Type.SD21_Turbo
        scheduler_type = Scheduler_Type.DDIM
        perform_noise_correction = False
        noise_regularization_lambda_ac = 0.0
    config = RunConfig(
        model_type=model_type,
        scheduler_type=scheduler_type,
        inversion_max_step=inversion_max_step,
        perform_noise_correction=perform_noise_correction,
        noise_regularization_lambda_ac=noise_regularization_lambda_ac,
    )
    return config


def invert(pipe_inversion, pipe_inference, init_image, prompt, cfg, is_sdxl=True):
    generator = torch.Generator().manual_seed(cfg.seed)
    if is_sdxl:
        noise = create_noise_list(
            cfg.model_type, cfg.num_inversion_steps, generator=generator
        )
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)

    pipe_inversion.cfg = cfg
    pipe_inference.cfg = cfg
    res = pipe_inversion(
        prompt=prompt,
        num_inversion_steps=cfg.num_inversion_steps,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator,
        image=init_image,
        guidance_scale=cfg.guidance_scale,
        strength=cfg.inversion_max_step,
        denoising_start=1.0 - cfg.inversion_max_step,
        num_renoise_steps=cfg.num_renoise_steps,
    )
    latents = res[0][0]
    return latents

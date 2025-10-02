from fastapi import FastAPI, Response
import asyncio
import os
import random

from modules.async_worker import AsyncTask, async_tasks
import modules.config
import modules.flags as flags
import modules.constants as constants
from modules.util import get_enabled_loras

app = FastAPI()

def create_default_task(prompt: str):
    """
    Creates an AsyncTask with default values, mimicking the WebUI.
    """
    task = AsyncTask(args=[])

    # Set all default attributes that are normally parsed from Gradio inputs
    task.prompt = prompt
    task.negative_prompt = modules.config.default_prompt_negative
    task.style_selections = modules.config.default_styles
    task.performance_selection = flags.Performance(modules.config.default_performance)
    task.aspect_ratios_selection = modules.config.default_aspect_ratio
    task.image_number = 1
    task.output_format = modules.config.default_output_format
    task.seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)
    task.read_wildcards_in_order = False
    task.sharpness = modules.config.default_sample_sharpness
    task.cfg_scale = modules.config.default_cfg_scale
    task.base_model_name = modules.config.default_base_model_name
    task.refiner_model_name = modules.config.default_refiner_model_name
    task.refiner_switch = modules.config.default_refiner_switch
    task.loras = get_enabled_loras(modules.config.default_loras)
    task.input_image_checkbox = False
    task.current_tab = 'txt2img'
    task.uov_method = flags.disabled.casefold()
    task.uov_input_image = None
    task.outpaint_selections = []
    task.inpaint_input_image = None
    task.inpaint_additional_prompt = ''
    task.inpaint_mask_image_upload = None
    task.disable_preview = False
    task.disable_intermediate_results = False
    task.disable_seed_increment = True
    task.black_out_nsfw = modules.config.default_black_out_nsfw
    task.adm_scaler_positive = 1.5
    task.adm_scaler_negative = 0.8
    task.adm_scaler_end = 0.3
    task.adaptive_cfg = modules.config.default_cfg_tsnr
    task.clip_skip = modules.config.default_clip_skip
    task.sampler_name = modules.config.default_sampler
    task.scheduler_name = modules.config.default_scheduler
    task.vae_name = modules.config.default_vae
    task.overwrite_step = modules.config.default_overwrite_step
    task.overwrite_switch = modules.config.default_overwrite_switch
    task.overwrite_width = -1
    task.overwrite_height = -1
    task.overwrite_vary_strength = -1
    task.overwrite_upscale_strength = modules.config.default_overwrite_upscale
    task.mixing_image_prompt_and_vary_upscale = False
    task.mixing_image_prompt_and_inpaint = False
    task.debugging_cn_preprocessor = False
    task.skipping_cn_preprocessor = False
    task.canny_low_threshold = 64
    task.canny_high_threshold = 128
    task.refiner_swap_method = flags.refiner_swap_method
    task.controlnet_softness = 0.25
    task.freeu_enabled = False
    task.freeu_b1 = 1.01
    task.freeu_b2 = 1.02
    task.freeu_s1 = 0.99
    task.freeu_s2 = 0.95
    task.debugging_inpaint_preprocessor = False
    task.inpaint_disable_initial_latent = False
    task.inpaint_engine = modules.config.default_inpaint_engine_version
    task.inpaint_strength = 1.0
    task.inpaint_respective_field = 0.618
    task.inpaint_advanced_masking_checkbox = False
    task.invert_mask_checkbox = False
    task.inpaint_erode_or_dilate = 0
    task.save_final_enhanced_image_only = False
    task.save_metadata_to_images = modules.config.default_save_metadata_to_images
    task.metadata_scheme = flags.MetadataScheme(modules.config.default_metadata_scheme)
    task.cn_tasks = {x: [] for x in flags.ip_list}
    task.debugging_dino = False
    task.dino_erode_or_dilate = 0
    task.debugging_enhance_masks_checkbox = False
    task.enhance_input_image = None
    task.enhance_checkbox = False
    task.enhance_uov_method = modules.config.default_enhance_uov_method
    task.enhance_uov_processing_order = modules.config.default_enhance_uov_processing_order
    task.enhance_uov_prompt_type = modules.config.default_enhance_uov_prompt_type
    task.enhance_ctrls = []
    task.should_enhance = False
    task.images_to_enhance_count = 0
    task.enhance_stats = {}
    task.generate_image_grid = False

    task.steps = task.performance_selection.steps()
    task.original_steps = task.steps

    return task

@app.post("/v1/generation/text-to-image")
async def text_to_image(prompt: str):
    """
    Generates an image from a text prompt and returns the image as a response.
    """
    task = create_default_task(prompt)
    async_tasks.append(task)

    while not task.yields:
        await asyncio.sleep(0.1)

    image_path = None
    while True:
        if not task.yields:
            await asyncio.sleep(0.1)
            continue

        flag, product = task.yields.pop(0)
        if flag == 'finish':
            if product:
                image_path = product[-1]
            break

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    else:
        return {"error": "Image generation failed or no image was produced."}

import time
from typing import List, Optional, Union, Tuple, Literal

import torch
import PIL.Image

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from compel import Compel, ReturnedEmbeddingsType
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.models.embeddings import MultiIPAdapterImageProjection
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, LCMScheduler, ControlNetXSAdapter, \
    UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from streamdiffusion.adapters.control_adapter import TensorUNetControlNetXSModel


class StreamDiffusion(UNet2DConditionLoadersMixin):
    def __init__(
        self,
        pipe: Optional[DiffusionPipeline],
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        device: Optional[str] = None,
        vae_scale_factor: Optional[int] = 8,
        original_inference_steps: Optional[int] = 50
    ) -> None:

        # compute
        self.device = pipe.device if pipe else device
        self.dtype = torch_dtype

        # image dimensions
        self.height = height
        self.width = width
        self.vae_scale_factor = pipe.vae_scale_factor if pipe else vae_scale_factor
        self.latent_height = int(height // self.vae_scale_factor )
        self.latent_width = int(width // self.vae_scale_factor )

        # pipe & components
        self.pipe = pipe
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        # self.image_processor = VaeImageProcessor(self.vae_scale_factor)
        self.pipe.scheduler.config['original_inference_steps'] = original_inference_steps
        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        # noise scheduler
        self.seed = 1
        self.sub_timesteps = None
        self.sub_timesteps_pt = None
        self.noise_generator = torch.Generator()
        self.scheduler.set_timesteps(original_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # text prompt encoding
        self.cached_prompt_embeds = None
        self.cached_add_text_embeds = None
        self.cached_add_time_ids = None
        if self.is_sdxl:
            self.compel = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
        else:
            self.compel = Compel(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder
            )

        # image prompt encoding
        self.cached_ip_embeds = None
        self.cached_ip_strength = None
        self.ip_encoder = None
        self.ip_projection = None
        self.ip_feature_extractor = None

        # control
        self.control_adapter = None
        self.control_cond_embedding = None
        self.cached_control_strength = None

        # guidance
        self.cfg_type = cfg_type
        self.init_noise = None
        self.stock_noise = None
        self.c_out = None
        self.c_skip = None
        self.alpha_prod_t_sqrt = None
        self.beta_prod_t_sqrt = None

        # inference steps
        self.t_list = t_index_list
        self.do_add_noise = do_add_noise

        # batching
        self.latent_buffer = None
        self.control_buffer = None
        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)
        self.batch_size = self.denoising_steps_num * frame_buffer_size

    @property
    def is_sdxl(self):
        return type(self.pipe) is StableDiffusionXLPipeline

    @property
    def ip_adapter_loaded(self):
        return all((self.ip_encoder, self.ip_feature_extractor, self.ip_projection))

    def load_ip_adapter(
        self,
        repo_id: str = "h94/IP-Adapter",
        encoder_folder: str = "sdxl_models/image_encoder",
        projection_weights_filename: str = "sdxl_models/ip-adapter_sdxl.safetensors",
        low_cpu_mem_usage: bool = True,
        local_files_only: bool = True
    ):
        # load image encoder
        self.ip_encoder = CLIPVisionModelWithProjection.from_pretrained(
            repo_id,
            subfolder=encoder_folder,
            low_cpu_mem_usage=low_cpu_mem_usage,
            local_files_only=local_files_only
        ).to(self.device, dtype=self.dtype)

        # load image processor (feature extractor)
        clip_size = self.ip_encoder.config.image_size if self.ip_encoder else 224
        self.ip_feature_extractor = CLIPImageProcessor(size=clip_size, crop_size=clip_size)

        # load projection layer
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=projection_weights_filename,
            local_files_only=local_files_only
        )
        state_dict = load_file(model_file, device='cpu')
        state_dict = {k.replace("image_proj.", ""): v for k, v in state_dict.items() if k.startswith("image_proj.")}
        image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(state_dict, low_cpu_mem_usage=low_cpu_mem_usage)
        self.ip_projection = MultiIPAdapterImageProjection([image_projection_layer]).to(self.device, dtype=self.dtype)

    def load_controlnet_adapter(self, control_type: Literal["canny", "depth"]):

        # model_id
        model_id = f'UmerHA/Testing-ConrolNetXS-SD2.1-{control_type}'
        if self.is_sdxl:
            model_id = f'UmerHA/Testing-ConrolNetXS-SDXL-{control_type}'

        # load adapter
        self.control_adapter = ControlNetXSAdapter.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            use_safetensors=True
        ).to(self.device)

        # check if vanilla unet
        if isinstance(self.unet, UNet2DConditionModel):

            # fuse control layers
            self.unet = TensorUNetControlNetXSModel.from_unet(
                unet=self.pipe.unet,
                controlnet=self.control_adapter
            ).to(self.device, dtype=self.dtype)

    @torch.inference_mode()
    def set_timesteps(self, t_list: List[int]):

        # check if inference steps have changed
        denoising_steps_num_changed = False
        if len(t_list) != len(self.t_list):
            denoising_steps_num_changed = True

        # update members
        self.t_list = t_list
        self.denoising_steps_num = len(t_list)

        # update batch size
        self.batch_size = self.denoising_steps_num * self.frame_bff_size

        # initialize buffers for batch denoising
        if self.denoising_steps_num > 1:
            # FIXME: What if processing is in progress?
            #  We kill the buffer and all partially denoised latents are discarded.
            self._initialize_latent_buffer()
            self._initialize_control_buffer()
        else:
            self.latent_buffer = None
            self.control_buffer = None

        # update sub timesteps
        self.sub_timesteps = [self.timesteps[t] for t in t_list]
        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_pt = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size,
            dim=0,
        )

        # update scheduler scalings
        self.scheduler_update_scalings()

        # repeat prompt to match inference steps num
        if denoising_steps_num_changed:
            self.set_noise(seed=self.seed)
            self.repeat_prompt()
            self.repeat_image_prompt()

    @torch.inference_mode()
    def _initialize_latent_buffer(self):
        self.latent_buffer = torch.zeros(
            (
                (self.denoising_steps_num - 1) * self.frame_bff_size,
                4,
                self.latent_height,
                self.latent_width,
            ),
            dtype=self.dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def _initialize_control_buffer(self):
        channels = self.control_adapter.config.block_out_channels[0]
        self.control_buffer = torch.zeros(
            (
                (self.denoising_steps_num - 1) * self.frame_bff_size,
                channels,
                self.latent_height,
                self.latent_width,
            ),
            dtype=self.dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def set_noise(self, seed: int = 1) -> None:

        # set seed
        self.seed = seed
        self.noise_generator.manual_seed(seed)

        # init noise
        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=self.noise_generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = self.init_noise.clone()

    @torch.inference_mode()
    def update_image_prompt(self, image: PIL.Image.Image) -> None:

        if not self.ip_adapter_loaded:
            raise ValueError(
                f"The pipeline doesn't have required image prompt components."
                f"{self.ip_encoder=}, {self.ip_feature_extractor=}, {self.ip_projection=}"
            )

        # Process the image
        image_features = self.ip_feature_extractor(
            images=image,
            return_tensors="pt",
        ).pixel_values.to(device=self.device, dtype=self.dtype)

        # Generate image embeddings with image encoder
        image_embeds = self.ip_encoder(image_features).image_embeds

        # projecting image embedding through ip adapter weights
        self.cached_ip_embeds = self.ip_projection(image_embeds)[0]
        # self.cached_ip_embeds = self.cached_ip_embeds.unsqueeze(0) # add batch dimension

        # repeat to fit batch size
        self.repeat_image_prompt()

    def repeat_image_prompt(self):

        # repeat for batching
        if self.cached_ip_embeds is not None:
            self.cached_ip_embeds = self.fit_to_dimension(self.cached_ip_embeds, self.batch_size)

    @torch.inference_mode()
    def set_image_prompt_scale(self, scale: float):
        self.cached_ip_strength = torch.tensor([scale], device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def set_control_scale(self, scale: float):
        self.cached_control_strength = torch.tensor([scale], device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def generate_pipeline_text_prompt_embedding(self, prompt: str):
        return self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    @torch.inference_mode()
    def update_prompt(self, prompt: str) -> None:

        # get embeddings
        embeds = self.compel.build_weighted_embedding(
            prompt=prompt,
            requires_pooled=self.is_sdxl
        )

        # check for sdxl mode
        if not self.is_sdxl:

            # unpack embeds
            text_embeds = embeds[0]

            # repeat embeds for batch size
            self.cached_prompt_embeds = text_embeds.to(dtype=self.dtype).unsqueeze(0)

        else:

            # unpack embeds
            text_embeds, pooled_embeds = embeds

            # repeat embeds for batch size
            self.cached_prompt_embeds = text_embeds.to(dtype=self.dtype).unsqueeze(0)

            # repeat embeds for batch size
            self.cached_add_text_embeds = pooled_embeds.to(dtype=self.dtype) # .unsqueeze(0)

            # Set up the additional time embeddings needed for SDXL
            self.cached_add_time_ids = self._get_add_time_ids(
                (self.height, self.width),
                (0, 0),
                (self.height, self.width)
            ).to(self.device)

            # repeat embeds for batch size
            self.cached_add_time_ids = self.cached_add_time_ids # .unsqueeze(0)

        # change dimension to fit batch size
        self.repeat_prompt()

    def repeat_prompt(self):

        # repeat normal prompt
        self.cached_prompt_embeds = self.fit_to_dimension(self.cached_prompt_embeds, self.batch_size)
        # self.cached_prompt_embeds = self.fit_to_dimension(self.cached_prompt_embeds.to(dtype=self.dtype), self.batch_size)

        if self.is_sdxl:

            # repeat sdxl special prompts
            self.cached_add_text_embeds = self.fit_to_dimension(self.cached_add_text_embeds, self.batch_size)
            # self.cached_add_text_embeds = self.fit_to_dimension(self.cached_add_text_embeds.to(dtype=self.dtype), self.batch_size)
            self.cached_add_time_ids = self.fit_to_dimension(self.cached_add_time_ids, self.batch_size)

    @torch.inference_mode()
    def generate_control_embedding(self, image_pt):

        # Generate canny
        # start_time = time.time()
        control_pt = self.control_adapter.controlnet_cond_embedding(image_pt)
        # print(f"Control adapter embedding took {time.time() - start_time:.4f} seconds.")

        # # Generate depth
        # start_time = time.time()
        # depth_pt = self.depth_feature_extractor.generate(image_tensor)
        # depth_pt = self._process_depth_for_adapter(depth_pt)
        # print(f"Depth feature extraction took {time.time() - start_time:.4f} seconds.")

        return control_pt

    # repeat image prompt
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if idx is None:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
            ) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
            ) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = (
                self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
            )

        return denoised_batch

    def scheduler_update_scalings(self):

        # c_skip / c_out
        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        # cumprod sqrt
        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size,
            dim=0,
        )

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        added_cond_kwargs,
        cross_attention_kwargs,
        controlnet_cond: Optional[torch.Tensor] = None,
        controlnet_cond_scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
        #     x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
        #     t_list = torch.concat([t_list[0:1], t_list], dim=0)
        # elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
        #     x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
        #     t_list = torch.concat([t_list, t_list], dim=0)
        # else:
        # x_t_latent_plus_uc = x_t_latent

        # control adapter
        ctl_kwargs = {}
        if controlnet_cond is not None:
            ctl_kwargs['controlnet_cond'] = controlnet_cond
            ctl_kwargs['conditioning_scale'] = controlnet_cond_scale
            ctl_kwargs['apply_control'] = True

        model_pred = self.unet(
            x_t_latent,
            t_list,
            encoder_hidden_states=self.cached_prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
            **ctl_kwargs
        )[0]

        # if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
        #     noise_pred_text = model_pred[1:]
        #     self.stock_noise = torch.concat(
        #         [model_pred[0:1], self.stock_noise[1:]], dim=0
        #     )  # ここコメントアウトでself out cfg
        # elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
        #     noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        # else:
        noise_pred_text = model_pred

        # if self.guidance_scale > 1.0 and (
        #     self.cfg_type == "self" or self.cfg_type == "initialize"
        # ):
        #     noise_pred_uncond = self.stock_noise * self.delta
        # if self.guidance_scale > 1.0 and self.cfg_type != "none":
        #     model_pred = noise_pred_uncond + self.guidance_scale * (
        #         noise_pred_text - noise_pred_uncond
        #     )
        # else:
        if self.cfg_type == "self" or self.cfg_type == "initialize":
            noise_pred_uncond = self.stock_noise * 1.5
            model_pred = noise_pred_uncond + 1.0 * (noise_pred_text - noise_pred_uncond)
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent)
        if self.cfg_type == "self" or self.cfg_type == "initialize":
            scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
            delta_x = self.scheduler_step_batch(model_pred, scaled_noise)
            alpha_next = torch.concat(
                [
                    self.alpha_prod_t_sqrt[1:],
                    torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                ],
                dim=0,
            )
            delta_x = alpha_next * delta_x
            beta_next = torch.concat(
                [
                    self.beta_prod_t_sqrt[1:],
                    torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                ],
                dim=0,
            )
            delta_x = delta_x / beta_next
            init_noise = torch.concat(
                [self.init_noise[1:], self.init_noise[0:1]], dim=0
            )
            self.stock_noise = init_noise + delta_x

        return denoised_batch, model_pred

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], device=self.device, dtype=self.dtype)
        return add_time_ids

    @torch.inference_mode()
    def encode_image(self, image_tensor: torch.Tensor, add_init_noise: bool = True) -> torch.Tensor:

        # pre-process image
        # normalize from [0, 1] (PyTorch standard) to [-1, 1] (StableDiffusion standard)
        sd_image_tensor = image_tensor * 2 - 1

        img_latent = retrieve_latents(self.vae.encode(sd_image_tensor), self.noise_generator)
        img_latent = img_latent * self.vae.config.scaling_factor

        if add_init_noise:
            img_latent = self.add_noise(img_latent, self.init_noise[0], 0)

        return img_latent

    @torch.inference_mode()
    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        output_latent = self.vae.decode(x_0_pred_out / self.vae.config.scaling_factor, return_dict=False)[0]
        return output_latent

    def predict_x0_batch(self, x_t_latent: torch.Tensor, control_cond: torch.Tensor) -> torch.Tensor:

        # prepare args for unet
        added_cond_kwargs = {}
        cross_attention_kwargs = {}

        # get previous batch buffers
        prev_latent_batch = self.latent_buffer
        prev_control_batch = self.control_buffer

        # image-prompt adapter
        if self.cached_ip_strength is not None and self.cached_ip_embeds is not None:
            added_cond_kwargs["image_embeds"] = self.cached_ip_embeds
            cross_attention_kwargs["ip_adapter_scale"] = self.cached_ip_strength

        # control adapter
        ctl_kwargs = {}
        if control_cond is not None:
            ctl_kwargs['controlnet_cond'] = control_cond
            ctl_kwargs['controlnet_cond_scale'] = self.cached_control_strength

        # SDXL specific additional conditioning
        if self.is_sdxl:
            base_added_cond_kwargs = {
                "text_embeds": self.cached_add_text_embeds,
                "time_ids": self.cached_add_time_ids
            }
            added_cond_kwargs.update(base_added_cond_kwargs)

        # handle batching
        if self.denoising_steps_num > 1:

            # handle noise accumulation
            self.stock_noise = torch.cat(
                (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
            )

            # handle batching
            x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
            if control_cond is not None:
                control_cond = torch.cat((control_cond, prev_control_batch), dim=0)

        else:
            # Single-step: Reset stock_noise to prevent accumulation
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                # Reset stock_noise for single-step to prevent accumulation
                self.stock_noise = self.init_noise.clone()

        # prepare arguments for unet
        t_list = self.sub_timesteps_pt
        t_list = t_list.to(self.device)

        # unet
        x_0_pred_batch, model_pred = self.unet_step(
            x_t_latent,
            t_list,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            **ctl_kwargs
        )

        # handle batching for next iteration
        if self.denoising_steps_num > 1:

            # get latest denoised latent
            x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)

            # update latent buffer
            if self.do_add_noise:
                self.latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                )
            else:
                self.latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                )

            # update control buffer
            if control_cond is not None:
                self.control_buffer = control_cond[:-1]

        else:

            x_0_pred_out = x_0_pred_batch
            self.latent_buffer = None

        return x_0_pred_out

    @torch.inference_mode()
    def __call__(
        self,
        input: torch.Tensor = None,
        control_image: torch.Tensor = None,
        encode_input: bool = True,
        decode_output: bool = True
    ) -> torch.Tensor:

        # Process control image if provided
        control_cond = None
        if control_image is not None:
            control_cond = self.generate_control_embedding(control_image)

        # check if input should be encoded
        if encode_input:

            # encode with VAE
            x_t_latent = self.encode_image(input)
        else:
            x_t_latent = input

        # diffusion
        x_0_pred_out = self.predict_x0_batch(
            x_t_latent,
            control_cond
        )

        # check if output should be decoded
        if decode_output:

            # decode with VAE
            x_output = self.decode_image(x_0_pred_out).detach().clone()
        else:
            x_output = x_0_pred_out

        return x_output

    @classmethod
    def fit_to_dimension(cls, tensor, dimension):

        if tensor is None:
            return None

        # Get the original batch size
        original_batch_size = tensor.shape[0]

        # If the original batch size is already equal to the desired dimension, return as is
        if original_batch_size == dimension:
            return tensor

        # Take the first element if there are multiple elements in the batch
        if original_batch_size > 1:
            tensor = tensor[0].unsqueeze(0)

        # Create a repeat tuple with ones for all dimensions except the first
        # For a tensor with ndim=3, this creates (dimension, 1, 1)
        # For a tensor with ndim=4, this creates (dimension, 1, 1, 1)
        repeat_dims = (dimension,) + (1,) * (tensor.ndim - 1)

        return tensor.repeat(*repeat_dims)

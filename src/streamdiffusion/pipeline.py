import time

from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import torch
import PIL.Image
import numpy as np

from compel import Compel, ReturnedEmbeddingsType
from diffusers.image_processor import VaeImageProcessor
from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents

from streamdiffusion.image_filter import SimilarImageFilter


class StreamDiffusion:
    def __init__(
        self,
        pipe: Optional[DiffusionPipeline],
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        device: Optional[str] = None,
        vae_scale_factor: Optional[int] = 8,
        original_inference_steps: Optional[int] = 50
    ) -> None:

        # optional arguments
        self.device = pipe.device if pipe else device
        self.vae_scale_factor = pipe.vae_scale_factor if pipe else vae_scale_factor

        self.dtype = torch_dtype
        self.generator = None

        self.timer_event = getattr(torch, str(self.device).split(':', 1)[0])

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        # Text Embeddings
        self.prompt_embeds = None
        self.add_text_embeds = None
        self.add_time_ids = None

        # Image Embeddings
        self.ip_adapter_image_embeds = None
        self.ip_adapter_scale = 0.0

        self.cfg_type = cfg_type
        self.alpha_prod_t_sqrt = None
        self.beta_prod_t_sqrt = None

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.image_processor = VaeImageProcessor(self.vae_scale_factor)
        self.inference_time_ema = 0

        # check if sd pipeline instance provided
        if pipe:

            # save pipe
            self.pipe = pipe

            # save pipeline components
            self.vae = pipe.vae
            self.unet = pipe.unet
            self.text_encoder = pipe.text_encoder

            # scheduler
            self.pipe.scheduler.config['original_inference_steps'] = original_inference_steps
            self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

            # advanced prompt encoding
            self.compel = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)

        self.sdxl = type(self.pipe) is StableDiffusionXLPipeline

        # override compel if we're in SDXL mode
        if self.sdxl:

            # For SDXL-Turbo, we use ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
            # We need pooled embeddings too
            self.compel = Compel(
                tokenizer=pipe.tokenizer,
                text_encoder=pipe.text_encoder,
                # returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                # requires_pooled=True
            )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:

        if not self.pipe:
            raise Exception("No pipeline")

        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:

        if not self.pipe:
            raise Exception("No pipeline")

        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.inference_mode()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        strength: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)

        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        # update prompt
        if prompt:
            self.update_prompt(prompt)

        # update image prompt
        if self.ip_adapter_image_embeds is None:
            black_image = PIL.Image.new('RGB', (224, 224), color=0)
            self.generate_image_embedding(black_image)
            self.ip_adapter_scale = 0.0

        self.scheduler.set_timesteps(num_inference_steps, self.device, strength=strength)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

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
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

    @torch.inference_mode()
    def generate_image_embedding(self, image: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> None:
        """
        Generate image embeddings for IP-Adapter from a reference image.

        Args:
            image: A reference image to control generation. Can be a PIL Image, tensor, or numpy array.
        """
        if not hasattr(self.pipe, "image_encoder") or self.pipe.image_encoder is None:
            raise ValueError(
                "The pipeline doesn't have an image_encoder. Make sure IP-Adapter is properly loaded.")

        # Use the pipeline's image processing methods directly
        if hasattr(self.pipe, "feature_extractor") and self.pipe.feature_extractor is not None:

            # Process the image using the pipeline's feature extractor
            if not isinstance(image, list):
                image = [image]

            # Process the image
            image = self.pipe.feature_extractor(
                images=image,
                return_tensors="pt",
            ).pixel_values.to(device=self.device, dtype=self.dtype)

            # Generate image embeddings using the pipeline's image encoder
            image_embeds = self.pipe.image_encoder(image).image_embeds

            # Store the image embeddings in the format expected by IP-Adapter
            self.ip_adapter_image_embeds = image_embeds

            print(f"Generated image embeddings with shape: {image_embeds.shape}")
        else:
            raise ValueError("Feature extractor not found. IP-Adapter may not be properly loaded.")

    @torch.inference_mode()
    def generate_prompt_embeddings(self, prompt: str):
        return self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    def update_prompt(self, prompt: str) -> None:

        # Get embeddings from Compel
        embeds = self.generate_prompt_embeddings(prompt) # self.compel(prompt)

        if not self.sdxl:

            # repeat embeds for batch size and store
            self.prompt_embeds = embeds.to(dtype=self.dtype).repeat(self.batch_size, 1, 1)

        else:

            # unpack embeds
            text_embeds, pooled_embeds = embeds[0], embeds[2]

            # repeat embeds for batch size and store
            self.prompt_embeds = text_embeds.to(dtype=self.dtype).repeat(self.batch_size, 1, 1)

            # Process pooled embeddings
            self.add_text_embeds = pooled_embeds.to(dtype=self.dtype)

            # Set up the additional time embeddings needed for SDXL
            self.add_time_ids = self._get_add_time_ids(
                (self.height, self.width),
                (0, 0),
                (self.height, self.width),
                dtype=self.prompt_embeds.dtype,
                text_encoder_projection_dim=int(self.add_text_embeds.shape[-1]),
            )

        # if self.use_denoising_batch and self.cfg_type == "full":
        #     uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        # elif self.cfg_type == "initialize":
        #     uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)
        #
        # if self.guidance_scale > 1.0 and (
        #         self.cfg_type == "initialize" or self.cfg_type == "full"
        # ):
        #     self.prompt_embeds = torch.cat(
        #         [uncond_prompt_embeds, self.prompt_embeds], dim=0
        #     )

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

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        added_cond_kwargs,
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        model_pred = self.unet(
            x_t_latent_plus_uc,
            t_list,
            cross_attention_kwargs={"ip_adapter_scale": torch.tensor([0.8888], device=self.device)},
            encoder_hidden_states=self.prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
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

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        # passed_add_embed_dim = (
        #     self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        # )
        # expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        #
        # if expected_add_embed_dim != passed_add_embed_dim:
        #     raise ValueError(
        #         f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        #     )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.inference_mode()
    def encode_image(self, image_tensors: torch.Tensor, add_init_noise: bool = True) -> torch.Tensor:
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor

        if add_init_noise:
            img_latent = self.add_noise(img_latent, self.init_noise[0], 0)

        return img_latent

    @torch.inference_mode()
    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        output_latent = self.vae.decode(
            x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return output_latent

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        added_cond_kwargs = {}
        prev_latent_batch = self.x_t_latent_buffer

        # Set the IP-Adapter scale directly on the pipe BEFORE inference
        # This is the critical step that was missing
        if hasattr(self, "ip_adapter_scale") and self.ip_adapter_scale is not None:

            # This sets the scale on the UNet's attention processors directly
            self.pipe.set_ip_adapter_scale(self.ip_adapter_scale)

        # Add IP-Adapter embeddings if available
        if self.ip_adapter_image_embeds is not None:

            # Add the image embeddings to added_cond_kwargs
            added_cond_kwargs["image_embeds"] = self.ip_adapter_image_embeds

            # Add the scale for IP-Adapter if it's needed by the implementation
            if hasattr(self.pipe, "config") and getattr(self.pipe.config, "requires_adapter_scale", False):
                added_cond_kwargs["adapter_scale"] = self.ip_adapter_scale

        # Handle SDXL specific added conditions
        if self.sdxl:
            base_added_cond_kwargs = {
                "text_embeds": self.add_text_embeds.to(self.device),
                "time_ids": self.add_time_ids.to(self.device)
            }
            added_cond_kwargs.update(base_added_cond_kwargs)

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                )

            x_t_latent = x_t_latent.to(self.device)
            t_list = t_list.to(self.device)
            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list, added_cond_kwargs=added_cond_kwargs)

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx=idx, added_cond_kwargs=added_cond_kwargs)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred

        return x_0_pred_out

    @torch.inference_mode()
    def __call__(
        self,
        x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None,
        encode_input: bool = True,
        decode_output: bool = True
    ) -> torch.Tensor:

        start = self.timer_event.Event(enable_timing=True)
        end = self.timer_event.Event(enable_timing=True)
        start.record()

        # pre process
        x = self.image_processor.preprocess(x, self.height, self.width).to(
            device=self.device, dtype=self.dtype
        )

        # similarity filter
        if self.similar_image_filter:
            x = self.similar_filter(x)
            if x is None:
                time.sleep(self.inference_time_ema)
                return self.prev_image_result

        # check if input should be encoded
        if encode_input:

            # encode with VAE
            x_t_latent = self.encode_image(x)
        else:
            x_t_latent = x

        # diffusion
        x_0_pred_out = self.predict_x0_batch(x_t_latent)

        # check if output should be decoded
        if decode_output:

            # decode with VAE
            x_output = self.decode_image(x_0_pred_out).detach().clone()
        else:
            x_output = x_0_pred_out

        self.prev_image_result = x_output
        end.record()
        self.timer_event.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time

        return x_output


    @staticmethod
    def slerp(v1, v2, t, DOT_THR=0.9995, zdim=-1):
        """SLERP for pytorch tensors interpolating `v1` to `v2` with scale of `t`.

        `DOT_THR` determines when the vectors are too close to parallel.
            If they are too close, then a regular linear interpolation is used.

        `zdim` is the feature dimension over which to compute norms and find angles.
            For example: if a sequence of 5 vectors is input with shape [5, 768]
            Then `zdim = 1` or `zdim = -1` computes SLERP along the feature dim of 768.

        Theory Reference:
        https://splines.readthedocs.io/en/latest/rotation/slerp.html
        PyTorch reference:
        https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
        Numpy reference:
        https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
        """

        # take the dot product between normalized vectors
        v1_norm = v1 / torch.norm(v1, dim=zdim, keepdim=True)
        v2_norm = v2 / torch.norm(v2, dim=zdim, keepdim=True)
        dot = (v1_norm * v2_norm).sum(zdim)

        # if the vectors are too close, return a simple linear interpolation
        if (torch.abs(dot) > DOT_THR).any():
            print(f'warning: v1 and v2 close to parallel, using linear interpolation instead.')
            res = (1 - t) * v1 + t * v2

        # else apply SLERP
        else:
            # compute the angle terms we need
            theta = torch.acos(dot)
            theta_t = theta * t
            sin_theta = torch.sin(theta)
            sin_theta_t = torch.sin(theta_t)

            # compute the sine scaling terms for the vectors
            s1 = torch.sin(theta - theta_t) / sin_theta
            s2 = sin_theta_t / sin_theta

            # interpolate the vectors
            res = (s1.unsqueeze(zdim) * v1) + (s2.unsqueeze(zdim) * v2)

        return res

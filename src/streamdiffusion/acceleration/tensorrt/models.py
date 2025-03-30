import torch


class BaseModel:
    def __init__(
        self,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=None,
        text_maxlen=None,
    ):
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose

        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.min_image_shape = 256      # min image resolution: 256x256
        self.max_image_shape = 1024     # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def check_dims(self, batch_size, image_height, image_width):
        assert self.min_batch <= batch_size <= self.max_batch, f"Invalid batch_size: {self.min_batch=} <= {batch_size=} <= {self.max_batch=}"
        assert image_height % 8 == 0 or image_width % 8 == 0, f"Invalid image_height: {image_height} % 8 != 0 or image_width: {image_width} % 8 != 0"
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert self.min_latent_shape <= latent_height <= self.max_latent_shape, f"Invalid image_height: {self.min_latent_shape=} <= {latent_height=} <= {self.max_latent_shape=}"
        assert self.min_latent_shape <= latent_width <= self.max_latent_shape, f"Invalid image_width: {self.min_latent_shape=} <= {latent_width=} <= {self.max_latent_shape=}"
        return latent_height, latent_width

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


class CLIP(BaseModel):
    def __init__(self, device, max_batch_size, embedding_dim, min_batch_size=1):
        super(CLIP, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
        )
        self.name = "CLIP"

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    # def optimize(self, onnx_graph):
    #     opt = Optimizer(onnx_graph)
    #     opt.info(self.name + ": original")
    #     opt.select_outputs([0])  # delete graph output#1
    #     opt.cleanup()
    #     opt.info(self.name + ": remove output[1]")
    #     opt.fold_constants()
    #     opt.info(self.name + ": fold constants")
    #     opt.infer_shapes()
    #     opt.info(self.name + ": shape inference")
    #     opt.select_outputs([0], names=["text_embeddings"])  # rename network output
    #     opt.info(self.name + ": remove output[0]")
    #     opt_onnx_graph = opt.cleanup(return_onnx=True)
    #     opt.info(self.name + ": finished")
    #     return opt_onnx_graph


class UNet(BaseModel):
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch_size=1,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
    ):
        super(UNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNet"

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "B", 2: "H", 3: "W"},
            "timestep": {0: "T"},
            "encoder_hidden_states": {0: "B"},
            "latent": {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [
                (batch_size, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (batch_size, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [
                (min_batch,),
                (batch_size,),
                (max_batch,)
            ],
            "encoder_hidden_states": [
                (batch_size, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (1, 3),
            "encoder_hidden_states": (batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (batch_size, self.unet_dim, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32

        # input latent
        sample_input = torch.randn(
            batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device
        )

        # timestep (allowing for 1 to 3 timesteps)
        timestep_input = torch.randint(1, 4, (1,), dtype=torch.float32, device=self.device)

        # Generate the encoder hidden states input
        encoder_hidden_states_input = torch.randn(
            batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device
        )

        return sample_input, timestep_input, encoder_hidden_states_input


class UNetXLTurbo(BaseModel):
    def __init__(
        self,
        fp16=True,
        device="cuda",
        max_batch_size=1,
        min_batch_size=1,
        encoder_hidden_states_dim=2048,   # Updated for SDXL-Turbo
        text_maxlen=77,                   # Updated for SDXL-Turbo
        text_embeds_dim=1280,             # SDXL-Turbo-specific
        time_ids_maxlen=6,                # SDXL-Turbo-specific
        embedding_dim=768,
        unet_dim=4
    ):
        super(UNetXLTurbo, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.text_embeds_dim = text_embeds_dim
        self.time_ids_maxlen = time_ids_maxlen
        self.name = "UnetXLTurbo"

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "B", 2: "H", 3: "W"},
            "timestep": {0: "T"},
            "encoder_hidden_states": {0: "B"},
            "latent": {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [
                (batch_size, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (batch_size, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [
                (min_batch,),
                (batch_size,),
                (max_batch,)
            ],
            "encoder_hidden_states": [
                (batch_size, self.text_maxlen, self.encoder_hidden_states_dim),
                (batch_size, self.text_maxlen, self.encoder_hidden_states_dim),
                (batch_size, self.text_maxlen, self.encoder_hidden_states_dim),
            ],
            "text_embeds": [
                (batch_size, self.text_embeds_dim),
                (batch_size, self.text_embeds_dim),
                (batch_size, self.text_embeds_dim),
            ],
            "time_ids": [
                (batch_size, self.time_ids_maxlen),
                (batch_size, self.time_ids_maxlen),
                (batch_size, self.time_ids_maxlen),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (1, 3),
            "encoder_hidden_states": (batch_size, self.text_maxlen, self.encoder_hidden_states_dim),
            "text_embeds": (batch_size, self.text_embeds_dim),
            "time_ids": (batch_size, self.time_ids_maxlen),
            "latent": (batch_size, self.unet_dim, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32

        # input latent
        sample_input = torch.randn(
            batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device
        )

        # timestep (allowing for 1 to 3 timesteps)
        timestep_input = torch.randint(1, 4, (1,), dtype=torch.float32, device=self.device)

        # encoder hidden states input
        encoder_hidden_states_input = torch.randn(
            batch_size, self.text_maxlen, self.encoder_hidden_states_dim, dtype=dtype, device=self.device
        )

        # additional text embeds
        add_text_embeds = torch.randn(batch_size, self.text_embeds_dim, dtype=dtype, device=self.device)

        # time ids
        add_time_ids = torch.randint(0, 1000, (batch_size, self.time_ids_maxlen), dtype=torch.int32, device=self.device)

        return sample_input, timestep_input, encoder_hidden_states_input, add_text_embeds, add_time_ids


class UNetXLTurboIPAdapter(BaseModel):
    def __init__(
        self,
        fp16=True,
        device="cuda",
        max_batch_size=1,
        min_batch_size=1,
        encoder_hidden_states_dim=2048,   # Updated for SDXL-Turbo
        text_maxlen=77,                   # Updated for SDXL-Turbo
        text_embeds_dim=1280,             # SDXL-Turbo-specific
        time_ids_maxlen=6,                # SDXL-Turbo-specific
        embedding_dim=768,
        unet_dim=4
    ):
        super(UNetXLTurboIPAdapter, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.text_embeds_dim = text_embeds_dim
        self.time_ids_maxlen = time_ids_maxlen
        self.name = "UNetXLTurboIPAdapter"

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids", "image_embeds", "ip_adapter_scale"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return None

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [
                (batch_size, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (batch_size, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [
                (min_batch,),
                (batch_size,),
                (max_batch,)
            ],
            "encoder_hidden_states": [
                (batch_size, self.text_maxlen, self.encoder_hidden_states_dim),
                (batch_size, self.text_maxlen, self.encoder_hidden_states_dim),
                (batch_size, self.text_maxlen, self.encoder_hidden_states_dim),
            ],
            "text_embeds": [
                (batch_size, self.text_embeds_dim),
                (batch_size, self.text_embeds_dim),
                (batch_size, self.text_embeds_dim),
            ],
            "time_ids": [
                (batch_size, self.time_ids_maxlen),
                (batch_size, self.time_ids_maxlen),
                (batch_size, self.time_ids_maxlen),
            ],
            "image_embeds": [
                (batch_size, 1, 4, self.encoder_hidden_states_dim),
                (batch_size, 1, 4, self.encoder_hidden_states_dim),
                (batch_size, 1, 4, self.encoder_hidden_states_dim),
            ],
            "ip_adapter_scale": [
                (1,),
                (1,),
                (1,)
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (1, 3),
            "encoder_hidden_states": (batch_size, self.text_maxlen, self.encoder_hidden_states_dim),
            "text_embeds": (batch_size, self.text_embeds_dim),
            "time_ids": (batch_size, self.time_ids_maxlen),
            "image_embeds": (batch_size, 1, 4, self.encoder_hidden_states_dim),
            "ip_adapter_scale": (1,),
            "latent": (batch_size, self.unet_dim, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32

        # input latent
        sample_input = torch.randn(
            batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device
        )

        # timestep (allowing for 1 to 3 timesteps)
        timestep_input = torch.randint(1, 4, (1,), dtype=torch.float32, device=self.device)

        # encoder hidden states input
        encoder_hidden_states_input = torch.randn(
            batch_size, self.text_maxlen, self.encoder_hidden_states_dim, dtype=dtype, device=self.device
        )

        # additional text embeds
        text_embeds = torch.randn(batch_size, self.text_embeds_dim, dtype=dtype, device=self.device)

        # time ids
        time_ids = torch.randint(0, 1000, (batch_size, self.time_ids_maxlen), dtype=torch.int32, device=self.device)

        # image embeds
        image_embeds = torch.randn(
            batch_size, 1, 4, self.encoder_hidden_states_dim, dtype=dtype, device=self.device
        )

        # ip_adapter_scale
        ip_adapter_scale = torch.randint(1, 4, (1,), dtype=dtype, device=self.device)

        return sample_input, timestep_input, encoder_hidden_states_input, text_embeds, time_ids, image_embeds, ip_adapter_scale


class VAE(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAE, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE decoder"

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {
            "latent": {0: "B", 2: "H", 3: "W"},
            "images": {0: "B", 2: "8H", 3: "8W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            4,
            latent_height,
            latent_width,
            dtype=torch.float32,
            device=self.device,
        )


class VAEEncoder(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAEEncoder, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE encoder"

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "images": {0: "B", 2: "8H", 3: "8W"},
            "latent": {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            3,
            image_height,
            image_width,
            dtype=torch.float32,
            device=self.device,
        )

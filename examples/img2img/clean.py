import torch
from PIL import Image
from PIL import  ImageDraw, ImageFont

from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
from diffusers.utils import load_image, make_image_grid

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.ip_adapter import prepare_unet_for_onnx_export


def add_label(image, text):
    draw = ImageDraw.Draw(image)
    # Use a default font if custom font is not available
    font_size = 40
    try:
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Add white background for text
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
    draw.rectangle([(0, 0), (text_width + 10, text_height + 10)], fill=(255, 255, 255))
    draw.text((5, 5), text, fill=(0, 0, 0), font=font)
    return image

# torch.cuda.empty_cache()

device = 'mps'
dtype = torch.float16

# load vae
print('Loading Tiny VAE')
vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesdxl",
    torch_dtype=dtype,
).to(device)

# load pipeline
print('Loading Pipeline')
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/sdxl-turbo',
    vae=vae,
    torch_dtype=dtype,
    variant='fp16'
).to(device)

# Load and add IP-Adapter to the pipeline
print('Loading IP Adapter')
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.safetensors",
    torch_dtype=dtype
)

# replace attention processors
print('Replacing attention processors')
prepare_unet_for_onnx_export(pipe)

# StreamDiffusion
print('Loading StreamDiffusion')
stream = StreamDiffusion(
    pipe,
    device=device,
    t_index_list=[33],
    original_inference_steps=100,
    torch_dtype=dtype,
    do_add_noise=False,
    height=512,
    width=904
)

print('Loading Images')
# Load images
input_image = load_image("https://images.saatchiart.com/saatchi/2158365/art/10132877/9195647-KGRGTRFG-7.jpg").resize((512, 512))
reference_image = load_image("https://kavkaartbooks.s3.cdn-upgates.com/_cache/8/b/8b57c4db665a0b35b683babd11c1e860-img-3650.jpg")  # Image for IP-Adapter

# Generate image embedding for IP-Adapter
print('Generating IP Embeddings')
black_image = Image.new('RGB', (224, 224), color=0)
stream.generate_image_embedding(black_image)
# stream.generate_image_embedding(reference_image)

# Set up generation parameters
prompt = "ibex, high quality, best quality"

# Loop through different scales
ip_scales = [0.6, 0.7, 0.8, 0.9]
strengths = [0.4, 0.5, 0.6, 0.7]
all_images = []

for ip in ip_scales:
    for s in strengths:

        print(f'Preparing for {ip=}, {s=}')
        stream.ip_adapter_scale = ip

        # prepare
        stream.t_list = [99 - int(100 * s)]
        stream.denoising_steps_num = len(stream.t_list)
        stream.prepare(
            prompt=prompt,
            num_inference_steps=100,
            guidance_scale=0.0,
            seed=int(123)
        )

        print(f'Generating for {ip=}, {s=}')
        for _ in range(stream.denoising_steps_num - 1):
            stream(input_image, encode_input=True, decode_output=True)
        img_pt = stream(input_image, encode_input=True, decode_output=True)
        img_pil = postprocess_image(img_pt)[0]
        img_pil = add_label(img_pil, f'ip={ip}, str={s}')
        all_images.append(img_pil)

grid = make_image_grid(all_images, rows=1, cols=len(all_images))
# grid = make_image_grid(all_images, rows=4, cols=len(all_images) // 4)
grid.save('grid.jpg')
grid.show()

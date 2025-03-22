import re
import time
import torch

from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionPipeline


pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16
).to("mps")
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, device='mps')
# compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
#                 text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
#                 returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
#                 requires_pooled=[False, True])

prompt = 'sunrise over snowy mountains, (huge dog floating in the sky)1.5, 8k'

# compel
times = []
for i in range(10):
    start = time.perf_counter()
    encoder_output = compel.build_weighted_embedding(prompt)
    times.append(time.perf_counter() - start)
print(f"Average compel time: {sum(times) / len(times)}")

import re
import time
import torch

from compel import Compel
from diffusers import StableDiffusionPipeline


pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16
).to("mps")
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, device='mps')


def parse_prompt_subprompts(prompt: str):
    """
    Parses a prompt to extract subprompts and their weights.

    Example:
        Input: "sunset on (snowy)1.5 mountain"
        Output: [("sunset on", 1.0), ("snowy", 1.5), ("mountain", 1.0)]
    """
    pattern = r"\(([^)]+)\)([\d.]+)"
    segments = []
    last_end = 0

    for match in re.finditer(pattern, prompt):
        text_before = prompt[last_end:match.start()].strip()  # Strip spaces
        if text_before:  # Avoid adding empty or whitespace-only segments
            segments.append((text_before, 1.0))

        bracketed_text = match.group(1).strip()  # Strip spaces from inside the parentheses
        weight = float(match.group(2))
        if bracketed_text:  # Ensure there's actual text
            segments.append((bracketed_text, weight))

        last_end = match.end()

    text_after = prompt[last_end:].strip()  # Strip spaces at the end
    if text_after:
        segments.append((text_after, 1.0))

    return segments

def faster_compel(prompt):

    segments = parse_prompt_subprompts(prompt)
    fragments = [segment[0] for segment in segments]
    weights = [segment[1] for segment in segments]

    this_conditioning = compel.conditioning_provider.get_embeddings_for_weighted_prompt_fragments(
        text_batch=[fragments],
        fragment_weights_batch=[weights],
        device='mps'
    )
    return this_conditioning


prompt = 'sunrise over snowy mountains, (huge dog floating in the sky)1.5, 8k'

# compel
times = []
for i in range(10):
    start = time.perf_counter()
    # compel.build_conditioning_tensor(prompt)
    faster_compel(prompt)
    times.append(time.perf_counter() - start)
print(f"Average compel time: {sum(times) / len(times)}")


# # Create a LineProfiler instance and tell it to profile the __call__ method.
# lp = line_profiler.LineProfiler()
# lp.add_function(compel.__call__)
#
# # Run the __call__ method inside the profiler.
# result = lp.runcall(compel.__call__, prompt)
#
# # Print the line-by-line profiling results.
# lp.print_stats()
#

# # pipeline encode
# times = []
# for i in range(100):
#     start = time.perf_counter()
#     pipe.encode_prompt(
#         prompt=prompt,
#         device='mps',
#         num_images_per_prompt=1,
#         do_classifier_free_guidance=False
#     )
#     times.append(time.perf_counter() - start)
# print(f"Average pipe time: {sum(times) / len(times)}")

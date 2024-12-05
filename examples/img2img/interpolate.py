import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderTiny

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


def pil_to_cv2(pil_image):
    """Convert a PIL Image to an OpenCV-compatible NumPy array."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(frame):
    """Convert an OpenCV frame (NumPy array) to a PIL Image."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def frames_to_video(frames, frame_rate, video_file_path):

    # Read the first image to determine the video frame size
    height, width = frames[0].height, frames[0].width

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video_writer = cv2.VideoWriter(video_file_path, fourcc, frame_rate, (width, height))

    # Write each image into the video
    for frame in frames:
        frame_cv2 = pil_to_cv2(frame)
        video_writer.write(frame_cv2)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {video_file_path}")

def init():
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo").to(
        device=torch.device("mps"),
        dtype=torch.float16
    )

    # Wrap the pipeline in StreamDiffusion
    stream = StreamDiffusion(
        pipe,
        t_index_list=[33],
        torch_dtype=torch.float16
    )

    # Use Tiny VAE for further acceleration
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

    return pipe, stream

def video_to_frames(video_path):

    frames = []

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break  # Exit loop if no more frames or error

        # Convert the frame to a PIL Image
        pil_image = cv2_to_pil(frame)
        frames.append(pil_image)

    # Release the video capture object
    cap.release()

    return frames

def interpolate(image1, image2, num_interpolated, stream):

    height, width = image1.height, image1.width

    image1_pt = stream.image_processor.preprocess(image1, height, width)
    image2_pt = stream.image_processor.preprocess(image2, height, width)

    image1_lat = stream.encode_image(image1_pt, add_init_noise=False)
    image2_lat = stream.encode_image(image2_pt, add_init_noise=False)

    interpolated_latents = []
    for weight in np.linspace(0, 1, num_interpolated + 1, endpoint=False)[1:]:

        # spherical interpolation
        inter_lat = stream.slerp(image1_lat, image2_lat, weight)

        # decode to latent to tensor
        interpolated_latents.append(inter_lat)

    interpolated_images = []
    for img_lat in interpolated_latents:
        img_pt = stream.decode_image(img_lat).detach().clone()
        img_pil = postprocess_image(img_pt, output_type="pil")
        interpolated_images.append(img_pil[0])

    return interpolated_images


def blend(image, blend_image, weight, stream):

    height, width = image.height, image.width

    image_pt = stream.image_processor.preprocess(image, height, width)
    blend_image_pt = stream.image_processor.preprocess(blend_image, height, width)

    image_lat = stream.encode_image(image_pt, add_init_noise=False)
    blend_image_lat = stream.encode_image(blend_image_pt, add_init_noise=False)

    # blend
    res_lat = stream.slerp(image_lat, blend_image_lat, weight)

    # decode
    res_pt = stream.decode_image(res_lat).detach().clone()

    # convert back to image
    res_pil = postprocess_image(res_pt, output_type="pil")

    return res_pil[0]


def blend_video(stream, weight, fps, source_video_path, target_video_path):

    # result
    all_images = []

    # read frames
    frames = video_to_frames(source_video_path)

    # iterate frame pairs
    for idx, (image1_pil, image2_pil) in enumerate(zip(frames, frames[1:])):

        # interpolate
        print(f'Interpolating frames: {idx} and {idx + 1}')
        blended_frame = blend(image1_pil, image2_pil, weight, stream)

        # add to list
        all_images.append(blended_frame)

    # create video
    frames_to_video(all_images, frame_rate=fps, video_file_path=target_video_path)


def interpolate_video(stream, num_inter, fps, source_video_path, target_video_path):

    # result
    all_images = []

    # read frames
    frames = video_to_frames(source_video_path)

    # iterate frame pairs
    for idx, image1_pil, image2_pil in enumerate(zip(frames, frames[1:])):

        # interpolate
        print(f'Interpolating frames: {idx} and {idx + 1}')
        inter_images = interpolate(image1_pil, image2_pil, num_inter, stream)

        # add to list
        all_images.extend(inter_images)
        all_images.append(image2_pil)

    # insert first frame
    all_images.insert(0, frames[0])

    # create video
    frames_to_video(all_images, frame_rate=fps, video_file_path=target_video_path)


if __name__ == "__main__":

    pipe, stream = init()



    # interpolate_video(
    #     stream,
    #     num_inter=1,
    #     fps=24,
    #     source_video_path='/Users/himmelroman/Desktop/interp/duchess/input_duchess.12fps.15s.mp4',
    #     target_video_path='/Users/himmelroman/Desktop/interp/duchess/input_duchess.24fps.i1.mp4'
    # )
    #
    # blend_video(
    #     stream,
    #     weight=0.25,
    #     fps=24,
    #     source_video_path='/Users/himmelroman/Desktop/interp/duchess/input_duchess.30fps.15s.mp4',
    #     target_video_path='/Users/himmelroman/Desktop/interp/duchess/input_duchess.30fps.b25.mp4'
    # )

import sys

sys.path.append("../")

import os
import json
import time
import psutil
import argparse
import torch
import torchvision
import cv2
import numpy as np
import gradio as gr

from tools.painter import mask_painter

from track_anything import TrackingAnything
from model.misc import get_device
from utils.download_util import load_file_from_url


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=8000, help="only useful when running gradio applications")
    parser.add_argument('--mask_save', default=False)
    parser.add_argument('--checkpoint_folder', type=str, default='weights')
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()

    if not args.device:
        args.device = str(get_device())

    return args


args = parse_augment()

checkpoint_folder = args.checkpoint_folder
pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
sam_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
ckpt_folder = checkpoint_folder

sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type], ckpt_folder)
cutie_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'cutie-base-mega.pth'), ckpt_folder)
propainter_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'ProPainter.pth'), ckpt_folder)
raft_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'raft-things.pth'), ckpt_folder)
flow_completion_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), ckpt_folder)

# initialize sam, cutie, propainter models
trackAnything = TrackingAnything(sam_checkpoint, cutie_checkpoint, propainter_checkpoint, raft_checkpoint, flow_completion_checkpoint, args)

title = r"""<h1>Ai remove tools</h1>"""

description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/biaggii/AI-remove-anything' target='_blank'><b>Improving Propagation and Transformer for Video Inpainting (ICCV 2023)</b></a>.<br>
"""
article = r"""
If is is helpful, please help to ⭐ the <a href='https://github.com/biaggii/AI-remove-anything' target='_blank'>Github Repo</a>. Thanks! 
---

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>costa.biaggi@gmail.com</b>.
<div>
    🤗 Find Me:
    costa.biaggi@gmail.com
</div>

"""
css = """
.gradio-container {width: 85% !important}
.gr-monochrome-group {border-radius: 5px !important; border: revert-layer !important; border-width: 2px !important; color: black !important}
button {border-radius: 8px !important;}
.add_button {background-color: #4CAF50 !important;}
.remove_button {background-color: #f44336 !important;}
.mask_button_group {gap: 10px !important;}
.video {height: 300px !important;}
.image {height: 300px !important;}
.video .wrap.svelte-lcpz3o {display: flex !important; align-items: center !important; justify-content: center !important;}
.video .wrap.svelte-lcpz3o > :first-child {height: 100% !important;}
.margin_center {width: 50% !important; margin: auto !important;}
.jc_center {justify-content: center !important;}
"""


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.

    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path


# tracking vos
def vos_tracking_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("", ""), ("Tracking finished! Try to click the Inpainting button to get the inpainting result.", "Normal")]
    trackAnything.cutie.clear_memory()
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1, len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            template_mask = np.clip(template_mask + interactive_state["multi_mask"]["masks"][mask_number] * (mask_number + 1), 0, mask_number + 1)
        video_state["masks"][video_state["select_frame_number"]] = template_mask
    else:
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    # operation error
    if len(np.unique(template_mask)) == 1:
        template_mask[0][0] = 1
        operation_log = [("Please add at least one mask to track by clicking the image in step2.", "Error"), ("", "")]
        # return video_output, video_state, interactive_state, operation_error
    masks, logits, painted_images = trackAnything.generator(images=following_frames, template_mask=template_mask)
    # clear GPU memory
    trackAnything.cutie.clear_memory()

    if interactive_state["track_end_number"]:
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        video_state["logits"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = logits
        video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
    else:
        video_state["masks"][video_state["select_frame_number"]:] = masks
        video_state["logits"][video_state["select_frame_number"]:] = logits
        video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

    video_output = generate_video_from_frames(
            video_state["painted_images"],
            output_path="./result/track/{}".format(video_state["video_name"]),
            fps=fps
            )  # import video_input to name the output video
    interactive_state["inference_times"] += 1


    # convert points input to prompt state
def get_prompt(c_state, click_input):
    inputs = json.loads(click_input)
    points = c_state[0]
    labels = c_state[1]
    for i in inputs:
        points.append(i[:2])
        labels.append(i[2])
    c_state[0] = points
    c_state[1] = labels
    prompt = {
        "prompt_type"     : ["click"],
        "input_point"     : c_state[0],
        "input_label"     : c_state[1],
        "multimask_output": "True",
        }
    return prompt


def sam_refine(video_states, point_prompt, click_state, interactive_state, evt: gr.SelectData):
    if point_prompt == "Positive":
        coordinate = f'[[{evt.index[0]},{evt.index[1]},1]]'
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = f'[[{evt.index[0]},{evt.index[1]},0]]'
        interactive_state["negative_click_times"] += 1

    # prompt for sam model
    trackAnything.samcontroler.sam_control.reset_image()
    trackAnything.samcontroler.sam_control.set_image(video_states["origin_images"][video_states["select_frame_number"]])
    prompt = get_prompt(c_state=click_state, click_input=coordinate)

    mask, logit, painted_image = trackAnything.first_frame_click(
            image=video_states["origin_images"][video_states["select_frame_number"]],
            points=np.array(prompt["input_point"]),
            labels=np.array(prompt["input_label"]),
            multimask=prompt["multimask_output"],
            )
    video_states["masks"][video_states["select_frame_number"]] = mask
    video_states["logits"][video_states["select_frame_number"]] = logit
    video_states["painted_images"][video_states["select_frame_number"]] = painted_image

    operation_log = [("[Must Do]", "Add mask"), (": add the current displayed mask for video segmentation.\n", None),
                     ("[Optional]", "Remove mask"), (": remove all added masks.\n", None),
                     ("[Optional]", "Clear clicks"), (": clear current displayed mask.\n", None),
                     ("[Optional]", "Click image"), (": Try to click the image shown in step2 if you want to generate more masks.\n", None)]
    return painted_image, video_states, interactive_state, operation_log, operation_log


# Get video frames
def get_frames_from_video(input_video, video_status):
    """
    Args:
        [input]
        input_video:str
        video_status
    Return
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
        [output]
        video_state, video_info, template_frame, image_selection_slider,
        track_pause_number_slider, point_prompt, clear_button_click, Add_mask_button, template_frame,
        tracking_video_predict_button, tracking_video_output, inpainting_video_output, remove_mask_button,
        inpaint_video_predict_button, step2_title, step3_title, mask_dropdown, run_status, run_status2
    """
    frames = []
    video_id = time.time()
    operation_log = [("[Must Do]", "Click image"), (": Video uploaded! Try to click the image shown in step2 to add masks.\n", None)]
    fps = 30
    try:
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if psutil.virtual_memory().percent > 90:
                operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    except (OSError, TypeError, ValueError, KeyError, SystemError) as e:
        print(f'read frames from video error: {e}')

    image_size = (frames[0].shape[0], frames[0].shape[1])
    # initialize the video_state
    video_status = {
        "video_id"           : video_id,
        "video_name"         : os.path.split(input_video)[-1],
        "origin_images"      : frames,
        "painted_images"     : frames.copy(),
        "masks"              : [np.zeros((frames[0].shape[0], frames[0].shape[1]), np.uint8)] * len(frames),
        "logits"             : [None] * len(frames),
        "select_frame_number": 0,
        "fps"                : fps
        }
    info = f"Video Name: {video_status['video_name']},\nFPS: {fps},\nTotal Frames: {len(frames)},\nImage Size:{image_size}"
    """
        [output]
        video_state, video_info, template_frame, image_selection_slider,
        track_pause_number_slider, point_prompt, clear_button_click, Add_mask_button, template_frame,
        tracking_video_predict_button, tracking_video_output, inpainting_video_output, remove_mask_button,
        inpaint_video_predict_button, step2_title, step3_title, mask_dropdown, run_status, run_status2
    """

    trackAnything.samcontroler.sam_control.reset_image()
    trackAnything.samcontroler.sam_control.set_image(video_status["origin_images"][0])

    return (video_status, info, video_status["origin_images"][0], gr.update(visible=True, maximum=len(frames), value=1),
            gr.update(visible=True, maximum=len(frames), value=len(frames)), gr.update(visible=True),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=True, choices=[], value=[]),
            gr.update(visible=True, value=operation_log), gr.update(visible=True, value=operation_log))


# Webui
with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as app:
    gr.Markdown(title)

    click_state = gr.State([[], []])

    interactive_state = gr.State(
            {
                "inference_times"     : 0,
                "negative_click_times": 0,
                "positive_click_times": 0,
                "mask_save"           : args.mask_save,
                "multi_mask"          : {
                    "mask_names": [],
                    "masks"     : []
                    },
                "track_end_number"    : None,
                }
            )

    video_state = gr.State(
            {
                "video_id"           : "",
                "video_name"         : "",
                "origin_images"      : None,
                "painted_images"     : None,
                "masks"              : None,
                "inpaint_masks"      : None,
                "logits"             : None,
                "select_frame_number": 0,
                "fps"                : 30
                }
            )
    # Parameters _________________________________________________________________
    with gr.Group(elem_classes="gr-monochrome-group"):
        with gr.Row():
            with gr.Accordion('ProPainter Parameters', open=False):
                with gr.Row():
                    resize_ratio_number = gr.Slider(
                            label='Resize ratio',
                            minimum=0.01,
                            maximum=1.0,
                            step=0.01,
                            value=1.0
                            )
                    raft_iter_number = gr.Slider(
                            label='Iterations for RAFT inference.',
                            minimum=5,
                            maximum=20,
                            step=1,
                            value=20, )
                with gr.Row():
                    dilate_radius_number = gr.Slider(
                            label='Mask dilation for video and flow masking.',
                            minimum=0,
                            maximum=10,
                            step=1,
                            value=8, )

                    subvideo_length_number = gr.Slider(
                            label='Length of sub-video for long video inference.',
                            minimum=40,
                            maximum=200,
                            step=1,
                            value=80, )
                with gr.Row():
                    neighbor_length_number = gr.Slider(
                            label='Length of local neighboring frames.',
                            minimum=5,
                            maximum=20,
                            step=1,
                            value=10, )

                    ref_stride_number = gr.Slider(
                            label='Stride of global reference frames.',
                            minimum=5,
                            maximum=20,
                            step=1,
                            value=10, )

    # Input and Output Video _________________________________________________________________
    with gr.Column():
        # input video
        gr.Markdown("## Step1: Upload video")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                video_input = gr.Video(elem_classes="video")
                extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary")
            with gr.Column(scale=2):
                run_status = gr.HighlightedText(
                        value=[("", ""), ("Try to upload your video and click the Get video info button to get started!", "Normal")],
                        color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"}
                        )
                video_info = gr.Textbox(label="Video Info")

        # add masks
        step2_title = gr.Markdown("---\n## Step2: Add masks", visible=False)
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                template_frame = gr.Image(type="pil", interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track start frame", visible=False)
                track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
            with gr.Column(scale=2, elem_classes="jc_center"):
                run_status2 = gr.HighlightedText(
                        value=[("", ""), ("Try to upload your video and click the Get video info button to get started!", "Normal")],
                        color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"}
                        )
                with gr.Row():
                    with gr.Column(scale=2, elem_classes="mask_button_group"):
                        clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False)
                        remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=False, elem_classes="remove_button")
                        Add_mask_button = gr.Button(value="Add mask", interactive=True, visible=False, elem_classes="add_button")
                    point_prompt = gr.Radio(
                            choices=["Positive", "Negative"],
                            value="Positive",
                            label="Point prompt",
                            interactive=True,
                            visible=False,
                            min_width=100,
                            scale=1
                            )
                mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)

        # output video
        step3_title = gr.Markdown("---\n## Step3: Track masks and get the inpainting result", visible=False)
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                tracking_video_output = gr.Video(visible=False, elem_classes="video")
                tracking_video_predict_button = gr.Button(value="1. Tracking", visible=False, elem_classes="margin_center")
            with gr.Column(scale=2):
                inpainting_video_output = gr.Video(visible=False, elem_classes="video")
                inpaint_video_predict_button = gr.Button(value="2. Inpainting", visible=False, elem_classes="margin_center")

        # first step: get the video information
        extract_frames_button.click(
                fn=get_frames_from_video,
                inputs=[video_input, video_state],
                outputs=[video_state, video_info, template_frame,
                         image_selection_slider, track_pause_number_slider, point_prompt, clear_button_click, Add_mask_button, template_frame,
                         tracking_video_predict_button, tracking_video_output, inpainting_video_output, remove_mask_button, inpaint_video_predict_button, step2_title, step3_title,
                         mask_dropdown, run_status, run_status2]
                )

        # click select image to get mask using sam
        template_frame.select(
                fn=sam_refine,
                inputs=[video_state, point_prompt, click_state, interactive_state],
                outputs=[template_frame, video_state, interactive_state, run_status, run_status2]
                )

        # tracking video from select image and mask
        tracking_video_predict_button.click(
                fn=vos_tracking_video,
                inputs=[video_state, interactive_state, mask_dropdown],
                outputs=[tracking_video_output, video_state, interactive_state, run_status, run_status2]
                )

    gr.Markdown(description)
    gr.Markdown(article)

# app.queue(concurrency_count=99)
app.launch(enable_queue=True, show_error=True, show_tips=True, show_api=True, debug=args.debug)

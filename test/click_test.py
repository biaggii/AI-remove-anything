import json
import time

import gradio as gr
import psutil

import cv2
from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np

from mask_painter import mask_painter

# gr = importlib.import_module('gradio')

SAM_checkpoint = '../weights/sam_vit_h_4b8939.pth'
model_type = 'vit_h'


class SamModel:
    def __init__(self, model_path, device='cuda'):
        self.original_image = None
        self.model_path = model_path
        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=self.model_path)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
        self.embedded = False
        # print(self.predictor.model)

    @torch.no_grad()
    def set_image(self, img: np.ndarray):
        self.original_image = img
        if self.embedded:
            print('repeat embedding, please reset_image.')
            return
        self.predictor.set_image(img)
        self.embedded = True
        return

    @torch.no_grad()
    def reset_image(self):
        # reset image embedding
        self.predictor.reset_image()
        self.embedded = False

    def predict(self, prompts, mode, multimask=True):
        """
        image: numpy array, h, w, 3
        prompts: dictionary, 3 keys: 'point_coords', 'point_labels', 'mask_input'
        prompts['point_coords']: numpy array [N,2]
        prompts['point_labels']: numpy array [1,N]
        prompts['mask_input']: numpy array [1,256,256]
        mode: 'point' (points only), 'mask' (mask only), 'both' (consider both)
        mask_outputs: True (return 3 masks), False (return 1 mask only)
        when mask_outputs=True, mask_input=logits[np.argmax(scores), :, :][None, :, :]
        """
        assert self.embedded, 'prediction is called before set_image (feature embedding).'
        assert mode in ['point', 'mask', 'both'], 'mode must be point, mask, or both'

        if mode == 'point':
            masks, scores, logits = self.predictor.predict(
                    point_coords=prompts['point_coords'],
                    point_labels=prompts['point_labels'],
                    multimask_output=multimask
                    )
        elif mode == 'mask':
            masks, scores, logits = self.predictor.predict(
                    mask_input=prompts['mask_input'],
                    multimask_output=multimask
                    )
        elif mode == 'both':  # both
            masks, scores, logits = self.predictor.predict(
                    point_coords=prompts['point_coords'],
                    point_labels=prompts['point_labels'],
                    mask_input=prompts['mask_input'],
                    multimask_output=multimask
                    )
        else:
            raise "Not implement now!"
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks, scores, logits


def get_prompt(c_state, click_input):
    inputs = json.loads(click_input)
    points = c_state[0]
    labels = c_state[1]
    for i in inputs:
        points.append(i[:2])
        labels.append(i[2])
    # c_state[0] = np.array(points)
    # c_state[1] = np.array(labels)
    c_state[0] = points
    c_state[1] = labels
    prompt = {
        "prompt_type"     : ["click"],
        "input_point"     : c_state[0],
        "input_label"     : c_state[1],
        "multimask_output": "True",
        }
    return prompt


clicks = json.loads('[]')
model = SamModel(SAM_checkpoint, device='cuda')
status = {
    'original_image' : None,
    'original_video' : None,
    'selected_images': [],
    'points'         : [],
    'mask'           : None,
    'masks'          : [],
    }

mask = ''


# get_prompt = importlib.import_module('..webui','app.get_prompt')
def onSelectedImg(img, state, interactive, point_prompt, evt: gr.SelectData):
    global mask
    global model
    # coordinate = f'[{evt.index[0]},{evt.index[1]},1]'
    # clicks.append(evt.index)
    # print(point_prompt)
    coordinate = f'[[{evt.index[0]},{evt.index[1]},1]]'
    interactive["positive_click_times"] += 1
    prompt = get_prompt(state, coordinate)
    prompts = {
        'point_coords': np.array(prompt['input_point']),
        'point_labels': np.array(prompt['input_label']),
        }
    model.reset_image()
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    model.set_image(img)
    masks, scores, logits = model.predict(prompts, 'point', multimask=True)
    mask = masks[np.argmax(scores)].astype(np.uint8)
    p_img = mask_painter(img, mask, background_alpha=0.8)
    return p_img, gr.update(label='mask', value=mask), gr.update(label='logits', value=logits)


def onTrackClick(video, img, btn):
    global mask
    global model
    model.clear()


frames = []


def onGetFrames(video_path):
    # print(f'Video path ---- {video_path}')
    video_path = video_path
    user_name = time.time()
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # frames.append(frame)
                if current_memory_usage > 90:
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0], frames[0].shape[1])
    # return frames[20] if len(frames) > 20 else frames[0], image_size, fps
    return frames[0], image_size, fps


css = """
.gradio-container {width: 85% !important}
.gr-monochrome-group {border-radius: 5px !important; border: revert-layer !important; border-width: 2px !important; color: black !important}
button {border-radius: 8px !important;}
.add_button {background-color: #4CAF50 !important;}
.remove_button {background-color: #f44336 !important;}
.mask_button_group {gap: 10px !important;}
.video {height: 600px !important;}
.image {height: 600px !important;}
.video .wrap.svelte-lcpz3o {display: flex !important; align-items: center !important; justify-content: center !important;}
.video .wrap.svelte-lcpz3o > :first-child {height: 100% !important;}
.margin_center {width: 50% !important; margin: auto !important;}
.jc_center {justify-content: center !important;}
.console {height: 300px !important;}
"""


def onSelectedInput(selected):
    # print(selected)
    if selected == 'video':
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    if selected == 'image':
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as app:
    click_state = gr.State([[], []])
    interactive_state = gr.State(
            {
                "inference_times"     : 0,
                "negative_click_times": 0,
                "positive_click_times": 0,
                "mask_save"           : True,
                "multi_mask"          : {
                    "mask_names": [],
                    "masks"     : []
                    },
                "track_end_number"    : None,
                }
            )
    with gr.Row():
        modeSelector = gr.Radio(choices=['video', 'image'], label='Select Mode', interactive=True, value='video')
        pointPromptSelector = gr.Radio(choices=['Positive', 'Negative'], label='Select Prompt', interactive=True, value='Positive')
    with gr.Row():
        inputVideo = gr.Video(label='input video', elem_classes='image', visible=True)
        inputImage = gr.Image(label='input image', type='numpy', elem_classes='image', visible=True)

    with gr.Row(elem_classes='jc_center'):
        btnTrack = gr.Button(value='Track', elem_classes='add_button')
        btnGetFrames = gr.Button(value='Get Frames', elem_classes='add_button')

    with gr.Row():
        outputVideo = gr.Video(label='output video', elem_classes='video', visible=True)
        outputImage = gr.Image(label='output image', type='numpy', elem_classes='image', interactive=False)

    with gr.Row():
        console = gr.Textbox(text_align='left', label="Console", elem_classes='console', show_label=True)
        console2 = gr.Textbox(text_align='left', label="Console", elem_classes='console', show_label=True)

    modeSelector.change(
            fn=onSelectedInput,
            inputs=[modeSelector],
            outputs=[inputVideo, inputImage, outputVideo],
            )
    btnTrack.click(
            fn=onTrackClick,
            inputs=[inputImage, click_state, interactive_state, inputVideo],
            outputs=[outputVideo],
            )

    inputImage.select(
            fn=onSelectedImg,
            inputs=[inputImage, click_state, interactive_state, pointPromptSelector],
            outputs=[outputImage, console, console2],
            )

    btnGetFrames.click(
            fn=onGetFrames,
            inputs=[inputVideo],
            outputs=[inputImage, console, console2],
            )

app.launch(debug=True)

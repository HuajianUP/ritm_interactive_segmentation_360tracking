import torch
import json
import os
import cv2
import numpy as np
from tkinter import messagebox

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.annotations = []
        self.refined = False
        self.target = 0
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    def load_annotations(self, target_id=0, anno_file=None, mask_dir=None, downsample=1):
        if anno_file is not None:
            with open(anno_file, "r") as outfile:
                annotations = json.load(outfile)
            self.annotations = []
            for anno in annotations:
                target = anno["target"]
                clicks = anno["clicks"]
                clicker_list = []
                for click in clicks:
                    coords = click["coords"]
                    coords = [x/downsample for x in coords]
                    clicker_list.append(clicker.Click(click["is_positive"], coords, click["indx"]))
                mask = cv2.imread(os.path.join(mask_dir, anno["mask"]))
                if downsample > 1:
                    mask = cv2.resize(mask, (mask.shape[1]//downsample, mask.shape[0]//downsample), cv2.INTER_AREA)
                mask = mask[:, :, 0] > 127
                #self.controller.set_mask(mask)
                self.annotations.append((clicker_list, mask, False))
        self.target = target_id
        self.object_count = target_id
        if len(self.annotations)>0:
            click, mask,_ = self.annotations[target_id]
            self.set_click_mask(click, mask)
        return len(self.annotations)


    def save_annotations(self, file_path="test.json", mask_dir=None, downsample=1):
        # multiple targets
        if not self.refined:
            return
        annotations = []
        for i, (clicks, object_mask, brefine) in enumerate(self.annotations):
            SN = file_path.split("/")[-1].split(".")[0]
            mask_file = SN+"_{}.png".format(i)
            anno = {"target": i,
                    "mask": mask_file}
            clicks_dict = []
            for click in clicks:
                coords = click.coords 
                coords = [x * downsample for x in coords]
                buff={"is_positive": click.is_positive,
                    "coords": coords,
                    "indx": click.indx
                }
                clicks_dict.append(buff)
            anno.update({"clicks": clicks_dict})
            annotations.append(anno) 
            if mask_dir is not None and brefine:
                #print("save", object_mask, object_mask.shape)#;exit()
                if downsample > 1:
                    object_mask = cv2.resize(object_mask.astype(np.uint8), (object_mask.shape[1]*downsample, object_mask.shape[0]*downsample), cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(mask_dir, mask_file), object_mask)

        if len(annotations) > 0:
            with open(file_path, "w") as outfile:
                json.dump(annotations, outfile)

    def set_click_mask(self, clicks, mask):
        #if len(self.probs_history) > 0:
        #    self.reset_last_object()
        # previous state
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })
        self.clicker.set_state(clicks)
        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)

    def get_latest_click(self):
        #print(self.clicker.get_state()[-1].coords);exit()
        click = self.clicker.get_state()
        if len(click) > 0:
            return click[-1].coords
        else:
            return None

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.annotations = []
        self.refined = False
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, is_positive):
        #print(x, y)
        self.refined = True
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            print("double prediction")
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        torch.cuda.empty_cache()
        #print(pred.shape, pred.type)

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return
        
        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            #print("reset init mask")
            self.reset_init_mask()
        self.update_image_callback()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.refined and len(self.clicker) > 0 and self.probs_history:
            #print(self.probs_history[-1][-1])
            object_mask = self.probs_history[-1][-1] > self.prob_thresh
            #print("finish object 1", object_mask)
            object_mask = np.array(object_mask, dtype=np.uint8)
            cv2.normalize(object_mask, object_mask, 0, 255, cv2.NORM_MINMAX)
            #if object_mask.max() < 256:
            #    object_mask = object_mask.astype(np.uint8)
            #    object_mask *= 255 // object_mask.max()
            #print("finish object 2", object_mask, object_mask.shape)
            if self.target + 1 > len(self.annotations):
                self.annotations.append((self.clicker.get_state(), object_mask, True))
            else:
                self.annotations[self.target] = (self.clicker.get_state(), object_mask, True)

        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.target += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)
        #if self.probs_history:
        #    total_mask = self.probs_history[-1][0] > self.prob_thresh
        #    results_mask_for_vis[np.logical_not(total_mask)] = 0
        #    vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        return vis

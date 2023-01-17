import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import os
import cv2
import json
import glob
import numpy as np
from PIL import Image, ImageTk

from interactive_omni_anno.canvas import CanvasImage
from interactive_omni_anno.controller import InteractiveController
from interactive_omni_anno.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame


class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, model):
        super().__init__(master)
        self.master = master
        master.title("Refined interactive annotation tool for 360tracking")
        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = args.limit_longest_size
        self.downsample = args.downsample

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image)
        
        self._load_sequences(args.dir)
        self._init_state()
        #self._add_menu()
        self._add_canvas()
        self._add_buttons()

        self._update_sequence()
        


        master.bind('n', lambda event: self.controller.finish_object())
        #master.bind('z', lambda event: self.controller.partially_finish_object())
        # undo
        master.bind('z', lambda event: self.controller.undo_click())
        # 
        master.bind('f', lambda event: self._next_frame_callback())
        master.bind('e', lambda event: self._last_frame_callback())

        # image canvas zoom
        master.bind('c', lambda event: self._canvas_zoomout_callback())
        master.bind('<space>', lambda event: self._canvas_focus_callback())



    def _load_sequences(self, dir):
        self.dir = dir
        self.sequences = sorted(os.listdir(dir))
        self.current_sequence_id = 0
        self.corresponding_anno_file = None
        while(self._check_labeling_process()):
            if self.current_sequence_id < len(self.sequences) - 1:
                self.current_sequence_id += 1
            else:
                break
        
    def _check_labeling_process(self):
        self.images_path = os.path.join(self.dir, self.sequences[self.current_sequence_id], "image")
        self.image_files = [os.path.join(self.images_path, path) for path in sorted(os.listdir(self.images_path))]
        # load init bbox
        inti_bboxs_path = os.path.join(self.dir, self.sequences[self.current_sequence_id], "bbox_init")
        if os.path.isdir(inti_bboxs_path):
            self.inti_bboxs_files = [os.path.join(inti_bboxs_path, path) for path in sorted(os.listdir(inti_bboxs_path))]
            assert len(self.inti_bboxs_files) == len(self.image_files)
        # check if have refined masks and clicks.
        self.masks_path = os.path.join(self.dir, self.sequences[self.current_sequence_id], "mask") #  mask
        self.annotations_path = os.path.join(self.dir, self.sequences[self.current_sequence_id], "anno") # polygon_init_new clicks
        self.annotations_files = []
        self.visit_record_files = os.path.join(self.dir, self.sequences[self.current_sequence_id], "record.json") 

        # need to support multiple objects annotation, suppose the mask is of .png format and the click is 
        if os.path.exists(self.masks_path) and os.path.exists(self.annotations_path):
            annotations_files = [os.path.join(self.annotations_path, path) for path in sorted(os.listdir(self.annotations_path))]
            if os.path.isfile(self.visit_record_files):
                with open(self.visit_record_files, "r") as outfile:
                    self.visit_record = json.load(outfile)

                for i, image_file in enumerate(self.image_files):
                    # check if exits
                    image_SN = image_file.split("/")[-1].split(".")[0]
                    if image_SN in self.visit_record and self.visit_record[image_SN]:
                        continue
                    else:
                        self.current_frame_id = i
                        return False
                self.current_frame_id = len(self.image_files) - 1
                return True
            else:
                self.visit_record = {}
                self.current_frame_id = 0
                return False 
            """        
            elif len(annotations_files) == len(self.image_files):
                self.current_frame_id = len(self.image_files) - 1
                #image_SN = self.image_files[self.current_frame_id].split("/")[-1].split(".")[0]
                #self.corresponding_anno_file = os.path.join(self.annotations_path, image_SN+".json")
                return True
            else:
                for i, image_file in enumerate(self.image_files):
                    # check if exits
                    image_SN = image_file.split("/")[-1].split(".")[0]
                    corresponding_anno_file = os.path.join(self.annotations_path, image_SN+".json")
                    if os.path.isfile(corresponding_anno_file):
                        continue
                    else:
                        self.current_frame_id = i
                        return False
            """
        else:
            os.makedirs(self.masks_path, exist_ok=True)
            os.makedirs(self.annotations_path, exist_ok=True)
            self.current_frame_id = 0
            #image_SN = self.image_files[self.current_frame_id].split("/")[-1].split(".")[0]
            #self.corresponding_anno_file = os.path.join(self.annotations_path, image_SN+".json")
            return False

    def _next_sequence_callback(self):
        #print("_next_sequence_callback")
        if self.current_sequence_id < len(self.sequences)-1:
            self.current_sequence_id += 1
            self._update_sequence()
        else:
            with open(self.visit_record_files, "w") as outfile:
                json.dump(self.visit_record, outfile)
            messagebox.showinfo(title="Thank you", message="Congratulations!!! You have finished all annotations.")
    
    def _last_sequence_callback(self):
        if self.current_sequence_id > 0:
            self.current_sequence_id -= 1
            self._update_sequence()
        else:
            with open(self.visit_record_files, "w") as outfile:
                json.dump(self.visit_record, outfile)

    def _index2iid(self, index):
        iid = list(str(hex(index+1)))[2:]
        iid = "".join(iid)
        for i in range(3 - len(iid)):
            iid = '0' + iid
        #print("I"+iid)
        return "I"+iid.upper()

    def _select_sequence_callback(self):
        id = self.sequences_table.index(self.sequences_table.focus())
        self._index2iid(id)
        self.current_sequence_id = id
        self._update_sequence()
        #print(self.sequences_table.focus(), id)

    def _select_frame_callback(self):
        #self.frames_table.see(self._index2iid(self.current_frame_id))
        id = self.frames_table.index(self.frames_table.focus())
        self.current_frame_id = id
        self._update_frame()
        #print(self.sequences_table.focus(), id, self._index2iid(id), self.current_frame_id)

    def _select_target_callback(self):
        id = self.targets_table.index(self.targets_table.focus())
        self.controller.load_annotations(id)
        self._update_image()
        #print("target", id)

    def _next_frame_callback(self):
        if self.current_frame_id < len(self.image_files)-1:
            self.current_frame_id += 1
            self._update_frame()
        else:
            if_next = messagebox.askokcancel(title="Process", message="Great! You have finished current sequence. Continue?")
            if(if_next):
                self._next_sequence_callback()


    def _last_frame_callback(self):
        if self.current_frame_id > 0:
            self.current_frame_id -= 1
            self._update_frame()
        else:
            if_next = messagebox.askokcancel(title="Process", message="Go to last sequence?")
            if(if_next):
                self._last_sequence_callback()


    def _update_sequence(self):
        # save visit record
        target_image = glob.glob(os.path.join(self.dir, self.sequences[self.current_sequence_id], "*.jpg"))
        #print(target_image);exit()
        image1 = Image.open(target_image[0])
        image1 = image1.resize((160, 160), Image.ANTIALIAS)
        tk_img = ImageTk.PhotoImage(image1)
        self.target_label.configure(image=tk_img)
        self.target_label.image = tk_img

        with open(self.visit_record_files, "w") as outfile:
            json.dump(self.visit_record, outfile)

        self.sequences_table.see(self._index2iid(self.current_sequence_id))
        self._check_labeling_process()
        # update frames table
        items = self.frames_table.get_children()
        if len(items) > len(self.image_files):
            for i in range(len(self.image_files)):
                self.frames_table.item(items[i], values=(i+1, self.image_files[i].split("/")[-1]))
            for i in range(len(self.image_files), len(items)):
                self.frames_table.delete(items[i])
                #self.frames_table.insert('','end',values=[i+1, self.image_files[i].split("/")[-1]])
        else:
            for i, item in enumerate(items): ## Changing all children from root item
                self.frames_table.item(item, values=(i+1, self.image_files[i].split("/")[-1]))
            for i in range(len(items), len(self.image_files)):
                self.frames_table.insert('','end',values=[i+1, self.image_files[i].split("/")[-1]])
        self._update_frame()

    def _save_anno(self):
        self.controller.finish_object()
        if self.corresponding_anno_file is not None:
            corresponding_mask_path = os.path.join(self.dir, self.corresponding_anno_file.split("/")[-3], "mask")
            #print(self.corresponding_anno_file.replace("anno", "mask"));exit()
            self.controller.save_annotations(self.corresponding_anno_file, corresponding_mask_path, downsample=self.downsample)

    def _update_frame(self):
        # finish_object
        #print("_update_frame", self.current_frame_id)
        #if self.current_frame_id < len(self.frames_table.get_children())-1:
        #    self.frames_table.see(self._index2iid(0))
        #self.frames_table.focus(self._index2iid(0))
        #self.frames_table.see(self._index2iid(self.current_frame_id))
        self._save_anno()
        # update recording
        #image = cv2.cvtColor(cv2.imread(self.image_files[self.current_frame_id]), cv2.COLOR_BGR2RGB)
        image = cv2.imread(self.image_files[self.current_frame_id])
        if self.downsample > 1:
            image = cv2.resize(image, (image.shape[1]//self.downsample, image.shape[0]//self.downsample), cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.controller.set_image(image)
        self._update_label_state()
        # if have annotation

        image_SN = self.image_files[self.current_frame_id].split("/")[-1].split(".")[0]
        self.corresponding_anno_file = os.path.join(self.annotations_path, image_SN+".json")
        for item in self.targets_table.get_children():
            self.targets_table.delete(item)

        #self.targets_table.insert('','end',values=["target 0"])

        if os.path.isfile(self.corresponding_anno_file) and image_SN in self.visit_record:
            anno_count = self.controller.load_annotations(anno_file=self.corresponding_anno_file, mask_dir=self.masks_path, downsample=self.downsample)
            self._update_image()
            for i in range(anno_count):
                self.targets_table.insert('','end',values=["target {}".format(i)])
        else:
            # only have inti bbox
            #print("bbox_init")
            bbox = np.loadtxt(self.inti_bboxs_files[self.current_frame_id])
            lx, ly, bbox_w, bbox_h = bbox
            center_x = int(lx+bbox_w*0.5)//self.downsample
            center_y = int(ly+bbox_h*0.5)//self.downsample
            #sample_point_x = center_x if center_x < img_w else center_x - img_w
            #sample_point_y = center_y if center_y < img_h else center_y - img_h
            self.controller.add_click(center_x, center_y, is_positive=True)

        self.visit_record.update({image_SN: True})

    def _canvas_focus_callback(self):
        self.canvas.focus_set()
        click = self.controller.get_latest_click()
        if click is not None:
            for i in range(3):
                self.image_on_canvas.set_focus(1.2, click)

    def _canvas_zoomout_callback(self):
        self.canvas.focus_set()
        click = self.controller.get_latest_click()
        if click is not None:
            for i in range(3):
                self.image_on_canvas.set_focus(0.5, click)     

    def _update_label_state(self):
        self.current_sequence_label["text"] = "current sequence: {}/{}".format(self.current_sequence_id+1, len(self.sequences))
        self.current_frame_label["text"] ="current frame: {}/{}".format(self.current_frame_id+1, len(self.image_files))


    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'fixed_crop': tk.BooleanVar(value=True),
                'skip_clicks': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=min(400, self.limit_longest_size)),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },

            'predictor_params': {
                'net_clicks_limit': tk.IntVar(value=8)
            },
            'brs_mode': tk.StringVar(value='NoBRS'),
            'prob_thresh': tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20),

            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=3),
            'sequences': tk.StringVar(value=self.sequences[self.current_sequence_id]),
        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill='x')

        button = FocusButton(self.menubar, text='Load image', command=self._load_image_callback)
        button.pack(side=tk.LEFT)
        self.save_mask_btn = FocusButton(self.menubar, text='Save mask', command=self._save_mask_callback)
        self.save_mask_btn.pack(side=tk.LEFT)
        self.save_mask_btn.configure(state=tk.DISABLED)

        self.load_mask_btn = FocusButton(self.menubar, text='Load mask', command=self._load_mask_callback)
        self.load_mask_btn.pack(side=tk.LEFT)
        self.load_mask_btn.configure(state=tk.DISABLED)

        button = FocusButton(self.menubar, text='About', command=self._about_callback)
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text='Exit', command=self.master.quit)
        button.pack(side=tk.LEFT)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.control_frame

        self.clicks_options_frame = FocusLabelFrame(master, text="Clicks management")
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.finish_object_button = \
            FocusButton(self.clicks_options_frame, text='Finish\nobject', bg='#b6d7a8', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.finish_object)
        #self.finish_object_button.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=3)
        self.finish_object_button.grid(row=0, column=0, sticky=tk.W, padx=22)

        self.undo_click_button = \
            FocusButton(self.clicks_options_frame, text='Undo click', bg='#ffe599', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.undo_click)
        #self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=3)
        self.undo_click_button.grid(row=0, column=1, padx=0)

        self.reset_clicks_button = \
            FocusButton(self.clicks_options_frame, text='Reset clicks', bg='#ea9999', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._reset_last_object)
        #self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=3)
        self.reset_clicks_button.grid(row=0, column=2, sticky=tk.E, padx=22)
        
        ##############################################
        self.sequences_options_frame = FocusLabelFrame(master, text="Sequences management")
        self.sequences_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        col_head = ("#", "sequences")
        col_head_anchor = (tk.CENTER, "w")
        yscroll = tk.Scrollbar(self.sequences_options_frame, orient=tk.VERTICAL)

        self.sequences_table=ttk.Treeview(self.sequences_options_frame, columns=col_head, show='headings', yscrollcommand=yscroll.set, 
                height=5)

        for i in range(len(col_head)):
            self.sequences_table.column(col_head[i],width=i*100+36,anchor=col_head_anchor[i])
            self.sequences_table.heading(col_head[i],text=col_head[i])

        yscroll.config(command=self.sequences_table.yview)
        yscroll.grid(row=0, rowspan=2, column=1, padx=1, sticky=tk.N+tk.S)
        self.sequences_table.grid(row=0, rowspan=2, column=0, padx=1, sticky=tk.N+tk.S)
        
        self.select_sequence_button = \
            FocusButton(self.sequences_options_frame, text='select', bg='#ea9999', fg='black', width=5, height=1,
                        state=tk.NORMAL, command=self._select_sequence_callback)
        self.select_sequence_button.grid(row=2, column=0, columnspan=2, pady=2)

        for i in range(len(self.sequences)):
            self.sequences_table.insert('','end',values=[i+1, self.sequences[i]])

        self.current_sequence_label = tk.Label(self.sequences_options_frame, text="current sequence: 000/0000")
        self.current_sequence_label.grid(row=0, column=2, columnspan=2, pady=0, sticky='w')

        self.last_sequence_button = \
            FocusButton(self.sequences_options_frame, text='Last\nSeqence', bg='#ea9999', fg='black', width=5, height=2,
                        state=tk.NORMAL, command=self._last_sequence_callback)
        self.last_sequence_button.grid(row=1, column=2, pady=0)
        self.next_sequence_button = \
            FocusButton(self.sequences_options_frame, text='Next\nSequence', bg='#ea9999', fg='black', width=5, height=2,
                        state=tk.NORMAL, command=self._next_sequence_callback)
        self.next_sequence_button.grid(row=1, column=3, pady=0)


        ##############################################
        self.frames_options_frame = FocusLabelFrame(master, text="Frames management")
        self.frames_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        col_head = ("#", "name")
        col_head_anchor = (tk.CENTER, "w")
        frames_yscroll = tk.Scrollbar(self.frames_options_frame, orient=tk.VERTICAL)
        self.frames_table=ttk.Treeview(self.frames_options_frame, columns=col_head, show='headings', yscrollcommand=frames_yscroll.set, 
                height=5)

        for i in range(len(col_head)):
            self.frames_table.column(col_head[i],width=i*100+36,anchor=col_head_anchor[i])
            self.frames_table.heading(col_head[i],text=col_head[i])
        
        frames_yscroll.config(command=self.frames_table.yview)
        frames_yscroll.grid(row=0, rowspan=3, column=1, padx=1, sticky=tk.N+tk.S)
        self.frames_table.grid(row=0, rowspan=3, column=0, padx=1, sticky=tk.W)
        
        self.select_frame_button = \
            FocusButton(self.frames_options_frame, text='select', bg='#ea9999', fg='black', width=5, height=1,
                        state=tk.NORMAL, command=self._select_frame_callback)
        self.select_frame_button.grid(row=3, column=0, columnspan=2, pady=2)

        #for item in self.frames_table.get_children():
        #    self.frames_table.delete(item)
        for i in range(len(self.image_files)):
            self.frames_table.insert('','end',values=[i+1, self.image_files[i].split("/")[-1]])

        self.current_frame_label = tk.Label(self.frames_options_frame, text="current sequence: 000/0000")
        self.current_frame_label.grid(row=0, column=2, columnspan=2, sticky=tk.W)

        self.last_frame_button = \
            FocusButton(self.frames_options_frame, text='Last\nFrame', bg='#ea9999', fg='black', width=5, height=2,
                        state=tk.NORMAL, command=self._last_frame_callback)
        self.last_frame_button.grid(row=1, column=2, pady=2)
        self.next_frame_button = \
            FocusButton(self.frames_options_frame, text='Next\nFrame', bg='#ea9999', fg='black', width=5, height=2,
                        state=tk.NORMAL, command=self._next_frame_callback)
        #self.next_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=3, pady=3)
        self.next_frame_button.grid(row=1, column=3, pady=2)

        self.targets_table=ttk.Treeview(self.frames_options_frame, columns="#", show='headings', height=2)
        self.targets_table.column("#", width=70)
        self.targets_table.heading("#", text="anno")
        self.targets_table.grid(row=2, rowspan=2, column=2, padx=1, sticky=tk.W)
        self.select_target_button = \
            FocusButton(self.frames_options_frame, text='select', bg='#ea9999', fg='black', width=5, height=1,
                        state=tk.NORMAL, command=self._select_target_callback)
        self.select_target_button.grid(row=2, rowspan=2, column=3, pady=2)

        self.target_frame = FocusLabelFrame(master, text="Target")
        self.target_frame.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        #image1 = Image.open("./test.jpg")
        #image1 = image1.resize((130, 130), Image.ANTIALIAS)
        #target_image = ImageTk.PhotoImage(image1)

        self.target_label = tk.Label(self.target_frame)
        self.target_label.pack(side=tk.TOP, fill=tk.X, padx=1, pady=1)

        #self.target_label.place(x=1, y=1)

        self.info_frame = FocusLabelFrame(master, text="Shortcuts")
        self.info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        
        tk.Label(self.info_frame, text="Positive: <left click>").grid(row=0, pady=1, sticky='w')
        tk.Label(self.info_frame, text="Negative: <right click>").grid(row=1, pady=1, sticky='w')
        tk.Label(self.info_frame, text="Undo: <z>").grid(row=2, pady=1, sticky='w')
        tk.Label(self.info_frame, text="Finish/new object: <n>").grid(row=3, pady=1, sticky='w')
        tk.Label(self.info_frame, text="Next frame: <f>").grid(row=4, pady=1, sticky='w')
        tk.Label(self.info_frame, text="Last frame: <e>").grid(row=5, pady=1, sticky='w')
        tk.Label(self.info_frame, text="Auto zoom in: <space>").grid(row=6, pady=1, sticky='w')
        tk.Label(self.info_frame, text="Auto zoom out: <c>").grid(row=7, pady=1, sticky='w')
        tk.Label(self.info_frame, text="Zoom: <mouse wheel>").grid(row=8, pady=1, sticky='w')
        tk.Label(self.info_frame, text="Scroll image: <w a s d>").grid(row=9, pady=1, sticky='w')




    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                image = cv2.imread(filename)
                if self.downsample > 1:
                    print("downsample")
                    image = cv2.resize(image, (image.shape[1]//self.downsample, image.shape[0]//self.downsample), cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.controller.set_image(image)
                self.save_mask_btn.configure(state=tk.NORMAL)
                self.load_mask_btn.configure(state=tk.NORMAL)
    """
    def _save_mask_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            mask = self.controller.result_mask
            if mask is None:
                return

            filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
                ("PNG image", "*.png"),
                ("BMP image", "*.bmp"),
                ("All files", "*.*"),
            ], title="Save the current mask as...")

            if len(filename) > 0:
                if mask.max() < 256:
                    mask = mask.astype(np.uint8)
                    mask *= 255 // mask.max()
                cv2.imwrite(filename, mask)

    def _load_mask_callback(self):
        if not self.controller.net.with_prev_mask:
            messagebox.showwarning("Warning", "The current model doesn't support loading external masks. "
                                              "Please use ITER-M models for that purpose.")
            return

        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Binary mask (png, bmp)", "*.png *.bmp"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                mask = cv2.imread(filename)[:, :, 0] > 127
                self.controller.set_mask(mask)
                self._update_image()
    """

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    def _change_brs_mode(self, *args):
        if self.state['brs_mode'].get() == 'NoBRS':
            self.net_clicks_entry.set('INF')
            self.net_clicks_entry.configure(state=tk.DISABLED)
            self.net_clicks_label.configure(state=tk.DISABLED)
            self.lbfgs_iters_entry.configure(state=tk.DISABLED)
            self.lbfgs_iters_label.configure(state=tk.DISABLED)
        else:
            if self.net_clicks_entry.get() == 'INF':
                self.net_clicks_entry.set(8)
            self.net_clicks_entry.configure(state=tk.NORMAL)
            self.net_clicks_label.configure(state=tk.NORMAL)
            self.lbfgs_iters_entry.configure(state=tk.NORMAL)
            self.lbfgs_iters_label.configure(state=tk.NORMAL)

        self._reset_predictor()

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = self.state['brs_mode'].get()
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None if brs_mode == 'NoBRS' else self.state['predictor_params']['net_clicks_limit'].get()

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None

        predictor_params = {
            'brs_mode': brs_mode,
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': net_clicks_limit,
                'max_size': self.limit_longest_size
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': self.state['lbfgs_max_iters'].get()}
        }
        self.controller.reset_predictor(predictor_params)

    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()
        #print("_click_callback")
        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self._check_entry(self):
            #print("add_click")
            self.controller.add_click(x, y, is_positive)

    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get())
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED
        before_1st_click_state = tk.DISABLED if self.controller.is_incomplete_mask else tk.NORMAL

        self.finish_object_button.configure(state=after_1st_click_state)
        self.undo_click_button.configure(state=after_1st_click_state)
        self.reset_clicks_button.configure(state=after_1st_click_state)

        #self.next_clicks_button.configure(state=after_1st_click_state)

        """
        #self.zoomin_options_frame.set_frame_state(before_1st_click_state)
        #self.brs_options_frame.set_frame_state(before_1st_click_state)
        
        if self.state['brs_mode'].get() == 'NoBRS':
            self.net_clicks_entry.configure(state=tk.DISABLED)
            self.net_clicks_label.configure(state=tk.DISABLED)
            self.lbfgs_iters_entry.configure(state=tk.DISABLED)
            self.lbfgs_iters_label.configure(state=tk.DISABLED)
        """

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked

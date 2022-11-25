""" 
it reads images not frames 
images are extracted at 16fps.
default dataset if using images
"""


import os
import glob
import numpy as np
import random
import ffmpeg
import json
import pandas as pd
from joblib import Parallel, delayed
from backend.image_base import VideoDataset

# ROOT = "D:\\datasets\\Vision\\AVCAffe_DB\\public_release_pritam"
# subset='train'

def train_val_split(fold=1):
    """
    it return the validation split; 
    and rest of them are training split;
    """
    if fold==1:
        return ['aiim001', 'aiim002', 'aiim013', 'aiim014', 'aiim043', 'aiim044', 'aiim053', 'aiim054', 'aiim057', 'aiim058',
                'aiim063', 'aiim064', 'aiim067', 'aiim068', 'aiim077', 'aiim078', 'aiim099', 'aiim100', 'aiim107', 'aiim108']

def list_to_dict(self_score, name_list):
    dd = {}
    for l in self_score:
        if name_list is not None: # for arousal and valence
            dd.update({l[0]:name_list.index(l[1])})
        else: # for cogloads
            dd.update({l[0]:float(l[1])})
    return dd

def lists_to_dict(self_score, name_list):
    assert isinstance(self_score, list)
    self_score = np.concatenate(self_score, axis=1)
    dd = {}
    for l in self_score:
        assert l[0]==l[2]==l[4]==l[6]==l[8]
        dd.update({
            l[0]:[name_list[0].index(l[1]), name_list[1].index(l[3]), float(l[5]), float(l[7]), float(l[9])]
            })
    return dd

def read_anno_files(filename):
    holder, task_ids = [], []
    with open(filename, mode='r') as file:
        for line in file:
            if ',' in line:
                name, val = line.split(',')
                holder.append(val.strip('\n').strip())
                task_ids.append(name)
                
    self_score =  np.vstack((task_ids, holder)).T
    return self_score

class AVCAffe(VideoDataset):
    def __init__(self,
                 ROOT,
                 subset, 
                 fold=1,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=16.,
                 video_transform=None,
                 return_audio=True,
                 audio_must=True,
                 audio_clip_duration=1.,
                 audio_fps=44100,
                 audio_fps_out=100,
                 audio_transform=None,
                 return_labels=True,
                 class_name='mental_demand',
                 cogload_class_type='binary',
                 video_type='short_face', 
                 return_index=True,
                 mode='clip', # clip, video
                 clips_per_video=1,
                 return_pid_task=True,
                 task_categories='all',
                 ):
        
        """
        this dataloader loads the face crops corresponidng to shorter segments 
        and their respective audio streams.

        Parameters
        ----------
        ROOT : string
            root path of AVCAffe; example: your_path/avcaffe.
        subset : string
            either 'train' or 'val'
            for training or validation split respectively.
        fold : int
            mention as 1, currently AVCAffe has only one fold.
        return_video : binary
            set True if want to fetch video frames.
        video_clip_duration : float
            clip durations for visual output.
        video_fps : int
            video frames per second, note original fps is 16, please set desired
            video fps as either 16, 8, 4, 2, or 1
        video_transform : None or function
            to apply transformations on visual frames, pass the transformation functions;
            if None is set, it returns PIL Images.
        return_audio : binary
            set True if want to fetch audio.
        audio_must : binary
            some of the clips doesn't have audio, as the participant is silent,
            if want to extract those clips when participant is speaking set to True,
            otherwise set False.
        audio_clip_duration : int
            clip durations for audio output.
        audio_fps : int
            audio frequency. note, the original clips have 44100 Hz,
        audio_fps_out : int
            shape of the spectrogram in time axis.
        audio_transform : None or function
            to apply transformations on audio waveforms, pass the transformation functions;
            if None is set, it returns raw audio waveforms.
        return_labels : binary
            set true if want to fetch the labels.
        class_name : string
            mention either arousal/valence/mental_demand/effort/temporal_demand.
        cogload_class_type : str 
            pass either binary, continuous, original
            binary: convert >10 as 1 and <=10 as 0;
            continuous: in between 0 to 1 -> basically true_score/21;
            original: as it is, 0-21;
        return_index : binary
            set true if want to fetch the indices of the clips.
        mode : string
            either clip or video.
            if clip is set, it randomly samples clips_per_video per sample
            if video is set, it uniformly samples clips_per_video per sample
        return_pid_task : binary
            set true if want to fetch 'participant id and task id' of the respective clips.
        task_categories : string
            either pass 'all' which will return all clips of all tasks or,
            pass a list for particular task categories. e.g., ['task_1', 'task_2',...]

        Returns
        -------
        Dataset Object.

        """        
                        
        assert cogload_class_type in ['binary', 'continuous', 'original']
        assert video_type in ['short_face']
        assert subset in ['train', 'val']

        classes = {
                    'arousal': ['Excited', 'Wide-awake', 'Neutral', 'Dull', 'Calm'],
                    'valence': ['Pleasant', 'Pleased', 'Neutral', 'Unsatisfied', 'Unpleasant'],
                    }
        
        avbl_classes = ['arousal', 'valence', 'mental_demand', 'temporal_demand', 'effort']
        assert class_name in avbl_classes or class_name == 'all'
                
        if isinstance(task_categories, list):
            for k in task_categories:
                assert k in ['task_1', 'task_2', 'task_3', 'task_4', 'task_5', 'task_6', 'task_7', 'task_8', 'task_9']
        else:
            task_categories=='all'
                    
        if video_type=='short_face':
            DATA_PATH = os.path.join(f"{ROOT}", 'images', 'shorter_segments_face')
        else:
            raise NotImplementedError()

        # path for self-reported annotations
        ANNO_PATH = os.path.join(f"{ROOT}", 'ground_truths') 


        all_subjects = os.listdir(DATA_PATH)
        if subset == 'val':
            subjects = train_val_split(fold=fold)
        elif subset == 'train':
            subjects = set(all_subjects) - set(train_val_split(fold=fold))

        # loading list of subdirs to read
        CACHE_FILE = os.path.join(f"{ROOT}", 'images', 'shorter_segments_dirs.txt')
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                all_dirs=json.loads(f.read())
        else:
            print('fetching list of dirs...')
            all_dirs = ['/'.join(fn.replace('\\','/').split('/')[-4:]) \
                                  for fn in glob.glob(f'{DATA_PATH}/*/*/*/**/', recursive=True) \
                                      ]
            with open(CACHE_FILE, 'w') as f:
                f.write(json.dumps(all_dirs))
             
        
        if task_categories=='all': # just filtering the train test subjs
            media_files = sorted(['/'.join(fn.split('/')[-4:]) \
                                  for fn in all_dirs \
                                      if fn.split('/')[-4] in subjects])

        elif isinstance(task_categories, list): # extracting a subset of files
            media_files = sorted(['/'.join(fn.split('/')[-4:]) \
                                  for fn in all_dirs \
                                       if fn.split('/')[-4] in subjects \
                                           and fn.split('/')[-3] in task_categories
                                          ])

        if audio_must: # load files which must have audios
            no_audio_files = []
            with open(os.path.join(f"{ROOT}", 'info', 'no_audio_files.txt'), 'r') as f:
                for item in f.readlines():
                    no_audio_files.append(item.strip('\n'))

            media_files = [mf for mf in media_files if mf.split('/')[-2]+'.avi' not in no_audio_files]


        # if return_labels:
        if class_name=='all':
            self_score, name_list = [], []
            for k in avbl_classes:
                self_score.append(read_anno_files(os.path.join(ANNO_PATH, k + '.txt')))
                name_list.append(classes[k] if k in classes.keys() else None)
            lookup_table =  lists_to_dict(self_score, name_list)
        else:
            self_score = read_anno_files(os.path.join(ANNO_PATH, class_name + '.txt'))
            name_list = classes[class_name] if class_name in classes.keys() else None
            lookup_table =  list_to_dict(self_score, name_list)
        
        labels, ids, pid_task_names = [], [], []
        for mf in media_files:
            # labels.append(label_group(lookup_table['_'.join(mf.split('/')[:-1])], class_name=class_name))
            labels.append(lookup_table['_'.join(mf.split('/')[:-2])])
            ids.append(int(mf.split('/')[0][4:]))
            pid = float(mf.split('/')[0][4:])+float(mf.split('/')[1][5:])/10 # this makes aiim005_task_1 --> 5.1
            pid_task_names.append(pid)


        if class_name not in ['arousal', 'valence']:
            if cogload_class_type=='binary':
                labels = [int(k>10) for k in labels]
            elif cogload_class_type=='continuous':
                labels = [k/21 for k in labels]
            elif cogload_class_type=='original':
                labels = [k for k in labels]

        self.class_name = class_name
        super(AVCAffe, self).__init__(
            return_video=return_video,
            video_root=DATA_PATH,
            video_fns=media_files,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_root=os.path.join(f"{ROOT}", 'videos', 'shorter_segments'),
            audio_fns=media_files,
            audio_clip_duration=audio_clip_duration,
            audio_fps=audio_fps,
            audio_fps_out=audio_fps_out,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            ids=ids,
            return_index=return_index,
            mode=mode,
            clips_per_video=clips_per_video,
            return_pid_task=return_pid_task,
            pid_task_names=pid_task_names,
        )

        self.name = 'AVCAffe dataset'
        self.root = ROOT
        self.subset = subset
        self.num_videos = len(media_files)
        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in media_files])

if __name__ == "__main__":
    ROOT = "/mnt/PS6T/datasets/AVCAffe_DB/public_release_pritam"
    db = AVCAffe(
        ROOT = ROOT,
         subset='train',
         return_video=True,
         video_clip_duration=2,
         video_type='short_face',
         video_fps=8,
         video_transform=None,
         return_audio=True,
         audio_clip_duration=2,
         audio_must=True,
         return_labels=True,
         mode='clip',
         clips_per_video=1,
         return_pid_task=False,
         )  
    
    
    op = db.__getitem__(1)

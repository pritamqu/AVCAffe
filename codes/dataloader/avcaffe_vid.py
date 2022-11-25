# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import os
import glob
import numpy as np
from backend.video_base import VideoDataset

def train_val_split(fold=1):
    # it returns the validation split; and rest of them are training split
    if fold==1:
        return ['aiim001', 'aiim002', 'aiim013', 'aiim014', 'aiim043', 'aiim044', 'aiim053', 'aiim054', 'aiim057', 'aiim058',
                'aiim063', 'aiim064', 'aiim067', 'aiim068', 'aiim077', 'aiim078', 'aiim099', 'aiim100', 'aiim107', 'aiim108']
    else:
        raise ValueError

def list_to_dict(self_score, name_list):
    # convert a list to dict for quick fetch
    dd = {}
    for l in self_score:
        if name_list is not None:
            dd.update({l[0]:name_list.index(l[1])})
        else:
            dd.update({l[0]:float(l[1])})
    return dd


class AVCAffe(VideoDataset):
    def __init__(self,
                 ROOT,
                 subset,
                 fold=1,
                 return_video=True,
                 video_clip_duration=2,
                 video_fps=25.,
                 video_transform=None,
                 return_audio=True,
                 audio_must=False,
                 audio_clip_duration=2,
                 audio_fps=44100,
                 audio_fps_out=100,
                 audio_transform=None,
                 return_labels=True,
                 class_name='mental_demand',
                 cogload_class_type='binary',
                 return_index=False,
                 mode='clip',
                 clips_per_video=5,
                 return_pid_task=False,
                 task_categories='all',
                 ):
        """


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
            video frames per second, note original fps is 25,
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
        # arousal valence class names
        classes = {
                    'arousal': ['Excited', 'Wide-awake', 'Neutral', 'Dull', 'Calm'],
                    'valence': ['Pleasant', 'Pleased', 'Neutral', 'Unsatisfied', 'Unpleasant'],
                    }

        # path for video
        DATA_PATH = os.path.join(f"{ROOT}", 'videos', 'shorter_segments')
        # path for self-reported annotations
        ANNO_PATH = os.path.join(f"{ROOT}", 'ground_truths')

        self.class_name = class_name

        # separate subjects based on split
        all_subjects = os.listdir(DATA_PATH)
        if subset == 'val':
            subjects = train_val_split(fold=fold)
        elif subset == 'train':
            subjects = set(all_subjects) - set(train_val_split(fold=fold))
        else:
            raise ValueError(f"subset values should be either train or val - given {subset}")


        # choosing the required files
        if task_categories=='all': # just filtering the train test subjs
            media_files = sorted(['/'.join(fn.replace('\\','/').split('/')[-3:]) \
                                  for fn in glob.glob(os.path.join(f"{DATA_PATH}", "*", "*", "*.avi")) \
                                      if fn.replace('\\','/').split('/')[-3] in subjects])
        # extracting a subset of files
        elif type(task_categories) == list:
            media_files = sorted(['/'.join(fn.replace('\\','/').split('/')[-3:]) \
                                  for fn in glob.glob(os.path.join(f"{DATA_PATH}", "*", "*", "*.avi")) \
                                       if fn.replace('\\','/').split('/')[-3] in subjects \
                                           and fn.replace('\\','/').split('/')[-2] in task_categories
                                          ])
        # load files which must have audios
        if audio_must:
            no_audio_files = []
            with open(os.path.join(f"{ROOT}", 'info', 'no_audio_files.txt'), 'r') as f:
                for item in f.readlines():
                    no_audio_files.append(item.strip('\n'))

            media_files = [mf for mf in media_files if mf.split('/')[-1] not in no_audio_files]



        # if return_labels:
        holder, task_ids = [], []
        with open(os.path.join(ANNO_PATH, class_name + '.txt'), mode='r') as file:
            for line in file:
                if ',' in line:
                    name, val = line.split(',')
                    holder.append(val.strip('\n').strip())
                    task_ids.append(name)

        self_score =  np.vstack((task_ids, holder)).T
        name_list = classes[class_name] if class_name in classes.keys() else None
        lookup_table =  list_to_dict(self_score, name_list)
        labels, ids, pid_task_names = [], [], []
        for mf in media_files:
            # labels.append(label_group(lookup_table['_'.join(mf.split('/')[:-1])], class_name=class_name))
            labels.append(lookup_table['_'.join(mf.split('/')[:-1])])
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


        super(AVCAffe, self).__init__(
            return_video=return_video,
            video_root=DATA_PATH,
            video_fns=media_files,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_root=DATA_PATH,
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

        self.name = 'AVCAffe Dataset'
        self.root = ROOT
        self.subset = subset
        self.num_videos = len(media_files)
        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in media_files])

if __name__ == "__main__":
    ROOT = "/mnt/PS6T/datasets/AVCAffe_DB/public_release_pritam"

    subset='train'
    db = AVCAffe(ROOT, subset)
    op = db.__getitem__(1)
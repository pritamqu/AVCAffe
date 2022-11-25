import os
import random
import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict
import glob
from backend import av_wrappers
from torchvision.datasets.folder import default_loader as image_loader

def load_video_frames(path, start_idx, total_frames, skip_frames):
    frames = []
    for k in range(int(start_idx), int(total_frames+start_idx), int(skip_frames)):
        frames.append(image_loader(path[k]))
    return frames

def chararray(fn_list):
    charr = np.chararray(len(fn_list), itemsize=max([len(fn) for fn in fn_list]))
    for i in range(len(fn_list)):
        charr[i] = fn_list[i]
    return charr


class VideoDataset(data.Dataset):
    def __init__(self,
                 return_video=True,
                 video_root=None,
                 video_fns=None,
                 video_clip_duration=1.,
                 video_fps=25,
                 video_transform=None,
                 return_audio=True,
                 audio_root=None,
                 audio_fns=None,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_fps_out=None,
                 audio_transform=None,
                 return_labels=False,
                 labels=None,
                 ids=None,
                 return_index=False,
                 mode='clip',
                 clips_per_video=1,
                 return_pid_task=False,
                 pid_task_names=None,
                 
                 ):
        super(VideoDataset, self).__init__()
        
        # when videos are converted to imgs, the fps was 16
        self.ORIGINAL_FPS = 16
        # e.g., if want to reduce the fps from 16 to 8 skip 1 frames 
        assert video_fps in [16, 8, 4, 2]
        self.skip_frames = (self.ORIGINAL_FPS//video_fps)

        self.num_samples = 0
        self.return_video = return_video
        self.video_root = video_root
        if return_video:
            self.video_fns = chararray(video_fns)
            self.num_samples = self.video_fns.shape[0]
        self.video_fps = video_fps
        self.video_transform = video_transform

        self.return_audio = return_audio
        self.audio_root = audio_root
        if return_audio:
            self.audio_fns = chararray(audio_fns)
            self.num_samples = self.audio_fns.shape[0]
        self.audio_fps = audio_fps
        self.audio_fps_out = audio_fps_out
        self.audio_transform = audio_transform

        self.return_labels = return_labels
        if return_labels:
            self.labels = np.array(labels)
            self.labels = self.labels.astype(np.int64)
        self.ids = np.array(ids)
        self.return_index = return_index

        self.video_clip_duration = video_clip_duration
        self.audio_clip_duration = audio_clip_duration
        self.clips_per_video = clips_per_video
        self.mode = mode
        self.return_pid_task = return_pid_task
        self.pid_task_names = pid_task_names
                         

    def _load_sample(self, sample_idx):
        """ it loads a sample audio to a container"""
        
        video_ctr = None
        if self.return_video:
            video_fn = os.path.join(self.video_root, self.video_fns[sample_idx].decode())
            video_ctr = [fn for fn in glob.glob(os.path.join(f"{video_fn}", "*.jpg"))]
        
        audio_ctr = None
        if self.return_audio:
            # loading audio directly from the videos
            audio_fn = os.path.join(self.audio_root, self.audio_fns[sample_idx].decode()[:-1]+'.avi')
            audio_ctr = av_wrappers.av_open(audio_fn)

        return video_ctr, audio_ctr

    def __getitem__(self, index):
        
        ########### just one clip for regular use
        #########################################
        
        if self.mode == 'clip':
           
            try:
                sample_idx = index % self.num_samples
                video_ctr, audio_ctr = self._load_sample(sample_idx)
                a_ss, a_dur = self._sample_snippet(audio_ctr)   
                # video_ctr = list of file names
                # number of frames conisering original fps and video duration
                if self.return_video:
                    v_dur = self.ORIGINAL_FPS*self.video_clip_duration
                    # choose the random starting point
                    if len(video_ctr)< v_dur:
                        v_ss=0
                    else:
                        v_ss = random.randint(0, len(video_ctr)-v_dur)
                else:
                    v_ss, v_dur = None, None
                
                # return frames, audio, labels etc.
                sample = self._get_clip(sample_idx, audio_ctr=audio_ctr, audio_start_time=a_ss, 
                                                    video_ctr=video_ctr, video_start_time=v_ss, 
                                                    audio_clip_duration=a_dur, video_clip_duration=v_dur)
                if sample is None:
                    return self[(index+1) % len(self)]
                return sample
            except Exception:
                return self[(index+1) % len(self)]
            
                       
        ########### return clips_per_video number of clips from whole video
        ###################################################################

        elif self.mode == 'video':
            
            video_ctr, audio_ctr = self._load_sample(index)
            # Load entire video
            ss, sf = self._get_time_lims(audio_ctr)
            if self.return_video:
                vs, vf = 0, len(video_ctr)
            else:
                vs, vf = None, None
            
            audio_dur, video_dur = None, None
            if self.return_audio:
                if sf <= ss:
                    sf = ss + self.audio_clip_duration
                audio_dur = sf - ss

            if self.return_video:
                video_dur = vf
                
            # sample = self._get_clip(index, audio_ctr, start_time, audio_clip_duration=audio_dur)
            sample = self._get_clip(index, audio_ctr=audio_ctr, audio_start_time=ss, 
                                    video_ctr=video_ctr, video_start_time=vs, 
                                    audio_clip_duration=audio_dur, video_clip_duration=video_dur)

            # Split video into overlapping chunks
            chunks = defaultdict(list)
            if self.return_video:
                nf = sample['frames'].shape[1]
                chunk_size = int(self.video_clip_duration * self.video_fps)
                if chunk_size >= nf:
                    chunks['frames'] = torch.stack([sample['frames'] for _ in range(self.clips_per_video)])
                else:
                    timestamps = np.linspace(0, max(nf - chunk_size, 1), self.clips_per_video).astype(int)
                    chunks['frames'] = torch.stack([sample['frames'][:, ss:ss+chunk_size] for ss in timestamps])

            if self.return_audio:
                nf = sample['audio'].shape[-1] # time dim [1, freq, time]
                chunk_size = int(self.audio_clip_duration * self.audio_fps_out)
                if chunk_size >= nf:
                    chunks['audio'] = torch.stack([sample['audio'] for _ in range(self.clips_per_video)])
                else:
                    timestamps = np.linspace(0, max(nf - chunk_size, 1), self.clips_per_video).astype(int)
                    chunks['audio'] = torch.stack([sample['audio'][:, :, int(ss):int(ss+chunk_size)] for ss in timestamps])
                    
            if self.return_labels:
                chunks['label'] = sample['label']

            if self.return_index:
                chunks['index'] = torch.stack([torch.tensor(sample['index']).repeat(self.clips_per_video)], dim=1)
                
            if self.return_pid_task:
                chunks['pid_task'] = self.pid_task_names[index]
                
            return chunks
        

    def __len__(self):
        if self.mode == 'clip':
            return self.num_samples * self.clips_per_video
        else:
            return self.num_samples

    def __repr__(self):
        desc = "{}\n - Root: {}\n - Subset: {}\n - Num videos: {}\n - Num samples: {}\n".format(
            self.name, self.root, self.subset, self.num_videos, self.num_videos * self.clips_per_video)
        if self.return_video:
            desc += " - Example video: {}/{}\n".format(self.video_root, self.video_fns[0].decode())
        if self.return_audio:
            desc += " - Example audio: {}/{}\n".format(self.audio_root, self.audio_fns[0].decode())
        return desc

    def _get_time_lims(self, audio_ctr):
        audio_st, audio_ft = None, None
        if audio_ctr is not None:
            audio_stream = audio_ctr.streams.audio[0]
            tbase = audio_stream.time_base
            audio_st = 0
            audio_dur = audio_stream.duration * tbase
            audio_ft = audio_st + audio_dur
            
        return audio_st, audio_ft
    
    def _sample_snippet(self, audio_ctr):
        if self.return_audio:
            audio_st, audio_ft = self._get_time_lims(audio_ctr)
            audio_duration = audio_ft - audio_st
            if self.audio_clip_duration > audio_duration:
                return 0., audio_duration
            else:
                min_d, max_d = self.audio_clip_duration, min(self.audio_clip_duration, audio_duration)
                duration = random.uniform(min_d, max_d)
                sample_ss_a = random.uniform(audio_st, audio_ft - duration)
                return sample_ss_a, duration            
        else:
            return None, None

    def _get_clip(self, clip_idx, audio_ctr, audio_start_time, 
                  video_ctr, video_start_time, 
                  audio_clip_duration=None, video_clip_duration=None,
                  ):
        if audio_clip_duration is None:
            audio_clip_duration = self.audio_clip_duration
        if video_clip_duration is None:
            video_clip_duration = self.video_clip_duration

        sample = {}
        
        if self.return_video:
            frames = load_video_frames(path=video_ctr, start_idx=video_start_time, 
                                total_frames=video_clip_duration, 
                                skip_frames=self.skip_frames)
            
        
            if self.video_transform is not None:
                frames = self.video_transform(frames)
                
            sample['frames'] = frames      
        
        
        if self.return_audio:
            samples, rate = av_wrappers.av_laod_audio(
                audio_ctr,
                audio_fps=self.audio_fps,
                start_time=audio_start_time,
                duration=audio_clip_duration,
            )
            if self.audio_transform is not None:
                samples = self.audio_transform(samples)
            sample['audio'] = samples

        if self.return_labels:
            lbl = self.labels[clip_idx]
            if isinstance(lbl, np.ndarray):
                sample['label'] = torch.from_numpy(lbl)
            else:
                sample['label'] = lbl

        if self.return_index:
            sample['index'] = clip_idx
            
        if self.return_pid_task:
            sample['pid_task'] = self.pid_task_names[clip_idx]

        return sample
# This script is prepared using the source code available at:
# https://github.com/facebookresearch/AVID-CMA/blob/main/datasets/video_db.py

import os
import random
import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict
from backend import av_wrappers


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
        """ it loads a sample video to a container"""
        video_ctr = None
        if self.return_video:
            # video_fn = '{}/{}'.format(self.video_root, self.video_fns[sample_idx].decode())
            video_fn = os.path.join(self.video_root, self.video_fns[sample_idx].decode())
            video_ctr = av_wrappers.av_open(video_fn)

        audio_ctr = None
        if self.return_audio:
            # audio_fn = '{}/{}'.format(self.audio_root, self.audio_fns[sample_idx].decode())
            audio_fn = os.path.join(self.audio_root, self.audio_fns[sample_idx].decode())
            if self.return_video and audio_fn == video_fn:
                audio_ctr = video_ctr
            else:
                audio_ctr = av_wrappers.av_open(audio_fn)

        return video_ctr, audio_ctr


    def __getitem__(self, index):

        ########### just one clip for regular use
        #########################################

        if self.mode == 'clip':
            try:
                sample_idx = index % self.num_samples
                video_ctr, audio_ctr = self._load_sample(sample_idx)
                v_ss, v_dur, a_ss, a_dur = self._sample_snippet(video_ctr, audio_ctr)
                sample = self._get_clip(sample_idx, video_ctr, audio_ctr, v_ss, a_ss, video_clip_duration=v_dur, audio_clip_duration=a_dur)
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
            vs, vf, ss, sf = self._get_time_lims(video_ctr, audio_ctr)

            if self.return_video and self.return_audio:
                start_time = max(vs, ss) if ss < 0 else vs
                final_time = min(vf, sf) if ss < 0 else vf
                if final_time <= start_time:
                    final_time = start_time + max(self.video_clip_duration, self.audio_clip_duration)
            elif self.return_video and not self.return_audio:
                start_time = vs
                final_time = vf
                if final_time <= start_time:
                    final_time = start_time + self.video_clip_duration
            elif self.return_audio and not self.return_video:
                start_time = ss
                final_time = sf
                if final_time <= start_time:
                    final_time = start_time + self.audio_clip_duration

            video_dur = final_time - start_time
            sample = self._get_clip(index, video_ctr, audio_ctr, start_time, start_time, video_clip_duration=video_dur, audio_clip_duration=video_dur)

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

    def _get_time_lims(self, video_ctr, audio_ctr):
        video_st, video_ft, audio_st, audio_ft = None, None, None, None
        if video_ctr is not None:
            video_stream = video_ctr.streams.video[0]
            tbase = video_stream.time_base
            video_st = video_stream.start_time * tbase
            video_dur = video_stream.duration * tbase
            video_ft = video_st + video_dur

        if audio_ctr is not None:
            audio_stream = audio_ctr.streams.audio[0]
            tbase = audio_stream.time_base
            audio_st = audio_stream.start_time * tbase
            audio_dur = audio_stream.duration * tbase
            audio_ft = audio_st + audio_dur

        return video_st, video_ft, audio_st, audio_ft

    def _sample_snippet(self, video_ctr, audio_ctr):
        video_st, video_ft, audio_st, audio_ft = self._get_time_lims(video_ctr, audio_ctr)
        if not self.return_audio:
            video_duration = video_ft - video_st
            if self.video_clip_duration > video_duration:
                return 0., video_duration, 0., video_duration
            else:
                min_d, max_d = self.video_clip_duration, min(self.video_clip_duration, video_duration)
                duration = random.uniform(min_d, max_d)
                sample_ss_v = random.uniform(video_st, video_ft - duration)
                return sample_ss_v, duration, sample_ss_v, duration

        elif not self.return_video:
            audio_duration = audio_ft - audio_st
            if self.audio_clip_duration > audio_duration:
                return 0., audio_duration, 0., audio_duration
            else:
                min_d, max_d = self.audio_clip_duration, min(self.audio_clip_duration, audio_duration)
                duration = random.uniform(min_d, max_d)
                sample_ss_a = random.uniform(audio_st, audio_ft - duration)
                return sample_ss_a, duration, sample_ss_a, duration

        else:
            video_duration = video_ft - video_st
            audio_duration = audio_ft - audio_st
            if self.video_clip_duration > video_duration and self.audio_clip_duration > audio_duration:
                return 0., video_duration, 0, audio_duration

            elif self.video_clip_duration > video_duration and not self.audio_clip_duration > audio_duration:
                # when vid_dur is less than what we need but aud_dur is fine
                min_d, max_d = self.audio_clip_duration, min(self.audio_clip_duration, audio_duration)
                duration = random.uniform(min_d, max_d)
                sample_ss_a = random.uniform(audio_st, audio_ft - duration)
                return 0., video_duration, sample_ss_a, duration

            elif not self.video_clip_duration > video_duration and self.audio_clip_duration > audio_duration:
                # when vid_dur is fine but aud_dur is less than what we need
                min_d, max_d = self.video_clip_duration, min(self.video_clip_duration, video_duration)
                duration = random.uniform(min_d, max_d)
                sample_ss_v = random.uniform(video_st, video_ft - duration)
                return sample_ss_v, duration, 0, audio_duration

            else:
                # when both are okay
                min_ss = max(audio_st, video_st)
                max_ss = min(audio_ft - self.audio_clip_duration, video_ft - self.video_clip_duration)
                assert max_ss > min_ss, f"max_ss is {max_ss} and min_ss is {min_ss}"
                if self.audio_clip_duration > self.video_clip_duration:
                    sample_ss_a = random.uniform(min_ss, max_ss)
                    sample_tt_a = sample_ss_a + self.audio_clip_duration

                    win_min = max(sample_ss_a, video_st)
                    win_max = min(sample_tt_a, video_ft) - self.video_clip_duration
                    sample_ss_v = random.uniform(win_min, win_max)

                    return sample_ss_v, self.video_clip_duration, sample_ss_a, self.audio_clip_duration
                else:
                    sample_ss_v = random.uniform(min_ss, max_ss)
                    sample_tt_v = sample_ss_v + self.video_clip_duration

                    win_min = max(sample_ss_v, audio_st)
                    win_max = min(sample_tt_v, audio_ft) - self.audio_clip_duration
                    sample_ss_a = random.uniform(win_min, win_max)

                    return sample_ss_v, self.video_clip_duration, sample_ss_a, self.audio_clip_duration


    def _get_clip(self, clip_idx, video_ctr, audio_ctr, video_start_time, audio_start_time, video_clip_duration=None, audio_clip_duration=None):
        if video_clip_duration is None:
            video_clip_duration = self.video_clip_duration
        if audio_clip_duration is None:
            audio_clip_duration = self.audio_clip_duration

        sample = {}
        if self.return_video:
            frames, fps, start_time = av_wrappers.av_load_video(
                video_ctr,
                video_fps=self.video_fps,
                start_time=video_start_time,
                duration=video_clip_duration,
            )

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

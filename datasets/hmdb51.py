import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import h5py
import cv2
import torchvision #import transforms

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1]) - 1

    @property
    def label(self):
        return int(self._data[2])


class HMDB51(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='frame{:06d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.h5 = '/DATACENTER_SSD/ysd/hmdb51_rgb.h5'

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _decode(self, img, flag='rgb'):
        if flag == 'rgb':
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = img[..., ::-1] # bgr -> rgb
        elif flag == 'flow':
            img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
            img = img[..., np.newaxis]
        return torchvision.transforms.ToPILImage()(img)

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            path = os.path.join(directory, self.image_tmpl.format(idx))
            reader = h5py.File(self.h5, 'r')
            rtr = [(self._decode(reader[path].value))]
            reader.close()
            return rtr
            # return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            path_x = os.path.join(directory, self.image_tmpl).format('u', idx)
            reader = h5py.File('/DATACENTER_SSD/ysd/hmdb51_flow_u.h5', 'r')
            x_img = self._decode(reader[path_x].value, flag='flow')
            
            path_y = os.path.join(directory, self.image_tmpl).format('v', idx)
            reader = h5py.File('/DATACENTER_SSD/ysd/hmdb51_flow_v.h5', 'r')
            y_img = self._decode(reader[path_y].value, flag='flow')

            # x_img = Image.open(os.path.join(directory, self.image_tmpl).format('u', idx)).convert('L')
            # y_img = Image.open(os.path.join(directory, self.image_tmpl).format('v', idx)).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if record.num_frames > self.num_segments+1:
            rand_end = max(0, record.num_frames - self.num_segments - 1)
            begin_ind = randint(0, rand_end)
            end_index = min(begin_ind + self.num_segments, record.num_frames)
            rand = list(range(1,record.num_frames))
            res = rand[begin_ind:end_index]
        else:
            rand_end = record.num_frames
            res = list(range(1,record.num_frames))
        for index in res:
            if len(res) == self.num_segments:
                break
            res.append(index)
        assert len(res) == self.num_segments
        return res

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        return _sample_indices(record)
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        if self.transform is not None:
            self.transform.randomize_parameters()
            images = [self.transform(image) for image in images]
        process_data = torch.stack(images, 0).permute(1, 0, 2, 3)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

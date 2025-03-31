import base64
import io
import json
import os.path as osp
import os

import PIL.Image
import pickle
from labelme import __version__
from labelme.logger import logger
from labelme import PY2
from labelme import QT4
from labelme import utils
import numpy as np

def pillow2array(img,flag='color'):
    # Handle exif orientation tag
    #if flag in ['color', 'grayscale']:
        #img = ImageOps.exif_transpose(img)
    # If the image mode is not 'RGB', convert it to 'RGB' first.
    if flag in ['color', 'color_ignore_orientation']:
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = PIL.Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        array = np.array(img)
    elif flag in ['grayscale', 'grayscale_ignore_orientation']:
        img = img.convert('L')
        array = np.array(img)
    else:
        raise ValueError(
            'flag must be "color", "grayscale", "unchanged", '
            f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
            f' but got {flag}')
    return array

def decode_img(buffer,fmt='rgb'):
    buff = io.BytesIO(buffer)
    img = PIL.Image.open(buff)

    if fmt=='rgb':
        img = pillow2array(img, 'color')
    else:
        img = pillow2array(img, 'grayscale')

    return img

def mci_read(file_path):
    with open(file_path,"rb") as f:
        data = pickle.load(f)
        shape = data['shape']
        datas = []
        for d in data['imgs']:
            d = decode_img(d,fmt="grayscale")
            datas.append(d)
        datas = np.stack(datas,axis=-1)
        return datas

PIL.Image.MAX_IMAGE_PIXELS = None


class LabelFileError(Exception):
    pass


class LabelFile(object):

    suffix = '.json'

    current_img_idx = 0
    total_imgs_nr = 1
    cur_file_name = ""
    cur_img = None
    def __init__(self, filename=None,imgpath=None):
        self.shapes = []
        self.imagePath = imgpath
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        print(f"Load img {filename}")
        try:
            if osp.splitext(filename)[1].lower() == ".mci":
                if LabelFile.cur_file_name == filename and LabelFile.cur_img is not None:
                    img = LabelFile.cur_img
                else:
                    img = mci_read(filename)
                    LabelFile.cur_img = img
                LabelFile.cur_file_name = filename
                LabelFile.current_img_idx = min(max(0,LabelFile.current_img_idx),img.shape[2]-1)
                LabelFile.total_imgs_nr = img.shape[2]
                print(LabelFile.current_img_idx,img.shape,LabelFile.total_imgs_nr)
                img = img[:,:,LabelFile.current_img_idx]
                image_pil = PIL.Image.fromarray(img)
            else:
                LabelFile.cur_file_name = filename
                
                suffix = ['.jpg','.jpeg','.png','.bmp']
                if not os.path.exists(filename):
                    filename1 = filename+".jpg"
                    filename2 = filename + ".jpeg"
                    if os.path.exists(filename1):
                        filename = filename1
                    elif os.path.exists(filename2):
                        filename = filename2
    
                LabelFile.cur_img = None
                image_pil = PIL.Image.open(filename)
                LabelFile.current_img_idx = 0
                LabelFile.total_imgs_nr = 1
        except IOError:
            logger.error('Failed opening image file: {}'.format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = 'PNG'
            elif ext in ['.jpg', '.jpeg']:
                format = 'JPEG'
            else:
                format = 'PNG'
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            'version',
            'imageData',
            'imagePath',
            'shapes',  # polygonal annotations
            'flags',   # image level flags
            'imageHeight',
            'imageWidth',
        ]
        shape_keys = [
            'label',
            'points',
            'group_id',
            'shape_type',
            'flags',
        ]
        try:
            with open(filename, 'rb' if PY2 else 'r') as f:
                data = json.load(f)
            version = data.get('version')
            if version is None:
                logger.warn(
                    'Loading JSON file ({}) of unknown version'
                    .format(filename)
                )
            elif version.split('.')[0] != __version__.split('.')[0]:
                logger.warn(
                    'This JSON file ({}) may be incompatible with '
                    'current labelme. version in file: {}, '
                    'current version: {}'.format(
                        filename, version, __version__
                    )
                )

            '''if 'imageData' in data and data['imageData'] is not None:
                imageData = base64.b64decode(data['imageData'])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data['imagePath'])
                imageData = self.load_image_file(imagePath)'''
            imagePath = osp.splitext(filename)[0]
            imageData = self.load_image_file(self.imagePath)
            flags = data.get('flags') or {}
            imagePath = data['imagePath']
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode('utf-8'),
                data.get('imageHeight'),
                data.get('imageWidth'),
            )
            shapes = [
                dict(
                    label=s['label'],
                    points=s['points'],
                    shape_type=s.get('shape_type', 'polygon'),
                    flags=s.get('flags', {}),
                    group_id=s.get('group_id'),
                    other_data={
                        k: v for k, v in s.items() if k not in shape_keys
                    }
                )
                for s in data['shapes']
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                'imageHeight does not match with imageData or imagePath, '
                'so getting imageHeight from actual image.'
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                'imageWidth does not match with imageData or imagePath, '
                'so getting imageWidth from actual image.'
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        '''if imageData is not None:
            imageData = base64.b64encode(imageData).decode('utf-8')
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )'''

        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            #imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filename, 'wb' if PY2 else 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix

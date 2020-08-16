import os
from evaluate_on_coco import *
from torchsummary import summary
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms as trn
from tool import utils
from matplotlib import pyplot as plt
import subprocess as sp
from pymediainfo import MediaInfo

try:
    import accimage
except ImportError:
    accimage = None


class ListDataset(Dataset):
    def __init__(self, l, transform=None, load=True):
        self.l = l
        self.loader = default_loader  # self.feature_loader
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.load = load
        if transform is not None:
            self.transform = transform

    def __getitem__(self, idx):
        path = self.l[idx]
        if self.load:
            frame = self.transform(self.loader(path))
            return path, frame
        else:
            frame = self.transform(path)
            return frame



    def __len__(self):
        return len(self.l)


class VideoDataet(Dataset):
    def __init__(self, video, transform=None):
        self.video = video
        self.meta = self.parse_meta()
        self.frames = self.decode()
        self.transform = trn.Compose([
            trn.ToTensor(),
            # trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

    def parse_meta(self):
        media_info = MediaInfo.parse(self.video)
        meta = dict()
        for track in media_info.tracks:
            # print(track.to_data())
            if track.track_type == 'General':
                meta['file_name'] = track.file_name + '.' + track.file_extension
                meta['file_extension'] = track.file_extension
                meta['format'] = track.format
                meta['duration'] = track.duration
                meta['frame_count'] = track.frame_count
                meta['frame_rate'] = track.frame_rate
            elif track.track_type == 'Video':
                meta['width'] = int(track.width)
                meta['height'] = int(track.height)
                meta['rotation'] = float(track.rotation) if track.rotation is not None else 0.
                meta['codec'] = track.codec
        return meta

    def decode(self):
        video = cv2.VideoCapture(self.video)
        frames = []
        while video.isOpened():
            ret, f = video.read()
            if not ret:
                break
            else:
                frames.append(f)
        video.release()

        return frames

    def __getitem__(self, idx):
        frame = self.transform(self.frames[idx])
        return idx, frame

    def __len__(self):
        return self.meta['frame_count']


def decode_video(video_path, frame_path):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    command = ['ffmpeg',
               '-hide_banner', '-loglevel', 'panic',
               # '-skip_frame', 'nokey',
               '-i', video_path,
               '-pix_fmt', 'bgr24',  # color space
               # '-r', '1',
               '-vcodec', 'rawvideo',  # origin video
               '-f', 'image2pipe',  # output format : image to pipe
               'pipe:1']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=width * height * 3)

    return_list = []
    count = 0
    while True:
        # print(count)
        frame_name = "%06d" % count + ".jpg"
        dst = os.path.join(frame_path, frame_name)
        raw_image = pipe.stdout.read(width * height * 3)
        pipe.stdout.flush()

        image1 = np.frombuffer(raw_image, dtype='uint8')
        if image1.shape[0] == 0:
            break
        image2 = image1.reshape((height, width, 3))
        cv2.imwrite(dst, image2)
        return_list.append(dst)
        count += 1
    return return_list


if __name__ == '__main__':

    logging = init_logger(log_dir='log')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Darknet('cfg/yolov4.cfg')
    model.print_network()

    model.load_weights('yolov4.weights')
    model = model.cuda()
    summary(model, (3, 608, 608))
    model.eval()

    model = torch.nn.DataParallel(model)
    class_names = load_class_names('data/coco.names')

    transform = trn.Compose([trn.Resize((608, 608)), trn.ToTensor()])

    video = '/workspace/vlog.mp4'
    video_name = os.path.basename(video)
    frame_path = f'/workspace/frames/{video_name}'
    detect_frame_path = f'/workspace/frames/{video_name}_detect'
    if not os.path.exists(detect_frame_path):
        os.makedirs(detect_frame_path)

    # frames = decode_video(video, frame_path=frame_path')
    frames = [os.path.join(frame_path, f) for f in sorted(os.listdir(frame_path))]

    # loader = DataLoader(VideoDataet(video,transform=transform), batch_size=4, shuffle=False, num_workers=2)
    # images = [os.path.join('/nfs_shared/food/images', i.strip()) for i in
    #           open('/nfs_shared/food/meta/test.txt', 'r').readlines()]
    # print(images[:5])
    loader = DataLoader(ListDataset(frames[:1000], transform=transform), batch_size=16, shuffle=False, num_workers=2)

    inference_time = 0
    pure_inference_time = 0
    c = 0
    with torch.no_grad():
        for i, (path, imgs) in enumerate(loader):
            start = time.time()
            outputs = model(imgs)
            pure_inference_time += (time.time() - start)
            boxes = utils.post_processing(imgs, 0.25, 0.5, outputs)
            inference_time += (time.time() - start)
            for n, p in enumerate(path):
                box_im = utils.plot_boxes_cv2(cv2.imread(p), boxes[n],
                                              savename=os.path.join(detect_frame_path, f'{c:06}.jpg'),
                                              class_names=class_names)

                c += 1

            # print(output[0].shape)
            # print(output[1].shape)
            # utils.post_processing(img, conf_thresh, nms_thresh, output)
    print(f'inference_time: {inference_time}, fps: {len(loader.dataset.l) / inference_time}')
    print(f'pure_inference_time: {pure_inference_time}, fps: {len(loader.dataset.l) / pure_inference_time}')

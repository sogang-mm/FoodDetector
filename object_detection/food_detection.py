from object_detection.pytorch_YOLOv4.evaluate_food_data import *
from classification.evaluate import *
import matplotlib.pyplot as plt

from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageColor


def draw_bounding_box_on_image(image, boxes, labels, thickness=4):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/Nanum/NanumMyeongjoExtraBold.ttf", 25)
    w, h = image.size
    colors = list(ImageColor.colormap.values())

    for box in boxes:
        conf = box[5]
        label = box[-1]
        label_str = [f'{labels[label]}: {int(conf * 100)}%']
        color = colors[hash(labels[label]) % len(colors)]

        (left, top, right, bottom) = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness,
                  fill=color)
        print(label_str, (left, top, right, bottom))
        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = [font.getsize(ds)[1] for ds in label_str]

        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > (1 + 2 * 0.01) * sum(display_str_heights):
            text_bottom = top
        else:
            # text_bottom = bottom + total_display_str_height
            text_bottom = top + total_display_str_height

        # Reverse list and print from bottom to top.
        for display_str in label_str[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
            # draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom + 2 * margin)], fill=color)
            # draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
            draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
            text_bottom -= text_height - 2 * margin

    return image


if __name__ == '__main__':
    model = Darknet('pytorch_YOLOv4/cfg/yolov4.cfg')
    model.load_weights('pytorch_YOLOv4/yolov4.weights')
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    # model.print_network()
    # summary(model, (3, 608, 608))

    class_names = load_class_names('pytorch_YOLOv4/data/coco.names')
    coco_food_classes = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 60]
    transform = trn.Compose([trn.Resize((608, 608)), trn.ToTensor()])

    food_model = Mobilenet_v2()
    ckpt = torch.load('/hdd/ms/food_run/records/food/mobilenet/state_95_acc_0.79_loss_0.75.pt')
    food_model.load_state_dict(ckpt['model_state_dict'])
    food_model.cuda()
    # food_model = nn.DataParallel(model)
    food_model.eval()
    food_transform = trn.Compose([trn.Resize((224,224)),
                                  # trn.CenterCrop(224),
                                  trn.ToTensor(),
                                  trn.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
    softmax = nn.Softmax()
    food_classes_name = [i.strip() for i in open('/nfs_shared/food/meta/classes.txt', 'r').readlines()]

    video = '/workspace/vlog.mp4'
    video_name = os.path.basename(video)
    frame_path = f'/workspace/frames/{video_name}'
    # detect_frame_path = f'/workspace/frames/{video_name}_detect'
    detect_frame_path = f'/workspace/food_frames/{video_name}_detect'
    if not os.path.exists(detect_frame_path):
        os.makedirs(detect_frame_path)

    frames = decode_video(video, frame_path=frame_path)
    # frames = [os.path.join(frame_path, f) for f in sorted(os.listdir(frame_path))]
    # loader = DataLoader(ListDataset(frames, transform=transform), batch_size=16, shuffle=False, num_workers=2)

    food_loader = DataLoader(ListDataset([], transform=food_transform, load=False), batch_size=32, shuffle=False,
                             num_workers=2)

    inference_time = 0
    pure_inference_time = 0

    with torch.no_grad():
        for i, (path, imgs) in enumerate(loader):
            start = time.time()
            outputs = model(imgs)
            pure_inference_time += (time.time() - start)
            boxes = utils.post_processing(imgs, 0.25, 0.5, outputs)
            inference_time += (time.time() - start)

            for n, p in enumerate(path):
                # print(boxes[n])
                food_boxes = [box for box in boxes[n] if box[-1] in coco_food_classes]
                # print(food_boxes)
                im = Image.open(p)
                if len(food_boxes):
                    w, h = im.size[0], im.size[1]

                    crop_imgs = [im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))) for box
                                 in food_boxes]
                    food_loader.dataset.l = crop_imgs
                    food_prob, food_indice = [], []
                    for i, imgs in enumerate(food_loader):
                        outputs = food_model(imgs.cuda())
                        outputs = softmax(outputs)
                        prob, indice = torch.topk(outputs.cpu(), k=1)
                        food_prob.extend(list(prob.numpy().flatten()))
                        food_indice.extend(list(indice.numpy().flatten()))
                    print(p)
                    food_boxes = [[*box[:5], food_prob[i], food_indice[i]] for i, box in enumerate(food_boxes)]
                    im = draw_bounding_box_on_image(im, food_boxes, food_classes_name, )

                im.save(os.path.join(detect_frame_path, f'{os.path.basename(p)}'))

    os.system('ffmpeg -i /workspace/food_frames/vlog.mp4_detect/%6d.jpg -framerate 29.97 -vcodec h264 /workspace/vlog_food.mp4')

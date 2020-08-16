import cv2
import json
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
import random

if __name__ == '__main__':
    result = json.load(open('result.json', 'r'))

    size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('result.avi', fourcc, 1, size)
    random.shuffle(result['result'])
    font_size = 20
    line = font_size + 5
    for ret in result['result'][:100]:
        path = ret['image']
        gt = ret['gt']
        predict = [(t[0], t[1]) for t in zip(ret['predict'], ret['probability'])]
        print(path, gt, predict)

        im = Image.open(path).convert('RGB')
        im.thumbnail(size)
        im = ImageOps.expand(im, (int((size[0] - im.width) / 2), int((size[1] - im.height) / 2)))
        im = im.resize(size)

        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("/usr/share/fonts/truetype/Nanum/NanumMyeongjoExtraBold.ttf", font_size)
        text = [f'{i[0]} : {i[1]:.4f}' for i in predict]
        draw.text((5, 0), gt, fill=(5, 255, 5), font=font, align="left")
        for n, t in enumerate(text):
            color = 'red' if ret['predict'][n] != gt else (5, 255, 5)
            draw.text((5, (n + 1) * line), t, fill=color, font=font, align="left")

        cv_im = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
        out.write(cv_im)
        out.write(cv_im)
        out.write(cv_im)
        out.write(cv_im)
        out.write(cv_im)

    out.release()

import numpy as np
import matplotlib
import pycocotools.mask as maskUtils
import torch
import asyncio

from mmdet.apis import init_detector, inference_detector, show_result_pyplot, async_inference_detector, show_result
from mmdet.apis.inference import inference_batch
from mmdet.utils.contextmanagers import concurrent
import mmcv

config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

def sync_main_batch(loops, warm_loops=10):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    img = ['demo/demo.jpg', 'demo/demo1.jpg']

    for _ in mmcv.track_iter_progress(range(warm_loops)):
        result = inference_batch(model, img)

    for _ in mmcv.track_iter_progress(range(loops)):
        result = inference_batch(model, img)

    for i in range(len(img)):
        show_result(img[i], result[i], model.CLASSES, out_file='result_sync_batch{}.jpg'.format(i))


def sync_main_single(loops, warm_loops=10):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img = 'demo/demo.jpg'

    for _ in mmcv.track_iter_progress(range(warm_loops)):
        result = inference_detector(model, img)

    for _ in mmcv.track_iter_progress(range(loops)):
        result = inference_detector(model, img)

    show_result(img, result[0], model.CLASSES, out_file='result_sync_single.jpg')


async def async_main(loops, warm_loops=10):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 5

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device='cuda:0'))

    img = 'demo/demo.jpg'

    for _ in mmcv.track_iter_progress(range(warm_loops)):
        async with concurrent(streamqueue):
            result = await async_inference_detector(model, img)

    for _ in mmcv.track_iter_progress(range(loops)):
        async with concurrent(streamqueue):
            result = await async_inference_detector(model, img)

    show_result(img, result, model.CLASSES, out_file='result_async.jpg')


if __name__ == "__main__":
    print('async_main:')
    #asyncio.run(async_main(10))

    print('sync_main_single:')
    sync_main_single(10)

    print('sync_main_batch:')
    sync_main_batch(10)
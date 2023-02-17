from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import warnings

warnings.filterwarnings("ignore")


config_file = 'configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
checkpoint_file = 'checkpoint/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
device = 'cuda:2'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
#print(model)
# 推理演示图像
img = '/GPFS/data/ziyili/diffusion/stable-diffusion-main/outputs/txt2img-samples/samples/00119.png'
result = inference_detector(model, img)
#print(len(result))
print(result)
bbox_result,seg_result=result
print("bbox_result",bbox_result[4])
print("seg_result",seg_result[4])
print("bbox_result",len(bbox_result),len(bbox_result[4]))
print("seg_result",len(seg_result),len(seg_result[4]))
print(type(seg_result[4][0]))
print(seg_result[4][0].shape)
show_result_pyplot(model, img, result, score_thr=0.3, out_file='00119_pred_seg.jpg')

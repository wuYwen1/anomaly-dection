from cv2 import COLORMAP_JET
import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet_ssp import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

def plot_fig(test_img, scores, gts, index, save_name, class_name, rec):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        ori_img = test_img[i]
        img = denormalization(ori_img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > 0.1] = 1
        mask[mask <= 0.1] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        # vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        vis_img = mark_boundaries(ori_img.transpose(1, 2, 0), mask, color=(0, 0, 255), mode='thick')

        fig_img, ax_img = plt.subplots(1, 6, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        rec_img =rec[i]

        # woxiede
        import cv2
        save_path = '/root/autodl-tmp/wyw/DRAEM-main-a/WYW/RESULT/' + class_name + '/'+'origin'
        save_path1 = '/root/autodl-tmp/wyw/DRAEM-main-a/WYW/RESULT/' + class_name + '/'+'gt'
        save_path2 = '/root/autodl-tmp/wyw/DRAEM-main-a/WYW/RESULT/' + class_name + '/'+'vis'
        save_path3 = '/root/autodl-tmp/wyw/DRAEM-main-a/WYW/RESULT/' + class_name + '/'+'mask'
        save_path4 = '/root/autodl-tmp/wyw/DRAEM-main-a/WYW/RESULT/' + class_name + '/'+'predicted'
        save_path5 = '/root/autodl-tmp/wyw/DRAEM-main-a/WYW/RESULT/' + class_name + '/'+'rec'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path1):
            os.makedirs(save_path1)
        if not os.path.exists(save_path2):
            os.makedirs(save_path2)
        if not os.path.exists(save_path3):
            os.makedirs(save_path3)
        if not os.path.exists(save_path4):
            os.makedirs(save_path4)
        if not os.path.exists(save_path5):
            os.makedirs(save_path5)
        
        cv2.imwrite(save_path + '/' + index + '.jpg' , np.uint8(ori_img.transpose(1, 2, 0)*255))
        cv2.imwrite(save_path1 + '/'+ index + '.jpg' , np.uint8(gt*255))
        cv2.imwrite(save_path2 + '/'+ index + '.jpg' , np.uint8(vis_img*255))
        cv2.imwrite(save_path3 + '/'+ index + '.jpg' , np.uint8(mask))
        cv2.imwrite(save_path5 + '/'+ index + '.jpg' , np.uint8(rec_img.transpose(1,2,0)*255))
        # H, W = heat_map.shape
        # heat = np.zeros((1,H,W))
        # heat[0,:,:] = heat_map
        heat_map1 = cv2.applyColorMap(np.uint8(heat_map),colormap=COLORMAP_JET)
        heat_map1 = heat_map1*0.5 + ori_img.transpose(1, 2, 0)*255*0.5
        cv2.imwrite(save_path4 + '/'+ index + '.jpg' , heat_map1)


        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(ori_img.transpose(1, 2, 0))
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(rec_img.transpose(1, 2, 0))
        ax_img[1].title.set_text('rec')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[3].title.set_text('Predicted heat map')
        ax_img[4].imshow(mask, cmap='gray')
        ax_img[4].title.set_text('residual mask')
        ax_img[5].imshow(vis_img)
        ax_img[5].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(save_name, dpi=100)
        plt.close()

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x

def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("/root/autodl-tmp/wyw/DRAEM-main-a/outputs/results.txt",'a+') as file:
        file.write(fin_str)


def test(obj_names, mvtec_path, checkpoint_path, base_model_name):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    for obj_name in obj_names:
        img_dim = 256
        # run_name = base_model_name+"_"+obj_name+'_'
        save_dir = '/root/autodl-tmp/wyw/DRAEM-main-a/result/'+obj_name+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        run_name = 'DRAEM_test_0.0001_700_bs8'+"_"+obj_name+'_'
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()
        mvtec_path ='/root/autodl-tmp/wyw/data/'
        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        cnt_display = 0
        display_indices = np.random.randint(len(dataloader), size=(16,))



        for i_batch, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)


            if i_batch in display_indices:
                t_mask = out_mask_sm[:, 1:, :, :]
                display_images[cnt_display] = gray_rec[0]
                display_gt_images[cnt_display] = gray_batch[0]
                display_out_masks[cnt_display] = t_mask[0]
                display_in_masks[cnt_display] = true_mask[0]
                cnt_display += 1

            plot_image = sample_batched["image"].cpu().detach().numpy()
            plot_score = out_mask_sm[:, 1:, :, :].cpu().detach().numpy().squeeze(0)
            plot_gt = sample_batched["mask"].cpu().detach().numpy()
            plot_rec = gray_rec.cpu().detach().numpy()
            gt_mask = np.asarray(plot_gt)
            # precision, recall, thresholds = precision_recall_curve(gt_mask.astype(np.uint8).flatten(), plot_score.flatten())
            # a = 2 * precision * recall
            # b = precision + recall
            # f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            # threshold = thresholds[np.argmax(f1)]
            save_name = save_dir+str(i_batch)+'.jpg'
            plot_fig(plot_image,plot_score,plot_gt,str(i_batch),save_name,obj_name,plot_rec)

            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

          
            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        print(obj_name)
        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        print("==============================")

    print(run_name)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))

    write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=False)
    parser.add_argument('--base_model_name', action='store', type=str, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=False)
    parser.add_argument('--checkpoint_path', action='store', type=str, default='/root/autodl-tmp/wyw/DRAEM-main-a/checkpoints_dataset',required=False)

    args = parser.parse_args()

    obj_list = [ 
                    # 'bottle',
                    'dataset',
                #  'guandao',
                #  'pipeline',
                # #  'piple',
                #  'object2',
                # #  'capsule',
                #  'bottle',
                #  'carpet',
                #  'leather',
                #  'pill',
                #  'transistor',
                #  'tile',
                #  'cable',
                #  'zipper',
                #  'toothbrush',
                #  'metal_nut',
                #  'hazelnut',
                 ]

    with torch.cuda.device(args.gpu_id):
        test(obj_list,args.data_path, args.checkpoint_path, args.base_model_name)

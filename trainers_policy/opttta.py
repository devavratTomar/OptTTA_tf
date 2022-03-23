import numpy as np
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import tensorflow_addons as tfa
import os
import random
import itertools
import matplotlib.pyplot as plt
import math
import tqdm
from dataloader import GenericNumpyVolumeLoader

from networks.unet import UNet
from utils import MetricTracker
from utils import *
from PIL import Image

from trainers_policy.diff_augmentations import Indentity, GaussainBlur, Contrast, Brightness, Gamma, RandomResizeCrop, RandomHorizontalFlip,\
    RandomVerticalFlip, RandomRotate, DummyAugmentor

DEBUG = True
STYLE_AUGMENTORS = [Gamma.__name__, GaussainBlur.__name__, Contrast.__name__, Brightness.__name__, Indentity.__name__]
SPATIAL_AUGMENTORS = [RandomResizeCrop.__name__, RandomHorizontalFlip.__name__, RandomVerticalFlip.__name__, RandomRotate.__name__, RandomScaledCenterCrop.__name__]

STRING_TO_CLASS = {
    'Identity': Indentity,
    'GaussianBlur': GaussainBlur,
    'Contrast': Contrast,
    'Brightness': Brightness,
    'Gamma': Gamma,
    'RandomResizeCrop': RandomResizeCrop,
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'RandomVerticalFlip': RandomVerticalFlip,
    'RandomRotate': RandomRotate
}


class OptTTA():
    def __init__(self, opt):
        self.opt = opt
        print("Test Time Data Augmentation")

    def initialize(self):
        self.target_test_dataloader = GenericNumpyVolumeLoader(self.opt.rootdir, self.opt.target_sites)

        ## pretrained source model
        self.unet = UNet(self.opt.n_classes)
        self.unet(np.zeros([1, 256, 256, 1]), training=False)
        self.unet.load_weights(self.opt.source_segmentor_path)

        ## criterian
        self.criterian_mse = tf.keras.losses.MeanSquaredError()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=4)

        ## metric tracker
        self.metric_tracker = MetricTracker()

        ## batch norm stats of unet
        self.bn_stats = self.get_batch_norm_stats()


    def save_policy(self, policy_name, augmentors):
        save_dir = os.path.join(self.opt.checkpoints_opttta, "saved_policies", policy_name)
        ensure_path(save_dir)

        # save augmentors
        for aug in augmentors:
            aug_name = type(aug).__name__
            aug.save_weights(os.path.join(save_dir, aug_name))

    def load_policy(self, policy_name, augmentors):
        save_dir = os.path.join(self.opt.checkpoints_opttta, 'saved_policies', policy_name)

        # get augmentors
        for aug in augmentors:
            aug_name = type(aug).__name__
            aug.load_weights(os.path.join(save_dir, aug_name))


    def get_batch_norm_stats(self):
        return self.unet.non_trainable_variables


    def get_slice_index(self, img, threshold):
        out = []
        for i in range(img.size()[0]):
            tmp_img = img[i].clone() # don't spoil input image
            min_val = np.quantile(tmp_img, 0.1)
            max_val = np.quantile(tmp_img, 0.9)
            tmp_img[tmp_img<min_val] = min_val
            tmp_img[tmp_img>max_val] = max_val

            tmp_img = (tmp_img - min_val)/(max_val - min_val + 1e-8)
            if tmp_img.mean() > threshold:
                out.append(i)
        
        return out


    def ensemble_predictions(self, augmentors, tgt_vol, batch_size=16):
    
        k = self.opt.k
        # self.unet.train()
        style_augmentors = [aug for aug in augmentors if type(aug).__name__ in STYLE_AUGMENTORS]
        spatial_augmentors = [aug for aug in augmentors if type(aug).__name__ in SPATIAL_AUGMENTORS]
        
        predictions_volume = []    
        ###### For visualization ######
        viz_preds = []
        viz_augs = []

        for i in range(tgt_vol.size()[0]):
            predictions = []
            for j in range(0, k//batch_size):
                aug_imgs = tgt_vol[i:(i+1)].repeat(batch_size, 1, 1, 1)

                spatial_affines = []
                for aug in style_augmentors:
                    aug_imgs = aug(aug_imgs)

                for aug in spatial_augmentors:
                    aug_imgs, affine = aug.test(aug_imgs)
                    spatial_affines.append(affine)
                
                # get predictions on the augmented images of the ith slice
                preds = self.unet(aug_imgs, training=False).numpy()

                # visualizations
                if j == 0:
                    viz_preds.append(np.argmax(preds, axis=-1))
                    viz_augs.append(aug_imgs.numpy())
                
                # invert affines in reverse order
                for aug, affine in zip(reversed(spatial_augmentors), reversed(spatial_affines)):
                    inv_affine = aug.invert_affine(affine)
                    preds, inv_affine = aug.test(preds, inv_affine)
                
                predictions.append(preds.numpy())

                ## end for j loop
            
            # gather all predictions
            predictions = np.concatenate(predictions, axis=0)
            predictions = tf.nn.softmax(predictions).numpy()
            predictions = np.mean(predictions, axis=0)

            predictions_volume.append(predictions)
            ## end for i loop
        
        # gather predictions of all images of the volume
        predictions_volume = np.stack(predictions_volume, axis=0)

        # visualizations
        viz_augs = np.stack(viz_augs, axis=0)
        viz_preds = np.stack(viz_preds, axis=0)
        
        return predictions_volume, viz_preds, viz_augs


    def optimize_augmentors(self, target_image, augmentors_list, n_steps, sample_size):
        print("Optimizing parameters for Augmentations: {}".format(', '.join([type(n).__name__ for n in augmentors_list])))
        
        if DEBUG:
            pred_visuals = []
            aug_visuals = []
            loss_curve = []
            
            ## trick if there are a lot of images
            tmp = []
            for j in range(0, target_image.shape[0], sample_size):
                batch_imgs = target_image[j:(j+sample_size)].copy()
                batch_imgs = tf.image.resize(batch_imgs, [256, 256], ResizeMethod.BILINEAR)
                pred = self.unet(batch_imgs, training=False).numpy()
                pred = np.argmax(pred, axis=-1)
                tmp.append(pred)
            
            pred = np.concatenate(tmp, axis=0)
            pred_visuals.append(pred)
            aug_visuals.append(batch_imgs.numpy())
        
        slice_indices = self.get_slice_index(target_image, 0.2)
        optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        for iter_step in tqdm.tqdm(range(n_steps)):
            # apply augmentations
            aug_imgs = target_image[random.choices(slice_indices, k=sample_size)].copy()
            with tf.GradientTape() as tape:
                for aug in augmentors_list:
                    aug_imgs = aug(aug_imgs)

                pred, feats = self.unet(aug_imgs, exp_stat=True, training=False)

                ##### compute losses
                ### batch norm loss
                curr_stats = [i for bn in feats for i in tf.nn.moments(bn, axes=[0, 1, 2])]
                loss_batch_norm = tf.reduce_sum(
                    [self.criterian_mse(curr_stats[i], self.bn_stats[i]) for i in range(len(self.bn_stats))]
                )

                ### entropy loss
                p = tf.nn.softmax(pred)
                loss_entropy = tf.reduce_mean(tf.reduce_sum(-p * tf.math.log(p + 1e-6), axis=-1)) * 100

                ### nuclear norm
                p_reduce = self.max_pool(p)
                shape = [tf.shape(p_reduce)[0], tf.shape(p_reduce)[1] * tf.shape(p_reduce)[2], tf.shape(p_reduce)[3]]
                p_reduce = tf.reshape(p_reduce, shape=shape)
                loss_nuclear = tf.reduce_mean(tf.reduce_sum(tf.linalg.svd(p_reduce, compute_uv=False), axis=-1)) * 0.5

                ###
                loss = loss_batch_norm + loss_entropy - loss_nuclear
            
            ## extract gradients
            gradients = tape.gradient(loss, [augment.trainable_variables for augment in augmentors_list])
            ## clip the gradients
            gradients = [tf.clip_by_norm(grad, 1.0) for grad in gradients]
            optimizer.apply_gradients(zip(gradients, [augment.trainable_variables for augment in augmentors_list]))

            self.metric_tracker.update_metrics({
                'total': loss.numpy(),
                'bn': loss_batch_norm.numpy(),
                'ent': loss_entropy.numpy(),
                'div': loss_nuclear.numpy()
            })

            loss_curve.append(list(self.metric_tracker.current_metrics().values()))

            # visualizations
            if DEBUG:
                if iter_step % 50 == 0:
                    pred_visuals.append(np.argmax(pred.numpy(), axis=-1))
                    aug_visuals.append(aug_imgs.numpy())
        if DEBUG:
            return np.concatenate(pred_visuals, axis=0), np.concatenate(aug_visuals, axis=0), loss_curve

        return loss_curve


    def test_time_optimize(self, target_image, target_image_name, batch_size=12, best_k_policies=3):
        n_augs = self.opt.n_augs
        sub_policies = []

        ## check if exploration is done!
        OPT_POLICY_CHECKPOINT = os.path.join(self.opt.checkpoints_opttta, 'OptimalSubpolicy.txt')
        if os.path.exists(OPT_POLICY_CHECKPOINT):
            print('\n\nOptimized Sub policies found. Performing exploitation........\n\n')
            with open(OPT_POLICY_CHECKPOINT, 'r') as f:
                OPT_POLICIES = f.readlines()

            for line in OPT_POLICIES:
                subpolicytxt = line.split('_')
                subpolicyclasses = []

                for policy in subpolicytxt:
                    policy = policy.strip()
                    subpolicyclasses.append(STRING_TO_CLASS[policy])

                sub_policies.append(subpolicyclasses)

        else:
            print('\n\nNo Optimized Sub policies exists. Performing exploration........\n\n')
            all_augmentations = [Gamma, GaussainBlur, Contrast, Brightness, Indentity, RandomResizeCrop, DummyAugmentor]
            for item in itertools.combinations(all_augmentations, n_augs):
                item = list(item)
                if DummyAugmentor in item:
                    item.remove(DummyAugmentor)
                    item += [RandomHorizontalFlip, RandomVerticalFlip, RandomRotate]

                sub_policies.append(item)

        print('\n\n')
        print(['_'.join([v.__name__ for v in sp]) for sp in sub_policies ])
        print('\n\n')

        optimized_subpolicies = []
        subpolicies_optimizers_state_dicts = []
        global_policy_losses = []

        for sub_policy in sub_policies:
            augmentations = []
            policy_name = '_'.join([n.__name__ for n in sub_policy])
            print("Optimizing for sub policies: ", policy_name)

            for policy in sub_policy:
                augmentations.append(policy()) ## create differentiable policies

            ####################################################################################################################
            ### load pre-trained augmentations if needed
            ### check if in exploration phase
            if os.path.exists(OPT_POLICY_CHECKPOINT):
                self.load_policy(policy_name, augmentations)

            ####################################################################################################################

            ####################################################################################################################

            ########## number of steps to fine tune 10 times less
            n_steps = self.opt.n_steps//10 if os.path.exists(OPT_POLICY_CHECKPOINT) else self.opt.n_steps

            if not DEBUG:
                loss_curve = self.optimize_augmentors(target_image, augmentations, n_steps, batch_size)
            else:
                pred_visuals, aug_visuals, loss_curve = self.optimize_augmentors(target_image, augmentations, n_steps, batch_size)
                self.visualize_segmentations(aug_visuals, pred_visuals, policy_name, target_image_name)
                self.visualize_losses(loss_curve, policy_name, target_image_name)

            
            global_policy_losses.append(loss_curve[-1][-1])

        best_policy_indices = np.argsort(global_policy_losses)[:best_k_policies]
        all_sub_policy_mean_predictions = {}
        all_sub_policy_uncertainty_estimation = {}
        all_sub_policy_viz_aug = {}
        all_sub_policy_viz_pred = {}

        ## remember what we discovered....
        names_opt_sub_polices = []
        for i in best_policy_indices:
            policy_name = '_'.join([type(n).__name__ for n in optimized_subpolicies[i]])
            print('Loss for policy %s %f'% (policy_name, global_policy_losses[i]))
            names_opt_sub_polices.append(policy_name)

            mean_pred, uncertainty_pred, viz_preds, viz_augs = self.ensemble_predictions(optimized_subpolicies[i], target_image)
            all_sub_policy_mean_predictions[policy_name] = mean_pred
            all_sub_policy_uncertainty_estimation[policy_name] = uncertainty_pred

            ## visualizations
            all_sub_policy_viz_aug[policy_name] = viz_augs
            all_sub_policy_viz_pred[policy_name] = viz_preds

            ## save policies if needed
            self.save_optimizer(policy_name, subpolicies_optimizers_state_dicts[i])
            self.save_policy(policy_name, optimized_subpolicies[i])

        # take average across all subpolicies
        final_prediction = np.stack(list(all_sub_policy_mean_predictions.values()), axis=0)
        final_prediction = np.mean(final_prediction, axis=-1, keepdim=False)
        final_prediction_labels = np.argmax(final_prediction, axis=-1)

        # save opt subpolicy names
        if not os.path.exists(OPT_POLICY_CHECKPOINT):
            with open(OPT_POLICY_CHECKPOINT, 'w') as f:
                for line in names_opt_sub_polices:
                    f.write("%s\n"%line)

        return final_prediction_labels, final_prediction, all_sub_policy_viz_aug, all_sub_policy_viz_pred


    def visualize_losses(self, x, policy_name, image_name):
        x = np.array(x)
        legend = list(self.metric_tracker.current_metrics().keys())

        for i in range(x.shape[1]):
            plt.plot(np.arange(x.shape[0]), x[:, i])

        plt.legend(legend)
        plt.grid(True)
        ensure_path(os.path.join(self.opt.checkpoints_opttta, 'visuals', 'loss_curves', policy_name))
        plt.savefig(os.path.join(self.opt.checkpoints_opttta, 'visuals', 'loss_curves', policy_name, image_name + '.png'))
        plt.clf()


    def make_grid(self, x, nrow):
        # x size if batch x h x w x ch
        if x.ndim == 3:
            batch, h, w = x.shape
        else:
            batch, h, w, ch = x.shape

        ncol = int(math.ceil(batch/nrow))

        if x.dim==3:
            out_array = np.zeros((ncol * h, nrow * w))
        else:
            out_array = np.zeros((ncol * h, nrow * w, ch))
        
        for i in range(batch):
            col_num = i % nrow
            row_num = i // nrow
            out_array[row_num * h: (row_num + 1) *h, col_num * w : (col_num + 1) * w, ...] = x[i].copy()

        return out_array


    def save_image(self, x, path):
        x = (255 * x).astype(np.uint8)
        Image.fromarray(x).save(path)


    def visualize_segmentations(self, imgs, segs, policy_name, img_name):
        img_grid = self.make_grid(imgs, nrow=4)
        seg_grid = self.make_grid(segs, nrow=4)
        overlay_grid = overlay_segs(img_grid, seg_grid)

        ensure_path(os.path.join(self.opt.checkpoints_opttta, 'visuals', 'segmentations', policy_name))
        self.save_image(overlay_grid, os.path.join(self.opt.checkpoints_opttta, 'visuals', 'segmentations', policy_name, img_name + '.png'))
        self.save_image(0.5*img_grid + 0.5, os.path.join(self.opt.checkpoints_opttta, 'visuals', 'segmentations', policy_name, img_name + '_img_' + '.png'))
    
    def save_pred_numpy(self, x, folder, name):
        ensure_path(os.path.join(self.opt.checkpoints_opttta, folder))
        np.save(os.path.join(self.opt.checkpoints_opttta, folder, name), x)


    def launch(self):
        self.initialize()
        ensure_path(os.path.join(self.opt.checkpoints_opttta, 'visuals', 'final_predictions'))

        for iter, (img, seg) in enumerate(self.target_test_dataloader):
            patient_name = self.target_test_dataloader.patient_names[iter]
            
            print('Predicting for image: ', patient_name)
            pred, pred_probs, all_sub_policy_mean_predictions, all_sub_policy_aug_imgs = self.test_time_optimize(img)

            viz = self.make_grid(viz, nrow=len(img))
            self.save_image(viz, os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'final_predictions', patient_name + '.png'))
            self.save_pred_numpy(pred, 'predictions', patient_name)
            self.save_pred_numpy(pred_probs, 'predictions_prob', patient_name)
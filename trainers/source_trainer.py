import tensorflow as tf
import numpy as np
from networks.unet import UNet
from dataloader import get_dataloader
import losses
import numpy as np
from utils import ensure_path, color_seg
import os
import tensorflow_addons as tfa

class SourceTrainer:
    def __init__(self, opt):
        self.opt = opt
        ## data loaders from numpy
        self.train_loader, self.val_loader = get_dataloader(opt.rootdir, opt.sites, opt.batch_size)

        ## model
        self.model = UNet(opt.n_classes)
        self.model(np.zeros([16, 256, 256, 1]), training = False)

        ## optimizers and schedular
        schedular = tf.keras.optimizers.schedules.CosineDecay(opt.lr, opt.n_steps//opt.batch_size, alpha=0.01)
        rmsprop = tfa.optimizers.extend_with_decoupled_weight_decay(tf.keras.optimizers.RMSprop)
        self.optimizer = rmsprop(learning_rate=schedular, weight_decay=1e-4)

        ## ensure paths
        ensure_path(os.path.join(opt.checkpoints_dir, "saved_models"))

        ## tensorboard
        self.summary_writer = tf.summary.create_file_writer(os.path.join(opt.checkpoints_dir, "tf_logs"))



    @tf.function
    def train_step(self, img, seg):
        with tf.GradientTape() as tape:
            loss = losses.compute_weighted_loss(self.model, img, seg, is_train=True)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss



    @tf.function
    def valid_step(self, image, label):
        loss = losses.compute_weighted_loss(self.model, image, label, is_train=False)
        return loss


    @tf.function
    def batch_dice_multiple(self, image, label):
        pred = self.model(image, training=False)
        pred_softmax = tf.nn.softmax(pred)
        pred_seg = tf.cast(tf.argmax(pred_softmax, -1), tf.float32)
        dsc_batch_list = list()

        # Note: Following for loop goes from 0 to (num_classes-1)
        # and ignore_index is num_classes, thus ignore_index is
        # not considered in computation of IoU.
        for sem_class in range(1, self.opt.n_classes):
            pred_inds = tf.cast(pred_seg == sem_class, tf.float32)
            target_inds = tf.cast(label == sem_class, tf.float32)
            dsc_batch = (tf.reduce_sum(2 * pred_inds * target_inds)+1e-7) / ((tf.reduce_sum(pred_inds) + tf.reduce_sum(target_inds))+1e-7)
            # print(dsc_batch)
            dsc_batch_list.append(dsc_batch)
        return dsc_batch_list



    def run(self):
        steps_so_far = 0
        train_iterator = iter(self.train_loader)

        while steps_so_far <= self.opt.n_steps:
            try:
                img, seg = next(train_iterator)
            except:
                train_iterator = iter(self.train_loader)
                img, seg = next(train_iterator)
            
            train_loss = self.train_step(img, seg)

            # needs printing
            if steps_so_far % self.opt.print_freq < self.opt.batch_size:
                with self.summary_writer.as_default():
                    tf.summary.scalar('training/loss', train_loss, step=steps_so_far)
                    tf.summary.scalar('training/lr', self.optimizer._decayed_lr(tf.float32), step=steps_so_far)
                print("Step: %10d | Training Loss: %.10f" % (steps_so_far, train_loss))

            # needs displaying
            if steps_so_far % self.opt.display_freq < self.opt.batch_size:
                with self.summary_writer.as_default():
                    predictions = self.model(img, training=False).numpy()
                    predictions = np.argmax(predictions, axis=-1)
                    predictions = color_seg(predictions)
                    ground_truths = color_seg(seg.numpy())

                    tf_image = (255*(0.5 * img.numpy() + 0.5)).astype(np.uint8)
                    tf.summary.image("train_img", tf_image, step=steps_so_far)
                    tf.summary.image("preds", predictions,  step=steps_so_far)
                    tf.summary.image("gts", ground_truths,  step=steps_so_far)

            # needs validation
            if steps_so_far % self.opt.evaluation_freq < self.opt.batch_size:
                losses = tf.keras.metrics.Mean()
                dic1 = []
                dic2 = []
                dice1 = tf.keras.metrics.Mean()
                dice2 = tf.keras.metrics.Mean()
                for (x_valid, y_valid) in self.val_loader:
                    eloss = self.valid_step(x_valid, y_valid)
                    v_dices = self.batch_dice_multiple(x_valid, y_valid)
                    losses(eloss)
                    dice1(v_dices[0])
                    dice2(v_dices[1])

                    dic1 += [v_dices[0]]
                    dic2 += [v_dices[1]]


                with self.summary_writer.as_default():
                    tf.summary.scalar('validation/loss', losses.result(),   step=steps_so_far)
                    tf.summary.scalar('validation/mdice_1', dice1.result(), step=steps_so_far)
                    tf.summary.scalar('validation/mdice_2', dice2.result(), step=steps_so_far)
                    tf.summary.scalar('validation/mean_dice', 0.5*dice1.result() + 0.5*dice2.result(), step=steps_so_far)
            
            
            # needs saving
            if steps_so_far % self.opt.save_freq < self.opt.batch_size:
                self.model.save_weights(os.path.join(self.opt.checkpoints_dir, "saved_models", str(steps_so_far)))
            
            # increment steps
            steps_so_far += self.opt.batch_size
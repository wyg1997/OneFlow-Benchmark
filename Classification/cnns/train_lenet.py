"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import oneflow as flow
import ofrecord_util
import config as configs
from util import Snapshot, Summary, InitNodes, Metric
from job_function_util import get_train_config, get_val_config
import resnet_model
import resnext_model
import vgg_model
import alexnet_model
import inception_model
import mobilenet_v2_model

parser = configs.get_parser()
args = parser.parse_args()
configs.print_args(args)

total_device_num = args.num_nodes * args.gpu_num_per_node
train_batch_size = total_device_num * args.batch_size_per_device
val_batch_size = total_device_num * args.val_batch_size_per_device
(C, H, W) = args.image_shape
epoch_size = math.ceil(args.num_examples / train_batch_size)
num_val_steps = int(args.num_val_examples / val_batch_size)


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv1",
        kernel_initializer=initializer,
    )
    pool1 = flow.nn.max_pool2d(
        conv1, ksize=2, strides=2, padding="SAME", name="pool1", data_format="NCHW"
    )
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
    )
    pool2 = flow.nn.max_pool2d(
        conv2, ksize=2, strides=2, padding="SAME", name="pool2", data_format="NCHW"
    )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="dense1",
    )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    output = flow.layers.dense(
        hidden,
        10,
        kernel_initializer=initializer,
        name="dense2"
    )
    return output


flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.enable_debug_mode(True)


def label_smoothing(labels, classes, eta, dtype):
    assert classes > 0
    assert eta >= 0.0 and eta < 1.0

    return flow.one_hot(labels, depth=classes, dtype=dtype,
                        on_value=1 - eta + eta / classes, off_value=eta/classes)


@flow.global_function("train", get_train_config(args))
def TrainNet():
    if args.train_data_dir:
        assert os.path.exists(args.train_data_dir)
        print("Loading data from {}".format(args.train_data_dir))
        (labels, images) = ofrecord_util.load_imagenet_for_training(args)

    else:
        print("Loading synthetic data.")
        (labels, images) = ofrecord_util.load_synthetic(args)
    # logits = model_dict[args.model](images,
    #                                 need_transpose=False if args.train_data_dir else True,
    #                                 )
    logits = lenet(images, True)

    if args.label_smoothing > 0:
        one_hot_labels = label_smoothing(labels, args.num_classes, args.label_smoothing, logits.dtype)
        loss = flow.nn.softmax_cross_entropy_with_logits(one_hot_labels, logits, name="softmax_loss")
    else:
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    flow.losses.add_loss(loss)
    predictions = flow.nn.softmax(logits)
    outputs = {"loss": loss, "predictions": predictions, "labels": labels}
    return outputs


def main():
    InitNodes(args)
    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)

    summary = Summary(args.log_dir, args)
    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    for epoch in range(args.num_epochs):
        metric = Metric(desc='train', calculate_batches=args.loss_print_every_n_iter,
                        summary=summary, save_summary_steps=epoch_size,
                        batch_size=train_batch_size, loss_key='loss')
        for i in range(epoch_size):
            TrainNet().async_get(metric.metric_cb(epoch, i))


if __name__ == "__main__":
    main()

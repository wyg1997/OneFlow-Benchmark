from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
from datetime import datetime

import oneflow as flow

import data_loader
import vgg_model
import resnet_model
import alexnet_model
import inceptionv3_model

parser = argparse.ArgumentParser(description="flags for oneflow cnn to onnx")

parser.add_argument("--model", type=str, default="alexnet", help="alexnet, vgg16 or resnet50")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--image_size", type=int, default=224, help="image size")
parser.add_argument("--model_load_dir", type=str,
                    default='/home/xiexuan/sandbox/OneFlow-Benchmark/snapshot_alexnet_init',
                    help="model load directory")
parser.add_argument("--log_dir", type=str, default="./output", help="log info save directory")

args = parser.parse_args()
(H, W, C) = (args.image_size, args.image_size, 3)

model_dict = {
    "resnet50": resnet_model.resnet50,
    "inceptionv3": inceptionv3_model.inceptionv3,
    "vgg16": vgg_model.vgg16,
    "alexnet": alexnet_model.alexnet,
}

flow.config.gpu_device_num(1)
flow.config.enable_debug_mode(True)
val_config = flow.function_config()
val_config.default_distribute_strategy(flow.distribute.consistent_strategy())
val_config.default_data_type(flow.float)

@flow.function(val_config)
def InferenceNet(images=flow.FixedTensorDef((args.batch_size, H, W, C), dtype=flow.float),
                 labels=flow.FixedTensorDef((args.batch_size, ), dtype=flow.int32)):
    logits = model_dict[args.model](images)
    softmax = flow.nn.softmax(logits)
    outputs = {"softmax":softmax, "labels": labels}
    return outputs


def main():
    print("=".ljust(66, "="))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("-".ljust(66, "-"))
    print("Time stamp: {}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))

    assert args.model_load_dir
    check_point = flow.train.CheckPoint()
    if args.model_load_dir:
        assert os.path.isdir(args.model_load_dir)
        print("Restoring model from {}.".format(args.model_load_dir))
        check_point.load(args.model_load_dir)
    else:
        print("Init model on demand.")
        check_point.init()
    onnx_model = flow.export_onnx(args.model_load_dir)
    print(type(onnx_model))


if __name__ == "__main__":
    main()

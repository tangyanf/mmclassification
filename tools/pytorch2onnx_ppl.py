import argparse
from functools import partial

import mmcv
import numpy as np
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

from mmcls.models import build_classifier

torch.manual_seed(3)

import onnxtool

def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cpu().eval()

    num_classes = model.head.num_classes
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :] for img in imgs]

    print(img_list)
    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, return_loss=False)

    register_extra_symbolics(opset_version)
    with torch.no_grad():

        input_tensors = { 'input_mmc': img_list[0] }
        output_names = ['output_mmc']

        output_tensors = onnxtool.ppl_export(model,
                  input_tensors,
                  output_file,
                  output_names,
                  opset_version)

        """
        
        torch.onnx.export(
            model, (img_list, ),
            output_file,
            export_params=True,
            keep_initializers_as_inputs=True,
            verbose=show,
            opset_version=opset_version)
            """
        print(f'Successfully exported ONNX model: {output_file}')
    model.forward = origin_forward

    pytorch_result = model(img_list, return_loss=False)[0]
    if not np.allclose(pytorch_result, output_tensors['output_mmc']):
        print('The outputs are different between Pytorch and ONNX')
        return 1
            
    print('The outputs are same between Pytorch and ONNX')
    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ONNX')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    #parser.add_argument('--cifromlist',  help="run ci from list path, list = [[/'1.py/', /'1.pth/'],[/'2.py/', /'2.pth/'] ,...]",default="")
    #parser.add_argument('--cifromrule',  help='run ci from rule: checkpoints match configs, such as resnet18_*.pth match to resnet18_*.py  ' )
    parser.add_argument('--all', action='store_true',
                        help='run ci from rule: checkpoints match configs, such as resnet18_*.pth match to resnet18_*.py  ')
    args = parser.parse_args()
    return args

def run(args,
        configs,
        checkpoints):
    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
                          1,
                          3,
                      ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')


    cfg = mmcv.Config.fromfile(configs)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)

    if checkpoints:
        load_checkpoint(classifier, checkpoints, map_location='cpu')

    # conver model to onnx file
    return pytorch2onnx(
        classifier,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify)


def runFolder(checkpointPath,configPath):
    checkpoints = os.listdir(checkpointPath)
    configs = os.listdir(configPath)

    numCases = 0
    numFailedCase = 0

    for checkpoint in checkpoints:
        netName = checkpoint.split('_')[0]
        for config in configs:
            if config.split('_')[0] == netName:
                # match
                numFailedCase += run(args,
                    configPath + config,
                    checkpointPath + checkpoint)
                numCases += 1
    return numCases,numFailedCase

if __name__ == '__main__':
    args = parse_args()

    if not args.all:
        run(args,
            args.config,
            args.checkpoint)

    else:
        #new ci codes

        #load ci lists from files
        #or rules for multi cases

        """
        ciList = [
            ['..//configs//imagenet//resnet50_b32x8.py', '..//checkpoint//res50.pth'] ,#case1
            ['..//configs//imagenet//resnet50_b32x8_coslr.py', '..//checkpoint//res50.pth']  # case2
        ]
        """

        import os

        checkpointPath = "./checkpoint/cifar10/"
        configPath = "../configs/cifar10/"

        numCases = 0
        numFailedCase = 0;
        numCasesThis,numFailedCaseThis = runFolder(checkpointPath,configPath)
        numCases += numCasesThis;
        numFailedCase += numFailedCaseThis


        checkpointPath = "./checkpoint/imagenet/"
        configPath = "../configs/imagenet/"
        numCasesThis,numFailedCaseThis = runFolder(checkpointPath,configPath)
        numCases += numCasesThis;
        numFailedCase += numFailedCaseThis

        exitWord = "Test mmclassfication done. There are " + str(numCases) +  "cases. Failed " + str(numFailedCase) + " cases!"

        print (exitWord )
        import sys
        sys.exit(numFailedCase)

        """
        a=$(/usr/bin/python /path/hooks.py "$1" "$2" )
        echo $a
        """

"""
../configs/imagenet/resnet50_b32x8.py
--checkpoint ./checkpoint/res50.pth
--all
"""

#!/usr/bin/env python3

from pathlib import Path
import argparse
import os

import d3.model.tools as mt
import functools as fc
from d3.model.basemodel import Vector

def check_path(path, should_exist):
    """ Check that a path (file or folder) exists or not and return it.
    """
    path = os.path.normpath(path)
    if should_exist != os.path.exists(path):
        msg = "path " + ("does not" if should_exist else "already") + " exist: " + path
        raise argparse.ArgumentTypeError(msg)
    return path

def main(args):

    if (args.from_up is None) != (args.to_up is None):
        raise Exception("from-up and to-up args should be both present or both absent")

    up_conversion = None
    if args.from_up is not None:
        up_conversion = (args.from_up, args.to_up)

    # output = args.output if args.output is not None else '.' + args.type
    for category_dir in os.listdir(args.data_root):
        category_path=Path(args.data_root).joinpath(category_dir)
        if not os.path.isdir(category_path):
            print(category_dir)
            continue 
        out_category_path=Path(args.output_root).joinpath(category_dir)
        if not os.path.isdir(out_category_path):
            os.makedirs(Path(out_category_path).joinpath('train'))
            os.makedirs(Path(out_category_path).joinpath('test'))
        for mode in os.listdir(category_path):
            with os.scandir(Path(category_path).joinpath(mode)) as entries:
                for entry in entries:
                    entry_path=os.path.join(category_path,mode,entry.name,f'{entry.name}.off')
                    entry_name=entry.name
                    # print(entry_path)
                    # if os.path.splitext(entry)[1]=='.off':
                    if os.path.exists(entry_path):
                        in_filename=entry_path
                        out_entry=entry.name+'.obj'
                        out_filename=Path(out_category_path).joinpath(mode,out_entry)
                        in_filename=str(in_filename)
                        out_filename=str(out_filename)
                        result = mt.convert(in_filename, out_filename, up_conversion)
                        # if args.output is None:
                            # print(result)
                        with open(out_filename, 'w') as f:
                            f.write(result)
                        print(f'{out_filename} complete...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=main)
    parser.add_argument('-v', '--version', action='version', version='1.0')
    parser.add_argument('-i', '--input', metavar='input',
                        type=fc.partial(check_path, should_exist=True), default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', metavar='output',
                        help='Output path')
    parser.add_argument('-t', '--type', metavar='type',
                        help='Export type, useless if output is specified')
    parser.add_argument('-fu', '--from-up', metavar='fup', default=None,
                        help="Initial up vector")
    parser.add_argument('-tu', '--to-up', metavar='fup', default=None,
                        help="Output up vector")
    parser.add_argument('--data-root',type=str,default="/eva_data_eva_data_Augmentation/datasets_raw/ModelNet40_auto_aligned")
    parser.add_argument('--output-root',type=str,default='/eva_data_eva_data_Augmentation/datasets_raw/ModelNet40_auto_aligned_obj')
    args = parser.parse_args()
    args.func(args)


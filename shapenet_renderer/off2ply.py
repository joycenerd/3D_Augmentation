import igl

from pathlib import Path
import argparse
import os


parser=argparse.ArgumentParser()
parser.add_argument('--data-root',type=str,default="/eva_data_7/modelnet40_auto_aligned")
parser.add_argument('--output-root',type=str,default='/eva_data_7/modelnet40_auto_aligned_ply')
args=parser.parse_args()

import os
import igl

def convert_output(input_filename,out_filename):
    input_filename=str(input_filename)
    out_filename=str(out_filename)
    vertices, faces = igl.read_triangle_mesh(input_filename)
    f=open(out_filename,"a")
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('comment by YZF\n')
    f.write('comment PCL generated\n')
    f.write('element vertex {}\n'.format(len(vertices)))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('element face {}\n'.format(len(faces)))
    f.write('property list uchar int vertex_index\n')
    f.write('end_header\n')
    for i in range(len(vertices)):
        x, y, z = vertices[i]
        f.write('{} {} {}\n'.format(x, y, z))
    for j in range(len(faces)):
        a, b, c = faces[j]
        f.write('{} {} {} {}\n'.format(3, a, b, c))    
    f.close             


for category_dir in os.listdir(args.data_root):
    category_path=Path(args.data_root).joinpath(category_dir)
    out_category_path=Path(args.output_root).joinpath(category_dir)
    if not os.path.isdir(out_category_path):
        os.makedirs(Path(out_category_path).joinpath('train'))
        os.makedirs(Path(out_category_path).joinpath('test'))
    for mode in os.listdir(category_path): 
        with os.scandir(Path(category_path).joinpath(mode)) as entries:
            for entry in entries:
                if entry.is_file():
                    if os.path.splitext(entry)[1]=='.off':
                        in_filename=Path(category_path).joinpath(mode,entry.name)
                        out_entry=entry.name[:-3]+'ply'
                        out_filename=Path(out_category_path).joinpath(mode,out_entry)
                        convert_output(in_filename,out_filename)
                        print(f'{out_filename} complete...')

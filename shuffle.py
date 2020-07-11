import functools
import os
import tempfile
import random
import time
from typing import List

import numpy as np
from PIL import Image
import tensorflow as tf

SATURATION = 255
VALUE = 255

HILBERT_PATTERN = [np.array([0, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0])]
HILBERT_ROTATIONS = [
    lambda p, side: np.array([p[1], p[0]]),
    lambda p, side: p,
    lambda p, side: p,
    lambda p, side: np.array([side - 1 - p[1], side - 1 - p[0]]),
]

OUTPUT_DIR = 'output'

def base4_range(n):
    '''Iterates over range(4 ** n) as little-endian base4 representation.'''
    current = [0] * n
    yield tuple(current)
    for i in range(4 ** n - 1):
        current[0] += 1
        # carryover
        for j in range(n-1):
            if current[j] == 4:
                current[j] = 0
                current[j+1] += 1
        yield tuple(current)

def hilbert_curve(n):
    '''Takes an order n and returns an iterable of 2-tuples.

    Curve of order 0 is [(0, 0)]
    Curve of order 1 is [(0, 0), (0, 1), (1, 1), (1, 0)]
    etc.
    '''
    for base4_i in base4_range(n):
        # Accumulate a list of offsets, then add them up to get final point
        accum = np.array([0, 0])
        for i in range(n):
            if i != 0:
                accum = HILBERT_ROTATIONS[base4_i[i]](accum, 2**i)
            accum += 2**i * HILBERT_PATTERN[base4_i[i]]
        yield tuple(accum)

def log_4(n):
    '''Returns the smallest power of 4 >= n.'''
    i = 1
    while 4 ** i < n:
        i += 1
    return i

def make_hilbert_png(ordering, png_filename):
    '''Takes a shuffled list of integers 0...n and visualizes it as an image.'''
    hilbert_order = log_4(len(ordering))
    hues = np.linspace(0, 255, len(ordering))
    image_dims = 2 ** hilbert_order
    image_array = np.zeros([image_dims, image_dims, 3])
    for index, coord in zip(ordering, hilbert_curve(hilbert_order)):
        image_array[coord][0] = hues[index]
    image_array[:, :, 1] = SATURATION
    image_array[:, :, 2] = VALUE
    image_array = image_array.astype(np.uint8)
    img = Image.fromarray(image_array, 'HSV')
    img = img.resize((128, 128), resample=Image.NEAREST)
    img = img.convert('RGB')
    img.save(png_filename)
    return png_filename

def make_hilbert_curve_svg(hilbert_order, svg_filename):
    '''Takes an iterable of 2-tuples and plots the hilbert curve.'''
    curve = list(hilbert_curve(hilbert_order))
    image_size = 128
    padding = 8
    canvas_size = image_size + 2 * padding
    cell_center = image_size / (2 ** hilbert_order) / 2
    svg = [
        '<?xml version="1.0" encoding="UTF-8" ?>',
        '<svg height="{canvas_size}" width="{canvas_size}" xmlns="http://www.w3.org/2000/svg" version="1.1">'.format(canvas_size=canvas_size)]
    curve_size = 2 ** log_4(len(curve))
    for (y1, x1), (y2, x2) in zip(curve[:-1], curve[1:]):
        x1 *= image_size / curve_size
        x2 *= image_size / curve_size
        y1 *= image_size / curve_size
        y2 *= image_size / curve_size
        x1 += padding + cell_center
        x2 += padding + cell_center
        y1 += padding + cell_center
        y2 += padding + cell_center
        svg.append('<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="stroke:rgb(255,0,0);stroke-width:2" />'.format(
            x1=x1, x2=x2, y1=y1, y2=y2))
    svg.append('</svg>')
    completed_svg = '\n'.join(svg)
    with open(svg_filename, 'w') as f:
        f.write(completed_svg)

def shard_list(numbers, num_shards, jitter=False):
    if jitter:
        # Deterministic sharding for fair comparisons
        random.seed(17)
        # Minimum shard size is 0.75x the desired shard_list size. Remaining elements
        # are distributed semi-randomly between the shards.
        set_aside = int(len(numbers) * 0.75)
        remaining = len(numbers) - set_aside
        splits = sorted(random.randint(0, remaining) for i in range(num_shards - 1))
        splits = [split + int(set_aside * (i + 1) / num_shards)
            for i, split in enumerate(splits)]
        shards = []
        for lower, upper in zip([0] + splits, splits + [len(numbers)]):
            shards.append(numbers[lower:upper])
        print("Sharded as {}".format(list(map(len, shards))))
        assert sum(map(len, shards)) == len(numbers)
    else:
        if len(numbers) % num_shards != 0:
            raise Exception("list cannot be evenly split into {}".format(num_shards))
        shard_size = len(numbers) // num_shards
        shards = [numbers[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
    return shards


def pseudoshuffle(
    shards: List[List[int]], buffer_size: int, num_chained_buffers: int = 1,
    parallel_reads=1):
    # Deterministic shuffling of file order for fair comparisons.
    random.seed(17)
    random.shuffle(shards)
    shards = [tf.data.Dataset.from_tensor_slices(s) for s in shards]
    shard_dataset = tf.data.Dataset.from_tensor_slices(shards)
    dataset = shard_dataset.interleave(
        lambda x: x,
        cycle_length=parallel_reads, block_length=1)

    for i in range(num_chained_buffers):
        dataset = dataset.shuffle(buffer_size=buffer_size)

    return list(dataset)


def composite_images(rows, columns, image_names):
    '''Takes a list of list of filenames and outputs HTML table of <img>.'''
    html_parts = ['<table>']
    html_parts.append('<tr><td />')
    html_parts.extend('<th>{}</th>'.format(column) for column in columns)
    html_parts.append('</tr>')
    for row_name, img_row in zip(rows, image_names):
        html_parts.append('<tr>')
        html_parts.append('<th>{}</th>'.format(row_name))
        for img in img_row:
            if img is None:
                html_parts.append('<td></td>')
            else:
                html_parts.append('<td><img src="{}"></td>'.format(img))
        html_parts.append('</tr>')
    html_parts.append('</table>')
    return '\n'.join(html_parts)

def create_img_table(test, dimension1, dimension2):
    '''Takes a function, and two lists of params, and generates an HTML table.

    The HTML table will be the cartesian product of the two param lists.'''
    images = []
    for param1 in dimension1:
        img_row = []
        for param2 in dimension2:
            img_name = '{}_{}_{}.png'.format(test.__name__, param1, param2)
            shuffled_list = test(param1, param2)
            if not shuffled_list:
                img_name = None
            else:
                make_hilbert_png(shuffled_list, os.path.join(OUTPUT_DIR, img_name))
            img_row.append(img_name)
        images.append(img_row)
    composite_html = composite_images(dimension1, dimension2, images)
    filename = '{}/{}.html'.format(OUTPUT_DIR, test.__name__)
    with open(filename, 'w') as f:
        f.write('<html><body>\n')
        f.write(composite_html)
        f.write('</body></html>\n')


def basic_scaling(data_size, buffer_size_ratio):
    buffer_size = int(data_size * buffer_size_ratio) + 1
    l = list(range(data_size))
    return pseudoshuffle([l], buffer_size=buffer_size)

def chained_scaling(buffer_size_ratio, num_chained_buffers):
    data_size = 2 ** 14
    buffer_size = int(data_size * buffer_size_ratio / num_chained_buffers) + 1
    l = list(range(data_size))
    return pseudoshuffle([l],
        buffer_size=buffer_size, num_chained_buffers=num_chained_buffers)

def sharded_scaling(buffer_size_ratio, num_shards):
    data_size = 2 ** 14
    buffer_size = int(data_size * buffer_size_ratio) + 1
    l = list(range(data_size))
    shards = shard_list(l, num_shards)
    return pseudoshuffle(shards, buffer_size=buffer_size)

def parallel_read_scaling(num_shards, parallel_reads):
    if parallel_reads > num_shards:
        return []
    data_size = 2 ** 14
    buffer_size = data_size // 100
    l = list(range(data_size))
    shards = shard_list(l, num_shards)
    return pseudoshuffle(shards, buffer_size=buffer_size, parallel_reads=parallel_reads)

def parallel_read_scaling_jittered(num_shards, parallel_reads):
    if parallel_reads > num_shards:
        return []
    data_size = 2 ** 14
    buffer_size = data_size // 100
    l = list(range(data_size))
    shards = shard_list(l, num_shards, jitter=True)
    return pseudoshuffle(shards, buffer_size=buffer_size, parallel_reads=parallel_reads)

def twice_shuffled(num_shards, parallel_reads):
    if parallel_reads > num_shards:
        return []
    data_size = 2 ** 14
    buffer_size = data_size // 100
    l = list(range(data_size))
    shards = shard_list(l, num_shards)
    shuffled_shards = []
    for shard in shards:
        subshards = shard_list(shard, num_shards)
        shuffled_shards.append(pseudoshuffle(subshards, buffer_size=buffer_size, parallel_reads=parallel_reads))
    return pseudoshuffle(shuffled_shards, buffer_size=buffer_size, parallel_reads=parallel_reads)

for i in range(1, 6):
    make_hilbert_curve_svg(i, os.path.join(OUTPUT_DIR, 'hilbert_curve_{}.svg'.format(i)))

create_img_table(basic_scaling, (1024, 4096, 16384), (0, 0.01, 0.1, 0.5, 1))
create_img_table(chained_scaling, (0, 0.01, 0.1, 0.5), (1, 2, 4))
create_img_table(sharded_scaling, (0, 0.01, 0.1, 0.5), (1, 2, 4, 8))
create_img_table(parallel_read_scaling, (1, 2, 4, 8), (1, 2, 4, 8))
create_img_table(parallel_read_scaling_jittered, (1, 2, 4, 8), (1, 2, 4, 8))
create_img_table(twice_shuffled, (1, 2, 4, 8), (1, 2, 4, 8))

'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-20 16:55:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-20 16:55:47
'''
import math

def conv_output_size(in_size, kernel_size, padding=0, stride=1, dilation=1):
    return math.floor((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

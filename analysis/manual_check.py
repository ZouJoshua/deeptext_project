#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/6/22 10:40 上午
@File    : manual_check.py
@Desc    : 人工检查指标统计

"""

"""
1.查看人工标注情况
每个类目随机打散，抽取100条信息（不足100条的抽取全部），人工check，
2.模型训练badcase查看
先查看人工标注情况的100条信息和机器打标的情况，重点查看预测错误或者预测正确的概率比较低的
随机抽取100条查看
1）预测错误的概率比较大
2）预测正确的概率比较低的
"""


import csv
import random
import json
from general_tools.utils import read_xlrd


def get_check_example(csv_file, example_file):
    """
    获取核对人工检查的数据
    """
    with open(csv_file, "r", encoding="utf-8") as f:
        f_csv = csv.reader(f, delimiter='\t')
        header = next(f_csv)
        label_index = header.index("label")
        all_count = dict()
        for row in f_csv:
            label = row[label_index]
            if label in all_count:
                all_count[label].append(row)
            else:
                all_count[label] = [row]
    with open(example_file, "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        for k, v in all_count.items():
            random.shuffle(v)
            new_v = v[:100]
            for r in new_v:
                writer.writerow(r)


def analysis_example(excel_file):
    head, table = read_xlrd(excel_file)
    if_real_index = head.index("true/false")
    label_index = head.index("label")
    manual_label_index = head.index("manual_check_one")
    predict_label_index = head.index("predict_label")
    label_count = dict()
    for row_num in range(1, table.nrows):
        row_value = table.row_values(row_num)
        if row_value[if_real_index] == "":
            continue
        if_real = int(row_value[if_real_index])
        label = row_value[label_index]
        manual_label = row_value[manual_label_index]
        predict_label = row_value[predict_label_index].split(",")
        if label in label_count:
            label_count[label]["right_count"] += if_real
            label_count[label]["all_count"] += 1
            if manual_label in predict_label:
                label_count[label]["predict_count"] += 1
            if manual_label in label_count[label]["manual_label"]:
                label_count[label]["manual_label"][manual_label] += 1
            else:
                label_count[label]["manual_label"][manual_label] = 1
        else:
            label_count[label] = dict()
            label_count[label]["right_count"] = if_real
            label_count[label]["predict_count"] = 0
            if manual_label in predict_label:
                label_count[label]["predict_count"] += 1
            label_count[label]["all_count"] = 1
            label_count[label]["manual_label"] = dict()
            if manual_label in label_count[label]["manual_label"]:
                label_count[label]["manual_label"][manual_label] += 1
            else:
                label_count[label]["manual_label"][manual_label] = 1
    label_analysis = dict()
    for k, v in label_count.items():
        v["percent"] = "%.2f%%" % (v["right_count"] / v["all_count"] * 100)
        v["predict_percent"] = "%.2f%%" % (v["predict_count"] / v["all_count"] * 100)
        label_analysis[k] = v
    print(json.dumps(label_analysis, ensure_ascii=False, indent=4))


def get_badcase(csv_file, example_file):
    """
    获取模型预测错误的badcase
    """
    with open(csv_file, "r", encoding="utf-8") as f:
        f_csv = csv.reader(f, delimiter='\t')
        header = next(f_csv)
        label_index = header.index("label")
        all_count = dict()
        for row in f_csv:
            label = row[label_index]
            if label in all_count:
                all_count[label].append(row)
            else:
                all_count[label] = [row]




def main():
    # file = "/work/data/result_check.csv"
    # outfile = "./result_sample_check.csv"
    # get_check_example(file, outfile)
    excel_file = "/Desktop/result_sample_check.xlsx"
    analysis_example(excel_file)

if __name__ == "__main__":
    main()

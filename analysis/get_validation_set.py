#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/6/30 11:06 上午
@File    : get_validation_set.py
@Desc    : 从点检数据制作文本分类高质量的验证集（测试集）

"""

import json
from analysis.utils import read_xlrd, read_json_format_file


def process_manual_category(tag, manual_tag):
    """
    处理category信息
    """
    replace_tag = str(manual_tag)

    if str(manual_tag) in ["1", "1.0"]:
        replace_tag = tag

    if "[" and "]" not in replace_tag:
        tag_list = replace_tag.split(",")
    else:
        tag_list = eval(replace_tag)

    return ",".join([t.split("|")[0] for t in tag_list])


def process_manual_tonality(tag, manual_tag):
    """
    处理tonality调性分类
    """
    if str(tag).lower() in ["0.0", "false"]:
        tag = "false"
    elif str(tag).lower() in ["1.0", "true"]:
        tag = "true"
    else:
        tag = tag

    if str(manual_tag) in ["1", "1.0"]:
        standard_tag = tag
    else:
        if str(manual_tag).lower() in ["0.0", "false"]:
            standard_tag = "false"
        else:
            standard_tag = manual_tag

    return standard_tag





def write_validation_set_to_file(excel_file, outfile):
    """
    将点检信息处理为文本
    """
    head, table = read_xlrd(excel_file)
    # 一二级分类
    alg_tag2_index = head.index("alg_tag-2.0")
    manual_tag2_index = head.index("manual_tag-2.0")
    alg_subtag2_index = head.index("alg_subtag-2.0")
    manual_subtag2_index = head.index("manual_subtag-2.0")
    alg_tag3_index = head.index("alg_tag-3.0")
    manual_tag3_index = head.index("manual_tag-3.0")
    alg_subtag3_index = head.index("alg_subtag-3.0")
    manual_subtag3_index = head.index("manual_subtag-3.0")
    # 低调性分类
    alg_vulgar_index = head.index("alg_vulgar")
    manual_vulgar_index = head.index("manual_vulgar")
    alg_gossip_index = head.index("alg_gossip")
    manual_gossip_index = head.index("manual_gossip")
    alg_clickbait_index = head.index("alg_clickbait")
    manual_clickbait_index = head.index("manual_clickbait")
    alg_advert_index = head.index("alg_advert")
    manual_advert_index = head.index("manual_advert")
    file = open(outfile, "w", encoding="utf-8")
    for row_num in range(1, table.nrows):
        row_value = table.row_values(row_num)
        # print(type(row_value[alg_tag2_index]))
        row_value[manual_tag2_index] = process_manual_category(row_value[alg_tag2_index], row_value[manual_tag2_index])
        row_value[manual_subtag2_index] = process_manual_category(row_value[alg_subtag2_index], row_value[manual_subtag2_index])
        row_value[manual_tag3_index] = process_manual_category(row_value[alg_tag3_index], row_value[manual_tag3_index])
        row_value[manual_subtag3_index] = process_manual_category(row_value[alg_subtag3_index], row_value[manual_subtag3_index])

        row_value[manual_vulgar_index] = process_manual_tonality(row_value[alg_vulgar_index], row_value[manual_vulgar_index])
        row_value[manual_gossip_index] = process_manual_tonality(row_value[alg_gossip_index], row_value[manual_gossip_index])
        row_value[manual_clickbait_index] = process_manual_tonality(row_value[alg_clickbait_index], row_value[manual_clickbait_index])
        row_value[manual_advert_index] = process_manual_tonality(row_value[alg_advert_index], row_value[manual_advert_index])

        row_line = dict(zip(head, row_value))
        file.write(json.dumps(row_line, ensure_ascii=False) + "\n")

    file.close()

def _label_add(key, label_dict):
    if key in label_dict:
        label_dict[key] += 1
    else:
        label_dict[key] = 1


def label_count(file):
    all_label = dict()
    all_label["tag2.0"] = {"top": {"one_label": 0, "multi_label": 0}, "sub": {"one_label": 0, "multi_label": 0}}
    all_label["tag3.0"] = {"top": {"one_label": 0, "multi_label": 0}, "sub": {"one_label": 0, "multi_label": 0}}
    for line in read_json_format_file(file):
        top_category2 = line["manual_tag-2.0"]
        if top_category2 == "0.0":
            print(line)
        sub_category2 = line["manual_subtag-2.0"]
        top_category3 = line["manual_tag-3.0"]
        sub_category3 = line["manual_subtag-3.0"]
        if "," not in top_category2 and top_category2 != "":
            all_label["tag2.0"]["top"]["one_label"] += 1
            # label = top_category2.split("_")[0]

        else:
            all_label["tag2.0"]["top"]["multi_label"] += 1
        _label_add(top_category2, all_label["tag2.0"]["top"])

        if "," not in sub_category2 and sub_category2 != "":
            all_label["tag2.0"]["sub"]["one_label"] += 1
        else:
            all_label["tag2.0"]["sub"]["multi_label"] += 1
        _label_add(sub_category2, all_label["tag2.0"]["sub"])

        if "," not in top_category3 and top_category3 != "":
            all_label["tag3.0"]["top"]["one_label"] += 1
        elif top_category3 == "":
            continue
        else:
            all_label["tag3.0"]["top"]["multi_label"] += 1
        _label_add(top_category3, all_label["tag3.0"]["top"])

        if "," not in sub_category3 and sub_category3 != "":
            all_label["tag3.0"]["sub"]["one_label"] += 1
        elif sub_category3 == "":
            continue
        else:
            all_label["tag3.0"]["sub"]["multi_label"] += 1
        _label_add(top_category3, all_label["tag3.0"]["sub"])

    print(json.dumps(all_label, ensure_ascii=False, indent=4))
    return all_label

def label_dist(all_label):
    tag3 = all_label["tag3.0"]["top"]
    for k, v in tag3.items():
        if k != "one_label" and k != "multi_label":






def main():
    excel_file = "/work/data/9-25周周点检数据/all_spot_check.xlsx"
    outfile = "./all_spot_check.txt"
    # write_validation_set_to_file(excel_file, outfile)
    label_count(outfile)

if __name__ == "__main__":
    main()



from predict import Predictor
from config import Config

import numpy as np
import sys
import copy
def load_txt(str_name):
    ff = open('data/data_main/trec_'+str_name+'.txt', encoding='utf8')
    txt = ff.read().split('\n')
    txt = txt[0:len(txt) - 1]
    ff.close()
    return txt

def load_tag(dict_name,conf):
    f_tag=open(conf.data.dict_dir+'/label.dict',encoding='utf8')
    tag_dict=f_tag.readlines()
    tag_list=[]
    for tag in (tag_dict):
        tag_list.append(tag.split('\t')[0])
    return tag_list


def part_data_gen():
    part_data=[]
    part_data_idx=[]
    for i in range(len(probs)):
        pre_label=(probs[i][0]).tolist()
        tag_index=pre_label.index(max(pre_label))
        #print (tag_index,tag_dict_all)
        pre_tag_name=tag_dict_all[tag_index]
        if pre_tag_name in tag_dict_part:
            txt_split=txt[i].split('\t')
            title=txt_split[1]
            content=txt_split[2]
            part_data.append(pre_tag_name+'\t'+title+'\t'+content+'\t'+'custom_feature')
            part_data_idx.append(i)
    return part_data,part_data_idx
def replace_result(pre_probs,after_probs,after_idx):
    for i in range(len(after_idx)):
        init_list=np.zeros(len(tag_dict_all))
        for k in range(len(tag_dict_part)):
             idx=tag_dict_all.index(tag_dict_part[k])
             init_list[idx]=after_probs[i][0][k]
             
        pre_probs[after_idx[i]]=[init_list.tolist()]
    return pre_probs


def result_save(str_name):
    ff2 = open('result_predict/'+str_name+'pred_result_detail.txt', 'w+', encoding='utf8')
    ff3 = open('result_predict/'+str_name+'pred_result.txt','w+',encoding='utf8')
    ff4 = open('result_predict/'+str_name+'pred_confusion_mat.txt','w+',encoding='utf-8')
    count=0
    right_count=np.zeros(len(tag_dict_all))
    predict_count=np.zeros(len(tag_dict_all))
    confusion_mat=np.zeros((len(tag_dict_all),len(tag_dict_all)))
    standard_dict={}
    Max_num=1   
    for i in range(len(probs_final)):
        pre_label=(probs_final[i][0]).tolist()
        
        pre_label_sort=copy.deepcopy(pre_label)
        pre_label_sort.sort()
        txt_split = txt[i].split('\t')
        strlist = txt_split[1:]
        tag = txt_split[0]
        taglist=[]
        for i in range(Max_num):
            tag_index=pre_label.index(pre_label_sort[-i-1])
            taglist.append(tag_dict_all[tag_index])
        if tag  not in standard_dict:
            standard_dict.update({tag:1})
        else:
            standard_dict[tag]+=1
        predict_count[tag_index]+=1
        tag=tag.split('/')
        try:
            standard_tag_index=tag_dict_all.index(tag[0])
            tag=tag[0]
        except:
            try:
                standard_tag_index=tag_dict_all.index(tag[1])
                tag=tag[1]
            except:
                print (tag)
                continue
        confusion_mat[standard_tag_index][tag_index]+=1
        if taglist[0]==tag:
            count+=1
            right_count[tag_index]+=1
        ff2.write(tag+'\t'+str(taglist)+'\t'+str(strlist)+'\n')
      
    ff3.write('Micro_Precision:'+str(count/len(probs_final))+'\n')
    conf_title=' '+'\t'+'\t'.join(st for st in tag_dict_all)
    ff4.write(conf_title+'\n')
    for i in range(len(tag_dict_all)):
        ff4.write(tag_dict_all[i]+'\t'+'\t'.join(str(i) for i in list(confusion_mat[i][:]))+'\n')
    ff4.close()
    for i in range(len(tag_dict_all)):
        try:
            standard_inums=standard_dict[tag_dict_all[i]]
        except:
            standard_inums=0;
        s1=tag_dict_all[i]+'\t'
        s2='precision:'+str(right_count[i]/predict_count[i])+'\t'
        
        s3='recall:'+str(right_count[i]/standard_inums)+'\t'
        s4='right_count:'+str(right_count[i])+'\t'+'predict_count:'+str(predict_count[i])+'\t'+'standard_count:'+str(standard_inums)+'\n'
        ff3.write(s1+s2+s3+s4)
    ff2.close()
    ff3.close()



if __name__ == '__main__':
    config1 = Config(config_file=sys.argv[1])
    config2 = Config(config_file=sys.argv[2])
    target_list=['validate']
    for str_name in target_list:
        predictor = Predictor(config1)
        txt=load_txt(str_name)
    
        tag_dict_all=load_tag('dict_main',config1)
        tag_dict_part=load_tag('part_dict',config2)
        probs = (predictor.predict(txt))

        part_data,part_data_idx=part_data_gen()
        predictor = Predictor(config2)
        probs_part=predictor.predict(part_data)
        probs=replace_result(probs,probs_part,part_data_idx) 
        if len(sys.argv)>=4:
            config3 = Config(config_file=sys.argv[3])
            config3 = Config(config_file=sys.argv[3])
            tag_dict_part=load_tag('part_dict',config3)
            part_data,part_data_idx=part_data_gen()
            predictor = Predictor(config3) 
            probs_part=predictor.predict(part_data)
        if len(sys.argv)>=5:
            config4 = Config(config_file=sys.argv[4])
            config4 = Config(config_file=sys.argv[4])
            tag_dict_part=load_tag('part_dict',config4)
            part_data,part_data_idx=part_data_gen()
            predictor = Predictor(config4)
            probs_part=predictor.predict(part_data)

    # 替换对应值
        probs=replace_result(probs,probs_part,part_data_idx) 

        probs_final=probs
        result_save(str_name)
   

import numpy as np
import pickle
from tqdm import tqdm
from itertools import combinations
from config_llm import *
from openai import OpenAI
import time

def ask_gpt(client, question, conversation_history,response_format,max_retries=5):
    retry_count=0
    while retry_count < max_retries:
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=conversation_history + [{"role": "user", "content": question}],
                response_format=response_format,
                temperature=0,
                seed=2024,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(e)
            print(f"第{retry_count + 1}次执行失败，进行重试...")
            retry_count += 1
            time.sleep(1)





def vector_to_text(vector,event_dict):
    text=""
    id_event={v:k for k,v in event_dict.items()}
    for j in range(len(vector)):
        if vector[j] >0:
            text+=id_event[j]+" \n"
    return text

def evaluate(topk, sample_count,training=True):
    top_value = [0] * 10
    for top in topk:
        for i in range(top - 1, 10):
            top_value[i] += 1
    top_prob = np.array(top_value) / sample_count
    avg_5 = np.array(top_prob[:5]).mean()
    result = {"top1": top_prob[0], "top3": top_prob[2], "top5": top_prob[4], "avg5": avg_5}
    return result


def retrieve(vector, data_vector,event_weight=None, k=5):
    vectors_a = np.array(vector).reshape(1, -1)
    vectors_b = np.array(data_vector)
    if event_weight is not None:
        event_weight=np.array(list(event_weight.values()))
        # 计算点积
        dot_product =np.dot(vectors_a*event_weight,vectors_b.T)
        norm_a = np.linalg.norm(vectors_a * event_weight, axis=1, keepdims=True)
        norm_b = np.linalg.norm(vectors_b, axis=1, keepdims=True)
    else:
        dot_product = np.dot(vectors_a , vectors_b.T)
        norm_a = np.linalg.norm(vectors_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(vectors_b, axis=1, keepdims=True)


    # 计算余弦相似度矩阵
    similarity_matrix = dot_product / (norm_a * norm_b.T)

    # 对于每个向量，找出相似度最高的top_k个向量的索引
    indices = np.argsort(-similarity_matrix, axis=1)[:, :k]

    return similarity_matrix, indices


def assign_entity_vector(e, items,  up=False, down=False,cumulate=True):
    prefix = ""
    if up:
        prefix = "upstream:"
    elif down:
        prefix = "downstream:"

    if len(e[0]) > 0:
        for l, value in e[0].items():
            items.append(prefix + l )
    std_scope=[1e5, 1e4,1000, 100,10, 1]
    mean_scope=[1e5, 1e4, 1000,100, 10,1,0.5,0.15,0.1,0.05]
    max_scope=[1e5, 1e4, 1000,100, 10,1,0.5,0.15,0.1,0.05]
    min_scope=[1e5, 1e4, 1000,100, 10,1,0.5,0.15,0.1,0.05]
    if len(e[1]) > 0 or len(e[2]) > 0:
        for m, value in e[1].items():
            value = np.nan_to_num(value, nan=0)
            m=''.join([char for char in m if not char.isdigit()])
            if value[:5].max() > 0:
                max_increase = (value[5:].max() - value[:5].max()) / value[:5].max()
                max_decrease = (value[:5].max() - value[5:].max()) / value[:5].max()
            else:
                max_increase = 0
                max_decrease = 0
                for s in max_scope:
                    if value[5:].max() > s:

                            items.append(prefix + m + " max_increase {} from 0".format(s))

                            if cumulate==False:
                                break

            if value[:5].min() > 0:
                min_increase = (value[5:].min() - value[:5].min()) / value[:5].min()
                min_decrease = (value[:5].min() - value[5:].min()) / value[:5].min()
            else:
                min_increase = 0
                min_decrease = 0
                for s in min_scope:
                    if value[5:].min() > s:

                            items.append(prefix + m + " min_increase {} from 0".format(s))
                            if cumulate==False:
                                break

            if value[:5].std() > 0:
                std_increase = (value[5:].std() - value[:5].std()) / value[:5].std()
                std_decrease = (value[:5].std() - value[5:].std()) / value[:5].std()
            else:
                std_increase = 0
                std_decrease = 0
                for s in std_scope:
                    if value[5:].std() > s:

                            items.append(prefix + m + " std_increase {} from 0".format(s))

                            if cumulate==False:
                                break


            if value[:5].mean() > 0:
                mean_increase = (value[5:].mean() - value[:5].mean()) / value[:5].mean()
                mean_decrease = (value[:5].mean() - value[5:].mean()) / value[:5].mean()
            else:
                mean_increase = 0
                mean_decrease = 0

                for s in mean_scope:
                    if value[5:].mean() > s:

                            items.append(prefix + m + " mean_increase {} from 0".format(s))

                            if cumulate == False:
                                break

            if mean_increase > 0:
                for s in mean_scope:

                    if mean_increase > s:

                            items.append(prefix + m + " mean_increase_{}".format(s))

                            if cumulate == False:
                                break
            if mean_decrease > 0:
                for s in mean_scope:
                    if mean_decrease > s:

                            items.append(prefix + m + " mean_decrease_{}".format(s))

                            if cumulate == False:
                                break
            if std_increase > 0:
                for s in std_scope:
                    if std_increase > s:

                            items.append(prefix + m + " std_increase_{}".format(s))

                            if cumulate == False:
                                break

            if std_decrease > 0:
                for s in std_scope:
                    if std_decrease > s:

                            items.append(prefix + m + " std_decrease_{}".format(s))

                            if cumulate == False:
                                break


            if max_increase > 0:
                for s in max_scope:
                    if max_increase > s:

                            items.append(prefix + m + " max_increase_{}".format(s))
                            if cumulate == False:
                                break

            if max_decrease > 0:
                for s in max_scope:
                    if max_decrease > s:

                            items.append(prefix + m + " max_decrease_{}".format(s))

                            if cumulate == False:
                                break

            if min_increase > 0:
                for s in min_scope:
                    if min_increase > s:

                        items.append(prefix + m + " min_increase_{}".format(s))

                        if cumulate == False:
                            break

            if min_decrease > 0:
                for s in min_scope:
                    if min_decrease > s:

                        items.append(prefix + m + " min_decrease_{}".format(s))


                        if cumulate == False:
                            break

    return items


def index_train_data(train_data, args):

    pod_dict = eval(args.dataset)["pod_dict"]
    event_tf_idf = {}
    failure_types=[]
    data_vector=[]



    for k in train_data.keys():
        failure_events={}
        non_failure_events={}
        non_failure_count=0
        failure_count=0
        for v in train_data[k]:
            events = v["system_info"]
            failure_type = v["failure_type"]
            root = v["root"]
            for p , e in events.items():
                items = []
                items = assign_entity_vector(e, items)
                # new_items = [' and '.join(pair) for pair in combinations(items, 2)]
                # items += new_items
                items=list(set(items))
                if p==root:
                    for item in items:
                        if item not in failure_events:
                            failure_events[item]=1
                        else:
                            failure_events[item]+=1
                    failure_count+=1
                else:

                    for item in items:
                        if item not in non_failure_events:
                            non_failure_events[item]=1
                        else:
                            non_failure_events[item]+=1
                    non_failure_count+=1
        for other in tqdm(train_data.keys()):
            if other==k:
                continue
            else:
                for v in train_data[k]:
                    events = v["system_info"]
                    failure_type = v["failure_type"]
                    root = v["root"]
                    for p, e in events.items():

                        items = []
                        items = assign_entity_vector(e, items)
                        # new_items = [' and '.join(pair) for pair in combinations(items, 2)]
                        # items += new_items
                        items = list(set(items))
                        for item in items:
                            if item not in non_failure_events:
                                non_failure_events[item] = 1
                            else:
                                non_failure_events[item] += 1
                        non_failure_count+=1
        for e,v in failure_events.items():
            tf_idf=(v/failure_count)* np.log(non_failure_count/(non_failure_events[e]+1))
            if (v/failure_count) <args.alpha:
                tf_idf=0
            failure_events[e]=tf_idf
        event_tf_idf[k]=failure_events

    with open("./{}/preprocessed_file/event_idf_{}_{}_{}.pkl".format(args.dataset,args.entity,args.alpha,args.lamb), "wb") as f:
        pickle.dump((event_tf_idf), f)

    with open("./{}/preprocessed_file/event_idf_{}_{}_{}.pkl".format(args.dataset,args.entity,args.alpha,args.lamb), "rb") as f:
        event_tf_idf=pickle.load(f)
    for k, v in event_tf_idf.items():

        filtered_dict = dict(sorted(v.items(), key=lambda item: item[1], reverse=True)[:args.lamb])

        filtered_dict={key: v  for key, v in filtered_dict.items() if v > 0  }
        event_tf_idf[k]=filtered_dict



    count=0
    event_dict={}
    tf_idf={}
    event_weight={}

    for k,v in event_tf_idf.items():

        for e in v.keys():
            if e not in event_dict:
                event_dict[e]=count
                count+=1
                tf_idf[e]=v[e]

            else:
                event_dict.pop(e)
                tf_idf.pop(e)
    event_dict={ list(event_dict.keys())[j]:j for j in range(len(event_dict))}
    for k,v in event_dict.items():
        event_weight[k]=tf_idf[k]

    for k in train_data.keys():

        for v in train_data[k]:
            events = v["system_info"]
            failure_type = v["failure_type"]
            root = v["root"]
            e=events[root]
            items = []
            items = assign_entity_vector(e, items)
            # new_items = [' and '.join(pair) for pair in combinations(items, 2)]
            # items += new_items
            pod_vec=np.zeros(len(event_dict))
            for item in items:
                if item in event_dict.keys():
                    # if item in event_tf_idf[failure_type].keys():
                    index=event_dict[item]
                    pod_vec[index]=1

            data_vector.append(pod_vec)
            failure_types.append(failure_type)

    with open("./{}/preprocessed_file/tf_idf_data_base_{}_{}_{}.pkl".format(args.dataset,args.entity,args.alpha,args.lamb), "wb") as f:
        pickle.dump((data_vector,event_weight,event_tf_idf,failure_types,event_dict),f)



    return data_vector,event_weight,event_tf_idf,failure_types,event_dict

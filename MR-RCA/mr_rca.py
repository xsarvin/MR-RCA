import os
import pickle

import numpy as np
from pydantic import BaseModel
from prompt import *
import time
from utils import *
from sklearn.metrics import precision_score, recall_score, f1_score
from openai import OpenAI

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from collections import Counter

# 'https://api.openai.com/v1'


failure_cls_client = OpenAI(api_key="", base_url="")
summary_generate_client = OpenAI(api_key="", base_url="")
fault_cls_client = OpenAI(api_key="", base_url="")


path = list[str]



class failure_cls(BaseModel):
    failure_type: str
    chain_of_thought: str


class summary_generate(BaseModel):
    summary: str


class decision(BaseModel):
    fault_type: str
    reason: str




def inference(data, test_index, data_vector, event_weight, event_tf_idf, failure_types, event_dict, args):
    type_acc = 0
    top_k = []
    sample_count = 0
    predict_type = []
    actual_type = []
    entity_cls_time = 0
    entity_identify_time = 0
    root_cls_time = 0
    for k in tqdm(test_index):
        test_sample = data[k]
        if args.entity == "node":
            adj = "None"
        else:
            adj = test_sample["entity_relation"]
        events = test_sample["system_info"]
        failure_type = test_sample["failure_type"]
        root = test_sample["root"]
        entity_diagnose = {}
        entity_text = {}
        event_num = {}


        ################### entity failure identification ##############################
        for r in list(events.keys()):
            entity_identify_st = time.time()
            if r not in events.keys():
                continue
            e = events[r]
            items = []
            items = assign_entity_vector(e, items)
            # new_items = [' and '.join(pair) for pair in combinations(items, 2)]
            # items += new_items
            Flag = False
            for f_type in event_tf_idf.keys():
                type_count = 0
                for item in items:
                    if item in event_tf_idf[f_type].keys():
                        type_count += 1
                if type_count > len(event_tf_idf[f_type]) * args.beta:
                    Flag = True
                    break
            entity_identify_time+=time.time()-entity_identify_st


            ################### entity failure classification##############################
            entity_cls_st=time.time()
            entity_vector = np.zeros(len(event_dict))
            for item in items:
                if item in event_dict.keys():
                    index = event_dict[item]
                    entity_vector[index] += 1
            text = vector_to_text(entity_vector, event_dict)
            entity_text[r] = text
            event_num[r] = entity_vector.sum()
            if not Flag:
                entity_diagnose[r] =  "non_failure"
            else:
                similarity, indices = retrieve(entity_vector, np.array(data_vector), event_weight, args.k)
                similarity = np.nan_to_num(similarity, 0)

                sim_types = [failure_types[index] for index in indices[0]]
                sim_text = [vector_to_text(data_vector[index], event_dict) for index in indices[0]]
                sim = [similarity[0][index] for index in indices[0]]
                context_prompt = ""

                for j in range(len(sim_text)):
                    context_prompt += "input:{}".format(sim_text[j])
                    context_prompt += "failure_type:{} similarity: {}\n\n".format(sim_types[j], sim[j])
                context_prompt += "input:{}".format(text) + "failure_type:?"

                answers = []
                history_context = [
                    {"role": "system", "content": root_classification["failure_cls_{}".format(args.entity)]}]
                for _ in range(args.consistency_num):
                    event = ask_gpt(client=failure_cls_client, question=context_prompt, conversation_history=history_context,
                                    response_format=failure_cls)
                    answers.append(event.failure_type)
                    history_context.append(
                        {"role": "assistant", "content": event.failure_type + " " + event.chain_of_thought})

                most_consistent_answer = Counter(answers).most_common(1)[0][0]
                entity_diagnose[r] = most_consistent_answer
                print("{} : diagnose result is {}".format(r, most_consistent_answer))
                entity_text[r] = text

            entity_cls_time += time.time() - entity_cls_st

        possible_root = []

        system_info = ""

        for p, v in entity_diagnose.items():

            if v != "non_failure":
                possible_root.append(p)
            system_info += "the initial diagnose result of {} is {},and the fault related event number of {} is {}\n".format(p, entity_diagnose[p],p,event_num[p])


        ######################### fault classification ################################
        root_cls_st=time.time()

        if args.entity=="node":
            propagation_context = "Entity Failure Propagation: no information.\n"

        else:
            if args.entity == "pod":
                entity_dict = eval(args.dataset)["pod_dict"]
            elif args.entity == "service":
                entity_dict = eval(args.dataset)["service_dict"]
            id_entity = {v: k for k, v in entity_dict.items()}
            root_cls_st = time.time()
            anomaly_entities = possible_root
            anomaly_index = [entity_dict[e] for e in anomaly_entities]
            propagation_context = "Entity Failure Propagation:"
            for j in anomaly_index:
                up_stream = np.where(adj[:, j])[0]
                for up_index in up_stream:
                    if up_index in anomaly_index:
                        call_e = id_entity[up_index]
                        callee = id_entity[j]
                        propagation_context += "{} possibly propagate anomaly to {}.\n".format(callee, call_e)
            if propagation_context == "Entity Failure Propagation:":
                propagation_context += "no information.\n"

        root_classify_prompt = system_info +  propagation_context+ "\n"

        answers = []
        history_context = [
            {"role": "system", "content": root_classification["fault_cls_{}".format(args.entity)]}]
        for _ in range(args.consistency_num):  # collect multiple answers
            event = ask_gpt(client=fault_cls_client, question=root_classify_prompt, conversation_history=history_context,
                            response_format=decision)
            answers.append(event.fault_type)
            history_context.append({"role": "assistant", "content": event.fault_type + event.reason})

        # 选择最一致的答案（这里简单地选择出现次数最多的答案）

        most_consistent_answer = Counter(answers).most_common(1)[0][0]
        predict_root_type=most_consistent_answer
        root_cls_time += time.time() - root_cls_st

        actual_type.append(failure_type)
        predict_type.append(predict_root_type)

        sample_count += 1

    precision = precision_score(actual_type, predict_type, average="weighted")
    recall = recall_score(actual_type, predict_type, average="weighted")
    f1 = f1_score(actual_type, predict_type, average="weighted")


    with open("./{}/result/fault_record_summary.txt".format(args.dataset), "a") as f:
        f.write("alpha:{}_beta:{}_lamb:{}_k:{}_entity:{} the precision is {}, recall is {}, f-score is {}\n".format(args.alpha,args.beta, args.lamb,args.k,args.entity,precision, recall, f1))



def root_cause(args):
    with open("./{}/fault_data_llm_{}.pkl".format(args.dataset, args.entity), "rb") as f:
        data = pickle.load(f)
    index = list(data.keys())
    train_data = {}
    train_index = []
    for k, v in data.items():
        if v["failure_type"] not in train_data.keys():
            train_data[v["failure_type"]] = []
        if len(train_data[v["failure_type"]]) < 5:
            train_data[v["failure_type"]].append(v)
            train_index.append(k)
    test_index = list(set(index) - (set(train_index)))


    if os.path.exists("./{}/preprocessed_file/tf_idf_data_base_{}_{}_{}.pkl".format(args.dataset, args.entity, args.alpha,
                                                                    args.lamb)):
        with open("./{}/preprocessed_file/tf_idf_data_base_{}_{}_{}.pkl".format(args.dataset, args.entity, args.alpha,
                                                                  args.lamb), "rb") as f:
            data_vector, event_weight, event_tf_idf, failure_types, event_dict = pickle.load(f)
    else:
        data_vector, event_weight, event_tf_idf, failure_types, event_dict = index_train_data(train_data, args)

    inference(data, test_index, data_vector, event_weight, event_tf_idf, failure_types, event_dict, args)

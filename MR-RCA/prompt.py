root_classification = {

    "failure_cls_pod": "As an experienced operations person, Now give you an entity information and similar examples "
                     "(including similar history entity information, corresponding failure type and similarity degree compared to input entity information.) "
                     "please refer to the similar entity examples and input entity information,  "
                     "output failure type of input entity sample which is selected from k8s容器网络资源包重复发送', 'k8s容器网络丢包', 'k8s容器cpu负载', 'k8s容器读io负载', 'k8s容器网络延迟', 'k8s容器写io负载', 'k8s容器网络资源包损坏', 'k8s容器内存负载', 'k8s容器进程中止', 'non_failure'. think step by step and give the chain of thought. \n",

    "failure_cls_service": "As an experienced operations person, Now give you an entity information and similar examples "
                     "(including similar history entity information, corresponding failure type and similarity degree compared to input entity information.) "
                     "please refer to the similar entity examples and input entity information,  "
                     "output failure type of input entity sample which is selected from k8s容器网络资源包重复发送', 'k8s容器网络丢包', 'k8s容器cpu负载', 'k8s容器读io负载', 'k8s容器网络延迟', 'k8s容器写io负载', 'k8s容器网络资源包损坏', 'k8s容器内存负载', 'k8s容器进程中止', 'non_failure'. think step by step and give the chain of thought. \n",

    "failure_cls_node": "As an experienced operations person, Now give you an entity information and similar examples "
                        "(including similar history entity information, corresponding failure type and similarity degree compared to input entity information.) "
                        "please refer to the similar entity examples and input entity information,  "
                        "output failure type of input entity sample which is selected from 'node节点CPU故障', 'node 磁盘读IO消耗', 'node 内存消耗', 'node 磁盘空间消耗', 'node 磁盘写IO消耗', 'node节点CPU爬升', 'no fault'. think step by step and give the chain of thought. \n",




 # "fault_cls_pod": "You are an experienced operator, give you the initial analysis results of each entity of microservice,"
 #                     "fault related event number, and possible failure propagation context, The possible failure propagation "
 #                     "context may not be entirely accurate and further judgment is needed . you need to give one most possible system root"
 #                "cause fault type comprehensively which is selected from which is selected from  k8s容器网络资源包重复发送', 'k8s容器网络丢包', 'k8s容器cpu负载', 'k8s容器读io负载', 'k8s容器网络延迟', 'k8s容器写io负载', 'k8s容器网络资源包损坏', 'k8s容器内存负载', 'k8s容器进程中止'. think step by step and give the reasons  \n",
 #
 #
 #
 #    "fault_cls_service": "You are an experienced operator, give you the initial analysis results of each entity of microservice,"
 #                     "fault related event number, and possible failure propagation context, The possible failure propagation "
 #                     "context may not be entirely accurate and further judgment is needed . you need to give one most possible system root"
 #                "cause fault type comprehensively which is selected from which is selected from  k8s容器网络资源包重复发送', 'k8s容器网络丢包', 'k8s容器cpu负载', 'k8s容器读io负载', 'k8s容器网络延迟', 'k8s容器写io负载', 'k8s容器网络资源包损坏', 'k8s容器内存负载', 'k8s容器进程中止'. think step by step and give the reasons  \n",
 #
 #
 #
 #    "fault_cls_node": "You are an experienced operator, give you the initial analysis results of each entity of microservice,"
 #                     "fault related event number, and possible failure propagation context, The possible failure propagation "
 #                     "context may not be entirely accurate and further judgment is needed .  you need to give one most possible system root"
 #                "cause fault type comprehensively which is selected from which is selected from 'node节点CPU故障', 'node 磁盘读IO消耗', 'node 内存消耗', 'node 磁盘空间消耗', 'node 磁盘写IO消耗', 'node节点CPU爬升'. think step by step and give the reasons  \n"

    "fault_cls_pod": "You are an experienced operator, give you the initial analysis results of each entity of microservice,"
                     "fault related event number, and possible failure propagation context, The possible failure propagation "
                     "context may not be entirely accurate and further judgment is needed .  The bigger fault related event "
                     "number, the more likely to be the root cause. you need to give one most possible system root"
                "cause fault type comprehensively which is selected from which is selected from  k8s容器网络资源包重复发送', 'k8s容器网络丢包', 'k8s容器cpu负载', 'k8s容器读io负载', 'k8s容器网络延迟', 'k8s容器写io负载', 'k8s容器网络资源包损坏', 'k8s容器内存负载', 'k8s容器进程中止'. think step by step and give the reasons  \n",



    "fault_cls_service": "You are an experienced operator, give you the initial analysis results of each entity of microservice,"
                     "fault related event number, and possible failure propagation context, The possible failure propagation "
                     "context may not be entirely accurate and further judgment is needed .  The bigger fault related event "
                     "number, the more likely to be the root cause. you need to give one most possible system root"
                "cause fault type comprehensively which is selected from which is selected from  k8s容器网络资源包重复发送', 'k8s容器网络丢包', 'k8s容器cpu负载', 'k8s容器读io负载', 'k8s容器网络延迟', 'k8s容器写io负载', 'k8s容器网络资源包损坏', 'k8s容器内存负载', 'k8s容器进程中止'. think step by step and give the reasons  \n",



    "fault_cls_node": "You are an experienced operator, give you the initial analysis results of each entity of microservice,"
                     "fault related event number, and possible failure propagation context, The possible failure propagation "
                     "context may not be entirely accurate and further judgment is needed .  The bigger fault related event "
                     "number, the more likely to be the root cause. you need to give one most possible system root"
                "cause fault type comprehensively which is selected from which is selected from 'node节点CPU故障', 'node 磁盘读IO消耗', 'node 内存消耗', 'node 磁盘空间消耗', 'node 磁盘写IO消耗', 'node节点CPU爬升'. think step by step and give the reasons  \n"
}


rag_ablation={
    "summary_generation":"As a experienced operator, give you the failure data of the resource entity, "
                         "please generate the failure summary precisely and no more than 50 words",

    "summary_diagnose_node":"As a experienced operator, give you the failure summary and some history samples, give one most possible failure type of the entity which is selected from 'node节点CPU故障', 'node 磁盘读IO消耗', 'node 内存消耗', 'node 磁盘空间消耗', 'node 磁盘写IO消耗', 'node节点CPU爬升'. think step by step and give the reasons",

    "summary_diagnose_pod": "As a experienced operator, give you the failure summary and some history samples, give one most possible failure type of the entity comprehensively which is selected from which is selected from  k8s容器网络资源包重复发送', 'k8s容器网络丢包', 'k8s容器cpu负载', 'k8s容器读io负载', 'k8s容器网络延迟', 'k8s容器写io负载', 'k8s容器网络资源包损坏', 'k8s容器内存负载', 'k8s容器进程中止'. think step by step and give the reasons",
    "summary_diagnose_service": "As a experienced operator, give you the failure summary and some history samples, give one most possible failure type of the entity comprehensively which is selected from which is selected from  k8s容器网络资源包重复发送', 'k8s容器网络丢包', 'k8s容器cpu负载', 'k8s容器读io负载', 'k8s容器网络延迟', 'k8s容器写io负载', 'k8s容器网络资源包损坏', 'k8s容器内存负载', 'k8s容器进程中止'. think step by step and give the reasons",
}

whole_system_ablation={
    "whole_diagnose_node":"As a experienced operator, give you the failure events in the system and some example samples which including the failure events and corresponding failure type , give the failure type of microservice system which is selected from 'node节点CPU故障', 'node 磁盘读IO消耗', 'node 内存消耗', 'node 磁盘空间消耗', 'node 磁盘写IO消耗', 'node节点CPU爬升'. think step by step and give the reasons. Output_format is: failure_type:XXX.reason:XXX. no other information.",
    "whole_diagnose_pod":"As a experienced operator, give you the failure events in the system and some  example samples which including the failure events and corresponding failure type, give the failure type of  microservice system comprehensively which is selected from which is selected from  k8s容器网络资源包重复发送', 'k8s容器网络丢包', 'k8s容器cpu负载', 'k8s容器读io负载', 'k8s容器网络延迟', 'k8s容器写io负载', 'k8s容器网络资源包损坏', 'k8s容器内存负载', 'k8s容器进程中止'.  Output_format is: failure_type:XXX.reason:XXX. no other information.",
    "whole_diagnose_service":"As a experienced operator, give you the failure events in the system and some  example samples which including the failure events and corresponding failure type, give the failure type of microservice system comprehensively which is selected from which is selected from  k8s容器网络资源包重复发送', 'k8s容器网络丢包', 'k8s容器cpu负载', 'k8s容器读io负载', 'k8s容器网络延迟', 'k8s容器写io负载', 'k8s容器网络资源包损坏', 'k8s容器内存负载', 'k8s容器进程中止'.  Output_format is: failure_type:XXX.reason:XXX no other information.",
}
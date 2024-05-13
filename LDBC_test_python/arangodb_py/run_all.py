# -*- coding: utf-8 -*-
from arango import ArangoClient
from arango.http import DefaultHTTPClient
import time
import subprocess
import os
import os.path as osp
import sys
import datetime

config_file_path = "./initial.config"   # �����ļ�·�� 
# ����ȫ�ֱ���
title_v = ""    # �������ͷ�ļ�·��
title_e_weighted = ""   # ��Ȩ�߱���ͷ�ļ�·��
title_e_unweighted = "" # ��Ȩ�߱���ͷ�ļ�·��
title_e_unweighted_reverse = ""
title_e_weighted_reverse = ""

arangodb_url = ""
arangodb_db = ""
arangodb_username = ""
arangodb_password = ""

DATA_SETS = []
DATA_HOME = ""
db = None

is_directed = {}
is_weighted = {}

def initialize():
    global title_v, title_e_weighted, title_e_unweighted, title_e_unweighted_reverse, title_e_weighted_reverse, arangodb_url, arangodb_db, arangodb_username, arangodb_password, DATA_SETS, DATA_HOME,db
    # ��ȡ�����ļ�
    with open(config_file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    # ����������Ϣ
    config = {}
    for line in lines:
        if line.strip() and not line.strip().startswith("#"):  # �ǿ����Ҳ���ע����
            key, value = line.strip().split("=")
            config[key.strip()] = value.strip().strip('"')
    # ��ȡ·����Ϣ
    test_graph_list_path = config.get("test_graph_list_path")

    GRAPH_DATA_HOME = config.get("GRAPH_DATA_HOME")

    # ��ȡ����ͼ�б��ļ�
    with open(test_graph_list_path, 'r') as file:
        test_graph_list = file.readlines()
    test_graph_list = [graph.strip() for graph in test_graph_list]
    # ��ȡ����ͷ�ļ���·�����������ݵ���
    title_v = config.get("title_v_path")
    title_e_unweighted = config.get("title_e_unweighted_path")
    title_e_weighted = config.get("title_e_weighted_path")
    title_e_unweighted_reverse = config.get("title_e_unweighted_reverse_path")
    title_e_weighted_reverse = config.get("title_e_weighted_reverse_path")

    arangodb_url = config.get("arangodb_url")
    arangodb_db = config.get("arangodb_db")
    arangodb_username = config.get("arangodb_username")
    arangodb_password = config.get("arangodb_password")
    DATA_HOME = GRAPH_DATA_HOME
    DATA_SETS = test_graph_list

    # Ҫȷ������Ȩ������Ȩͼ����������дimportʱ�����headerҲ��һ��
    for NOW_GRAPH in DATA_SETS:
        is_weighted[NOW_GRAPH] = False # Ĭ������Ȩͼ��ֻ�������ļ��д���weight�����Ż�ת����True
        file_name = NOW_GRAPH + '.properties'
        file_path = osp.join(DATA_HOME, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            if line.strip() and not line.strip().startswith("#"):  # �ǿ����Ҳ���ע����
                key, value = line.strip().split("=")
                li = key.strip().split(".")
                if(len(li) >= 3 and li[2] == 'directed'):
                    if(value.strip() == 'true'):
                        is_directed[NOW_GRAPH] = True
                    else:
                        is_directed[NOW_GRAPH] = False
                if(len(li) >= 3 and li[2] == 'edge-properties' and len(li) >= 4 and li[3] == 'names'):
                    if(value.strip() == 'weight'):
                        is_weighted[NOW_GRAPH] = True
                    # Ĭ�ϲ��ı䣬��ΪĬ����False
    # ֹͣArangoDB����
    print(f"Stop ArangoDB service...", flush=True)
    subprocess.run(['sudo', 'systemctl', 'stop', 'arangodb3'])                
    # ����ArangoDB����
    print(f"Start ArangoDB service...", flush=True)
    subprocess.run(['sudo', 'systemctl', 'start', 'arangodb3'])
    client = ArangoClient(hosts=arangodb_url,http_client=DefaultHTTPClient(request_timeout=1000))
    try:
        # �������ӵ����ݿ�
        db = client.db('rucgraph_new', username='root', password='root')
        print("successfully connected to database")
    except ArangoError as e:
        print(f"Failed to connect to database: {e}")

def import_load():
    # ѡ�����ݼ�
    for NOW_GRAPH in DATA_SETS:
        DATA_e = NOW_GRAPH + '.e'
        DATA_v = NOW_GRAPH + '.v'
        DATA_v_path = osp.join(DATA_HOME, DATA_v)
        DATA_e_path = osp.join(DATA_HOME, DATA_e)
        # �������ƣ����ڵ������ݣ���ֹ��ͻ������ÿһ��Ҫȡ��ͬ����
        v_name = NOW_GRAPH + '_v'
        e_name = NOW_GRAPH + '_e'
            
        # ��ӡ��ʾ��Ϣ
        print(f"Import data: {NOW_GRAPH}...")
        # ����Ѵ��ڵļ��ϣ���������Ϊ�˱����ظ���������
        is_exist_collection_v = db.has_collection(v_name)
        if(not is_exist_collection_v):
            subprocess.run([
            'arangoimport',
            '--file', DATA_v_path,
            '--type', 'tsv',
            '--headers-file', title_v,
            '--server.database', arangodb_db,
            '--collection', v_name,
            '--create-collection', 'true',
            '--create-collection-type', 'document',
            '--server.password', arangodb_password
            ])
        else:
            print(f"Collection {v_name} already exists.There is no need to import it again.")
        is_exist_collection_e = db.has_collection(e_name)
        if(not is_exist_collection_e):
            if(not is_directed[NOW_GRAPH]):
                if(not is_weighted[NOW_GRAPH]):
                    subprocess.run([
                    'arangoimport',
                    '--file', DATA_e_path,
                    '--type', 'tsv',
                    '--headers-file', title_e_unweighted,
                    '--server.database', arangodb_db ,
                    '--collection', e_name,
                    '--create-collection', 'true',
                    '--create-collection-type', 'edge',
                    '--from-collection-prefix', v_name,
                    '--to-collection-prefix', v_name,
                    '--separator', ' ',
                    '--server.password', arangodb_password
                    ])
                else:
                    # ע�����ж���һ���weightת��Ϊ���֣�������sssp�㷨ִ�е�ʱ������������
                    # ���Ҹ�����ͷ�ļ�
                    subprocess.run([
                    'arangoimport',
                    '--file', DATA_e_path,
                    '--type', 'tsv',
                    '--headers-file', title_e_weighted,
                    '--datatype', 'weight=number',
                    '--server.database', arangodb_db ,
                    '--collection', e_name,
                    '--create-collection', 'true',
                    '--create-collection-type', 'edge',
                    '--from-collection-prefix', v_name,
                    '--to-collection-prefix', v_name,
                    '--separator', ' ',
                    '--server.password', arangodb_password
                    ])
            if(is_directed[NOW_GRAPH]): # ���������ͼ������Ҫִ��һ�η���ĵ���
                if(is_weighted[NOW_GRAPH]):
                    subprocess.run([
                    'arangoimport',
                    '--file', DATA_e_path,
                    '--type', 'tsv',
                    '--headers-file', title_e_weighted_reverse,
                    '--datatype', 'weight=number',
                    '--server.database', arangodb_db ,
                    '--collection', e_name,
                    '--create-collection', 'true',
                    '--create-collection-type', 'edge',
                    '--from-collection-prefix', v_name,
                    '--to-collection-prefix', v_name,
                    '--separator', ' ',
                    '--server.password', arangodb_password
                    ])
                else:
                    subprocess.run([
                    'arangoimport',
                    '--file', DATA_e_path,
                    '--type', 'tsv',
                    '--headers-file', title_e_unweighted_reverse,
                    '--server.database', arangodb_db ,
                    '--collection', e_name,
                    '--create-collection', 'true',
                    '--create-collection-type', 'edge',
                    '--from-collection-prefix', v_name,
                    '--to-collection-prefix', v_name,
                    '--separator', ' ',
                    '--server.password', arangodb_password
                    ])

        else:
            print(f"Collection {e_name} already exists.There is no need to import it again.")
        # ����ͼ
            print(f"Create graph... {NOW_GRAPH}")
        # ����Ѵ��ڵ�ͼ����������Ϊ�˱����ظ�����ͼ
        is_exist_graph = db.has_graph(NOW_GRAPH)
        if(not is_exist_graph):
            start_time = time.time()
            graph = db.create_graph(NOW_GRAPH)
            graph.create_edge_definition(
                edge_collection=e_name,
                from_vertex_collections=[v_name],
                to_vertex_collections=[v_name]
            )
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Create graph time: {total_time} seconds")
        else:
            print(f"Graph {NOW_GRAPH} already exists.There is no need to create it again.")
def check_over(algorithm,start_time, job_id):
    # ������ѯ���ʱ�䣨��ֹ��ѯռ�ô�����Դ������������
    interval = 0.1
    run_start_time = 0
    oldstatus = 'start'
    # ��ѯ�����������ҵ״̬�����仯����ӡ״̬��������loading״̬ת��Ϊrunning״̬��ʱ��ʼ��ʱ
    while True:
        status = db.pregel.job(job_id)
        if status['state'] in ['done', 'canceled', 'failed']:
            break
        if oldstatus == 'loading' and status['state'] == 'running':
            run_start_time = time.time()
        if oldstatus != status['state']:
            print(f"{algorithm} job state: {oldstatus} -> {status['state']}")
            print(f"{algorithm} job state: {status['state']}...")
        oldstatus = status['state']
        time.sleep(interval)
        end_time = time.time()
    loading_time = run_start_time - start_time
    running_time = end_time - run_start_time

    return loading_time, running_time

def parse_single_config(NOW_GRAPH):
    file_name = NOW_GRAPH + '.properties'
    file_path = osp.join(DATA_HOME, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    # ����������Ϣ
    single_property_dict = {}
    for line in lines:
        if line.strip() and not line.strip().startswith("#"):  # �ǿ����Ҳ���ע����
            key, value = line.strip().split("=")
            li = key.strip().split(".")
            if len(li) >= 3 and li[2] == "algorithms":
                single_property_dict["algorithms"] = value.strip() # ֮��Ҫת��Ϊ�б����ʽ��������
            if len(li) >= 3 and li[2] == "bfs":
                if len(li) >= 4 and li[3] == "source-vertex":
                    single_property_dict["bfs_source_vertex"] = value.strip()
            if len(li) >= 3 and li[2] == "cdlp":
                if len(li) >= 4 and li[3] == "max-iterations":
                    single_property_dict["cdlp_max_iterations"] = value.strip()
            if len(li) >= 3 and li[2] == "pr":
                if len(li) >= 4 and li[3] == "damping-factor":
                    single_property_dict["pr_damping_factor"] = value.strip()
                if len(li) >= 4 and li[3] == "num-iterations":
                    single_property_dict["pr_num_iterations"] = value.strip()
            if len(li) >= 3 and li[2] == "sssp":
                if len(li) >= 4 and li[3] == "weight-property":
                    single_property_dict["sssp_weight_property"] = value.strip()
                if len(li) >= 4 and li[3] == "source-vertex":
                    single_property_dict["sssp_source_vertex"] = value.strip()
            # wcc Ӧ�ö�û�в���
            # ���ò�lcc�������ڵ�ľ���ָ����
    algorithms_list = single_property_dict["algorithms"].split(",")
    return single_property_dict, algorithms_list

def run_test():
    # ����DATA_HOME��·��������DATA_SETS�����ݣ���ȡÿ�����ݼ��������ļ�
    for NOW_GRAPH in DATA_SETS:
        v_name = NOW_GRAPH + '_v'
        print(f"================== Start to run the test on: {NOW_GRAPH}... ======================")
        # �����������ݼ��������ļ�
        single_property_dict, algorithms_list = parse_single_config(NOW_GRAPH)
        for algorithm in algorithms_list:
            algorithm = algorithm.strip()
            if algorithm == "bfs":
                # ���� AQL ��ѯ��䣬ֱ��û�п��Է��ʵĽڵ㣬��������������Ϊһ���ܴ��ֵ
                # ������ܴ���bug������ʱ������
                # ���������store���̣����Ҫ����ʱ�䣬������Ҫת��Ϊaql��return
                max_depth = 1000000
                aql_query = f"""
                    FOR vertex,edge,path IN 0..{max_depth} OUTBOUND '{v_name}/{single_property_dict["bfs_source_vertex"]}' GRAPH '{NOW_GRAPH}'
                        OPTIONS {{order: "bfs", uniqueVertices: 'global'}}
                        RETURN {{"vertex": vertex._key, "depth": LENGTH(path.edges)}}
                """
                # ִ�� BFS �� AQL ��ѯ
                print(f"Run BFS algorithm...")
                start_time = time.time()
                cursor = db.aql.execute(aql_query)
                end_time = time.time()
                total_time = end_time - start_time
                # �����������������ݻ�ȡaql��ѯ���ķ���ֵ���Ӷ����Խ�����ȷ��У��
                # ans = [doc for doc in cursor]
                # print(ans)
                # ��ӡBFSʱ��
                print(f"BFS time: {total_time} seconds")
            if algorithm == "cdlp":
                print(f"Run CDLP algorithm...")
                # ִ��CDLP�㷨,��������ָ���߳���Ŀ�ģ���������߳���ĿӦ����python�ű�ִ�е�ʱ���ͬ���첽�����㷨ִ���޹�(��python arrango��github��Դ��)
                start_time = time.time()
                job_id = db.pregel.create_job(
                    graph=NOW_GRAPH,
                    algorithm="labelpropagation",
                    store=False,
                    max_gss= int(single_property_dict["cdlp_max_iterations"]),
                    thread_count = 1,
                    async_mode=False,
                    result_field="community"
                )
                loading_time, running_time = check_over(algorithm=algorithm, start_time=start_time, job_id=job_id)
                # �ֱ��ӡ����ʱ�������ʱ��
                print(f"{algorithm} loading time: {loading_time} seconds")
                print(f"{algorithm} running time: {running_time} seconds")
            if algorithm == "pr":
                # ִ��pagerank�㷨��max_gss����num_iterations��Ȼ��threhold��ʹ�ã���Ҫȷ��
                # ����������ֹͣ�㷨��damping_factor�������ã���ΪĬ�Ͼ���0.85������������������������ݼ�����������
                start_time = time.time()
                job_id = db.pregel.create_job(
                    graph=NOW_GRAPH,
                    algorithm="pagerank",
                    store = False,
                    max_gss = int(single_property_dict["pr_num_iterations"]),
                    thread_count = 1,
                    async_mode = False,
                    result_field = "rank"
                )
                loading_time, running_time = check_over(algorithm=algorithm,start_time=start_time, job_id=job_id)
                # �ֱ��ӡ����ʱ�������ʱ��
                print(f"{algorithm} loading time: {loading_time} seconds")
                print(f"{algorithm} running time: {running_time} seconds")
            if algorithm == "sssp":
                # ����Ǻ���Ȩ�صģ�ֻ��ִ�����aql�е�sssp�㷨
                # ����ǲ�����Ȩ�صģ���ô������ִ��pregel�е�sssp�㷨
                if(is_weighted[NOW_GRAPH]):
                    # ִ�� sssp �� AQL ��ѯ
                    print(f"Run sssp algorithm...")
                    start_time = time.time()
                    aql_query = f"""
                    FOR target IN `{v_name}` FILTER target._key != '{single_property_dict["sssp_source_vertex"]}'
                        LET sum = (
                            FOR vetex, edge IN OUTBOUND SHORTEST_PATH '{v_name}/{single_property_dict["sssp_source_vertex"]}' TO target._key GRAPH '{NOW_GRAPH}'
                                OPTIONS {{weightAttribute: 'weight', defaultWeight: 1}}
                                COLLECT AGGREGATE total_weight = SUM(edge.weight)
                                RETURN total_weight
                        )
                        UPDATE target WITH {{total_weight: sum[0]}} IN `{v_name}`
                    """
                    cursor = db.aql.execute(aql_query)

                    end_time = time.time()
                    total_time = end_time - start_time
                    print(f"sssp time: {total_time} seconds")
                else:
                    # �������Ȩͼ��ִ��pregel�е�sssp�㷨
                    start_time = time.time()
                    job_id = db.pregel.create_job(
                        graph=NOW_GRAPH,
                        algorithm="sssp",
                        store=False,
                        algorithm_params={'source': v_name+"/"+single_property_dict["sssp_source_vertex"]},
                        result_field = "distance"
                    )
                    loading_time, running_time = check_over(algorithm=algorithm,start_time=start_time, job_id=job_id)
                    # �ֱ��ӡ����ʱ�������ʱ��
                    print(f"{algorithm} loading time: {loading_time} seconds")
                    print(f"{algorithm} running time: {running_time} seconds")
            if algorithm == "wcc":
                # ִ��WCC�㷨�����������ͼ����ô����ִ��wcc����(��֤��ȷ�ԣ������������ͼ������ִ��cc�������μ��ĵ���cc���ܶ�
                # ����cc�����������ͼ��������ͬ
                if(is_directed[NOW_GRAPH]):
                    start_time = time.time()
                    job_id = db.pregel.create_job(
                        graph=NOW_GRAPH,
                        algorithm="wcc",
                        store=False,
                        thread_count=1,
                        async_mode=False,
                        result_field="component_weak"
                    )
                else:
                    start_time = time.time()
                    job_id = db.pregel.create_job(
                        graph=NOW_GRAPH,
                        algorithm="connectedcomponents",
                        store=False,
                        thread_count=1,
                        async_mode=False,
                        result_field="component_weak"
                    )
                loading_time, running_time = check_over(algorithm=algorithm,start_time=start_time, job_id=job_id)
                # �ֱ��ӡ����ʱ�������ʱ��
                print(f"{algorithm} loading time: {loading_time} seconds")
                print(f"{algorithm} running time: {running_time} seconds")

if __name__ == "__main__":
    seed = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_file ="logs/" +seed + '.log'

    with open(log_file, 'a',buffering=1) as file:
        sys.stdout = file
        initialize()
        import_load()
        run_test()
        print("All tests are finished.")
        sys.stdout = sys.__stdout__
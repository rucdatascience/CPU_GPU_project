# -*- coding: utf-8 -*-
from arango import ArangoClient
from arango.http import DefaultHTTPClient
import time
import subprocess
import os
import os.path as osp
import sys
import datetime

config_file_path = "./initial.config"   # 配置文件路径 
# 声明全局变量
title_v = ""    # 顶点标题头文件路径
title_e_weighted = ""   # 有权边标题头文件路径
title_e_unweighted = "" # 无权边标题头文件路径
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
    # 读取配置文件
    with open(config_file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    # 解析配置信息
    config = {}
    for line in lines:
        if line.strip() and not line.strip().startswith("#"):  # 非空行且不是注释行
            key, value = line.strip().split("=")
            config[key.strip()] = value.strip().strip('"')
    # 提取路径信息
    test_graph_list_path = config.get("test_graph_list_path")

    GRAPH_DATA_HOME = config.get("GRAPH_DATA_HOME")

    # 读取测试图列表文件
    with open(test_graph_list_path, 'r') as file:
        test_graph_list = file.readlines()
    test_graph_list = [graph.strip() for graph in test_graph_list]
    # 获取标题头文件的路径，用于数据导入
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

    # 要确认是有权还是无权图，这样才能写import时的命令，header也不一样
    for NOW_GRAPH in DATA_SETS:
        is_weighted[NOW_GRAPH] = False # 默认是无权图，只有配置文件中存在weight字样才会转化成True
        file_name = NOW_GRAPH + '.properties'
        file_path = osp.join(DATA_HOME, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            if line.strip() and not line.strip().startswith("#"):  # 非空行且不是注释行
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
                    # 默认不改变，因为默认是False
    # 停止ArangoDB服务
    print(f"Stop ArangoDB service...", flush=True)
    subprocess.run(['sudo', 'systemctl', 'stop', 'arangodb3'])                
    # 启动ArangoDB服务
    print(f"Start ArangoDB service...", flush=True)
    subprocess.run(['sudo', 'systemctl', 'start', 'arangodb3'])
    client = ArangoClient(hosts=arangodb_url,http_client=DefaultHTTPClient(request_timeout=1000))
    try:
        # 尝试连接到数据库
        db = client.db('rucgraph_new', username='root', password='root')
        print("successfully connected to database")
    except ArangoError as e:
        print(f"Failed to connect to database: {e}")

def import_load():
    # 选择数据集
    for NOW_GRAPH in DATA_SETS:
        DATA_e = NOW_GRAPH + '.e'
        DATA_v = NOW_GRAPH + '.v'
        DATA_v_path = osp.join(DATA_HOME, DATA_v)
        DATA_e_path = osp.join(DATA_HOME, DATA_e)
        # 集合名称，用于导入数据，防止冲突，所以每一个要取不同名字
        v_name = NOW_GRAPH + '_v'
        e_name = NOW_GRAPH + '_e'
            
        # 打印提示信息
        print(f"Import data: {NOW_GRAPH}...")
        # 检查已存在的集合，这样做是为了避免重复导入数据
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
                    # 注意其中多了一项，将weight转化为数字，这样在sssp算法执行的时候才能正常相加
                    # 并且更换了头文件
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
            if(is_directed[NOW_GRAPH]): # 如果是有向图，则还需要执行一次反向的导入
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
        # 创建图
            print(f"Create graph... {NOW_GRAPH}")
        # 检查已存在的图，这样做是为了避免重复创建图
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
    # 设置轮询间隔时间（防止轮询占用大量资源）和其他参数
    interval = 0.1
    run_start_time = 0
    oldstatus = 'start'
    # 轮询，如果发现作业状态发生变化，打印状态，并且在loading状态转化为running状态的时候开始计时
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
    # 解析属性信息
    single_property_dict = {}
    for line in lines:
        if line.strip() and not line.strip().startswith("#"):  # 非空行且不是注释行
            key, value = line.strip().split("=")
            li = key.strip().split(".")
            if len(li) >= 3 and li[2] == "algorithms":
                single_property_dict["algorithms"] = value.strip() # 之后要转化为列表的形式，别忘了
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
            # wcc 应该都没有参数
            # 不用测lcc（单个节点的聚类指数）
    algorithms_list = single_property_dict["algorithms"].split(",")
    return single_property_dict, algorithms_list

def run_test():
    # 根据DATA_HOME的路径，还有DATA_SETS的内容，读取每个数据集的配置文件
    for NOW_GRAPH in DATA_SETS:
        v_name = NOW_GRAPH + '_v'
        print(f"================== Start to run the test on: {NOW_GRAPH}... ======================")
        # 解析单个数据集的配置文件
        single_property_dict, algorithms_list = parse_single_config(NOW_GRAPH)
        for algorithm in algorithms_list:
            algorithm = algorithm.strip()
            if algorithm == "bfs":
                # 定义 AQL 查询语句，直到没有可以访问的节点，所以设置最大深度为一个很大的值
                # 这里可能存在bug，导致时间增加
                # 这里添加了store过程，如果要减少时间，可能需要转化为aql的return
                max_depth = 1000000
                aql_query = f"""
                    FOR vertex,edge,path IN 0..{max_depth} OUTBOUND '{v_name}/{single_property_dict["bfs_source_vertex"]}' GRAPH '{NOW_GRAPH}'
                        OPTIONS {{order: "bfs", uniqueVertices: 'global'}}
                        RETURN {{"vertex": vertex._key, "depth": LENGTH(path.edges)}}
                """
                # 执行 BFS 的 AQL 查询
                print(f"Run BFS algorithm...")
                start_time = time.time()
                cursor = db.aql.execute(aql_query)
                end_time = time.time()
                total_time = end_time - start_time
                # 后续可以用以下内容获取aql查询语句的返回值，从而可以进行正确性校验
                # ans = [doc for doc in cursor]
                # print(ans)
                # 打印BFS时间
                print(f"BFS time: {total_time} seconds")
            if algorithm == "cdlp":
                print(f"Run CDLP algorithm...")
                # 执行CDLP算法,本来可以指定线程数目的，但是这个线程数目应该是python脚本执行的时候的同步异步而与算法执行无关(看python arrango在github的源码)
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
                # 分别打印加载时间和运行时间
                print(f"{algorithm} loading time: {loading_time} seconds")
                print(f"{algorithm} running time: {running_time} seconds")
            if algorithm == "pr":
                # 执行pagerank算法，max_gss就是num_iterations，然后threhold不使用，我要确切
                # 迭代次数后停止算法。damping_factor不用设置，因为默认就是0.85，后续如果遇到了其他的数据集，可以设置
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
                # 分别打印加载时间和运行时间
                print(f"{algorithm} loading time: {loading_time} seconds")
                print(f"{algorithm} running time: {running_time} seconds")
            if algorithm == "sssp":
                # 如果是含有权重的，只能执行这个aql中的sssp算法
                # 如果是不含有权重的，那么还可以执行pregel中的sssp算法
                if(is_weighted[NOW_GRAPH]):
                    # 执行 sssp 的 AQL 查询
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
                    # 如果是无权图，执行pregel中的sssp算法
                    start_time = time.time()
                    job_id = db.pregel.create_job(
                        graph=NOW_GRAPH,
                        algorithm="sssp",
                        store=False,
                        algorithm_params={'source': v_name+"/"+single_property_dict["sssp_source_vertex"]},
                        result_field = "distance"
                    )
                    loading_time, running_time = check_over(algorithm=algorithm,start_time=start_time, job_id=job_id)
                    # 分别打印加载时间和运行时间
                    print(f"{algorithm} loading time: {loading_time} seconds")
                    print(f"{algorithm} running time: {running_time} seconds")
            if algorithm == "wcc":
                # 执行WCC算法，如果是有向图，那么必须执行wcc函数(保证正确性），如果是无向图，可以执行cc函数，参见文档，cc会快很多
                # 但是cc如果用于有向图，会结果不同
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
                # 分别打印加载时间和运行时间
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
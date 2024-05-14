from neo4j import GraphDatabase
import time
import subprocess
import os
import os.path as osp
import datetime
import getpass
import sys

# 和neo4j有关的一些路径的配置
NEO4J_HOME = os.environ["NEO4J_HOME"]
IMPORT_DIR_PATH = osp.join(NEO4J_HOME, "import")
INIT_CONF_PATH = osp.join(NEO4J_HOME, "conf", "neo4j.conf.init")
CONF_PATH = osp.join(NEO4J_HOME, "conf", "neo4j.conf")
DATA_DIR_PATH = osp.join(NEO4J_HOME, "data", "databases")
TRANS_DIR_PATH = osp.join(NEO4J_HOME, "data", "transactions")

config_file_path = "./initial.config"   # 配置文件路径 
# 声明全局变量
title_v = ""    # 顶点标题头文件路径
title_e_weighted = ""   # 有权边标题头文件路径
title_e_unweighted = "" # 无权边标题头文件路径
title_e_unweighted_reverse = ""
title_e_weighted_reverse = ""

neo4j_url = ""
auth = ("neo4j", "987654321")
neo4j_username = ""
neo4j_password = ""

DATA_SETS = []
DATA_HOME = ""

is_directed = {}
is_weighted = {}

def initialize():
    global title_v, title_e_weighted, title_e_unweighted, title_e_unweighted_reverse, title_e_weighted_reverse, neo4j_url, neo4j_username, neo4j_password, DATA_SETS, DATA_HOME
    # 读取配置文件
    with open(config_file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    # 解析配置信息
    config = {}
    # log中打印用到的配置信息
    print("================== using the following configurations: ==================",flush=True)
    for line in lines:
        print(line)
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
    # log中打印测试的图文件
    print("================== testing the following graphs: ==================")
    for graph in test_graph_list:
        print(graph)
    # 获取标题头文件的路径，用于数据导入
    title_v = config.get("title_v_path")
    title_e_unweighted = config.get("title_e_unweighted_path")
    title_e_weighted = config.get("title_e_weighted_path")
    title_e_unweighted_reverse = config.get("title_e_unweighted_reverse_path")
    title_e_weighted_reverse = config.get("title_e_weighted_reverse_path")

    neo4j_url = config.get("neo4j_url")
    neo4j_username = config.get("neo4j_username")
    neo4j_password = config.get("neo4j_password")
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

def import_data():
    print("================== Start to import all data... ==================")
    for NOW_GRAPH in DATA_SETS:
        DATA_e = NOW_GRAPH + '.e'
        DATA_v = NOW_GRAPH + '.v'
        DATA_v_path = osp.join(DATA_HOME, DATA_v)
        DATA_e_path = osp.join(DATA_HOME, DATA_e)

        
        if os.path.exists(osp.join(DATA_DIR_PATH, "neo4j")):
            os.system("rm -rf %s" % (osp.join(DATA_DIR_PATH, "neo4j")))
        if os.path.exists(osp.join(TRANS_DIR_PATH, "neo4j")):
            os.system("rm -rf %s" % (osp.join(TRANS_DIR_PATH, "neo4j")))

        # 每个图的node 和edge名字不能相同，不然后续创建图的时候会有问题！
        node_label = "Node" + deal_name(NOW_GRAPH)
        edge_label = "Edge" + deal_name(NOW_GRAPH)
        if(is_weighted[NOW_GRAPH] and is_directed[NOW_GRAPH]):
            os.system(
                f"""
                neo4j-admin database import full --delimiter=' ' --nodes={node_label}={title_v},{DATA_v_path} --id-type=integer --relationships={edge_label}={title_e_weighted},{DATA_e_path} --relationships={edge_label}={title_e_weighted_reverse},{DATA_e_path}
                """
            )
        if(is_weighted[NOW_GRAPH] and not is_directed[NOW_GRAPH]):
            os.system(
                f"""
                neo4j-admin database import full --delimiter=' ' --nodes={node_label}={title_v},{DATA_v_path} --id-type=integer --relationships={edge_label}={title_e_weighted},{DATA_e_path} 
                """
            )
        if(not is_weighted[NOW_GRAPH] and is_directed[NOW_GRAPH]):
            os.system(
                f"""
                neo4j-admin database import full --delimiter=' ' --nodes={node_label}={title_v},{DATA_v_path} --id-type=integer --relationships={edge_label}={title_e_unweighted},{DATA_e_path} --relationships:{edge_label}={title_e_unweighted_reverse},{DATA_e_path}
                """
            )
        if(not is_weighted[NOW_GRAPH] and not is_directed[NOW_GRAPH]):
            os.system(
                f"""
                neo4j-admin database import full --delimiter=' ' --nodes={node_label}={title_v},{DATA_v_path} --id-type=integer --relationships={edge_label}={title_e_unweighted},{DATA_e_path}
                """
            )
    print("================== Data import is finished. ==================")

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


def deal_name(name):
    return name.replace("-", "").replace("_", "")


def run_test():
    for NOW_GRAPH in DATA_SETS:
        DATA_e = NOW_GRAPH + '.e'
        DATA_v = NOW_GRAPH + '.v'
        DATA_v_path = osp.join(DATA_HOME, DATA_v)
        DATA_e_path = osp.join(DATA_HOME, DATA_e)

        
        if os.path.exists(osp.join(DATA_DIR_PATH, "neo4j")):
            os.system("rm -rf %s" % (osp.join(DATA_DIR_PATH, "neo4j")))
        if os.path.exists(osp.join(TRANS_DIR_PATH, "neo4j")):
            os.system("rm -rf %s" % (osp.join(TRANS_DIR_PATH, "neo4j")))

        # 每个图的node 和edge名字不能相同，不然后续创建图的时候会有问题！
        if(is_weighted[NOW_GRAPH] and is_directed[NOW_GRAPH]):
            os.system(
                f"""
                neo4j-admin database import full --delimiter=' ' --nodes=Node={title_v},{DATA_v_path} --id-type=integer --relationships=Edge={title_e_weighted},{DATA_e_path} --relationships=Edge={title_e_weighted_reverse},{DATA_e_path}
                """
            )
        if(is_weighted[NOW_GRAPH] and not is_directed[NOW_GRAPH]):
            os.system(
                f"""
                neo4j-admin database import full --delimiter=' ' --nodes=Node={title_v},{DATA_v_path} --id-type=integer --relationships=Edge={title_e_weighted},{DATA_e_path} 
                """
            )
        if(not is_weighted[NOW_GRAPH] and is_directed[NOW_GRAPH]):
            os.system(
                f"""
                neo4j-admin database import full --delimiter=' ' --nodes=Node={title_v},{DATA_v_path} --id-type=integer --relationships=Edge={title_e_unweighted},{DATA_e_path} --relationships:Edge={title_e_unweighted_reverse},{DATA_e_path}
                """
            )
        if(not is_weighted[NOW_GRAPH] and not is_directed[NOW_GRAPH]):
            os.system(
                f"""
                neo4j-admin database import full --delimiter=' ' --nodes=Node={title_v},{DATA_v_path} --id-type=integer --relationships=Edge={title_e_unweighted},{DATA_e_path}
                """
            )
        print("================== Data import is finished. ==================")
        
        print(f"================== Start to run the test on: {NOW_GRAPH}... ==================")
        
        single_property_dict, algorithms_list = parse_single_config(NOW_GRAPH)

        os.system("neo4j-admin server restart")
        print("Waiting for neo4j to ready...")
        while True:
            try:
                with GraphDatabase.driver(neo4j_url, auth=auth) as driver:
                    driver.verify_connectivity()
                break
            except Exception as e:
                if str(e) == "Unable to retrieve routing information":
                    time.sleep(3)
                else:
                    raise e
        print("neo4j is ready.")
        
        with GraphDatabase.driver(neo4j_url, auth=auth, database="neo4j") as driver:
            with driver.session() as session:
                result = session.run(f"MATCH (n:Node) RETURN count(n)")
                nodes = result.single()[0]
                result = session.run(f"MATCH ()-[r]->() RETURN count(r)")
                edges = result.single()[0]
                print("neo4j: V={}, E={}".format(nodes, edges))
                if(is_weighted[NOW_GRAPH]):
                    session.run(
                            f"""
                        CALL gds.graph.project(
                            '{deal_name(NOW_GRAPH)}',
                            'Node',
                            'Edge',
                            {{
                                relationshipProperties: 'weight'
                            }}
                            )
                        """    
                    )
                else:
                    session.run(
                            f"""
                        CALL gds.graph.project(
                            '{deal_name(NOW_GRAPH)}',
                            'Node',
                            'Edge'
                            )
                        """    
                    )

                for algorithm in algorithms_list:
                    algorithm = algorithm.strip()
                    if algorithm == "bfs":
                        print("bfs now running...")
                        tic = time.time()
                        result = session.run(
                            f"""
                            MATCH (source:Node{{ID:{int(single_property_dict["bfs_source_vertex"])}}})
                            CALL gds.bfs.stream(
                            '{deal_name(NOW_GRAPH)}',{{
                                sourceNode: source
                            }})
                            YIELD path
                            RETURN path
                            """
                        )
                        toc = time.time()
                        print("bfs cost time: ", toc - tic)
                    if algorithm == "cdlp":
                        print("cdlp now running...")
                        tic = time.time()
                        result = session.run(
                            f"""
                            CALL gds.labelPropagation.stream('{deal_name(NOW_GRAPH)}')
                            YIELD nodeId, communityId AS Community
                            RETURN gds.util.asNode(nodeId).ID AS ID, Community
                            ORDER BY Community, ID
                            """
                        )
                        toc = time.time()
                        print("cdlp cost time: ", toc - tic)
                    if algorithm == "pr":
                        print("pagerank now running...")
                        tic = time.time()
                        result = session.run(
                            f"""
                            CALL gds.pageRank.stream('{deal_name(NOW_GRAPH)}')
                            YIELD nodeId, score
                            RETURN gds.util.asNode(nodeId).ID AS ID, score
                            ORDER BY score DESC, ID ASC
                            """
                        )
                        toc = time.time()
                        print("pagerank cost time: ", toc - tic)
                    if algorithm == "sssp":
                        print("sssp now running...")
                        tic = time.time()
                        result = session.run(
                            f"""
                            MATCH (source:Node {{ID: {int(single_property_dict["sssp_source_vertex"])}}})
                            CALL gds.allShortestPaths.dijkstra.stream('{deal_name(NOW_GRAPH)}', {{
                            sourceNode: source,
                            relationshipWeightProperty: 'weight'
                            }})
                            YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
                            RETURN
                                index,
                                gds.util.asNode(sourceNode).ID AS sourceNodeid,
                                gds.util.asNode(targetNode).ID AS targetNodeid,
                                totalCost,
                                [nodeId IN nodeIds | gds.util.asNode(nodeId).ID] AS nodeIDs,
                                costs,
                                nodes(path) as path
                            ORDER BY index
                            """
                        )
                        toc = time.time()
                        print("sssp cost time: ", toc - tic)
                    if algorithm == "wcc":
                        print("wcc now running...")
                        tic = time.time()
                        result = session.run(
                            f"""
                            CALL gds.wcc.stream('{deal_name(NOW_GRAPH)}')
                            YIELD nodeId, componentId
                            RETURN gds.util.asNode(nodeId).ID AS ID, componentId
                            ORDER BY componentId, ID
                            """
                        )
                        toc = time.time()
                        print("wcc cost time: ", toc - tic)
    os.system("neo4j stop")

if __name__ == "__main__":
    seed = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_file ="logs/" +seed + '.log'

    with open(log_file, 'a',buffering=1) as file:
        sys.stdout = file
        initialize()
        import_data()
        run_test()
        print("All tests are finished.")
        sys.stdout = sys.__stdout__                       
                
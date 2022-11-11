
import collections
import random
import math
from utils.xfg_util.xfg_CFG_DFG import *
import utils.xfg_util.rgx_utils as rgx
import pickle
from tools import Parameters
import tensorflow as tf



def get_tag_dict():
    tag_dict = collections.OrderedDict()
    list_tags = list()
    for fam in rgx.llvm_IR_stmt_families_DFG[1:]:
        list_tags.append(fam[1])
    list_tags = sorted(set(list_tags))

    for i in range(1, len(list_tags) + 1):
        tag_dict[list_tags[i - 1]] = i
    return tag_dict


def get_regex_dict():
    regex_dic = collections.OrderedDict()
    for fam in rgx.llvm_IR_stmt_families_DFG:
        regex_dic[fam[3]] = fam[1]
    return regex_dic



def get_embeddings_dict():
    with open("embeddings/emb.p", 'rb') as f:
        embedding_matrix = pickle.load(f)
    with open("embeddings/dic_pickle", 'rb') as f:
        stmt_dict = pickle.load(f)
    return embedding_matrix, stmt_dict

def load_data(PARALLEL_DATA_FOLDER, UNPARALLEL_DATA_FOLDER, split=0.75, cut=False):

    regex_dict = get_regex_dict()
    tag_dict = get_tag_dict()
    embedding_matrix, stmt_dict = get_embeddings_dict()
    print('loading data...')
    random.seed(100)
    parallel_g_list = []
    unparallel_g_list = []

    parallel_file_list = [f for f in os.listdir(PARALLEL_DATA_FOLDER)]
    unparallel_file_list = [f for f in os.listdir(UNPARALLEL_DATA_FOLDER)]
    for f1 in parallel_file_list:
        G1 = load_xfg(PARALLEL_DATA_FOLDER, f1, 0, regex_dict, tag_dict, embedding_matrix, stmt_dict)
        parallel_g_list.append(G1)
    for f2 in unparallel_file_list:
        G2 = load_xfg(UNPARALLEL_DATA_FOLDER, f2, 1, regex_dict, tag_dict, embedding_matrix, stmt_dict)
        unparallel_g_list.append(G2)


    if cut:
        parallel_g_list, unparallel_g_list = cut_data(parallel_g_list, unparallel_g_list)


    split_i = math.ceil(min(len(parallel_g_list), len(unparallel_g_list)) * split)
    train_graph_list = parallel_g_list[:split_i]
    train_graph_list.extend(unparallel_g_list[:split_i])
    test_graph_list = parallel_g_list[split_i:]
    test_graph_list.extend(unparallel_g_list[split_i:])
    random.shuffle(train_graph_list)
    random.shuffle(test_graph_list)
    params = Parameters()
    params.set('class_num', 2)
    params.set('node_label_dim', 36)  # maximum node label (tag) -------------------------only DFG
    params.set('node_feature_dim', 200)  # dim of node features (attributes)
    params.set('edge_feat_dim', 0)
    params.set('feature_dim', params.node_feature_dim + params.node_label_dim)

    return train_graph_list, test_graph_list, params

def load_multi_data():
    regex_dict = get_regex_dict()
    tag_dict = get_tag_dict()
    embedding_matrix, stmt_dict = get_embeddings_dict()
    print('loading data')
    random.seed(100)
    parallel_g_list1 = []
    unparallel_g_list1 = []
    # 读parallel
    # PARALLEL_DATA_FOLDER1 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_rose/training_xfg/1/"
    # UNPARALLEL_DATA_FOLDER1 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_rose/training_xfg/2/"
    PARALLEL_DATA_FOLDER1 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_data/training_xfg/1/"
    UNPARALLEL_DATA_FOLDER1 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_data/training_xfg/2/"
    parallel_file_list1 = [f for f in os.listdir(PARALLEL_DATA_FOLDER1)]
    unparallel_file_list1 = [f for f in os.listdir(UNPARALLEL_DATA_FOLDER1)]
    # print(parallel_file_list)
    # print(len(parallel_file_list))
    for f1 in parallel_file_list1:
        G1 = load_graph(PARALLEL_DATA_FOLDER1, f1, 0, regex_dict, tag_dict, embedding_matrix, stmt_dict)
        parallel_g_list1.append(G1)
    for f2 in unparallel_file_list1:
        G2 = load_graph(UNPARALLEL_DATA_FOLDER1, f2, 1, regex_dict, tag_dict, embedding_matrix, stmt_dict)
        unparallel_g_list1.append(G2)
    # parallel_g_list1, unparallel_g_list1 = cut_data(parallel_g_list1, unparallel_g_list1)

    g_train = parallel_g_list1
    g_train.extend(unparallel_g_list1)

    #####划分验证集
    # split = 0.78
    # split_i = math.ceil(min(len(parallel_g_list1), len(unparallel_g_list1)) * split)
    # g_train = parallel_g_list1[:split_i]
    # g_train.extend(unparallel_g_list1[:split_i])
    # g_val = parallel_g_list1[split_i:]
    # g_val.extend(unparallel_g_list1[split_i:])



    parallel_g_list2 = []
    unparallel_g_list2 = []
    # 读parallel
    # PARALLEL_DATA_FOLDER2 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_rose/test_xfg/1/"
    # UNPARALLEL_DATA_FOLDER2 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_rose/test_xfg/2/"
    PARALLEL_DATA_FOLDER2 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_pluto/test_xfg/1/"
    UNPARALLEL_DATA_FOLDER2 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_pluto/test_xfg/2/"
    # PARALLEL_DATA_FOLDER2 = r"/home/syy/Code/Github/Parallelismprediction/data/Parallel_NPB_105_79/1/"
    # UNPARALLEL_DATA_FOLDER2 = r"/home/syy/Code/Github/Parallelismprediction/data/Parallel_NPB_105_79/2/"
    parallel_file_list2 = [f for f in os.listdir(PARALLEL_DATA_FOLDER2)]
    parallel_file_list2.sort(key=lambda x: str.lower(x[:2]))
    unparallel_file_list2 = [f for f in os.listdir(UNPARALLEL_DATA_FOLDER2)]
    unparallel_file_list2.sort(key=lambda x: str.lower(x[:2]))
    # print(unparallel_file_list2)
    # UA_testing_list = parallel_file_list2
    # for f in unparallel_file_list2:
    #     UA_testing_list.append(f)
    # print("UA_testing_list =", UA_testing_list)
    for f1 in parallel_file_list2:
        G1 = load_graph(PARALLEL_DATA_FOLDER2, f1, 0, regex_dict, tag_dict, embedding_matrix, stmt_dict)
        parallel_g_list2.append(G1)
    for f2 in unparallel_file_list2:
        G2 = load_graph(UNPARALLEL_DATA_FOLDER2, f2, 1, regex_dict, tag_dict, embedding_matrix, stmt_dict)
        unparallel_g_list2.append(G2)

    g_test = parallel_g_list2
    g_test.extend(unparallel_g_list2)
    # print(g_test)


    # parallel_g_list3 = []
    # unparallel_g_list3 = []
    # PARALLEL_DATA_FOLDER3 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_pluto/val/1/"
    # UNPARALLEL_DATA_FOLDER3 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_pluto/val/2/"
    # # PARALLEL_DATA_FOLDER3 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_rose/val/1/"
    # # UNPARALLEL_DATA_FOLDER3 = r"/home/syy/Code/Github/Parallelismprediction/data/parallel_multi_rose/val/2/"
    #
    # parallel_file_list3 = [f for f in os.listdir(PARALLEL_DATA_FOLDER3)]
    # parallel_file_list3.sort(key=lambda x: str.lower(x[:2]))
    # unparallel_file_list3 = [f for f in os.listdir(UNPARALLEL_DATA_FOLDER3)]
    # unparallel_file_list3.sort(key=lambda x: str.lower(x[:2]))
    # for f1 in parallel_file_list3:
    #     G1 = load_graph(PARALLEL_DATA_FOLDER3, f1, 0, regex_dict, tag_dict, embedding_matrix, stmt_dict)
    #     parallel_g_list3.append(G1)
    # for f2 in unparallel_file_list3:
    #     G2 = load_graph(UNPARALLEL_DATA_FOLDER3, f2, 1, regex_dict, tag_dict, embedding_matrix, stmt_dict)
    #     unparallel_g_list3.append(G2)
    #
    # g_val = parallel_g_list3
    # g_val.extend(unparallel_g_list3)





    random.shuffle(g_train)
    # random.shuffle(g_val)
    # random.shuffle(g_test)

    params = Parameters()
    params.set('class_num', 2)
    params.set('node_label_dim', 36)  # maximum node label (tag) -------------------------only DFG
    params.set('node_feature_dim', 200)  # dim of node features (attributes)
    params.set('edge_feat_dim', 0)
    params.set('feature_dim', params.node_feature_dim + params.node_label_dim)
    # return g_train, g_test, params, g_val
    return g_train, g_test, params



def data_info(g_list):

    node_sum = 0
    edge_sum = 0
    max1 = 0
    max2 = 0
    min1 = 9999999999
    min2 = 9999999999
    for G in g_list:
        if G.num_nodes < min1:
            min1 = G.num_nodes
        if G.num_edges < min2:
            min2 = G.num_edges
        if G.num_nodes > max1:
            max1 = G.num_nodes
        if G.num_edges > max2:
            max2 = G.num_edges
        node_sum += G.num_nodes
        edge_sum += G.num_edges

    n_avg = node_sum / len(g_list)
    e_avg = edge_sum / len(g_list)
    return n_avg, e_avg


def del_ne(g_list, n_avg, e_avg):
    n = 0
    e = 0
    l1 = g_list[:]
    for G in l1:
        if G.num_edges > e_avg:
            g_list.remove(G)
            e += 1

    l2 = g_list[:]
    for G in l2:
        if G.num_nodes > n_avg:
            g_list.remove(G)
            n += 1

    return g_list


def cut_data(parallel_g_list, unparallel_g_list):

    g_list = []
    g_list.extend(parallel_g_list)
    g_list.extend(unparallel_g_list)

    n_avg, e_avg = data_info(g_list)

    parallel_g_list = del_ne(parallel_g_list, n_avg, e_avg)

    unparallel_g_list = del_ne(unparallel_g_list, n_avg, e_avg)
    g_list = []
    g_list.extend(parallel_g_list)
    g_list.extend(unparallel_g_list)

    n_avg, e_avg = data_info(g_list)
    return parallel_g_list, unparallel_g_list


def batching(graph_batch, params):


    def onehot(n, dim):
        one_hot = [0] * dim
        one_hot[n] = 1
        return one_hot


    if graph_batch[0].node_features is None:
        node_features = None
    else:
        node_features = [g.node_features for g in graph_batch]
        node_features = np.concatenate(node_features, 0)

    node_tag_features = None
    if params.node_label_dim > 1:
        node_labes = []
        for g in graph_batch:
            node_labes += g.node_tags
        node_tag_features = np.array([onehot(n, params.node_label_dim) for n in node_labes])

    if node_features is None and node_tag_features is None:
        node_dgrees = []
        for g in graph_batch:
            node_dgrees += g.degrees
        features = node_dgrees
    else:
        if node_features is None:
            features = node_tag_features
        elif node_tag_features is None:
            features = node_features
        else:
            features = np.concatenate([node_features, node_tag_features], 1)

    g_num_nodes = [g.num_nodes for g in graph_batch]
    graph_indexes = [[sum(g_num_nodes[0:i - 1]), sum(g_num_nodes[0:i])] for i in range(1, len(g_num_nodes) + 1)]


    batch_label = [onehot(g.label, params.class_num) for g in graph_batch]

    total_node_degree = []
    indices = []
    indices_append = indices.append
    for i, g in enumerate(graph_batch):
        total_node_degree.extend(g.degrees)
        start_pos = graph_indexes[i][0]
        for e in g.edges:
            node_from = start_pos + e[0]
            node_to = start_pos + e[1]
            indices_append([node_from, node_to])
            indices_append([node_to, node_from])
    total_node_num = len(total_node_degree)
    values = np.ones(len(indices), dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    shape = np.array([total_node_num, total_node_num], dtype=np.int32)
    ajacent = tf.SparseTensorValue(indices, values, shape)


    index_degree = [([i, i], 1.0 / degree if degree > 0 else 0) for i, degree in enumerate(total_node_degree)]
    index_degree = list(zip(*index_degree))
    degree_inv = tf.SparseTensorValue(index_degree[0], index_degree[1], shape)

    return ajacent, features, batch_label, degree_inv, graph_indexes


# load_multi_data()
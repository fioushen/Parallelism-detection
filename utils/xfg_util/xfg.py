
from utils.xfg_util.ir_preprocess import *
from multiprocessing import Pool
from functools import partial
import numpy as np
import networkx as nx


class XFG(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.node_labels = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degrees = list(dict(g.degree).values())
        self.edges = list(g.edges)

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(edge_features.values()[0]) == np.ndarray)
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.node_labels = node_tags  # 边标签
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degrees = list(dict(g.degree).values())
        self.edges = list(g.edges)  # 网络边列表

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(edge_features.values()[0]) == np.ndarray)
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

def construct_xfg(data_folder):
    """
    Preprocess raw LLVM IR code into XFGs (conteXtual Flow Graphs)

    :param data_folder: string containing the path to the parent directory of the subfolders containing raw LLVM IR code
    :return data_folders: list of subfolders containing raw LLVM IR code

    Input files:
        data_folder/*/*.ll

    Files produced:
        data_folder/*/data_read_pickle
        data_folder/*_preprocessed/data_preprocessed_pickle

    Folders produced:
        data_folder/*_preprocessed/data_transformed/
        data_folder/*_preprocessed/preprocessed/
        data_folder/*_preprocessed/structure_dictionaries/
        data_folder/*_preprocessed/xfg/
        data_folder/*_preprocessed/xfg_dual/
    """

    # Get raw data sub-folders
    assert os.path.exists(data_folder), "Folder " + data_folder + " does not exist"
    folders_raw = list()
    listing_to_explore = [os.path.join(data_folder, f) for f in os.listdir(data_folder + '/')]
    while len(listing_to_explore) > 0:
        f = listing_to_explore.pop()
        if os.path.isdir(f):
            f_contents = os.listdir(f + '/')
            for file in f_contents:
                # if it contains raw .ll files
                if file[-3:] == '.ll':
                    folders_raw.append(f)
                    break
                elif os.path.isdir(os.path.join(f, file)):
                    listing_to_explore.append(os.path.join(f, file))

    print('In folder', data_folder, ', found', len(folders_raw), 'raw data folder(s):\n', folders_raw)

    # Loop over raw data folders
    num_folder = len(folders_raw)
    if not DEBUG:
        # assign folders to different processes concurrently
        with Pool(processes=PROCESSES) as pool:
            _partial_func = partial(construct_xfg_single_raw_folder, num_folder=num_folder)
            pool.map(_partial_func, enumerate(folders_raw), chunksize=1)
    else:
        for folder_counter, folder_raw in enumerate(folders_raw):
            construct_xfg_single_raw_folder(params=(folder_counter, folder_raw), num_folder=num_folder)

    return folders_raw


def load_xfg(data_folder, graph_file_name, label, regex_dict, tag_dict, embedding_matrix, stmt_dict):

    g = nx.Graph()
    node_dict = {}
    with open(os.path.join(data_folder, graph_file_name), 'rb') as f:
        xfg = pickle.load(f)

    node_idx = 0
    for node in xfg.nodes():
        node_dict[node] = node_idx
        g.add_node(node_idx)
        node_idx += 1

    edge_pair_1, edge_pair_2 = zip(*xfg.edges())
    for x, y in zip(edge_pair_1, edge_pair_2):
        x = node_dict[x]
        y = node_dict[y]
        g.add_edge(x, y)

    node_tags = []
    node_features = []
    for node in xfg.nodes():

        stmt = node.split('§')[0]
        if stmt_dict.__contains__(stmt):
            node_features.append(np.array(embedding_matrix[stmt_dict[stmt]]))
        else:
            node_features.append(np.array(embedding_matrix[stmt_dict['!UNK']]))

        for regx in regex_dict.keys():
            res = re.match(regx, node)
            if res:
                node_tags.append(tag_dict[regex_dict[regx]])
                flag = True
                break
        if not flag:
            node_tags.append(0)
    G = XFG(g, label, node_tags, np.stack(np.array(node_features)))
    return G

def load_graph(data_folder, graph_file_name, lable, regex_dict, tag_dict, embedding_matrix, stmt_dict):
    g = nx.Graph()
    node_dict = {}
    with open(os.path.join(data_folder, graph_file_name), 'rb') as f:
        XFG = pickle.load(f)
    # 給节点编号
    node_idx = 0
    for node in XFG.nodes():
        node_dict[node] = node_idx
        g.add_node(node_idx)
        node_idx += 1
    # 处理边信息,将边转换为数值对
    # G = GNNGraph(XFG, lable, node_tags, node_features)
    edge_pair_1, edge_pair_2 = zip(*XFG.edges())
    for x, y in zip(edge_pair_1, edge_pair_2):
        x = node_dict[x]
        y = node_dict[y]
        g.add_edge(x, y)
    # 处理节点信息
    # type(node)=str
    # 先通过遍历regx_dict,得到node_tag,然后根据node得到node_features
    # CFG = collections.namedtuple('CFG', 'graph lable node_tags node_features')

    node_tags = []
    node_features = []

    for node in XFG.nodes():
        # 处理node_features
        stmt = node.split('§')[0]
        if stmt_dict.__contains__(stmt):
            node_features.append(embedding_matrix[stmt_dict[stmt]])
        else:
            node_features.append(embedding_matrix[stmt_dict['!UNK']])
        # 处理node_tags
        flag = False
        for regx in regex_dict.keys():
            res = re.match(regx, node)
            if res:
                node_tags.append(tag_dict[regex_dict[regx]])
                flag = True
                break
        if not flag:
            node_tags.append(0)

    G = GNNGraph(g, lable, node_tags, np.stack(np.array(node_features)))
    return G

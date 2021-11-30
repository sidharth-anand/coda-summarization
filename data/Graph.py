import ast

def get_node_type(node, node_types):
    if not node.__class__.__name__ in node_types:
        node_types[node.__class__.__name__] = len(node_types)
    return node_types[node.__class__.__name__]

def get_edge_type(edge, edge_types, adjacency_list):
    if not edge in edge_types:
        edge_types[edge] = len(edge_types)
    adjacency_list.append([])
    return edge_types[edge]

def add_predefined_edges(predefined_edge_types, edge_types, adjacency_list):
    for edge_type in predefined_edge_types:
        get_edge_type(edge_type, edge_types, adjacency_list)

def add_connection(child, parent_id, field, node_list, adjacency_list, variables_used, node_types, edge_types):
    node_list.append(get_node_type(child, node_types))

    adjacency_list[get_edge_type(field, edge_types, adjacency_list)].append([len(node_list) - 1, parent_id])
    adjacency_list[get_edge_type('parent_of', edge_types, adjacency_list)].append([parent_id, len(node_list) - 1])

    if isinstance(child, ast.Name):
        id_node_id = len(node_list)
        if getattr(child, 'id') in variables_used:
            adjacency_list[get_edge_type('next_use_of', edge_types, adjacency_list)].append([variables_used[getattr(child, 'id')], id_node_id])
            adjacency_list[get_edge_type('previous_use_of', edge_types, adjacency_list)].append([id_node_id, variables_used[getattr(child, 'id')]])
        variables_used[getattr(child, 'id')] = id_node_id


    tree_to_graph(child, node_list, adjacency_list, node_types, variables_used, edge_types, len(node_list) - 1)

def tree_to_graph(root, node_list, adjacency_list, node_types, variables_used, edge_types, parent_id = None):
    if parent_id is None:
        node_list.append(get_node_type(root, node_types))
        adjacency_list.append([])
        parent_id = len(node_list) - 1

    if not hasattr(root, '_fields'):
        return

    for field in root._fields:
        child = getattr(root, field)
        if isinstance(child, list):
            for grandchild in child:
                add_connection(grandchild, parent_id, field, node_list, adjacency_list, variables_used, node_types, edge_types)
        else:
            add_connection(child, parent_id, field, node_list, adjacency_list, variables_used, node_types, edge_types)

def add_sibling_connections(adjacency_list, edge_types):
    children = {}

    for i in range(len(adjacency_list[get_edge_type('parent_of', edge_types, adjacency_list)])):
        parent_id = adjacency_list[get_edge_type('parent_of', edge_types, adjacency_list)][i][0]
        child_id = adjacency_list[get_edge_type('parent_of', edge_types, adjacency_list)][i][1]

        if not parent_id in children:
            children[parent_id] = []
        else:
            for child in children[parent_id]:
                adjacency_list[get_edge_type('next_sibling_of', edge_types, adjacency_list)].append([child, child_id])
                adjacency_list[get_edge_type('previous_sibling_of', edge_types, adjacency_list)].append([child_id, child])
        
        children[parent_id].append(child_id)

def add_condition_connections(node_list, adjacency_list, node_types, edge_types):
    node_types = list(node_types.keys())

    for edge in adjacency_list[get_edge_type('test', edge_types, adjacency_list)]:
        child_id = edge[0]
        parent_id = edge[1]

        if not node_types[node_list[parent_id]] == 'If':
            continue

        for body_edge in adjacency_list[get_edge_type('body', edge_types, adjacency_list)]:
            if body_edge[1] == parent_id:
                adjacency_list[get_edge_type('condition_true', edge_types, adjacency_list)].append([child_id, body_edge[0]])

        for orelse_edge in adjacency_list[get_edge_type('orelse', edge_types, adjacency_list)]:
            if orelse_edge[1] == parent_id:
                adjacency_list[get_edge_type('condition_false', edge_types, adjacency_list)].append([child_id, orelse_edge[0]])

def add_while_connections(node_list, adjacency_list, node_types, edge_types):
    node_types = list(node_types.keys())

    for edge in adjacency_list[get_edge_type('test', edge_types, adjacency_list)]:
        child_id = edge[0]
        parent_id = edge[1]

        if not node_types[node_list[parent_id]] == 'While':
            continue

        for body_edge in adjacency_list[get_edge_type('body', edge_types, adjacency_list)]:
            if body_edge[1] == parent_id:
                adjacency_list[get_edge_type('while_exec', edge_types, adjacency_list)].append([child_id, body_edge[0]])
                adjacency_list[get_edge_type('while_next', edge_types, adjacency_list)].append([body_edge[0], child_id])

def run_tree(simple_code, node_types, edge_types):
    node_list = []
    adjacency_list = []
    variables_used = {}
    
    add_predefined_edges(['parent_of', 'next_sibling_of', 'previous_sibling_of', 'next_use_of', 'previous_use_of', 'condition_true', 'condition_false', 'while_execute', 'while_next'], edge_types, adjacency_list)
    print(edge_types)
    ast_tree = ast.parse(simple_code)
    tree_to_graph(ast_tree, node_list, adjacency_list, node_types, variables_used, edge_types)
    
    add_sibling_connections(adjacency_list, edge_types)
    add_condition_connections(node_list, adjacency_list, node_types, edge_types)
    add_while_connections(node_list, adjacency_list, node_types, edge_types)
    
    for i in range(len(node_list)):
        node_list[i] = [node_list[i]]

    return {
        'node_list': node_list,
        'adjacency_list': adjacency_list,
        'node_types': node_types,
        'edge_types': edge_types,
        'variables_used': variables_used
    }
    #return node_list, adjacency_list, node_types, edge_types
    # save_file_path = 'tree_data.pickle'
    # with open(save_file_path, 'wb') as f:
    #     pickle.dump(adjacency_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(node_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(edge_types, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(node_types, f, protocol=pickle.HIGHEST_PROTOCOL)

def merge_graphs(merged_tree, new_tree):
    node_list_1 = merged_tree['node_list']
    adjacency_list_1 = merged_tree['adjacency_list']
    
    node_list_2 = new_tree['node_list']
    adjacency_list_2 = new_tree['adjacency_list']

    node_list_1 += node_list_2
    
    for i, edge_type in enumerate(adjacency_list_2):
        modified_edges = [[edge[0] + len(node_list_1), edge[1] + len(node_list_1)] for edge in edge_type]
        
        while i >= len(adjacency_list_1):
            adjacency_list_1.append([])

        adjacency_list_1[i] += modified_edges
            

    return {
        'node_list': node_list_1,
        'adjacency_list': adjacency_list_1,
    }        


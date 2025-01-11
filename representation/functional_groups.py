from fgutils import FGQuery 

query = FGQuery() 

def get_groups_from_smiles(smiles:str): # e.g. acetylsalicyclic acid "O=C(C)Oc1ccccc1C(=O)O" 
    return query.get(smiles) # e.g. [("ester", [0, 1, 3]), ("carboxylic_acid", [10, 11, 12])] 


# https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.topology.subgraph_isomorphism.html#graph_tool.topology.subgraph_isomorphism 
# this can also be used to find functional groups maybe 




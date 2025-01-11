# NOTE THAT THESE WERE FROM ANOTHER PROJECT WITH DIFFERENT ATOM FEATURES AND BOND FEATURES AND ALLOWED ATOM TYPES SO SOME PARTS CAN'T DIRECTLY BE USED 


import os
os.environ['DGLBACKEND'] = 'pytorch'

from rdkit import Chem 
from rdkit.Chem.rdMolDescriptors import CalcMolFormula 
import molmass 

import matplotlib.pyplot as plt 

import networkx as nx 
import dgl 

import torch 










def SMILEStoMol(smiles): 
    mol = Chem.rdmolfiles.MolFromSmiles(smiles) 
    return mol 


def smiles_to_formula(smiles:str): 
    return CalcMolFormula(Chem.MolFromSmiles(smiles))  


# TODO: update these to allow for other elements 
def smiles_to_atom_counts(smiles:str, atomtypes=['C', 'H', 'N', 'O', 'P', 'S']): 
    smiles = smiles.lower() 
    counts = [] 
    for atom in atomtypes: 
        counts.append(smiles.count(atom.lower())) 
    
    return counts 

def formula_to_atom_counts(formula:str, include_H=True):
    if include_H: 
        atomtypes = ['C', 'H', 'N', 'O', 'P', 'S']
    else: 
        atomtypes = ['C', 'N', 'O', 'P', 'S']

    count_series = molmass.Formula(formula).composition().dataframe()['Count'] 
    # make sure no atom not in list was found 
    for atom in count_series.keys(): 
        if atom not in atomtypes: 
            if atom == 'H': continue 
            print("ERROR: ATOM "+str(atom)+" NOT IN ALLOWED LIST, SKIPPING ATOM") 
    
    counts = [] 
    for target in atomtypes: 
        try: 
            counts.append(count_series[target]) 
        except: # no such atom 
            counts.append(0) 

    return counts 







# FOR COMPARISON for the DGL Graphs in MolGraph 
'''
# also can use vertex n-colourability 

# this is tutte polynomial 
def state_get_invariants(state): 
    # Tutte polynomial - using networkx 
    g = dgl.to_networkx(state.graph) 
    return nx.tutte_polynomial(g.to_undirected(as_view=True)) 
'''
# NOTE: instead of using graph invariants, now comparing node feature sequence and edge feature sequence, and if they match, will fully check isomorphism 

def may_be_isomorphic(g1, g2): 
    # note: both are dgl graphs 
    nf1 = g1.ndata['features'].tolist() 
    ef1 = g1.edata['bondTypes'].tolist() 

    nf2 = g2.ndata['features'].tolist() 
    ef2 = g2.edata['bondTypes'].tolist() 

    nf1.sort() 
    nf2.sort() 
    ef1.sort() 
    ef2.sort() 

    #print(nf1) 
    #print(nf2) 
    #print(ef1) 
    #print(ef2) 

    diff = False 
    for i in range(len(nf1)): 
        for j in range(len(nf1[i])): 
            if nf1[i][j] != nf2[i][j]: 
                diff = True 
                #print(i, j, nf1[i][j], nf2[i][j]) 
                break 
    #print(diff) 
    if (diff): return False 
    
    for i in range(len(ef1)): 
        for j in range(len(ef1[i])): 
            if ef1[i][j] != ef2[i][j]: 
                diff = True 
                #print(i, j, ef1[i][j], ef2[i][j]) 
                break 
    
    #print(diff) 

    return (not diff) 

def dgl_to_networkx_for_isomorphism(g1): 
    G1 = nx.DiGraph(dgl.to_networkx(g1)) 

    #print(G1) 
    #print(type(G1))

    g1_n_attrs = {} 
    for i in range(len(g1.nodes())): 
        g1_n_attrs[i] = {'idx': i} 

    g1_e_attrs = {} 

    for i in range(len(g1.edges()[0])):
        g1_e_attrs[(g1.edges()[0][i].item(), g1.edges()[1][i].item())] = {'idx': i} 
        i += 1 
    
    #print(g1_n_attrs) 
    #print(g1_e_attrs) 

    nx.set_node_attributes(G1, g1_n_attrs) 
    nx.set_edge_attributes(G1, g1_e_attrs)

    return G1 

def is_isomorphic(g1, g2, G1=None, G2=None): 
    # note: both are dgl graphs 
    def node_match(n1, n2): 
        #print("NODE MATCH", n1, n2, ':', g1.ndata['features'][n1['idx']] == g2.ndata['features'][n2['idx']] ) 
        #print() 
        return (g1.ndata['features'][n1['idx']] == g2.ndata['features'][n2['idx']]).all() 
    
    def edge_match(e1, e2): 
        #print("EDGE MATCH", e1, e2, ':', g1.edata['bondTypes'][e1['idx']] == g2.edata['bondTypes'][e2['idx']] ) 
        #print() 
        return (g1.edata['bondTypes'][e1['idx']] == g2.edata['bondTypes'][e2['idx']]).all()
    
    if G1==None: 
        G1 = dgl_to_networkx_for_isomorphism(g1) 
    if G2==None: 
        G2 = dgl_to_networkx_for_isomorphism(g2) 
    
    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match) 
























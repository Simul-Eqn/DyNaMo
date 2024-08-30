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


def smiles_to_atom_counts(smiles:str, atomtypes=['C', 'H', 'N', 'O', 'P', 'S']): 
    smiles = smiles.lower() 
    counts = [] 
    for atom in atomtypes: 
        counts.append(smiles.count(atom.lower())) 
    
    return counts 

def formula_to_atom_counts(formula:str, include_H=True):
    if include_H: 
        atomtypes = FTreeNode.atomTypes 
    else: 
        atomtypes = params.atom_types 

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












# MOLGRAPH CLASS 



# structure of graph: 
'''
node features (H isn't node): [atomic number, atomic mass, R, S, ?, not_chiral, formal charge, number of Hs, is_in_aromatic, degree]
node features (H is node): [atomic number, atomic mass, R, S, ?, not_chiral, formal charge, degree] 
atomic number may be a one-hot encoding instead. 
All are numbers. R, S, ?, not_chiral, is_in_aromatic" are 1/0. 

bond features is just a one-hot encoding of what bondtype it is: see bondTypes variable 
It'll be [0/1, 0/1, 0/1, 0/1] 
'''

# encoders 
def chnops_encoder(atomNum, ignore_error:bool): 
    # if it's C, H, N, O, P, or S 
    if atomNum==6: 
        return [1, 0, 0, 0, 0, 0] 
    elif atomNum==1: 
        return [0, 1, 0, 0, 0, 0] 
    elif atomNum==7: 
        return [0, 0, 1, 0, 0, 0] 
    elif atomNum==8: 
        return [0, 0, 0, 1, 0, 0] 
    elif atomNum==15: 
        return [0, 0, 0, 0, 1, 0] 
    elif atomNum==16: 
        return [0, 0, 0, 0, 0, 1] 
    
    if ignore_error: 
        return [0, 0, 0, 0, 0, 0] 
    else: 
        raise ValueError("Atom is not CHNOPS; cannot use CHNOPS one hot encoder.")



class MolGraph: 

    # nodes 
    node_colour_schemes = {
        'CHNOPS': ['#aaaaaa', '#eeeeee', '#33dd33', '#dd3333', '#dd9300', '#dddd00'] 
    }
    node_one_hot_encoders = {
        'CHNOPS': chnops_encoder, 
    } # from atomic num to one-hot encoding 

    @classmethod 
    def _getChiralFeatsFromMol(cls, mol:Chem.rdchem.Mol): 
        centers = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True) 
        feats = [(0, 0, 0, 1) for _ in range(mol.GetNumAtoms())]
        for (idx, t) in centers: 
            if (t == 'R'): 
                feat = (1, 0, 0, 0) 
            elif (t == 'S'): 
                feat = (0, 1, 0, 0) 
            elif (t == '?'): 
                feat = (0, 0, 1, 0) 
            else: 
                raise ValueError("CANNOT IDENTIFY CHIRAL CENTER TYPE FOR ("+str(idx)+", "+str(t)+") in mol") 
            feats[idx] = feat 
        
        return feats 

    @classmethod 
    def _getNodeFeatsFromAtom(cls, atom, chiral_feats, h_is_node:bool, one_hot_encoding_scheme=None, ignore_one_hot_error=True, ignore_anum_feats_error=False, ): 
        if one_hot_encoding_scheme is None: 
            atomic_num_feats = [atom.GetAtomicNum()]  
        else: 
            try: 
                atomic_num_feats = MolGraph.node_one_hot_encoders[one_hot_encoding_scheme](atom.GetAtomicNum(), ignore_error=ignore_one_hot_error) 
            except Exception as e: 
                if ignore_anum_feats_error: 
                    atomic_num_feats = [atom.GetAtomicNum()] 
                else: 
                    print("ERROR GETTING ATOMIC NUM FEATS:", atom)
                    print(e) 
                    raise ValueError(e) 

        
        if h_is_node: return [*atomic_num_feats, atom.GetMass(), *chiral_feats, atom.GetFormalCharge(), atom.GetTotalNumHs(), atom.GetDegree()]
        return [*atomic_num_feats, atom.GetMass(), *chiral_feats, atom.GetFormalCharge(), atom.GetTotalNumHs(), int(atom.GetIsAromatic()), atom.GetDegree()] 
    
    @classmethod 
    def _getNodeFeatsFromMol(cls, mol, h_is_node:bool, one_hot_encoding_scheme="CHNOPS", ignore_one_hot_error=True, ignore_anum_feats_error=False, ): 
        '''Extract node features from molecule. The return type is not a tensor, but a 2d Python list. 
        h_is_node is a planned feature for deciding if H should be nodes, but is not implemented yet. 
        one_hot_encoding_scheme defines which one-hot encoding scheme to use. None means it just takes the atomic number. 
        ignore_one_hot_error makes any element not in the encoding scheme be encoded as [0,0,...,0,0]
        ignore_anum_feats_error means, if there's any error in one-hot encoding, then it'll revert to just taking the atomic number. '''
        chiral_feats = MolGraph._getChiralFeatsFromMol(mol) 
        atoms = mol.GetAtoms() 
        return [MolGraph._getNodeFeatsFromAtom(atoms[i], chiral_feats[i], h_is_node, one_hot_encoding_scheme, ignore_one_hot_error, ignore_anum_feats_error )  for i in range(mol.GetNumAtoms())  ]
        



    # edges 

    # lookup table for bond types: single, double, triple, aromatic 
    bondTypes = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    edgeColors = ['#11ee11', '#1111ee', '#ee0000', '#606060'] # for display purposes 

    bond_n_feats = len(bondTypes)+1 # bondType index and bond energy 

    @classmethod 
    def _bondType_idx_to_encoding(cls, bondtype_idx): 
        res = [] 
        for bt_idx in range(len(MolGraph.bondTypes)): 
            if bondtype_idx==bt_idx: res.append(1) 
            else: res.append(0) 

        return res 

    @classmethod 
    def _getEdgeFeatsFromMol(cls, mol): 
        '''Extract edge features from molecule. The return type is not a tensor, but a 2d Python list. '''
        bonds = mol.GetBonds() 

        bondFrom = [] 
        bondTo = [] 
        edgeFeats = [] 

        for b in bonds:
            # get bond data 
            begin = b.GetBeginAtomIdx()
            end = b.GetEndAtomIdx()
            bondtype_idx = MolGraph.bondTypes.index(b.GetBondType())

            # add bond 
            bondFrom.append(begin) 
            bondTo.append(end) 
            edgeFeats.append( MolGraph._bondType_idx_to_encoding(bondtype_idx) ) 

            # add backwards bond since networkx is directed 
            bondFrom.append(end) 
            bondTo.append(begin) 
            edgeFeats.append( MolGraph._bondType_idx_to_encoding(bondtype_idx) )

        return bondFrom, bondTo, edgeFeats 





    # "initializers/constructors" 
    def __init__(self, mol, graph, graph_encoding_scheme="CHNOPS"): 
        self.mol = mol 
        self.dgl_graph = graph 
        self.graph_encoding_scheme = graph_encoding_scheme 
    
    @classmethod 
    def from_mol(cls, mol, graph_encoding_scheme="CHNOPS"): 

        # node feats 
        nodeFeatures = torch.tensor( MolGraph._getNodeFeatsFromMol(mol, False, one_hot_encoding_scheme=graph_encoding_scheme) ) 

        # bond feats 
        bondFrom, bondTo, edgeFeats = MolGraph._getEdgeFeatsFromMol(mol) 
        edgeFeats = torch.tensor( edgeFeats )


        graph = dgl.graph((torch.tensor(bondFrom), torch.tensor(bondTo)), num_nodes=mol.GetNumAtoms(), idtype=torch.int32)
        graph.ndata['features'] = nodeFeatures 
        graph.edata['bondTypes'] = edgeFeats 
        #print('features', gnn.ndata['features'])
        #print('bond types', gnn.edata['bondtypes'])
        #gnn = nx.to_undirected(gnn) 

        return MolGraph(mol, graph, graph_encoding_scheme) 

    @classmethod 
    def from_smiles(cls, smiles, graph_encoding_scheme="CHNOPS"): 
        return MolGraph.from_mol(SMILEStoMol(smiles), graph_encoding_scheme) 
    






    # comparator 
    def __eq__(self, other): 
        if not isinstance(other, MolGraph): raise ValueError("CANNOT COMPARE MolGraph with object of type "+str(type(other))) 
        if (may_be_isomorphic(self.dgl_graph, other.dgl_graph)): # to save time 
            return is_isomorphic(self.dgl_graph, other.dgl_graph) 
        return False 
    




    # display 
    vis_k = 0.3 
    def show_visualization(self, title=None, pos=None, atomTypes=['C','H','N','O','P','S'], draw_edge_labels=False, block=True): 
        plt.figure() 

        graph = self.graph.cpu() 

        # prepare 
        g = dgl.to_networkx(graph) 

        #stdout.write("Generating graph... \n")

        if pos==None: pos = nx.spring_layout(g, k=MolGraph.vis_k, iterations=20) 

        #stdout.write("Drawing graph... \n")

        # draw each kind of node 
        node_options = {"node_size": 400, "node_shape": 'o'} 

        labels = {} 

        for nodeType in range(len(atomTypes)): 
            nt = nodeType 
            if nt==1: continue # Hydrogen 
            if nt>1: nt -= 1 

            nodes = [] 
            for nidx in range(len(graph.nodes())): 
                if graph.ndata['features'][nidx][nt+2].item() == 1: 
                    nodes.append(nidx) 
                    labels[nidx] = atomTypes[nodeType] 

            #print(nodeType, nt)
            #print("ATOM TYPE:", utils.FTreeNode.atomTypes[nodeType]) 
            #print("COLOUR:", utils.FTreeNode.atomColours[nodeType]) 
            nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=MolGraph.atomColours[nodeType], **node_options) 

        # draw each kind of edge 

        edge_options = {"alpha": 0.7} 
        graph_edge_list = list(graph.edges()) 
        #print(graph_edge_list) 
        for edgeType in range(len(MolGraph.bondTypes)): 
            edges = [] 
            for eidx in range(len(graph_edge_list[0])): 
                #print(graph.edata['bondTypes'][eidx][edgeType+1].item(), end=' ')
                if graph.edata['bondTypes'][eidx][edgeType+1].item() == 1: 
                    edges.append((graph_edge_list[0][eidx].item(), graph_edge_list[1][eidx].item())) 
            #print() 
            #print(edgeType, ":", edges)
            nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color=MolGraph.node_colour_schemes[self.graph_encoding_scheme][edgeType], **edge_options)

        nx.draw_networkx_labels(g, pos)

        if draw_edge_labels: # TODO: THIS IS VERY WRONG 
            edge_labels = {} 
            i = 0 
            for e in g.edges(): 
                edge_labels[e] = str(i//2) # THIS IS WRONG, E.G. BENZENE RING 
                i += 1 
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8, alpha=0.5)

        if title != None: 
            plt.title(title) 

        plt.show(block=block) 

    












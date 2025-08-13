import pandas as pd
'''
Serveix per carregar i gestionar dades bàsiques
'''

class BusNetworkData:
    def __init__(self, nodes_path, od_path):
        # Carreguem nodes i posicions
        self.nodes_df = pd.read_csv(nodes_path)
        self.NB = list(self.nodes_df['node'])  # Llista de parades disponibles
        self.pos = {}
        self.OD = pd.read_csv(od_path, index_col=0)  # Carreguem matriu OD

    def get_available_stops(self):
        return self.NB

    def get_position(self, node):
        return self.pos.get(node, (None, None))

    def get_demand(self, i, j):
        return self.OD.loc[i, j]

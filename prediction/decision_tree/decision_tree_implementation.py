import pandas as pd 
import numpy as np 

trainning_data = pd.read_csv('../../no_index_no_outlier_trainning.csv')
trainning_target = pd.read_csv('../../data/target.csv')['price']


class Node :
    population = None
    target = None
    attribute = None
    mediane = None
    children = []

    # mesure le efficient de variance du node
    def CV(self):
        self.target = trainning_target[self.population.index]
        
        if len(self.target) < 1: return 0
        return self.target.std()/trainning_target.mean()
    def pred(self):
        return self.target.mean(), self.target.std()

#Parametre de profondeur
depth = 1

initial_node = Node()
initial_node.population = trainning_data
initial_node.attribute = "root"
nodes = [initial_node]
print("==== root ====")

min_information_gain = 0
for i in range(depth):
    for node in nodes:
        print(node)
        if len(node.children) == 0:
            
            best_column = ""
            max_information_gain = 0
            children = []
            for column in trainning_data.columns:
                son_nodes = []
                min_std = 0
                #On creer deux nodes dont les elements sont separer 
                #par la mediane de la colonne courante

                median = np.median(trainning_data[column])
                new_left_node = Node()
                new_left_node.population = node.population[node.population[column] >= median]
                new_left_node.attribute = column
                new_left_node.mediane = median

                new_right_node = Node()
                new_right_node.population = node.population[node.population[column] < median]
                new_right_node.attribute = column
                new_right_node.mediane = median

                #Si les nouveau neouds contient des elements
                #On mets a jour la vatiable a minisé
                #On les sauvegardes dans Son_nodes
                if (len(new_left_node.population) != 0) or (len(new_right_node.population) != 0):
                    min_std += (len(new_left_node.population)/len(node.population))*new_left_node.CV()
                    min_std += (len(new_right_node.population)/len(node.population))*new_right_node.CV()
                    son_nodes = [new_left_node,new_right_node]
                    information_gain = node.CV() - min_std
                    if (information_gain > max_information_gain):
                        max_information_gain = information_gain
                        children = son_nodes

            if ( max_information_gain > min_information_gain ):
                node.children = children    
                nodes += children
    
            
def displayTree(r, l=0):
    print("   "*l, end="")
    print(f"{r.attribute} = {r.mediane} {r.pred()} pop: {len(r.population)}")
    for c in r.children:
        displayTree(c, l+ 1)

displayTree(nodes[0])
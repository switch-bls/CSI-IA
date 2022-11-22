import pandas as pd #pd
import numpy as np #no probleeem

trainning_data = pd.read_csv('../../data/training_data_set.csv')
trainning_target = pd.read_csv('../../data/target.csv')['price']


class Node :
    population = None
    target = None
    attribute = None
    mediane = None
    children = []

    # for binary attributes
    def CV(self):
        self.target = trainning_target[self.population.index]
        
        if len(self.target) == 1: return 0
        return self.target.std()/trainning_target.mean()
    def pred(self):
        return self.target.mean(), self.target.std()


depth = 4

initial_node = Node()
initial_node.population = trainning_data
initial_node.attribute = "root"
nodes = [initial_node]
print("==== root ====")
#print(f"Entropy: {initial_node.CV()}")

min_information_gain = 0.0001

for i in range(depth):
    for node in nodes:
        if len(node.children) == 0:
            print("node courrant : "+str(node))
            best_column = ""
            max_information_gain = 0
            children = []
            for column in trainning_data.columns:
                son_nodes = []
                min_std = 0
                median = np.median(trainning_data[column])
                #print(column)
                new_left_node = Node()
                new_left_node.population = node.population[node.population[column] >= median]
                new_left_node.attribute = column
                new_left_node.median = median
                #print("pop1 : "+str(len(new_left_node.population.index)))
                new_right_node = Node()
                new_right_node.population = node.population[node.population[column] < median]
                new_right_node.attribute = column
                new_right_node.median = median
                #print("pop2 : "+str(len(new_right_node.population.index)))
                if (len(new_left_node.population) != 0) and (len(new_right_node.population) != 0):
                    print(f"looking at {column}={median}, CV1={new_left_node.CV()} CV2={new_right_node.CV()}")
                    min_std += (len(new_left_node.population)/len(node.population))*new_left_node.CV()
                    min_std += (len(new_right_node.population)/len(node.population))*new_right_node.CV()
                    print("son1 : "+str(son_nodes))
                    son_nodes.append(new_left_node)
                    son_nodes.append(new_right_node)
                    print("son2 : "+str(son_nodes[0]))

                information_gain = node.CV() - min_std
                #print(f"Weighted average entropy for {column} is {min_std}, and information gain is {information_gain}")
                #print(f"{max_information_gain}")
                print("son3 : "+str(son_nodes))
                print("information_gain : "+str(information_gain))
                print("max_information_gain : "+str(max_information_gain))
                if (information_gain > max_information_gain):
                    max_information_gain = information_gain
                    best_column = column
                    print("son : "+str(son_nodes))
                    if len(son_nodes) == 2: 
                        children.append(son_nodes[0]) 
                        children.append(son_nodes[1])
                        print("children 1 : "+str(children))
            if ( max_information_gain > min_information_gain ):
                print("children : "+str(children))
                node.children = children
                nodes += children
    
            
def displayTree(r, l=0):
    print(" "*l, end="")
    print(f"{r.attribute} ={r.mediane} {r.pred()}")
    for c in r.children:
        displayTree(c, l+ 1)

print(nodes)
print(nodes[0].children)
displayTree(nodes[0])
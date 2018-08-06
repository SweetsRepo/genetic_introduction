import random
import operator

import numpy as np

__author__ = "Christopher Sweet"
__doc__ = "An example of the traveling salesman problem solved by a simple Genetic Algorithm"

#Create a population of size 50, with 5 carried over from the previous generation
ELITE_SIZE = 20
MATE_SIZE = 100
#Percent chance of muation occuring
MUTATION_RATE = 0.01
#Number of cities to create
NUM_CITIES = 25
GENERATIONS = 500

class City:
    '''
    Representation of a city
    :attribute x: X Coordinate
    :attribute y: Y Coordinate
    '''
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y

    def distance_to_city(self, city):
        '''
        Finds the Euclidean Distance to the city given
        :param city: Other city to find distance to
        :return euclidean: Euclidean Distance to city
        '''
        x_distance = abs(self.x - city.x)
        y_distance = abs(self.y - city.y)
        euclidean = np.sqrt(x_distance**2 + y_distance**2)
        return euclidean

class Salesman:
    '''
    Representation of the salesman
    :attribute route: Route to take between cities
    :attribute distance: Distance for fitness metric
    :attribute fitness: Inverse of distance
    :notes: The GA evaluates the fitness of each salesman as the inverse of the distance
    '''
    def __init__(self, route:list):
        self.route = route
        self.distance = 0.0
        self.fitness = 0.0

    def route_distance(self, route:list):
        '''
        Finds the total distance given the route selected
        :param route: Route to take as a list of city objects
        :returns total_distance: Total distance for the salesman to travel
        '''
        #Make the route circular
        route.append(route[0])
        total_distance = 0
        for i in range(len(route)-1):
            #Calculate the distance between the current city and the next to travel to
            total_distance += route[i].distance_to_city(route[i+1])
        return total_distance

    def route_fitness(self, distance:float):
        '''
        Calculates the fitness as the inverse of the total distance
        :param distance: Total distance for route
        :return fitness: Fitness given the distance
        '''
        return 1 / float(distance)

    def evalutate_salesman(self):
        '''
        Wraps the distance calculation and fitness calculation into one function and
        stores results in class attributes
        '''
        self.distance = self.route_distance(self.route)
        self.fitness = self.route_fitness(self.distance)

def generate_cities():
    '''
    Creates randomly located cities
    '''
    return [City(x=int(random.random()*200), y=int(random.random()*200)) for _ in range(NUM_CITIES)]

def generate_salesmen(routes:list):
    '''
    Creates salesmen with varying routes
    :param routes: List of the rotues to use
    :returns population: Randomly selected routes that visit each city exactly once
    '''
    population = []
    for i in range(len(routes)):
        individual = Salesman(routes[i])
        individual.evalutate_salesman()
        population.append(individual)
    return population

def rank_salesmen(population:list):
    '''
    Ranks the salemen in the given population based off fitness
    :param population: Population of salesmen to evalutate
    :return ranked_pop: Population ordered by best salesman to worst as list of tuples
    '''
    ranked_pop = {}
    for i in range(len(population)):
        ranked_pop[i] = population[i].fitness
    return sorted(ranked_pop.items(), key=lambda k: k[1], reverse=True)

def select_salesmen(ranked_pop:list):
    '''
    Selects the salesmen to make it to the next generation using fitness
    proportionate selection and elitism (top-3 selected).
    :param ranked_pop: Ranked population of salesmen
    :returns strongest_salesmen_ids: List of the salesmen to start next generation
    '''
    ranked_pop = [list(item) for item in ranked_pop]
    strongest_salesmen_ids = []

    #Adds the top 3 salesmen to the next generation
    strongest_salesmen_ids.extend(range(ELITE_SIZE))
    total_fitness = sum([salesman[1] for salesman in ranked_pop])

    #Expand the remaining population such that each salesman can be selected based on their Fitness
    fitness_expanse = 0
    # ranked_pop will be in form [sale_id, fitness_expanse]
    for i, salesman in enumerate(ranked_pop):
        if i != 0:
            salesman[1] = 100 * (salesman[1]/total_fitness) + ranked_pop[i-1][1]
        else:
            salesman[1] = 100 * (salesman[1]/total_fitness)
    for i in range(MATE_SIZE):
        select = 100*random.random()
        i = 0
        while i < len(ranked_pop) and ranked_pop[i][1] < select:
            i+=1
        strongest_salesmen_ids.append(ranked_pop[i][0])
    return strongest_salesmen_ids

def mate_salesmen(strongest_salesmen:list):
    '''
    Breeds the strongest salemen to create a new generation
    :param strongest_salesmen: List of strongest salement selected through elitism and fitness proportionate selection
    :returns children: List of child routes from mating
    '''
    children = []
    pool = random.sample(strongest_salesmen, len(strongest_salesmen))
    children.extend(strongest_salesmen[:ELITE_SIZE])
    for i in range(len(strongest_salesmen)):
        #Grab the parents for this child
        p1 = strongest_salesmen[i]
        p2 = strongest_salesmen[len(strongest_salesmen)-i-1]
        #Create random segmentation of the genes (routes)
        gene_1 = int(random.random() * len(p1))
        gene_2 = int(random.random() * len(p2))
        start = min(gene_1, gene_2)
        end = max(gene_1, gene_2)
        #Build Gene from cross between the two parents
        child = []
        for i in range(start, end):
            child.append(p1[i])
        #Grabs the remaining parts of the route
        child_remainder =[item for item in p2 if item not in child]
        child += child_remainder
        children.append(child)
    return children

def mutate_salesmen(children:list):
    '''
    Applies random mutation to children to avoid convergence to local optima
    :param children: List of children to potentially mutate
    :returns children: Children, with a few potentially exposed to radiation
    '''
    for route in children:
        for i,city in enumerate(route):
            if random.random() <= MUTATION_RATE:
                city_swap_idx = int(random.random() * len(route))
                city_swap = route[city_swap_idx]
                route[city_swap_idx] = city
                route[i] = city_swap
    return children

def get_next_generation(current_gen):
    '''
    Follows the procedure below to create a new generation of salesmen
    :param current_gen: Current generation of salesmen
    :returns next_gen_routes: New Generation of salesmen
    :notes:
      1. Rank
      2. Select
      3. Mate
      4. Mutate
    '''
    #Rank the salesmen and return list of tuples (id, fitness)
    ranked = rank_salesmen(current_gen)
    #Select salesmen based off their fitness and some random luck
    selected = select_salesmen(ranked)
    #Mate the selected individuals
    selected_routes = [current_gen[i].route for i in selected]
    children_routes = mate_salesmen(selected_routes)
    #Random Mutation
    next_gen_routes = mutate_salesmen(children_routes)
    #Filter out the duplicate city from making the path circulary earlier
    for i, route in enumerate(next_gen_routes):
        next_gen_routes[i] = list(set(route))
    return next_gen_routes

def main():
    cities = generate_cities()
    routes = [random.sample(cities, len(cities)) for _ in range(ELITE_SIZE+MATE_SIZE)]
    for i in range(GENERATIONS):
        print("Generation: %d" %(i))
        salesmen = generate_salesmen(routes)
        routes = get_next_generation(salesmen)

    #Final Salesmen to evaluate algorithm performance with
    salesmen = generate_salesmen(routes)
    print(sum(salesman.distance for salesman in salesmen) / len(salesmen))

if __name__ == '__main__':
    main()

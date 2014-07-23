#-------------------------------------------------------------------------------
# Name: cs-qsar-ann-modeling-research
# Purpose: Container module for evolutionary algorithm-based feature selection
# module.
#
# Created: 20140704
# Copyright: (c) 2014
# Licence: MIT
#-------------------------------------------------------------------------------

import numpy as np
import math
import random
from deap import base
from deap import creator
from deap import tools
from deap import *

def createIndividualAttribute():
    '''
    Creates the individual as a mask for model training.
    '''
    eps=0.015
    randChromosomeVal = 0;
    r = random.uniform(0,1)
    if (r < eps):
            randChromosomeVal = 1
    else:
            randChromosomeVal = 0
    return randChromosomeVal;

def evalGAOneMax(individual):
    '''Example evaluation function from DEAP'''
    return sum(individual),

class EA:
    '''Class container for an evolutionary algorithm'''
    def __init__(self
                 ,featureDescriptors = 396
                 ,populationsize     = 100
                 ,cross_prob         = .5
                 ,mut_prob           = .5
                 ,num_gen            = 5
                 ,crosschance        = .5
                 ,mutchance          = .5
                 ,gen_max            = 100):
        self.featureDescriptors = featureDescriptors;
        self.populationsize     = populationsize;
        #Fitness
        creator.create("FitnessMax", base.Fitness, weights=(1.0,));
        self.toolbox = base.Toolbox();
        # Register attribute
        self.toolbox.register("attr_bool", createIndividualAttribute);

        # Register agent definition as a list representing the binary
        # feature mask
        creator.create("Individual", list, fitness=creator.FitnessMax);

        # Register agent creation
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                    self.toolbox.attr_bool, featureDescriptors);
        
        # Register population creation
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual);

        # Initialize population
        self.gaPop = self.toolbox.population(populationsize);

        # Cross-over, mutation probability & number of generations, respectively
        self.cross_prob = cross_prob;
        self.mut_prob   = mut_prob;
        self.num_gen    = num_gen;
        self.crosschance= crosschance;
        self.mutchance  = mutchance;
        self.gen_max    = gen_max;

        """
        Evolution components
        """
        self.toolbox.register("evaluate",evalGAOneMax);
        # configure a two-point cross-over
        self.toolbox.register("mate", tools.cxESTwoPoint);
        # configure a .05 probability of mutation
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutchance);
        self.toolbox.register("select", tools.selTournament, tournsize = 3);

        # Collect population's initial fitness'
        self.fitnesses  = list(map(self.toolbox.evaluate,self.gaPop));

    def evolve(self):
        '''Evolves the feature descriptor selection mask with genetic prpogramming'''
        try:
            print("Start of GA evolution");
            # Evaluate the entire population
            for ind, fit in zip(self.gaPop, self.fitnesses):
                ind.fitness.values = fit;
            print("  Evaluated %i individuals" % len(self.gaPop));

            # Begin the evolution
            for g in xrange(self.num_gen):
                print("-- Generation %i --" % g);
                # Select the next generation individuals
                offspring = self.toolbox.select(self.gaPop, len(self.gaPop));
                # Clone the selected individuals
                offspring = list(map(self.toolbox.clone, offspring));

                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.Random().random < self.cross_prob:
                        self.toolbox.mate(child1, child2);
                        del child1.fitness.values;
                        del child2.fitness.values;

                for mutant in offspring:
                    if random.Random().random < self.mut_prob:
                        self.toolbox.mutate(mutant);
                        del mutant.fitness.values;

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses   = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit;

                print("  Evaluated %i individuals" % len(invalid_ind));
                # The population is entirely replaced by the offspring
                self.gaPop[:] = offspring;
                # Gather all the fitnesses in one list and print the stats
                fits   = [ind.fitness.values[0] for ind in self.gaPop]
                length = len(self.gaPop)
                mean   = sum(fits) / length
                sum2   = sum(x*x for x in fits)
                std    = abs(sum2 / length - mean**2)**0.5
                print("  Min %s" % min(fits))
                print("  Max %s" % max(fits))
                print("  Avg %s" % mean)
                print("  Std %s" % std)
            print("-- End of (successful) evolution --")
            best_ind = tools.selBest(self.gaPop, 1)[0]
            print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            return 0;
        except:
            print "error evolving feature parameterss"

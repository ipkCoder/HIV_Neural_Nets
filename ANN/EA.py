#-------------------------------------------------------------------------------
# Name: cs-qsar-ann-modeling-research
# Purpose: Container module for evolutionary algorithm-based feature selection
# module.
#
# Created: 20141704
# Copyright: (c) CSUSM 2014
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

def mainGAEvolve(eaPopulation):
    '''Evolves the feature descriptor mask with genetic prpogramming'''
    try:
        print("Start of GA evolution");
        # Evaluate the entire population
        for ind, fit in zip(eaPopulation.gaPop, eaPopulation.fitnesses):
            ind.fitness.values = fit;
        print("  Evaluated %i individuals" % len(eaPopulation.gaPop));

        # Begin the evolution
        for g in xrange(eaPopulation.num_gen):
            print("-- Generation %i --" % g);
            # Select the next generation individuals
            offspring = eaPopulation.toolbox.select(eaPopulation.gaPop, len(eaPopulation.gaPop));
            # Clone the selected individuals
            offspring = list(map(eaPopulation.toolbox.clone, offspring));

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.Random().random < eaPopulation.cross_prob:
                eaPopulation.toolbox.mate(child1, child2);
                del child1.fitness.values;
                del child2.fitness.values;

        for mutant in offspring:
            if random.Random().random < eaPopulation.mut_prob:
                eaPopulation.toolbox.mutate(mutant);
                del mutant.fitness.values;

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses   = map(eaPopulation.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit;

        print("  Evaluated %i individuals" % len(invalid_ind));
        
        # The population is entirely replaced by the offspring
        eaPopulation.gaPop[:] = offspring;
        # Gather all the fitnesses in one list and print the stats
        fits   = [ind.fitness.values[0] for ind in eaPopulation.gaPop]
        length = len(eaPopulation.gaPop)
        mean   = sum(fits) / length
        sum2   = sum(x*x for x in fits)
        std    = abs(sum2 / length - mean**2)**0.5
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print("-- End of (successful) evolution --")
        best_ind = tools.selBest(eaPopulation.gaPop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        return 0;
    except:
        print "error evolving feature parameters"

class EA:
    '''Class container for an evolutionary algorithm'''
    def __init__(this
                 ,featureDescriptors = 396
                 ,populationsize = 100
                 ,cross_prob = .5
                 ,mut_prob  = .5
                 ,num_gen = 5
                 ,crosschance  = .5
                 ,mutchance  = .5
                 ,gen_max = 100):
        this.featureDescriptors = featureDescriptors
        this.populationsize     = populationsize
        #Fitness
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        this.toolbox = base.Toolbox();
        #Individual Attribute
        this.toolbox.register("attr_bool", createIndividualAttribute)

        # Define an individual as a list representing the binary
        # feature mask
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Individual Creation
        this.toolbox.register("individual", tools.initRepeat, creator.Individual,
                    this.toolbox.attr_bool, featureDescriptors)
        
        #Population Creation
        this.toolbox.register("population", tools.initRepeat, list, this.toolbox.individual)

        # Initialize population
        this.gaPop = this.toolbox.population(populationsize)

        # Cross-over, mutation probability & number of generations, respectively
        this.cross_prob = cross_prob
        this.mut_prob   = mut_prob
        this.num_gen    = num_gen
        this.crosschance= crosschance
        this.mutchance  = mutchance
        this.gen_max    = gen_max

        """
        Evolution components
        """
        this.toolbox.register("evaluate",evalGAOneMax)
        # configure a single-point cross-over
        this.toolbox.register("mate", tools.cxOnePoint)
        # configure a .05 probability of mutation
        this.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        this.toolbox.register("select", tools.selBest)

        # Collect population's initial fitness'
        this.fitnesses  = list(map(this.toolbox.evaluate,this.gaPop))

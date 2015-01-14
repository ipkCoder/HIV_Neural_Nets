import time                 # provides timing for benchmarks
from numpy import *        # provides complex math and array functions
import csv
import math
import sys

#Local files
from qsarHelpers import Timer
import ANN
import FromDataFile_ANN_DE_BPSO
import FromFitnessFile_ANN_DE_BPSO

'''
* Purpose: create file to hold results
* Return: file created
'''
def createAnOutputFile():
    try:
        file_name = None
        algorithm = None
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if ( (file_name == None) and (algorithm != None)):
            file_name = "{}_{}_gen{}_{}.csv".format(alg.__class__.__name__,
                            alg.model.__class__.__name__, alg.gen_max,timestamp)
        elif file_name==None:
            file_name = "{}.csv".format(timestamp)
        fileOut = open(file_name, 'wb', 0)
        fileW = csv.writer(fileOut)
            
        try:
            fileW.writerow(['Model', 'Generation', 'Individual', 'Descriptor ID', 'No. Descriptors', 'Fitness', \
                'R2Pred_Train', 'R2Pred_Validation', 'R2Pred_Test','SDEP_Train', 'SDEP_Validation', \
                'SDEP_Test', 'y_Train', 'yHat_Train', 'y_validation', 'yHat_validation','y_Test', \
                'yHat_Test', 'In to Hidden Weights', 'In to Out Weights'])

        except:
            print("couldn't write to file")
        return fileW;
    except:
        print "error creting outout file"

'''
Purpose: determine which features to select (1) and not to select (0)
* @param: population (really just need the number of features)
* @param: epsilon - probability of selecting a feature
'''
def getAValidRow(population, epsilon=0.015):
  
    numOfFea = population.shape[1] # number of features (columns)
    features = zeros(numOfFea) # bool vector (which features were selected/not)

    #The following ensure that at least 3 features are selected in each population
    sum = 0;
    while (sum < 3):
       for j in range(numOfFea):
          r = random.uniform(0,1)
          if (r < epsilon):
             features[j] = 1
          else:
             features[j] = 0
       sum = features.sum()
    
    return features

'''
Purpose: create initial velocity matrix (used to determine an individual's change in feature space)
* @param: numOfIndividuals - number of individuals in the population
* @param: numOfFeatures - number of features (in dataset) each individual should represent
* @return: velocity matrix
'''
def createInitVelMat(numOfIndividuals,numOfFeatures):
    try:
        velocity = random.random((numOfIndividuals,numOfFeatures)) # random numbers between 0 and 1
        return velocity
    except:
        print "error creating intitial velocity matrix."

'''
Purpose: create initial population matrix
* @param: numOfIndividuals - number of individuals in the population
* @param: numOfFeatures - number of features (in dataset) each individual should represent
* @return: population matrix
'''
def createInitPopMat(numOfIndividuals, numOfFeatures):
    
    try:
        population = zeros((numOfIndividuals,numOfFeatures))
        
        for i in range(numOfIndividuals):

            row = getAValidRow(population)

            while (not theRowIsUniqueInPop(i,row, population)):
                row = getAValidRow(population)
            
            for j in range(numOfFeatures):
               population[i][j] = row[j]

        return population
    except:
        print "error getting initial population matrix"
        
'''
Purpose: copy given row
* @param: current best row (individual)
* @return: global best row
'''
def InitializeGlobalBestRow(populationRow):
    try:
        numOfFea      = populationRow.shape[0]
        globalBestRow = zeros(numOfFea)
        for j in range(numOfFea):
            globalBestRow[j] = populationRow[j]

        return globalBestRow;
    except:
        print "error initializing global best row"

'''
Purpose: copy given population
* @param: population to copy
* @return: copy of population
'''
def CreateInitialLocalBestMatrix(population):
    try:
        numOfPop        = population.shape[0]
        numOfFea        = population.shape[1]
        localBestMatrix = zeros((numOfPop, numOfFea))
        for i in range(numOfPop):
            for j in range(numOfFea):
                localBestMatrix[i][j] = population[i][j]
        return localBestMatrix;
    except:
        print "error creating initial local best matrix."

'''
Purpose: copy given fitness
* @param: fitness to copy
* @return: copy of population
'''
def CreateInitialLocalBestFitness(fitness):
    try:
        numOfPop         = fitness.shape[0]
        localBestFitness = zeros(numOfPop)
        for i in range(numOfPop):
            localBestFitness[i] = fitness[i]
    
        return localBestFitness;
    except:
        print "error creating initial local best fitness"


'''
* Purpose: calculate fitness of an individual
* @param: model - model used for prediction
* @param: features - vector of features selected and not selected
* @param: TrainX - training set data
* @param: TrainY - training set target values
* @param: ValidateX - validation set data
* @param: ValidateY - validation set target values
* @return: fitness (how well individual makes predictions)
'''
def findFitnessOfARow(model, features, TrainX, TrainY, ValidateX, ValidateY):
 
    # select indexes of selected features
    xi = FromFitnessFile_ANN_DE_BPSO.OnlySelectTheOnesColumns(features)

    # select features from samples (using indexe of selected features)
    X_train_masked = TrainX.T[xi].T
    X_validation_masked = ValidateX.T[xi].T

    # create neural network to fit new set of features
    model.create_network(X_train_masked.shape[1])

    # train new model
    try:
        # train model using selected data features (return name of model used)
        model_desc = model.train(X_train_masked, TrainY, X_validation_masked, ValidateY)
    except:
        print "Error training in findFitnessOfARow"

    # uncomment for cross validation
    #Yhat_cv = FromFitnessFile_ANN_DE_BPSO.cv_predict(X_train_masked, TrainY,
    #                                                 X_validation_masked, ValidateY, model)
    Yhat_train      = model.predict(X_train_masked)
    Yhat_validation = model.predict(X_validation_masked)

    Y_fitness = append(TrainY, ValidateY)
    Yhat_fitness = append(Yhat_train, Yhat_validation) # for cross validation, use Yhat_cv instead of Yhat_Train
    fitness = FromFitnessFile_ANN_DE_BPSO.calc_fitness(xi, Y_fitness, Yhat_fitness, c=2)

    return fitness

'''
* Purpose: are two vectors the same? (could also use np.array_equal(v1, v2)
'''
def equal(V1, V2):

   if (V1.shape[0] != V2.shape[0]): # same number of features?
       return 0

   for i in range(V1.shape[0]):
       if (V1[i] != V2[i]): # if not the same, return false
           return 0
   return 1 # all the same

'''
Purpose: makes sure a row is not already in population
* @param: RowI - check rows in population up to this row number
* @param: population - population matrix with rows to check
* @param: - vector of values (compare against population row values)
'''
def theRowIsUniqueInPop(RowI, V, population):

    unique = 1
    # compae V to each individual before it
    for i in range(RowI):
        if (equal (V, population[i])):
            return (not unique)
    
    return unique

'''
Purpose: find the global best fitness and individual
* @param: population - population matrix
* @param: fitness - vector of individual fitnesses in population
* @param: globalBestRow - individual that produced best fitness over all generations
* @param: globalBestFitness - best fitness found so far over all generations
* @return: global best fitness and individual
'''
def findGlobalBest(population, fitness, globalBestRow, globalBestFitness):
    try:
        numOfPop = population.shape[0]
        numOfFea = population.shape[1]
        
        min = argmin(fitness) # index of min fitness

        # get global best fitness and row
        if (fitness[min] < globalBestFitness):
            globalBestFitness = fitness[min]
            for j in range(numOfFea):
                globalBestRow[j] = population[min][j]
                
        return globalBestRow, globalBestFitness;
    except:
        print "error finding global best";

'''
Purpose: find local best fitness and matrix (population)
* @param: population (matrix) - matrix of individuals (vectors)
* @param: fitness (array)- vector of individual fitnesses in population
* @param: localBestFitness (array) - best fitness of each line of descendents (index in population)
* @param: localBestMatrix (matrix) - population of best individual in each line of descendents
* @param: local bes fitness and matrix
'''
def findLocalBestMatrix (population, fitness, localBestFitness, localBestMatrix):
  
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    
    # for each individual in the population
    for i in range(numOfPop):
        # if the individual's fitness is less (better) than the best fitness of
        # one of the indiviual's ancestors (same position in a previous generation)
        if (fitness[i] < localBestFitness[i]):
            # update individual's local best fitness and local population matrix
            localBestFitness[i] = fitness[i]
            for j in range(numOfFea):
                localBestMatrix[i][j] = population[i][j]

    return localBestFitness, localBestMatrix

'''
Purpose: update velocity of an individual used to update individual's position in feature space
* @param: velocity - how fast to move individuals
* @param: population - matrix of individuals (vectors)
* @param: localBestMatrix (matrix) - population of best individual in each line of descendents
* @param: globalBestRow - individual that produced best fitness over all generations
* @param: globalBestFitness - best fitness found so far over all generations
* @param: updated velocity
'''
def findVelocity(velocity, population, localBestMatrix, globalBestRow, globalBestFitness):
  
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    
    c1 = 2
    c2 = 2
    inertiaWeight = 0.9 # This is considered as costant inertia
    
    for i in range(numOfPop):
        for j in range(numOfFea): 
            term1 = c1 * random.random() * (localBestMatrix[i][j]- population[i][j])
            term2 = c2 * random.random() * (globalBestRow[j] - population[i][j])
            velocity[i][j] = (inertiaWeight * velocity[i][j]) + term1 + term2    

    return velocity

'''
Purpose: check if row vector is representative of an individual in population
* @param: row - vetcor (individual) to check if already in population
* @param: parentPop - population to check against
* @return: true if exists in population, else false
'''
def rowExistInParentPop(row, parentPop):
    numOfPop = parentPop.shape[0]
    for i in range(numOfPop):
        if (equal (row, parentPop[i])):
            return 1
    return 0

'''
Purpose: create new individual from an individual and provisional individual (additional mutation)
* @param: P - individual to change
* @param: V - provisional individual mutated from individuals in population
* @param: model - model used for prediction
* @param: TrainX - training set data
* @param: TrainY - training set target values
* @param: ValidateX - validation set data
* @param: ValidateY - validation set target values
* @return: new individual
'''
def crossover(P, V, model, TrainX, TrainY, ValidateX, ValidateY ):
  
    '''Core DE algoritm finds a candidate solution(agent) among the population.'''
    numOfFea = P.shape[0]
    U = zeros(numOfFea)

    CR_rate = 0.8 #it is a common value when we do DE algorithm
    
    for j in range(numOfFea):
        if (random.uniform(0, 1) < CR_rate):
            U[j] = P[j]
        else:
            U[j] = V[j]

    fitnessU = findFitnessOfARow(model, U, TrainX, TrainY, ValidateX, ValidateY)
    fitnessV = findFitnessOfARow(model, V, TrainX, TrainY, ValidateX, ValidateY)
    
    if (fitnessU < fitnessV): # pick better mutation
        return U, fitnessU
    else:
        return V, fitnessV 

'''
Purpose: (DE/rand/1) Creates a mutant variant feature descriptor row vector 
         from a set of three mutually distinct candidate row vectors.
* @param: V1 - row vector one
* @param: V2 - row vector two
* @param: V3 - row vector three
* @param: F - controls the length of the exploration vector (V2 - V1)
* @return: mutant individual
'''
def mutate(V1, V2, V3, F = 0.5):
    numOfFea = V1.shape[0]
    # it is a common value when we do DE algorithm.
    # Formally 'F' is defined as the differential weight,
    # a tuning parameter, i.e. a constant, applied in the context of DE 
    # crossover and mutation functions.
    V        = zeros(numOfFea)
    for i in range (numOfFea):
        V[i] = V3[i] + (F *(V2[i] - V1[i]))
    return V;

'''
Purpose: select random row (individual) from population
* @param: population - current population
* @return: random individual
'''
def selectARowFromPopulation(population):
  
   numOfPop = population.shape[0]
   numOfFea = population.shape[1]
   
   # select random row index
   r        = int(random.uniform(1, population))
   # copy row and return
   anyRow   = zeros(numOfFea)
   for j in range(numOfFea):
       anyRow[j] = population[r][j]
   
   return anyRow

'''
Purpose: select three different random rows (individual)s from population
* @param: population - current population
* @return: three random individuals
'''
def selectThreeRandomRows(parentPop):
  
    numOfPop = parentPop.shape[0]
    numOfFea = parentPop.shape[1]
    
    V1 = selectARowFromPopulation(parentPop)
    V2 = selectARowFromPopulation(parentPop)
    V3 = selectARowFromPopulation(parentPop)

    #The following section ensures that the rows are not the same
    while (equal(V1,V2)):
        V2 = selectARowFromPopulation(parentPop)
    while (equal(V3,V1)) or (equal(V3,V2)):
        V3 = selectARowFromPopulation(parentPop)
    
    return V1, V2, V3;

'''
Purpose: find new individual for next generation
* @param: rowI - individual index in population to mutate
* @param: parentPop - population being modified
* @param: fitness - vector of individual fitnesses in population
* @param: model - model used for prediction
* @param: TrainX - training set data
* @param: TrainY - training set target values
* @param: ValidateX - validation set data
* @param: ValidateY - validation set target values
* @return:
'''
def findTheRightVector(rowI, parentPop, fitness, model, TrainX, TrainY, ValidateX, ValidateY):
  
    numOfPop = parentPop.shape[0] # why have, not used ???
    numOfFea = parentPop.shape[1]
    P        = zeros(numOfFea)
    U        = zeros(numOfFea)
    fitnessP = fitness[rowI]
    fitnessU = 0
    
    # individual to mutate
    P = [x for x in parentPop[rowI]]

    # find new row combination by using mutation function on three randowm rows
    # and using cross over function. If new row has lower/better fitness,  
    # return new row, else return old row
    while (U.sum() < 3):
        # The provisional individual (x'_i) is determined by three 
        # mutually distinct individuals V1(x_r), V2(x_s) and V3(x_t) from the previous 
        # generation.
        V1, V2, V3   = selectThreeRandomRows(parentPop)
        # The scaling factor 'F' controls the length of the
        # exploration vector (x_r - x_s)
        provisional  = mutate(V1, V2, V3, F = 0.5)
        U, fitnessU  = crossover(P, provisional, model, TrainX, TrainY, ValidateX, ValidateY)
  
    if (fitnessU < fitnessP): #choose better same individual or mutated
        return U
    else:
        return P

'''
Purpose: copy individual (array)
'''
def getPopulationI(parentPopI):
    numOfFea = parentPopI.shape[0]
    popI = zeros(numOfFea)
    for j in range(numOfFea):
        popI[j] = parentPopI[j]
    return popI # [x for x in parentPopI]

'''
Purpose: find and return individual (vector) with minimum fitness
* @param: population - population to search
* @param: fitness - vector of individual fitnesses in population
* @return: individual with min fitness
'''
def theVecWithMinFitness(fitness, population):
  
    min_fitness = fitness.min()
    
    numOfIndividuals = population.shape[0]
    
    for i in range(numOfIndividuals):
        if (fitness[i] == min_fitness):
            return population[i]

#    return population[argmin(fitness)]

'''
Purpose: create new population and initialize first 20% plus one more (best from previous population)
'''
def getTheBestRowAndThreeRandomRows(fitness, parentPop):

    numOfPop = parentPop.shape[0]
    numOfFea = parentPop.shape[1]

    # new population
    population = zeros((numOfPop, numOfFea))
    
    # The following moves the best row from the parent population 
    # and move it to be the first row of the current population
    
    # population[0] = [for row in V]
    V = theVecWithMinFitness(fitness, parentPop)
    for j in range(numOfFea):
        population[0][j] = V[j]
        
    #It seems that this program converge very fast. What we can do is
    #to distract about 20% of the birds to fly in different direction than
    #where they think the food is
    
    num = int(numOfPop * 0.2)
    for i in range(1,num):
        V = getAValidRow(population)
        while (V.sum() < 3) or (rowExistInParentPop(V, parentPop)):
            V = getAValidRow(population)
        for j in range(numOfFea):
            population[i][j] = V[j]
            
    return population

'''
Purpose: create and initialize new population
* @param: model - model used for prediction
* @param: alpha - TODO
* @param: beta - TODO
* @param: fitness - vector of individual fitnesses in population
* @param: parentPop - matrix of individuals from previous generation
* @param: population - matrix of individuals
* @param: localBestMatrix (matrix) - population of best individual in each line of descendents
* @param: globalBestRow - individual that produced best fitness over all generations
* @param: TrainX - training set data
* @param: TrainY - training set target values
* @param: ValidateX - validation set data
* @param: ValidateY - validation set target values
* @return: new population
'''
def findNewPopulation(model, alpha, beta, fitness, velocity, parentPop,\
                        population,localBestMatrix, globalBestRow, \
                        TrainX, TrainY, ValidateX, ValidateY):

    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    p        = (.5) * (1+alpha)      # used to reflect local best position
    q        = (1 - beta)            # used to reflect global best position
    F        = 0.5                   # used in DE/rand/1

    # put best row from last population matrix into new population and
    # randomly select new rows for 20% of new population
    population = getTheBestRowAndThreeRandomRows(fitness, parentPop)
  
    num = int(numOfPop * 0.2)
    for i in range(num, numOfPop):
        
        popI           = getPopulationI(parentPop[i]) # TODO - confirm this isn't used
        
        # find vector based on mutation and crossover of population row
        error = 0
        try:
            with Timer() as t:
                theRightVector = findTheRightVector(i, parentPop, fitness, \
                                     model, TrainX, TrainY, ValidateX, ValidateY )
        except:
            print("Error finding new individual")
            error = 1
        finally:
            if not error:
                pass
                #print("Found individual {} in {:.03f} min".format(i, (t.interval/60)))

        
        # find new population row column values
        for j in range(numOfFea):
            if (velocity[i][j] > 0) and (velocity[i][j] <= alpha):
                population[i][j] = theRightVector[j]
            elif (velocity[i][j] > alpha) and (velocity[i][j] <= p):
                population[i][j] = localBestMatrix[i][j]
            elif (velocity[i][j] > p) and (velocity[i][j] <= q):
                population[i][j] = globalBestRow[j]
            elif (velocity[i][j] > q) and (velocity[i][j] <= 1):
                population[i][j] = (1 - theRightVector[j])
            else:
                population[i][j] = parentPop[i][j]
        
        # The following ensure that we still have a vector that some of
        # features still have been selected. So zero vector or the ones
        # with less than 1% is not good
        V = zeros(numOfFea)
        for j in range(numOfFea):
            V[j] = population[i][j]
        
        # check to make sure new row has at least three selected columns,
        # is unique in current population so far (not duplicated), and was not
        # in parent population
        unique = 1
        while (V.sum() < 3) or (not theRowIsUniqueInPop(i, V, population)) or (rowExistInParentPop(V, parentPop)):
            row = getAValidRow(population)
            for j in range(numOfFea):
              population[i][j] = row[j]

    return population

'''
Purpose: create copy of population
* @param: population - population to copy
* @return: numpy array copy of population
'''
def getParentPopulation(population):
    try:
        return array([[x for x in row] for row in population])
    except:
        print "error getting parent population"
    
'''
Purpose: decide whether the program should be terminated
         if the best fitness has not improved in the last 30 generations, terminate program
* @param: Times - the number of times the fitness has stayed the same so far
* @param: oldFitness - previous best fitness
* @param: globalBestFitness - current best fitness
'''
def checkterTerminationStatus(Times, oldFitness, globalBestFitness):
    # if 30 times, terminate
    if (Times == 30):
        print "***** No need to continue. The fitness not changed in the last 30 generation"
        exit(0)
    # if old is same as current, increase count (same fitness)
    elif (oldFitness == globalBestFitness):
        Times = Times + 1
    # if current best is less than old, set old to current best fitness, restart Times
    elif (globalBestFitness < oldFitness):
        oldFitness = globalBestFitness
        Times = 0
        print "\n***** time is = ", time.strftime("%H:%M:%S", time.localtime())
        print "******************** Times is set back to 0 ********************\n"
    return oldFitness, Times

'''
Purpose: evolve population over generations to find best model (individual)
* @param: model - model used for prediction
* @param fileW - file containing results
* @param: fitness - how well individual makes predictions
* @param: velocity - used to determine an individual's change in feature space
* @param: population - matrix of individuals (vectors)
* @param: parentPop - previous population
* @param: localBestFitness - TODO
* @param: localBestMatrix - TODO
* @param: globalBestRow - individual that produced best fitness over all generations
* @param: globalBestFitness - best fitness found so far over all generations
* @param: TrainX - training set data
* @param: TrainY - training set target values
* @param: ValidateX - validation set data
* @param: ValidateY - validation set target values
'''
def IterateNtimes(model, fileW, fitness, velocity, population, parentPop,
                  localBestFitness,localBestMatrix, globalBestRow, \
                  globalBestFitness, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    
    '''Performs core BPSO functions'''
    
    numOfGenerations = 2000            # should be 2000
    
    alphaStarts      = 0.5
    alphaEnds        = 0.33
    alphaDecre       = (alphaStarts - alphaEnds) / numOfGenerations
    
    betaStarts       = 0
    betaEnds         = 0.05
    betaIncre        = (betaEnds - betaStarts)/numOfGenerations

    beta             = betaStarts
    alpha            = alphaStarts
        
    oldFitness       = globalBestFitness
    Times            = 0 # best fitness hasn't been the same
    
    #############################################################
    
    # loop for n generations
    for i in range(numOfGenerations):
        try:
            with Timer() as t0:

                # terminate if golbalBestFitness hasn't changed in 30 generations
                oldFitness, Times = checkterTerminationStatus(Times, oldFitness, globalBestFitness)
                print "This is iteration {}, Global best fitness is: {}".format(i,globalBestFitness); 
                
                unfit         = 1000
                fittingStatus = unfit
                # find new population matrix and fitness vector
                while (fittingStatus == unfit):
                    try:    
                        print("Finding new population")
                        with Timer() as t1:
                            # new population
                            population = findNewPopulation(model, alpha, beta, fitness, velocity, parentPop,\
                                             population,localBestMatrix, globalBestRow, \
                                             TrainX, TrainY, ValidateX, ValidateY)
                    finally:
                        print "--------- New pop found in {} min --------".format((t1.interval/60))
            
                    try:
                        with Timer() as t1:
                            # fitness of new population
                            fittingStatus, fitness = FromFitnessFile_ANN_DE_BPSO.validate_model(i+1, model,fileW, \
                                        population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
                    finally:
                        if fittingStatus == unfit:
                            print("Error validating model: finding new population to validate.")
                        else:
                            print "Validated model in {} min".format((t1.interval/60))      
                
                #remember current population matrix
                parentPop                         = getParentPopulation(population)
                # find new global best row and fitness
                globalBestRow, globalBestFitness  = findGlobalBest(population, fitness, globalBestRow, globalBestFitness)
                # if current population has lower fitness associated with it than
                # local best matrix, remember current population as local best matrix
                localBestFitness, localBestMatrix = findLocalBestMatrix(population, fitness, localBestFitness, localBestMatrix)
                # update velocity information
                velocity                          = findVelocity(velocity, population, localBestMatrix, globalBestRow, globalBestFitness)
                alpha                             = alpha - alphaDecre
                beta                              = beta + betaIncre
                end_time                          = time.time()
        finally:
            print "Generation {} complete: {} min".format(i, (t0.interval/60))
    return

'''
Purpose: run DE-BPSO/ANN
'''
def main():
    
    print "********** start time is = ", time.strftime("%H:%M:%S", time.localtime())

    try: # load data
        with Timer() as t:
            
            fileW      = createAnOutputFile();
            model      = ANN.ANN();             # artificial neural network model
            
            numOfPop   = 50  # should be 50 population
            numOfFea   = 385  # old data (396), new data (385)
            unfit      = 1000 
            
            # Final model requirements
            R2req_train    = .6
            R2req_validate = .5
            R2req_test     = .5
            
            # get training, validation, test data and rescale
            TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFile_ANN_DE_BPSO.getAllOfTheData()
            TrainX, ValidateX, TestX                           = FromDataFile_ANN_DE_BPSO.rescaleTheData(TrainX, ValidateX, TestX)
            
            print TrainX.shape
            print TrainY.shape
    finally:
        print( "Time to load and rescale data: {:.03f} sec".format(t.interval) )
    
    velocity      = createInitVelMat(numOfPop, numOfFea)
    unfit         = 1000
    fittingStatus = unfit
    
    try: # create population and train models on individuals
        with Timer() as t:
            while (fittingStatus == unfit):
                # create inititial population and find fitness for each row in population
                population             = createInitPopMat(numOfPop, numOfFea)
                fittingStatus, fitness = FromFitnessFile_ANN_DE_BPSO.validate_model(0, model,fileW, population, 
                    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
    finally:
        print "Initialized population and validated model: {} min".format((t.interval/60))

    try: # initialize local and global best fitness, matrix
        with Timer() as t:
            # initialize global best row and fitness to first population row
            globalBestRow                    = InitializeGlobalBestRow(population[0])
            globalBestFitness                = fitness[0]
            # find actual global best row and fitness
            globalBestRow, globalBestFitness = findGlobalBest(population,fitness, globalBestRow,globalBestFitness)
            # initialze local best matrix (Pid) with current population matirix
            # initialize local best fitness with current fitness vector
            localBestMatrix                  = CreateInitialLocalBestMatrix(population)
            localBestFitness                 = CreateInitialLocalBestFitness(fitness)
            # parent population is current population
            parentPop = getParentPopulation(population)
    finally:
        print( "Time to initialize data: {:.03f} sec".format(t.interval) )

    print "Starting iteration loop at ", time.strftime("%H:%M:%S", time.localtime())
    IterateNtimes(model, fileW, fitness, velocity, population, parentPop,
                  localBestFitness,localBestMatrix, globalBestRow,
                  globalBestFitness, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
#------------------------------------------------------    
# #main program starts in here
main()

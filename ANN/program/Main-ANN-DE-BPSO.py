import time                 # provides timing for benchmarks
from numpy  import *        # provides complex math and array functions
import csv
import math
import sys
from qsarHelpers import Timer

#Local files created by me
import ANN
import FromDataFileMLR_DE_BPSO
import FromFinessFileMLR_DE_BPSO
#------------------------------------------------------
def getTwoDecPoint(x):
    return float("%.2f"%x)

#------------------------------------------------------
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
                'R2Pred_Train', 'Q2_Train', 'R2Pred_Validation', 'R2Pred_Test','SDEP_Train', 'SDEP_Validation', \
                'SDEP_Test', 'y_Train', 'yHat_Train', 'yHat_CV', 'y_validation', 'yHat_validation','y_Test', \
                'yHat_Test', 'In to Hidden Weights', 'In to Out Weights'])

        except:
            print("couldn't write to file")
        return fileW;
    except:
        print "error creting outout file"
#------------------------------------------------------------------------------

# calculate fitness of a new potential individual

def findFitnessOfARow(model, vector, TrainX, TrainY, ValidateX, ValidateY):

    xi = FromFinessFileMLR_DE_BPSO.OnlySelectTheOnesColumns(vector)

    # select features from samples
    X_train_masked = TrainX.T[xi].T
    X_validation_masked = ValidateX.T[xi].T

    # create network to fit new set of features (possible diff number of inputs)
    model.create_network(X_train_masked.shape[1])

    # train new model
    try:
        model_desc = model.train(X_train_masked, TrainY, X_validation_masked, ValidateY)
    except:
        print "Error training in findFitnessOfARow"

    Yhat_cv = FromFinessFileMLR_DE_BPSO.cv_predict(X_train_masked, TrainY, X_validation_masked, ValidateY, model)
    Yhat_validation = model.predict(X_validation_masked)

    Y_fitness = append(TrainY, ValidateY)
    Yhat_fitness = append(Yhat_cv, Yhat_validation)
    fitness = FromFinessFileMLR_DE_BPSO.calc_fitness(xi, Y_fitness, Yhat_fitness, c=2)

    return fitness
#------------------------------------------------------------------------------
def equal(V1, V2):
   numOfFea = V1.shape[0]
   true = 1
   false = 0
   for i in range(numOfFea):
       if (V1[i] != V2[i]):
           return false
   return true

#------------------------------------------------------
def theRowIsUniqueInPop(RowI,V, population):

    unique = 1
    # compae V to each individual before it
    for i in range(RowI):
        if (equal (V, population[i])):
            return (not unique)
    
    return unique

#------------------------------------------------------------------------------
def getAValidRow(population, eps=0.015):
    numOfFea = population.shape[1]
    sum = 0;
    #The following ensure that at least couple of features are
    #selected in each population
    unique = 0 # why have unique, never changes in loop????
    while (sum < 3) and (not unique):
       V = zeros(numOfFea)
       for j in range(numOfFea):
          r = random.uniform(0,1)
          if (r < eps):
             V[j] = 1
          else:
             V[j] = 0
       sum = V.sum()
       
    return V
#------------------------------------------------------------------------------
def createInitVelMat(numOfPop,numOfFea):
    try:
        velocity = random.random((numOfPop,numOfFea))
        for i in range(numOfPop):
            for j in range(numOfFea):
                velocity[i][j] = random.random() # any random number between 0 and 1
        return velocity;
    except:
        print "error creating intitial velocity matrix."
#------------------------------------------------------
def createInitPopMat(numOfPop, numOfFea):
    try:
        population = zeros((numOfPop,numOfFea))
        for i in range(numOfPop):
            V = getAValidRow(population)
            while (not theRowIsUniqueInPop(i,V, population)):
                V = getAValidRow(population)   
            for j in range(numOfFea):
                population[i][j] = V[j]

        return population
    except:
        print "error getting initial population matrix"
#------------------------------------------------------
def findGlobalBest(population, fitness, globalBestRow,globalBestFitness):
    try:
        numOfPop = population.shape[0]
        numOfFea = population.shape[1]
        k = -1
        min = fitness[0]
        # find min fitness and corresponding population row index (k)
        for i in range(numOfPop):
            if (fitness[i] < min):
                min = fitness[i]
                k = i

        # get global best row (combination of feature selections)
        if (min < globalBestFitness):
            globalBestFitness = min
            for j in range(numOfFea):
                globalBestRow[j] = population[k][j]
                
        return globalBestRow, globalBestFitness;
    except:
        print "error finding global best";

#------------------------------------------------------
def findLocalBestMatrix (population, fitness, localBestFitness, localBestMatrix):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    for i in range(numOfPop):
        if (fitness[i] < localBestFitness[i]):
            localBestFitness[i] = fitness[i]
            for j in range(numOfFea):
                localBestMatrix[i][j] = population[i][j]
    return localBestFitness, localBestMatrix

#------------------------------------------------------
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

#------------------------------------------------------
def rowExistInParentPop(V, parentPop):
    numOfPop = parentPop.shape[0]
    for i in range(numOfPop):
        if (equal (V, parentPop[i])):
            return 1
#------------------------------------------------------
def crossover(P, V, model, TrainX, TrainY, ValidateX, ValidateY ):
    '''Core DE algoritm finds a candidate solution(agent) 
        among the population.'''
    numOfFea = P.shape[0]
    U = zeros(numOfFea)

    CRrate   = 0.8 #it is a common value when we do DE algorithm
    for j in range(numOfFea):
        R = random.uniform(0, 1)
        if (R < CRrate):
            U[j] = P[j]
        else:
            U[j] = V[j]

    fitnessU = findFitnessOfARow(model, U, TrainX, TrainY, ValidateX, ValidateY)
    fitnessV = findFitnessOfARow(model, V, TrainX, TrainY, ValidateX, ValidateY)
    
    if (fitnessU < fitnessV):
        return U, fitnessU
    else:
        return V, fitnessV
    
#------------------------------------------------------
def mutate(V1, V2, V3, F = 0.5):
    '''Creates a mutant variant feature descriptor 
    row vector from a set of three mutually distinct 
    candidate row vectors.'''
    numOfFea = V1.shape[0]
    # it is a common value when we do DE algorithm.
    # Formally 'F' is defined as the differential weight,
    # a tuning parameter, i.e. a constant, applied in the context of DE 
    # crossover and mutation functions.
    V        = zeros(numOfFea)
    for i in range (numOfFea):
        V[i] = V3[i] + (F *(V2[i] - V1[i]))
    return V;
#------------------------------------------------------        
def selectARowFromPopulation(parentPop):
   numOfPop = parentPop.shape[0]
   numOfFea = parentPop.shape[1]
   # select random row index
   r        = int(random.uniform(1,numOfPop))
   # copy row and return
   anyRow   = zeros(numOfFea)
   for j in range(numOfFea):
       anyRow[j] = parentPop[r][j]
   return anyRow
#------------------------------------------------------
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

#------------------------------------------------------

# P = individual i from parent population
# V = new individual created from function of three random individuals
# U = individual created from P and V, OR V (whichever has better fitness)

def findTheRightVector(rowI, parentPop, fitness, model, \
                       TrainX, TrainY, ValidateX, ValidateY):
    '''This is the mutation function'''
    numOfPop = parentPop.shape[0] # why have, not used ???
    numOfFea = parentPop.shape[1]
    P        = zeros(numOfFea)
    U        = zeros(numOfFea)
    fitnessP = fitness[rowI]
    fitnessU = 0
    
    # P is same as parent row i
    for j in range(numOfFea):
       P[j] = parentPop[rowI][j]
  
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
        U, fitnessU  = crossover(P, provisional, model,TrainX, TrainY, ValidateX, ValidateY)
  
    if (fitnessU < fitnessP):
        return U
    else:
        return P

#------------------------------------------------------
def getPopulationI(parentPopI):
   numOfFea = parentPopI.shape[0]
   popI = zeros(numOfFea)
   for j in range(numOfFea):
      popI[j] = parentPopI[j]
   return popI

#------------------------------------------------------
def theVecWithMinFitness(fitness, parentPop):
    m = fitness.min()
    numOfPop = parentPop.shape[0]
    for i in range(numOfPop):
        if (fitness[i] == m):
            return parentPop[i]

#------------------------------------------------------
def getTheBestRowAndThreeRandomRows(fitness, parentPop):

    numOfPop = parentPop.shape[0]
    numOfFea = parentPop.shape[1]

    population = zeros((numOfPop, numOfFea))
    
    # The following moves the best row from the parent population 
    # and move it to be the first row of the current population
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
            V1 = getAValidRow(population) # what is V1??? ---------------
        for j in range(numOfFea):
            population[i][j] = V[j]
            
    return population
#------------------------------------------------------

def findNewPopulation(model, alpha, beta, fitness, velocity, parentPop,\
                        population,localBestMatrix, globalBestRow, \
                        TrainX, TrainY, ValidateX, ValidateY):

    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    p        = (.5) * (1+alpha)
    q        = (1 - beta)
    F        = 0.5

    # put best row from last population matrix into new population and
    # randomly select new rows for 20% of new population
    population = getTheBestRowAndThreeRandomRows(fitness, parentPop)
  
    num = int(numOfPop * 0.2)
    for i in range(num, numOfPop):
        
        popI           = getPopulationI(parentPop[i])
        
        # find vector based on mutation and crossover of population row
        theRightVector = findTheRightVector(i, parentPop, fitness, \
                                 model, TrainX, TrainY, ValidateX, ValidateY )
      
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
        unique = 1

        # check to make sure new row has at least three selected columns,
        # is unique in current population so far (not duplicated), and was not
        # in last population (parent population)
        while (V.sum() < 3) or (not theRowIsUniqueInPop(i,V, population) ) or (rowExistInParentPop(V, parentPop)):
            V = getAValidRow(population)
            for j in range(numOfFea):
                population[i][j] = V[j]

    return population

#------------------------------------------------------
def getParentPopulation(population):
    try:
        numOfPop  = population.shape[0]
        numOfFea  = population.shape[1]
        parentPop = zeros((numOfPop, numOfFea))
        for i in range(numOfPop):
            for j in range(numOfFea):
                parentPop[i][j]= population[i][j]
        return parentPop;
    except:
        print "error getting parent population"
    
#------------------------------------------------------
def checkterTerminationStatus(Times, oldFitness, globalBestFitness):
    if (Times == 30):
        print "***** No need to continue. The fitness not changed in the last 30 generation"
        exit(0)
    elif (oldFitness == globalBestFitness):
        Times = Times + 1
    elif (globalBestFitness < oldFitness):
        oldFitness = globalBestFitness
        Times = 0
        print "\n***** time is = ", time.strftime("%H:%M:%S", time.localtime())
        print "******************** Times is set back to 0 ********************\n"
    return oldFitness, Times

#------------------------------------------------------
def IterateNtimes(model, fileW, fitness, velocity, population, parentPop,
                  localBestFitness,localBestMatrix, globalBestRow, \
                  globalBestFitness, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    '''Performs core BPSO functions'''
    numOfGenerations = 2000            # should be 2000
    alphaStarts      = 0.5
    alphaEnds        = 0.33
    betaStarts       = 0
    betaEnds         = 0.05
    beta             = betaStarts
    alpha            = alphaStarts
    alphaDecre       = (alphaStarts - alphaEnds) / numOfGenerations
    betaIncre        = (betaEnds - betaStarts)/numOfGenerations
    oldFitness       = globalBestFitness
    Times            = 0
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
                        with Timer() as t1:
                            population = findNewPopulation(model, alpha, beta, fitness, velocity, parentPop,\
                                             population,localBestMatrix, globalBestRow, \
                                             TrainX, TrainY, ValidateX, ValidateY)
                    finally:
                        print "--------- New pop found in {} min --------".format((t1.interval/60))
            
                    try:
                        with Timer() as t1:
                            fittingStatus, fitness = FromFinessFileMLR_DE_BPSO.validate_model(i+1, model,fileW, \
                                        population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
                    finally:
                        print "Validated model in {} min".format((t1.interval/60))      
                #remember current population matrix
                parentPop                         = getParentPopulation(population)
                # find new global best row and fitness
                globalBestRow, globalBestFitness  = findGlobalBest(population, fitness, globalBestRow, globalBestFitness)
                # if current population has lower fitness associated with it than
                # local best matrix, remember current population as local best matrix
                localBestFitness, localBestMatrix = findLocalBestMatrix(population, fitness, localBestFitness, localBestMatrix)
                # update velocities
                velocity                          = findVelocity(velocity, population, localBestMatrix, globalBestRow, globalBestFitness)
                alpha                             = alpha - alphaDecre
                beta                              = beta + betaIncre
                end_time                          = time.time()
        finally:
            print "Generation {} complete: {} min".format(i, (t0.interval/60))
    return
#------------------------------------------------------
def InitializeGlobalBestRow(populationRow):
    try:
        numOfFea      = populationRow.shape[0]
        globalBestRow = zeros(numOfFea)
        for j in range(numOfFea):
            globalBestRow[j] = populationRow[j]

        return globalBestRow;
    except:
        print "error initializing global best row"

#------------------------------------------------------
# copy population matrix
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

#------------------------------------------------------
# copy fitness vector
def CreateInitialLocalBestFitness(fitness):
    try:
        numOfPop         = fitness.shape[0]
        localBestFitness = zeros(numOfPop)
        for i in range(numOfPop):
            localBestFitness[i] = fitness[i]
    
        return localBestFitness;
    except:
        print "error creating initial local best fitness"

#------------------------------------------------------
def main():
    print "********** start time is = ", time.strftime("%H:%M:%S", time.localtime())
    try:
        with Timer() as t:
            fileW      = createAnOutputFile();
            model      = ANN.ANN();
            numOfPop   = 50  # should be 50 population
            numOfFea   = 385  # old data (396), new data (385)
            unfit      = 1000
            # Final model requirements
            R2req_train    = .6
            R2req_validate = .5
            R2req_test     = .5
            # get training, validation, test data and rescale
            TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR_DE_BPSO.getAllOfTheData()
            TrainX, ValidateX, TestX                           = FromDataFileMLR_DE_BPSO.rescaleTheData(TrainX, ValidateX, TestX)
            
            #TrainX = TrainX[:10]
            #TrainY = TrainY[:10]
            print TrainX.shape
            print TrainY.shape

    
    finally:
        print( "Time to load and rescale data: {:.03f} sec".format(t.interval) )
    
    # initial velocities, numbers between 0 and 1
    velocity      = createInitVelMat(numOfPop, numOfFea)
    unfit         = 1000
    fittingStatus = unfit
    try:
        with Timer() as t:
            while (fittingStatus == unfit):
                # create inititial population and find fitness for each row in population
                population             = createInitPopMat(numOfPop, numOfFea)
                fittingStatus, fitness = FromFinessFileMLR_DE_BPSO.validate_model(0, model,fileW, population, 
                    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
    finally:
        print "Validated model: {} min".format((t.interval/60))

    try:
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
#main program starts in here
main()

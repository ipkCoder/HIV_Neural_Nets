import time                 # provides timing for benchmarks
from numpy  import *        # provides complex math and array functions
#from sklearn import svm	    # provides Support Vector Regression
import csv
import math
import sys
import mlr

#Local files
import FromDataFileMLR_BPSO
import FromFitnessFileMLR_BPSO

#------------------------------------------------------
def getTwoDecPoint(x):
    return float("%.2f"%x)

#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013
			
def writeTheHeader():
    with open('GA-SVR.csv', 'ab') as csvfile:
        modelwriter = csv.writer(csvfile)
        modelwriter.writerow(['Descriptor Ids', 'Num of desc',
                              'Fitness', 'RMSE','TrainR2', 'RMSEValidate',
                              'ValidateR2', 'TestR2', 'Model', 'Localtime'])
#------------------------------------------------------------------------------
def equal (row1, row2):
    numOfFea = row1.shape[0]
    for j in range(numOfFea):
        if (row1[j] <> row2[j]):
            return 0
    return 1

#------------------------------------------------------------------------------
def theRowIsUniqueInPop(RowI,V, population):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    unique = 1
    for i in range(RowI-1):
        for j in range(numOfFea):
            if (equal (V, population[i])):
                return (not unique)
    return unique

#------------------------------------------------------------------------------
def getAValidRow(population, eps=0.015):
    numOfFea = population.shape[1]
    sum = 0;
    #The following ensure that at least couple of features are
    #selected in each population
    unique = 0
    while (sum < 3) and (not unique):
       V = zeros(numOfFea)
       for j in range(numOfFea):
          if (random.uniform(0,1) < eps):
             V[j] = 1
          else:
             V[j] = 0
       sum = V.sum()
       
    return V

#---------------- Initialize --------------------------
def createInitPopMat(numOfPop, numOfFea):
    population = zeros((numOfPop,numOfFea))
    for i in range(numOfPop):
        V = getAValidRow(population)
        while (not theRowIsUniqueInPop(i,V, population)):
            V = getAValidRow(population)   
        for j in range(numOfFea):
            population[i][j] = V[j]

    return population

#---------------- Initialize --------------------------
def createInitVelMat(numOfPop,numOfFea):
    velocity = random.random((numOfPop,numOfFea))
    for i in range(numOfPop):
        for j in range(numOfFea):
            velocity[i][j] = random.random() # any random number between 0 and 1
    return velocity

#-------------- Update Velocity -----------------------
def findVelocity(velocity, population, localBestMatrix, globalBestRow, globalBestFitness):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    c1 = 2
    c2 = 2
    inertiaWeight = 0.9 # Gene has made it to be 1 - but the paper says setting
                        # it to 1 converges very fast that is not a good idea
    for i in range(numOfPop):
        for j in range(numOfFea): 
            term1 = c1 * random.random() * (localBestMatrix[i][j]- population[i][j])
            term2 = c2 * random.random() * (globalBestRow[j] - population[i][j])
            velocity[i][j] = (inertiaWeight * velocity[i][j]) + term1 + term2    
    return velocity

#---------------- Evolution ---------------------------
def findNewPopulation(population, parentPop, velocity, alpha, beta,localBestMatrix,globalBestRow):

    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    p = (.5) * (1+alpha)

    for i in range(numOfPop):
        for j in range(numOfFea): 
            if (velocity[i][j] > 0) and (velocity[i][j] <= alpha):
                population[i][j] = population[i][j]
            elif (velocity[i][j] > alpha) and (velocity[i][j] <= p):
                population[i][j] = localBestMatrix[i][j]
            elif (velocity[i][j] > p) and (velocity[i][j] <= 1):
                population[i][j] = globalBestRow[j]
            else:
                population[i][j] = population[i][j]

        # The following ensure that we still have a vector that some of
        # features still have been selected. So zero vector or the ones
        # with less than 1% is not good
        V = zeros(numOfFea)
        for j in range(numOfFea):
            V[j] = population[i][j]
        sum = population[i].sum()
        unique = 1
        if (sum < 3) or (not theRowIsUniqueInPop(i,V, population) ) or (rowExistInParentPop(V, parentPop)):
            V = getAValidRow(population)
            for j in range(numOfFea):
                population[i][j] = V[j]

    return population

#------------------------------------------------------
def findGlobalBest(population, fitness, globalBestRow,globalBestFitness):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    min = fitness[0]
    k=0
    for i in range(numOfPop):
        if (fitness[i] < min):
            min = fitness[i]
            k = i
            
    if (min < globalBestFitness):
        globalBestFitness = min
        for j in range(numOfFea):
            globalBestRow[j] = population[k][j]
                
    return globalBestRow, globalBestFitness

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
def rowExistInParentPop(V, parentPop):
    numOfPop = parentPop.shape[0]
    for i in range(numOfPop):
        if (equal (V, parentPop[i])):
            return 1

#------------------------------------------------------
def checkterTerminationStatus(Times, oldFitness, globalBestFitness):
    if (Times == 40):
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
def IterateNtimes(model, fileW, fitness, velocity, population, localBestFitness,
                  localBestMatrix, globalBestRow, globalBestFitness,
                  TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):

    numOfGenerations = 2000# it should be 2000
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    unfit = 1000
    alphaStarts = 0.5
    alphaEnds = 0.33
    
    betaStarts = 0
    betaEnds = 0.05

    beta = betaStarts
    alpha = alphaStarts
    alphaDecre = (alphaStarts - alphaEnds) / numOfGenerations
    betaIncre = (betaEnds - betaStarts)/numOfGenerations
    
    oldFitness = globalBestFitness
    Times = 0
    for i in range(numOfGenerations):   
        oldFitness, Times = checkterTerminationStatus(Times, oldFitness, globalBestFitness)
        print "This is generation ", i, "Fitness is: ", globalBestFitness        

        if (fitness.min()<0.005):
            print "***********************************"
            print "Good: Fitness is low enough to quit"
            print "***********************************"
            exit(0)
        fittingStatus = unfit
        parentPop = zeros((numOfPop, numOfFea))
        for j in range(numOfPop):                   # error: had i instead of j  <----
            for k in range(numOfFea):               # error: had j instead of k  <----
                parentPop[j][k] = population[j][k]  # error: had i,j instead of j, k <----

        while (fittingStatus == unfit):
            population = findNewPopulation(population, parentPop, velocity, alpha, beta,
                                        localBestMatrix,globalBestRow) 
            #find the fitness for this iteration
            fittingStatus, fitness = FromFitnessFileMLR_BPSO.validate_model((i+1), model,fileW, population,
                        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

        globalBestRow, globalBestFitness  = findGlobalBest(population, fitness, globalBestRow, globalBestFitness)
        localBestFitness, localBestMatrix = findLocalBestMatrix (population, fitness,localBestFitness, localBestMatrix)
        velocity                          = findVelocity(velocity, population, localBestMatrix,globalBestRow, globalBestFitness)
        alpha                             = alpha - alphaDecre
        beta                              = beta + betaIncre
        
#------------------------------------------------------
def InitializeGlobalBestRowAndFitness(fitness, population):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    globalBestFitness = fitness.min()
    for i in range(numOfPop):
        if (fitness[i] == globalBestFitness):
            globalBestRow = zeros(numOfFea)
            for j in range(numOfFea):
                globalBestRow[j] = population[i][j]
            return globalBestRow, globalBestFitness 

#------------------------------------------------------
def CreateInitialLocalBestMatrix(population):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    localBestMatrix = zeros((numOfPop, numOfFea))
    for i in range(numOfPop):
        for j in range(numOfFea):
            localBestMatrix[i][j] = population[i][j]
    return localBestMatrix

#------------------------------------------------------
def CreateInitialLocalBestFitness(fitness):
    numOfPop = fitness.shape[0]
    localBestFitness = zeros(numOfPop)
    for i in range(numOfPop):
        localBestFitness[i] = fitness[i]
    
    return localBestFitness    

#------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 19, 2013
def createAnOutputFile():
    file_name = None
    algorithm = None
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if ( (file_name == None) and (algorithm != None)):
        file_name = "{}_{}_gen{}_{}.csv".format(alg.__class__.__name__,
                        alg.model.__class__.__name__, alg.gen_max,timestamp)
    elif file_name==None:
        file_name = "{}.csv".format(timestamp)
    fileOut = file(file_name, 'wb')
    fileW = csv.writer(fileOut)
            
    fileW.writerow(['Model', 'Generation', 'Individual', 'Descriptor ID', 'No. Descriptors', 'Fitness', \
                'R2Pred_Train', 'R2Pred_Validation', 'R2Pred_Test','SDEP_Train', 'SDEP_Validation', \
                'SDEP_Test', 'y_Train', 'yHat_Train', 'y_validation', 'yHat_validation','y_Test', \
                'yHat_Test'])
    
    #fileW.writerow(['Descriptor IDi', 'No. oescriptors', 'Fitness', 'Model','R2', 'Q2', \
    #        'R2Pred_Validation', 'R2Pred_Test','SEE_Train', 'SDEP_Validation', 'SDEP_Test', \
    #        'y_Train', 'yHat_Train', 'yHat_CV', 'y_validation', 'yHat_validation','y_Test', 'yHat_Test'])
    
    return fileOut, fileW
#------------------------------------------------------------------------------
#Ahmad Hadaegh: Initial Prog: July 14, 2013
#Ahmad Hadaegh: Modified  on: July 16, 2013
#Ahmad Hadaegh: Modified  on: July 19, 2013

#main program starts in here
def main():
    fileOut, fileW = createAnOutputFile()
    model = mlr.MLR()

    #fileW.writerow("heeeeeeeeeeeeeeeeellllllllllllllllloooooooo")
    numOfPop = 50   # should be 50 population
    unfit = 1000

    # Final model requirements
    R2req_train    = .6
    R2req_validate = .5
    R2req_test     = .5

    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR_BPSO.getAllOfTheData()
    TrainX, ValidateX, TestX                           = FromDataFileMLR_BPSO.rescaleTheData(TrainX, ValidateX, TestX)
  
    numOfFea = TrainX.shape[1]  # should be 385 descriptors (new data, 396 for old)
    print TrainX.shape
    
    unfit = 1000
    fittingStatus = unfit
    velocity = createInitVelMat(numOfPop, numOfFea)
    print "time is = ", time.strftime("%H:%M:%S", time.localtime())
    while (fittingStatus == unfit):
        population = createInitPopMat(numOfPop, numOfFea)
        fittingStatus, fitness = FromFitnessFileMLR_BPSO.validate_model(0, model,fileW, population,
                        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    globalBestRow, globalBestFitness = InitializeGlobalBestRowAndFitness(fitness, population)
    globalBestRow, globalBestFitness = findGlobalBest(population, fitness,
                                        globalBestRow,globalBestFitness)
    localBestMatrix = CreateInitialLocalBestMatrix(population)
    localBestFitness = CreateInitialLocalBestFitness(fitness)

    print "Starting the Loop - time is = ", time.strftime("%H:%M:%S", time.localtime())
    IterateNtimes(model, fileW, fitness, velocity, population, localBestFitness,
                  localBestMatrix, globalBestRow, globalBestFitness,
                  TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
    fileOut.close()
#------------------------------------------------------    
main()   
#main program ends in here
#------------------------------------------------------

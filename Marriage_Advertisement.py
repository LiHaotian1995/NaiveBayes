import feedparser as fd
import operator
import numpy as np
import re

def textParse(bigString):
    listOfToken = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfToken if len(tok) > 2]


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB(trainMatrix, trainCategory):
    numTrainDot = len(trainMatrix)
    numWord = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDot)

    p0Num = np.ones(numWord)
    p1Num = np.ones(numWord)

    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDot):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p0Vec = np.log(p0Num / p0Denom)
    p1Vec = np.log(p1Num / p1Denom)

    return p0Vec, p1Vec, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, PAbusive):
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - PAbusive)
    p1 = sum(vec2Classify * p1Vec) + np.log(PAbusive)
    if p1 > p0:
        return 1
    else:
        return 0


def calculateMostFrequent(vocabLiat, fullTest):
    frequentDict = {}
    for token in vocabLiat:
        frequentDict[token] = fullTest.count(token)
    sortedFrequent = sorted(frequentDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFrequent[:30]


def localWords(feed0, feed1):
    #step 1: visit RSS one by one; and use min one as a sum
    docList = []
    fullTest = []
    classList = []
    minLen = min(len(feed0['entries']), len(feed1['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    # print("\nwordList: \n" + str(wordList))
    # print("\ndocList: \n" + str(docList))
    # print("\nfullTest: \n" + str(fullTest))
    # print("\nclassList: \n" + str(classList))
    # print("\nminLen: \n" + str(minLen))
    # print("feed0, feed1:  " + str(len(feed0['entries'])) + "  " + str(len(feed1['entries'])))


    #step 2: get vocabList and remove the words which are more than 30 times
    vocabList = createVocabList(docList)
    print("\nlenth :" + str(len(vocabList)))
    # print("\nvocabList: \n" + str(vocabList))
    top30words = calculateMostFrequent(vocabList, fullTest)
    for pairW in top30words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    print("\nlenth :" + str(len(vocabList)))
    # print("\ntop30words: \n" + str(top30words))


    #step 3: give trainingSet number is [0,49] except 20 number, and the testSet numebrs are 20 random
    trainingSet = range(2*minLen)
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    # print("\ntrainingSet: " + str(trainingSet))
    print("\ntestSet: " + str(testSet))


    #step 4:
    # first, according to a random trainingSet, get the value of trainingClassMat and trainMat
    # then, return each of vocabulary value in the vocabList: p0V & p1V; and the p(one class) = pSpam
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB(trainMat, trainClass)
    # print("\ntrainMat: \n" + str(trainMat))
    # print("\ntrainClass: \n" + str(trainClass))
    # print("\np0V: \n" + str(p0V))
    # print("\np1V: \n" + str(p1V))
    # print("\npSpam: \n" + str(pSpam))


    #step 5: get a wordVector which is need to be classified
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print '\nError Index:', docIndex
            if docIndex % 2 == 1:
                print("\nNY")
            else:
                print("\nS.F.")
            print '\nclassification error: \n', docList[docIndex]

    rate = float(errorCount) / len(testSet)
    print("\nThe error rate is: " + str(rate*100) + " %")
    return vocabList, p0V, p1V


def getTopWords(f0, f1):
    vocabList, p0V, p1V = localWords(f0, f1)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -4.5:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -4.5:
            topNY.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    print '\n------------------------SF------------------------'
    for item in sortedSF:
        print item[0]

    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "\n------------------------NY------------------------"
    for item in sortedNY:
        print item[0]


if __name__ == '__main__':
    ny = fd.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = fd.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocabList, p0V, p1V = localWords(ny,sf)
    # print vocabList, p0V, p1V
    print("\np0V: \n" + str(p0V))
    print("\np1V: \n" + str(p1V))
    print("\nvocabList: \n" + str(vocabList))
    getTopWords(ny, sf)

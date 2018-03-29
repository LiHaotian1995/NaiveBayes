import numpy as np
import re

# create a vocabulary
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# set of words: exist means 1; no exist means 0
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word %s is not in the vocabulary " % word
    return returnVec


# bag of words: exist: the count of the words; no exist: 0
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# cut the email to every token, use regular expression
def textParse(bigString):
    listOfToken = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfToken if len(tok) > 2]


def trainNB(trainMatrix, trainCategory):
    numTrainDot = len(trainMatrix)
    numWord = len(trainMatrix[0])  # the number of vocabList

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

    # print("\np0Num: \n" + str(p0Num))
    # print("\np1Num: \n" + str(p1Num))
    # print("\np0Denom: \n" + str(p0Denom))
    # print("\np1Denom: \n" + str(p1Denom))

    pAbusive = sum(trainCategory) / float(numTrainDot)
    p0Vec = np.log(p0Num / p0Denom)
    p1Vec = np.log(p1Num / p1Denom)

    return p0Vec, p1Vec, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, PAbusive):
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - PAbusive)
    p1 = sum(vec2Classify * p1Vec) + np.log(PAbusive)
    # print("\nsum(vec2Classify * p0Vec): \n" + str(sum(vec2Classify * p0Vec)))
    # print("\np0: \n" + str(p0))
    # print("\np1: \n" + str(p1))
    if p1 > p0:
        return 1
    else:
        return 0



# a small test example
def testNB():
     postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop','him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
     classVec = [0,1,0,1,0,1]

     myList = createVocabList(postingList)
     # print("\nVocabList: \n" + str(myList))

     # Vec0 = setOfWords2Vec(myList, postingList[0])
     # Vec1 = setOfWords2Vec(myList, postingList[1])
     # Vec2 = setOfWords2Vec(myList, postingList[2])
     # Vec3 = setOfWords2Vec(myList, postingList[3])
     # Vec4 = setOfWords2Vec(myList, postingList[4])
     # Vec5 = setOfWords2Vec(myList, postingList[5])
     #
     # print(str(Vec0))
     # print(str(Vec1))
     # print(str(Vec2))
     # print(str(Vec3))
     # print(str(Vec4))
     # print(str(Vec5))

     trainMat = []
     for postinDoc in postingList:
         trainMat.append(setOfWords2Vec(myList, postinDoc))
     # print("\ntrainMat[0]: \n" + str(trainMat[0]))

     p0,p1,pa = trainNB(trainMat, classVec)
     # print(" \n pAbuise: \n" + str(pa))
     # print("\n p0: \n" + str(p0))
     # print("\n p1: \n" + str(p1))

     testContent1 = ['I', 'love', 'my', 'dog']
     testContent2 = ['my', 'dog', 'stupid', 'garbage']

     testDoc1 = np.array(setOfWords2Vec(myList, testContent1))
     testDoc2 = np.array(setOfWords2Vec(myList, testContent2))
     # print("\ntestDoc1: \n" + str(testDoc1))
     # print("\ntestDoc2: \n" + str(testDoc2))

     print testContent1, 'classified as: ', classifyNB(testDoc1, p0, p1, pa)
     print testContent2, 'classified as: ', classifyNB(testDoc2, p0, p1, pa)



# a spam email test
def spamTest(emailNum):
    #step 1:
    #fullTest: all words in 46 email        \  classList: [1,0,1,0....1,0], according to a sequence
    #vocabList: a non-repeating vocabulary  \  docList: a list, include the words in every email
    # fullTest = []
    docList = []
    classList = []
    global wordList
    for i in range(1,emailNum+1):
        wordList = textParse(open('/Users/lihaotian/PycharmProjects/Bayes/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        # fullTest.extend(wordList)
        classList.append(1)
        wordList = textParse(open('/Users/lihaotian/PycharmProjects/Bayes/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        # fullTest.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    # print("\nfullTest: " + str(fullTest))
    # print("\ndocList: " + str(docList))
    # print("\nclassList: " + str(classList))
    # print("\nvocabList: " + str(vocabList))
    # print("\nlenthVocabList: " + str(len(vocabList)))

    #step 2:
    #give trainingSet & testSet number: trainingSet number is [0,43] except 10 number, and the testSet numebrs are 10 random
    trainingSet = range(2*emailNum)
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    # print("\ntrainingSet: " + str(trainingSet))
    # print("\ntestSet: " + str(testSet))


    #step 3:
    #first, according to a random trainingSet, get the value of trainingClassMat and trainMat
    #then, return each of vocabulary value in the vocabList: p0Vec & p1Vec; and the p(spam) = pSpam
    trainMat = []
    trainingClassMat = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainingClassMat.append(classList[docIndex])
    # trainMatrix = np.mat(trainMat)
    # print trainMatrix
    # print trainMatrix.shape
    # print trainMat
    print trainingClassMat
    p0V, p1V, pSpam = trainNB(trainMat, trainingClassMat)
    # print("\np0V: \n" + str(p0V))
    # print("\np1V: \n" + str(p1V))
    # print("\npSpam: \n" + str(pSpam))


    #step 4:
    #get a wordVector which is need to be classified,
    global wordVector
    errorCount = 0
    print 'testSet:\n', testSet
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        # print wordVector
        # print np.array(wordVector)
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print '\nError Index:', docIndex
            if docIndex %2 == 1:
                print("\nham, number: " + str(docIndex/2+1))
            else:
                print("\nspam, number: " + str(docIndex/2+1))
            print '\nclassification error: \n', docList[docIndex]

    rate = float(errorCount) / len(testSet)
    print("\nThe error rate is: " + str(rate * 100) + " %")



if __name__ == '__main__':
    # testNB()
    spamTest(22)

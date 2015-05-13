import os
import sys
import csv
import StringIO
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree

def loadRecord(line):
    input = StringIO.StringIO(line)
    reader = csv.reader(input)
    row = map(float, reader.next())
    return LabeledPoint(row[-1],row[:-1]) 

chf = open('data/CAhousing.csv','r')
header = chf.next().rstrip("\n").split(",")
for i,j in enumerate(header):
    print "%d: %s" % (i,j)

chrdd = sc.parallelize(chf).map(lambda line: loadRecord(line))
chrdd.persist()

(trainingData, testData) = chrdd.randomSplit([0.7, 0.3])


model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    impurity='variance', minInstancesPerNode=2500)

predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())

with open("trunk.txt", "w") as f:
    f.write('Test Mean Squared Error = ' + str(testMSE))
    f.write('Learned regression tree model:')
    f.write( model.toDebugString() )

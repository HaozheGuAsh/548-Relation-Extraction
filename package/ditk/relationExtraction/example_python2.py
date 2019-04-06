# !/usr/bin/env python2

from . import relation_extraction_2 as parent


class Example(parent.RelationExtractionModel):

    def __init__(self):
        print("Initializing Relation Extraction Example Model Using Python2")

    def read_dataset(self, input_file):
        print("Python2 Example Model is reading data from:" + input_file)

    def data_preprocess(self, input_data=None):
        print("Python2 Example Model is preprocessing data")

    def tokenize(self, input_data=None):
        print("Python2 Example Model is tokenizing data")

    def train(self, train_data=None):
        print("Python2 Example Model is trainning data")

    def predict(self, test_data=None):
        print("Python2 Example Model is predicting data")

    def evaluate(self, input_data=None):
        print("Python2 Example Model is evaluating data")

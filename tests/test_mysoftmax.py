"""
Perform tests on SoftMaxClassifier classifier
"""
import unittest

import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score
from commonftests import numerical_dj, compare_analitic_and_numerical_dj,\
    simple_optimizer
from mysoftmax import j, h, dj, SoftMaxClassifier
from common import binarithate


np.random.seed(seed=2)


class TestSoftMaxClassifier(unittest.TestCase):

    def setUp(self):
        with open('test_data.pickle', 'r') as f:
            self.data = pickle.load(f)

    def test_gradient(self):
        scaler = RobustScaler()
        train_labels = self.data['train_labels']
        train_dataset = self.data['train_dataset']
        train_dataset = scaler.fit_transform(train_dataset)

        gradient_error = compare_analitic_and_numerical_dj(train_dataset,
                                                           train_labels, 0,
                                                           j, dj, numerical_dj)
        self.assertTrue(gradient_error < 10**-8)

    def test_classification(self):
        scaler = RobustScaler()
        train_labels = self.data['train_labels']
        train_dataset = self.data['train_dataset']
        test_labels = self.data['train_labels']
        test_dataset = self.data['train_dataset']

        train_dataset = scaler.fit_transform(train_dataset)
        test_dataset = scaler.transform(test_dataset)

        clf = SoftMaxClassifier(alpha=0.001, optimizer=simple_optimizer).fit(
            train_dataset, train_labels)
        predictions = clf.predict(test_dataset)
        f1 = f1_score(test_labels, predictions, average='weighted')
        self.assertGreater(f1, 0.58)

if __name__ == '__main__':
    unittest.main()

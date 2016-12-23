"""
Perform tests on mysvm classifier
"""
import unittest

import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score
from commonftests import numerical_dj, compare_analitic_and_numerical_dj,\
    simple_optimizer
from mysvm import j, h, dj, SvmClassifier, OneVersusAllSVM
from commonftests import numerical_dj, compare_analitic_and_numerical_dj,\
    simple_optimizer

np.random.seed(seed=2)


class TestSVM(unittest.TestCase):

    def setUp(self):
        with open('test_data.pickle', 'r') as f:
            self.data = pickle.load(f)

    def test_gradient(self):
        train_labels = self.data['train_labels']
        train_labels = (train_labels == 9).astype('int')
        train_dataset = self.data['train_dataset']
        train_dataset = RobustScaler().fit_transform(train_dataset)
        start_w = np.random.randn(
            train_dataset.shape[1]) / np.sqrt(train_dataset.shape[1])

        gradient_error = compare_analitic_and_numerical_dj(train_dataset,
                                                           train_labels * 2 - 1, 0,
                                                           j, dj, numerical_dj)
        self.assertTrue(gradient_error < 10**-8)

    def test_2class_svm(self):
        scaler = RobustScaler()
        train_labels = self.data['train_labels']
        train_labels = (train_labels == 9).astype('int')
        train_dataset = self.data['train_dataset']
        test_labels = self.data['train_labels']
        test_labels = (test_labels == 9).astype('int')
        test_dataset = self.data['train_dataset']
        train_dataset = scaler.fit_transform(train_dataset)
        test_dataset = scaler.transform(test_dataset)
        clf = SvmClassifier(alpha=0.001, optimizer=simple_optimizer).fit(
            train_dataset, train_labels)
        train_accuracy = np.mean(clf.predict(train_dataset) == train_labels)
        prediction = clf.predict(test_dataset)
        test_accuracy = np.mean(test_labels == prediction)
        precision = (1.0 * np.sum(prediction & test_labels)) / \
            np.sum(prediction)
        recall = (1.0 * np.sum(prediction & test_labels)) / np.sum(test_labels)
        self.assertGreater(train_accuracy, 0.9)
        self.assertGreater(test_accuracy, 0.7)
        self.assertGreater(precision, 0.7)
        self.assertGreater(recall, 0.7)

    def test_one_versus_all_svm(self):
        scaler = RobustScaler()
        train_labels = self.data['train_labels']
        train_dataset = self.data['train_dataset']
        test_labels = self.data['train_labels']
        test_dataset = self.data['train_dataset']

        train_dataset = scaler.fit_transform(train_dataset)
        test_dataset = scaler.transform(test_dataset)

        clf = OneVersusAllSVM(alpha=0.001, optimizer=simple_optimizer).fit(
            train_dataset, train_labels)
        predictions = clf.predict(test_dataset)
        f1 = f1_score(test_labels, predictions, average='weighted')
        self.assertGreater(f1, 0.58)

if __name__ == '__main__':
    unittest.main()

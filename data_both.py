import random
import math
import numpy as np
from data import Data

# Classifying Activity and Person
class TrainingData(Data):
    def __init__(self, logger, window_size = 100, logging = True):
        super(TrainingData, self).__init__(logger)
        self.logging = logging
        self.current_data_index = 0
        self.current_test_index = 0
        self.window_size = window_size # 10 seconds
        self.classes = ["Bruce-game", "Bruce-music", "Bruce-reading", "Bruce-video",
             "Carl-game", "Carl-music", "Carl-reading", "Carl-video",
             "Jerry-game", "Jerry-music", "Jerry-reading", "Jerry-video",
             "Matt-game", "Matt-music", "Matt-reading", "Matt-video",
             "Toby-game", "Toby-music", "Toby-reading", "Toby-video"]
        self.indexes = {"Bruce-game": 0, "Bruce-music": 0, "Bruce-reading": 0, "Bruce-video": 0,
             "Carl-game": 0, "Carl-music": 0, "Carl-reading": 0, "Carl-video": 0,
             "Jerry-game": 0, "Jerry-music": 0, "Jerry-reading": 0, "Jerry-video": 0,
             "Matt-game": 0, "Matt-music": 0, "Matt-reading": 0, "Matt-video": 0,
             "Toby-game": 0, "Toby-music": 0, "Toby-reading": 0, "Toby-video": 0}
        self.filenames = open("data/data_files.txt", "r").read().splitlines()
        self.logger.log("Initilizing Data")

        self.init_test_files()

        self.get_files()
        self.shuffle_set()
        self.logger.log("Finished Initilizing Data")

    def get_text(self):
        return 'both'

    def init_test_files(self):
        self.test_filenames = []
        random.shuffle(self.filenames)
        break_point = int(len(self.filenames) * 0.8)
        carl_data = self.seperate_carl()
        # self.test_filenames = self.filenames[break_point: len(self.filenames)]
        self.get_equal_amount_test(break_point)
        self.filenames = self.filenames[0:break_point]
        self.filenames += carl_data
        self.test_filenames += carl_data
        self.logger.log('Training file size: ' + str(len(self.filenames)))
        self.logger.log('Test file size: ' + str(len(self.test_filenames)))

    def seperate_carl(self):
        carl_data = []
        for filename in self.filenames:
            if self.get_name(filename).split('-')[0] == "Carl":
                carl_data += [filename]
        return carl_data

    def get_equal_amount_test(self, break_point):
        indexes = {"Bruce-game": break_point, "Bruce-music": break_point, "Bruce-reading": break_point, "Bruce-video": break_point,
             "Carl-game": break_point, "Carl-music": break_point, "Carl-reading": break_point, "Carl-video": break_point,
             "Jerry-game": break_point, "Jerry-music": break_point, "Jerry-reading": break_point, "Jerry-video": break_point,
             "Matt-game": break_point, "Matt-music": break_point, "Matt-reading": break_point, "Matt-video": break_point,
             "Toby-game": break_point, "Toby-music": break_point, "Toby-reading": break_point, "Toby-video": break_point}
        while len(self.test_filenames) < len(self.filenames) - break_point:
            for key, value in indexes.iteritems():
                if value >= len(self.filenames):
                    value = 0
                filename = self.filenames[value]
                name = self.get_name(filename)
                while name != key:
                    value += 1
                    if value >= len(self.filenames):
                        value = 0
                    filename = self.filenames[value]
                    name = self.get_name(filename)
                self.test_filenames += [filename]
                indexes[key] = value + 1
                if value >= len(self.filenames):
                    value = 0


    def get_batch(self, batch_size):
        train = self.train_set[self.current_data_index:self.current_data_index + batch_size]
        valid = self.valid_train_set[self.current_data_index:self.current_data_index + batch_size]
        self.current_data_index += batch_size
        if self.current_data_index >= len(self.train_set):
            self.get_files()
            self.shuffle_set()
        return train, valid

    def get_test_batch(self):
        input_data = []
        output_data = []
        batch_size = len(self.test_filenames) // 10
        filenames = self.filenames[self.current_test_index:self.current_test_index + batch_size]
        for filename in filenames:
            name = self.get_name(filename)
            if name not in self.classes:
                continue
            correctAnwser = [0] * len(self.classes)
            correctAnwser[self.classes.index(name)] = 1
            data = open(filename).read().splitlines()
            data = data[5:len(data) - 5];
            numberExamples = 0
            for y in range(1, len(data) - self.window_size):
                input_data += self.computeData(data[y:y+self.window_size])
                numberExamples += 1
            output_data += [correctAnwser] * numberExamples
        self.current_test_index += batch_size
        return input_data, output_data

    def is_at_test_batch_end(self):
        return self.current_test_index >= len(self.test_filenames)

    def get_test_set(self):
        input_data = []
        output_data = []
        # self.test_filenames = self.test_filenames[0:len(self.test_filenames) // 3]
        for filename in self.test_filenames:
            name = self.get_name(filename)
            correctAnwser = [0] * len(self.classes)
            correctAnwser[self.classes.index(name)] = 1
            data = open(filename).read().splitlines()
            data = data[5:len(data) - 5];
            numberExamples = 0
            for y in range(1, len(data) - self.window_size):
                input_data += self.computeData(data[y:y+self.window_size])
                numberExamples += 1
            output_data += [correctAnwser] * numberExamples
        return input_data, output_data

    def get_min_file_length(self):
        min_length = 500000000000
        for key, value in self.indexes.iteritems():
            filename = self.filenames[value]
            data = open(filename).read().splitlines()
            if len(data) < min_length:
                min_length = len(data)
        return min_length

    def trim_data(self, data, min_length):
        trim_spot = (len(data) - min_length) // 2
        if trim_spot < 0:
            trim_spot = 0
        return data[trim_spot: len(data) - trim_spot]

    # Gets one from each person
    def get_files(self):
        self.train_set = []
        self.valid_train_set = []
        self.current_data_index = 0
        min_length = self.get_min_file_length();
        loadNEach = 2
        for z in range(0,loadNEach):
            for key, value in self.indexes.iteritems():
                while self.get_name(self.filenames[value]) != key:
                    value += 1
                    if value >= len(self.filenames):
                        value = 0
                filename = self.filenames[value]
                name = self.get_name(filename)

                correctAnwser = [0] * len(self.classes)
                correctAnwser[self.classes.index(name)] = 1
                data = open(filename).read().splitlines()
                data = self.trim_data(data, min_length)
                numberExamples = 0
                for y in range(1, len(data) - self.window_size):
                    self.train_set += self.computeData(data[y:y+self.window_size])
                    numberExamples += 1
                self.valid_train_set += [correctAnwser] * numberExamples
                self.indexes[key] = value + 1
                if self.indexes[key] >= len(self.filenames):
                    self.indexes[key] = 0

    def get_name(self, filename):
        return filename.split('/')[1] + '-' + filename.split('/')[2]


    def computeData(self, data):
        singleVector = []
        for i in data:
            singleVector += map(float, i.split(','))
        return [singleVector]

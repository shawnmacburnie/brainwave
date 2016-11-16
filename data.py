import random

# extend this class
# Only needing to impliment the following functions
#       get_batch(batch_size:INT)
#       is_at_test_batch_end()
#       get_test_batch(batch_size:INT)
# With a constructor that sets train_set valid_train_set,
# test_set, valid_test_set, and classes
class Data(object):
    def __init__(self, logger):
        self.logger = logger
        self.train_set = [] # set for training input
        self.valid_train_set = [] # Set for valid training output
        self.test_set = [] # Set for testing input
        self.valid_test_set = [] # Set for valid test output
        self.classes = []

    def get_num_out(self):
        return len(self.classes)

    def get_num_in(self):
        return len(self.train_set[0])

    def shuffle_set(self):
        if len(self.train_set) == 0 or len(self.valid_train_set) == 0:
            self.get_files()
        tmpList = list(zip(self.train_set, self.valid_train_set))
        random.shuffle(tmpList)
        self.train_set, self.valid_train_set = zip(*tmpList)

    def get_train_size(self):
        return len(self.train_set)

    def get_test_size(self):
        return len(self.test_set)

    def get_output_order(self):
        return self.classes

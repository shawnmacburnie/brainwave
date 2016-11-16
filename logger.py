
class Logger():
    def __init__(self, startingFileName = None, shouldLog = True):
        if startingFileName == None:
            self.f = startingFileName = None
        else:
            self.f = open(startingFileName + '.log', 'w')
        self.shouldLog = shouldLog

    def update(self, newLogFile):
        self.kill()
        self.f = open(newLogFile + '.log', 'w')

    def updateBase(self, base):
        self.base = base

    def kill(self):
        if self.f != None:
            self.f.close()

    def log(self, text):
        if self.shouldLog:
            self.f.write(text + '\n')

import numpy as np
import random
from bias_cnn import bias_cnn
class SA:
    def __init__(self, parameters):

        self.parameter = parameters
        self.op = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0])
        self.a = [0,0,0,0]
        self.temperature = 100
        self.delta = 0.98
        self.fitness = 0
        self.hisBestResult = parameters
        self.hisBestFitness = 0

        parameters[6] = np.log(parameters[6])/np.log(2)
        parameters[1] = np.log(parameters[1])/np.log(2)
        parameters[3] = np.log(parameters[3]) / np.log(2)
        parameters[5] = np.log(parameters[5]) / np.log(2)

        for i in range(0,7):
            self.a = self.codingInput(parameters[i])
            for j in range(0,4):
                self.op[4*i+j] = self.a[j]
        print(self.op)

        self.resultNum = [0,0,0,0,0,0,0]

        for i in range(0,7):
            for j in range(0,4):
                self.a[j] = self.op[4*i+j]
            self.resultNum[i] = self.decodingOutput(self.a)
        self.resultNum[6] = 2**self.resultNum[6]
        self.resultNum[5] = 2 ** self.resultNum[5]
        self.resultNum[3] = 2 ** self.resultNum[3]
        self.resultNum[1] = 2 ** self.resultNum[1]
        print (self.resultNum)

    def codingInput(self,inputNum):
        self.templist = [0,0,0,0]
        if inputNum > 15 or inputNum < 0 :
            return []
        else:
            for i in range(0,4):
                self.templist[3-i] = inputNum%2
                inputNum = (inputNum - self.templist[3-i])/2
            return self.templist

    def decodingOutput(self,outputNum):
        self.result = 0
        for i in range(0,4):
            self.result += 2**(3-i)*outputNum[i]
        return self.result


    def run(self):
        self.resultNum = [0, 0, 0, 0, 0, 0, 0]

        for i in range(0, 7):
            for j in range(0, 4):
                self.a[j] = self.op[4 * i + j]
            self.resultNum[i] = self.decodingOutput(self.a)
        self.resultNum[6] = 2 ** self.resultNum[6]
        self.resultNum[5] = 2 ** self.resultNum[5]
        self.resultNum[3] = 2 ** self.resultNum[3]
        self.resultNum[1] = 2 ** self.resultNum[1]

        pre_cnn = bias_cnn(cnn_para=self.resultNum)
        pre_cnn.train_net()
        self.fitness = pre_cnn.eva_net()
        self.hisBestFitness = self.fitness

        while(self.temperature>1):
            self.flag = 1
            while self.flag == 1:
                self.changByteNum = random.randint(4,7)
                self.temp = self.op
                for change in range(0,self.changByteNum):
                    self.byte = random.randint(0,27)
                    self.temp[self.byte] = 1 - self.temp[self.byte]

                for i in range(0, 7):
                    for j in range(0, 4):
                        self.a[j] = self.op[4 * i + j]
                    self.resultNum[i] = self.decodingOutput(self.a)

                self.resultNum[6] = 2 ** self.resultNum[6]
                self.resultNum[5] = 2 ** self.resultNum[5]
                self.resultNum[3] = 2 ** self.resultNum[3]
                self.resultNum[1] = 2 ** self.resultNum[1]
                if (self.resultNum[0] > self.resultNum[2] and self.resultNum[2] > self.resultNum[4])and(self.resultNum[5]<256 and self.resultNum[6]<1024):
                    self.flag = 0


            pre_cnn = bias_cnn(cnn_para=self.resultNum)
            pre_cnn.train_net()
            self.fitnesstemp = pre_cnn.eva_net()

            if self.fitnesstemp > self.fitness:
                self.op = self.temp
                self.fitness = self.fitnesstemp
                if self.fitness > self.hisBestFitness:
                    self.hisBestFitness = self.fitness
                    self.hisBestResult = self.resultNum
            elif random.random(1) < np.exp((self.fitness - self.fitnesstemp)/self.temperature):
                self.op = self.temp
                self.fitness = self.fitnesstemp
            self.temperature *= self.delta
        return self.hisBestResult

a = SA([8,32,4,64,2,64,256])
resultNum = a.run()
print resultNum

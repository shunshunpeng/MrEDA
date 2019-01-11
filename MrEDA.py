#coding:utf-8-
from __future__ import division
import random
import math
import os
import numpy
import multiprocessing
from multiprocessing import cpu_count
from multiprocessing import Process, Manager
import datetime
import shutil

#####The definiton of user data////  
Cmax = 100	#certain maximal value  
Cmin = 0	#certain minimum value  

# parameters on train
HIDEN_LENGTH = 1
DATA_LENGTH = 1
TRAIN_NUM = 4
CPU_NUM = cpu_count()
#the total length of chrom
#CHROMLENGTH = LENGHT1+LENGHT2+LENGHT3+LENGHT4+LENGHT5
Pc = 0.7			#jiaocha paorability
Pm = 0.5			#bianyi probability
rp = int(1/Pm)			#bianyi probability
w1 = 0.3
w2 = 0.3
w3 = 0.4
Popu_value = {}
#definiton of data struct
bestIndex = 0       #the index of the best individual
worstIndex = 0      #the index of the worst individual

def service_partition(ab_service):
    path = './services/%d/' % ab_service
    if os.path.exists(path) == True:
        return None
    os.makedirs(path)
    t = 0
    fr = open('../QWS_Dataset_v2.txt')
    for line in fr:
        tmp = (t % ab_service) + 1
        cols = line.split(',')
        list = int(t/ab_service) + 1
        dest = path + '%d.txt' % tmp
        fw = open(dest, 'a')
        content = str(list) + '\t' + cols[0] + '\t' + str(int(cols[1])/100) + '\t' + cols[2] + '\r\n'
        fw.write(content)
        fw.close()
        t += 1
    fr.close()
        
def service_initial(service_num, ab_service, C, task):
        name = './services/%d/' % ab_service + '%d.txt' % task
        fr = open(name, 'r')
        t = 0
        for line in fr:
                if t >= service_num:
                        break
                line = line.strip('\r\n')
                cols = line.split('\t')
                C[t][0] = cols[0]
                C[t][1] = cols[1]
                C[t][2] = cols[2]
                C[t][3] = cols[3]
                t += 1
        fr.close()
        return  C 

def max_min(C, M):
    for i in range(0, 3):
        T = []
        for j in range(0, service_num):
            try:
                T.append(float(C[j][i+1]))
            except:
                pass
        M[i*2+0] = max(T)
        M[i*2+1] = min(T)
    return M

def compute_c_max_min():
    for task in range(0, ab_service):
        tmp = 'C%d' % (task+1) + '= service_initial(service_num, ab_service, C%d' % (task+1) + ', task+1)'
        exec(tmp)
        tmp = 'MM%d' % (task+1) + '= max_min(C%d' % (task+1) + ', MM%d' % (task+1) + ')'
        exec(tmp)

    for task in range(0, ab_service):
        for service in range(service_num):
            command = 'C%d[%d][1] = (MM%d[0]-float(C%d[%d][1]))/(float(MM%d[0])-float(MM%d[1]))' % (task+1, service, task+1, task+1, service, task+1, task+1)
            try:
                exec(command)
            except:
                pass
            command = 'C%d[%d][3] = (MM%d[4]-float(C%d[%d][3]))/(float(MM%d[4])-float(MM%d[5]))' % (task+1, service, task+1, task+1, service, task+1, task+1)
            try:
                exec(command)
            except:
                pass

def generateInitialPopulation():
    for i in range(PopSize):
        for j in range(ab_service):
            while True:
                tmp = ''
                for k in range(LENGTH):
                    tmp = tmp + str(random.randint(0,1))
                data = decomdeChromosome(tmp, 0, LENGTH)
                if data < service_num:
                    break
            for k in range(LENGTH):
                command = 'population%d.chrom[%d] = tmp[%d]' % (i, j*LENGTH+k, k)
                exec(command)
def worker(core, pop_order):
    name = multiprocessing.current_process().name
    for i in range(core*WORKER_NUM, min((core+1)*WORKER_NUM, ab_service)):
		sum_weights = [[[0.0]*HIDEN_LENGTH for col in range(LENGTH)] for row in range(Train_sample)]
		sum_hiden = [[0.0]*HIDEN_LENGTH for col in range(Train_sample)]
		sum_visible = [[0.0]*LENGTH for col in range(Train_sample)]
		sum_hj = [[0.0]*HIDEN_LENGTH for col in range(Train_sample)]
		sum_obs = [[0]*LENGTH for col in range(Train_sample)]
		for j in range(Train_sample):
			value = ['']*LENGTH
			data = list()
			data_t = list()
			for k in range(i*LENGTH, i*LENGTH+LENGTH, 1):
				command = 'value[%d] = population%d.chrom[%d]' % (k-i*LENGTH,pop_order[j][0],k)
				exec(command)
			command = 'data_t = numpy.array([int(value[0])'
			for k in range(1, LENGTH, 1):
				command = command + ', int(value[%d])' % (k)
			command = command + '])'
			exec(command)
			for k in xrange(DATA_LENGTH):
				data.append(numpy.array([1 if x > numpy.random.random() else 0 for x in data_t]))
			rbm = RBM(LENGTH,HIDEN_LENGTH,.05)
			rbm.train(data, TRAIN_NUM, .1)
			for k in range(LENGTH):
				for l in range(HIDEN_LENGTH):
					sum_weights[j][k][l] = rbm.weights[k][l]
			for k in range(HIDEN_LENGTH):
				sum_hiden[j][k] = rbm.bias_hid[k]
			for k in range(LENGTH):
				sum_visible[j][k] = rbm.bias_vis[k]
			for k in range(LENGTH):
				sum_obs[j][k] = rbm.obs[k]
			for k in range(HIDEN_LENGTH):
				sum_hj[j][k] = rbm.hj[k]
		p_g = [0.0]*LENGTH
		for j in range(LENGTH):
			sum_delta_1 = 0.0
			sum_delta_0 = 0.0
			sum_delta = 0.0
			avg_delta = 0.0
			for k in range(Train_sample):
				delta_v_1 = 0.0
				delta_v_0 = 0.0
				for l in range(HIDEN_LENGTH):
					e_value_1 = - sum_obs[k][j]*sum_hj[k][l]*sum_weights[k][j][l] - sum_obs[k][j]*sum_visible[k][j] - sum_hj[k][l]*sum_hiden[k][l]
					e_value_0 = - sum_hj[k][l]*sum_hiden[k][l]
					delta_v_1 += math.exp(-(e_value_1))
					delta_v_0 += math.exp(-(e_value_0))
				sum_delta_1 += delta_v_1
				sum_delta_0 += delta_v_0
				sum_delta = sum_delta_1 + sum_delta_0
			avg_delta = sum_delta/Train_sample
			p_g[j] = (sum_delta_1+avg_delta)/(sum_delta_1+sum_delta_0+2*avg_delta)
		for j in range(PopSize):
			while True:
				tmp = ['']*LENGTH
				for k in range(LENGTH):
					p = random.uniform(0,1)
					if p <= p_g[k]:
						tmp[k] = '1'
					else:
						tmp[k] = '0'
				content = ''
				for k in range(LENGTH):
					content = content + tmp[k]
				value = decomdeChromosome(content, 0, LENGTH)
				if value < service_num:
					break
			for k in range(LENGTH):
				command = 'population%d.chrom[%d] = tmp[%d]' % (j, i*LENGTH+k, k)
				exec(command)
    '''
for j in range(PopSize):
        command = 'print name + str(population%d.chrom)' % j
        exec(command)
    '''

def generateNextPopulation():
    find_top_50_per()
    pop_order = sorted(Popu_value.items(), key=lambda d: d[1], reverse=True)
    for core in range(CPU_NUM-1):
        command = 'worker_%d =' % core + "multiprocessing.Process(name='worker %d', target=worker, args=(%d, pop_order, ))" % (core, core)
        exec(command)
    for core in range(CPU_NUM-1):
        command = 'worker_%d.start()' % core
        exec(command)
    for core in range(CPU_NUM-1):
        command = 'worker_%d.join()' % core
        exec(command)

def mutationOperator():
	# bit mutation  
	for i in range(PopSize):
		command = 'popu = population%d' % i
		exec(command)
		for j in range(ab_service):
			content = ''
			for k in range(j*LENGTH, j*LENGTH+LENGTH):
				content = content + popu.chrom[k]
			while True:
				tmp = ['']*LENGTH
				for k in range(LENGTH):
					p = random.randint(0,rp-1)/rp
					if p < Pm:
						if popu.chrom[j*LENGTH+k] == '0':
							tmp[k] = '1'
						else:
							tmp[k] = '0'
					else:
						command = 'tmp[%d] = popu.chrom[%d]' % (k,j*LENGTH+k)
						exec(command)
				content = ''
				for k in range(LENGTH):
					content = content + tmp[k]
				data = decomdeChromosome(content, 0, LENGTH)
				if data < service_num:
					break
			for k in range(LENGTH):
				command = 'population%d.chrom[%d] = tmp[%d]' % (i, j*LENGTH+k, k)
				exec(command)

def crossoverOperator():
    index = [0]*PopSize
    for i in range(PopSize):
        index[i] = i
    for i in range(PopSize):
        point = random.randint(0, PopSize-i-1)
        temp = index[i]
        index[i] = index[point+i]
        index[point+i] = temp
    for i in range(0, PopSize-1, 2):
        p = random.randint(0,rp-1)/rp
        if p < Pc:
            point = random.randint(0,CHROMLENGTH-2) + 1
            for j in range(point, CHROMLENGTH, 1):
                command = 'ch = population%d.chrom[%d]' % (index[i],j)
                exec(command)
                command = 'population%d.chrom[%d] = population%d.chrom[%d]' % (index[i],j,index[i+1],j)
                exec(command)
                command = 'population%d.chrom[%d] = ch' % (index[i+1],j)
                exec(command)

def selectionOperator():
    sum = 0.0
    cfitness = [0.0]*PopSize
    for i in range(PopSize):
        command = 'newpopulation%d = Individual()' % i
        exec(command)
    for i in range(PopSize):
        command = 'sum += population%d.fitness' % i
        exec(command)
    for i in range(PopSize):
        command = 'cfitness[%d] = population%d.fitness/sum' % (i,i)
        exec(command)
    # calculate cumulative fitness   
    for i in range(1, PopSize, 1):
        cfitness[i] = cfitness[i] + cfitness[i-1]
    for i in range(PopSize):
        p = random.randint(0,rp-1)/rp
        index = 0
        while p > cfitness[index]:
            index += 1
        for j in range(CHROMLENGTH):
            command = 'newpopulation%d.chrom[%d] = population%d.chrom[%d]' % (i,j,index,j)
            exec(command)
        command = 'newpopulation%d.value=population%d.value' % (i,index)
        exec(command)
        command = 'newpopulation%d.fitness = population%d.fitness' % (i,index)
        exec(command)
    for i in range(PopSize):
        for j in range(CHROMLENGTH):
            command = 'population%d.chrom[%d] = newpopulation%d.chrom[%d]' % (i,j,i,j)
            exec(command)
        command = 'population%d.value = newpopulation%d.value' % (i,i)
        exec(command)
        command = 'population%d.fitness = newpopulation%d.fitness' % (i,i)
        exec(command)

#evaluation by some metrix
def evaluatePopulation():
    calculateObjectValue()
    calculateFitnessValue()
    findBestAndWorstIndividual()

def findBestAndWorstIndividual():
    sum = 0.0
    for i in range(CHROMLENGTH):
        bestIndividual.chrom[i] = population0.chrom[i]
        worstIndividual.chrom[i] = population0.chrom[i]
    bestIndividual.value = population0.value
    worstIndividual.value = population0.value
    bestIndividual.fitness = population0.fitness
    worstIndividual.fitness = population0.fitness
    #currentBest = population0
    for i in range(PopSize):
        command = 'fit = population%d.fitness' % i
        exec(command)
        if fit > bestIndividual.fitness:
            for j in range(CHROMLENGTH):
                command = 'bestIndividual.chrom[%d] = population%d.chrom[%d]' % (j,i,j)
                exec(command)
            command = 'bestIndividual.value = population%d.value' % (i)
            exec(command)
            command = 'bestIndividual.fitness = population%d.fitness' % (i)
            exec(command)
            bestIndex = i
        elif fit < worstIndividual.fitness:
            for j in range(CHROMLENGTH):
                command = 'worstIndividual.chrom[%d] = population%d.chrom[%d]' % (j,i,j)
                exec(command)
            command = 'worstIndividual.value = population%d.value' % (i)
            exec(command)
            command = 'worstIndividual.fitness = population%d.fitness' % (i)
            exec(command)
            worstIndex = i
        command = 'sum += population%d.fitness' % i
        exec(command)
    if generation == 0:
        for i in range(CHROMLENGTH):
            currentBest.chrom[i] = bestIndividual.chrom[i]
        currentBest.value = bestIndividual.value
        currentBest.fitness = bestIndividual.fitness
    else:
        if bestIndividual.fitness > currentBest.fitness:
            for i in range(CHROMLENGTH):
                currentBest.chrom[i] = bestIndividual.chrom[i]
            currentBest.value = bestIndividual.value
            currentBest.fitness = bestIndividual.fitness

def calculateFitnessValue():
    for i in range(PopSize):
        for j in range(ab_service):
            command = 'temp%d = decomdeChromosome(population%d.chrom, %d*LENGTH, LENGTH)' % (j,i,j)
            exec(command)

        try:
            response_time = float(C1[temp0][1])
            availability = float(C1[temp0][2])
            throughout = float(C1[temp0][3])
        except:
            response_time = 0
            availability = 0
            throughout = 0
        for j in range(1, ab_service, 1):
            command = 'location = temp%d' % j
            exec(command)
            command = 'response_time = response_time + float(C%d[%d][1])' % (j+1,location)
            try:
                exec(command)
            except:
                pass
            command = 'availability = availability*float(C%d[%d][2])' % (j+1,location)
            try:
                exec(command)
            except:
                pass
            command = 'ava_tmp = float(C%d[%d][3])' % (j+1, location)
            try:
                exec(command)
            except:
                ava_tmp = 0
                pass
            if ava_tmp < throughout:
                throughout = ava_tmp

        #command = 'population%d.fitness = x1+x2+x3+x4+x5' % i
        command = 'population%d.fitness = w1*response_time+w2*availability+w3*throughout' % i
        exec(command)

#f(x) = x1* x1 +  x2*x2 + x3*x3 + x4*x4 + x5*x5
def calculateObjectValue():
    for i in range(PopSize):
        for j in range(ab_service):
            command = 'temp%d = decomdeChromosome(population%d.chrom, %d*LENGTH, LENGTH)' % (j,i,j)
            exec(command)

        try:
            response_time = float(C1[temp0][1])
        except:
            pass
        try:
            availability = float(C1[temp0][2])
            throughout = float(C1[temp0][3])
        except:
            availability = 0
            throughout = 0
            pass
        for j in range(1, ab_service, 1):
            command = 'location = temp%d' % j
            exec(command)
            command = 'response_time = response_time + float(C%d[%d][1])' % (j+1,location)
            try:
                exec(command)
            except:
                pass
            command = 'availability = availability*float(C%d[%d][2])' % (j+1,location)
            try:
                exec(command)
            except:
                pass
            command = 'ava_tmp = float(C%d[%d][3])' % (j+1, location)
            try:
                exec(command)
            except:
                ava_tmp = 0
                pass
            if ava_tmp < throughout:
                throughout = ava_tmp
        #command = 'population%d.fitness = x1+x2+x3+x4+x5' % i
        command = 'population%d.value = w1*response_time+w2*availability+w3*throughout' % i
        try:
            exec(command)
        except:
            pass

def decomdeChromosome(string, point, length):
    content = '0b'
    for i in range(point, point+length, 1):
        content = content + string[i]
    decimal = eval(content)
    return decimal

def performEvolution():
    if bestIndividual.fitness > currentBest.fitness:
        for i in range(CHROMLENGTH):
            command = 'currentBest.chrom[%d] = population%d.chrom[%d]' % (i,bestIndex,i)
            exec(command)
        command = 'currentBest.value = population%d.value' % (bestIndex)
        exec(command)
        command = 'currentBest.fitness = population%d.fitness' % (bestIndex)
        exec(command)
    else:
        for i in range(CHROMLENGTH):
            command = 'population%d.chrom[%d] = currentBest.chrom[%d]' % (worstIndex,i,i)
            exec(command)
        command = 'population%d.value = currentBest.value' % (worstIndex)
        exec(command)
        command = 'population%d.fitness = currentBest.fitness' % (worstIndex)
        exec(command)

def outputTextReport(ite_num):
    sum = 0.0
    for i in range(PopSize):
        command = 'sum += population%d.value' % i
        exec(command)
    average = sum/PopSize
    content = 'gen=%d, sum= %f, avg=%f, best=%f, chromosome=' % (generation, sum, average, currentBest.value)
    for i in range(CHROMLENGTH):
        content += currentBest.chrom[i]
    print content
    path = './log/ab_%d/ser_%d/pop_%d/hid_%d/' % (ab_service, service_num, PopSize, HIDEN_LENGTH)
    name = path + '%d.txt' % ite_num
    fp = open(name,'a')
    fp.write(content+'\r\n')
    fp.close()

class RBM:
  def __init__(self, num_vis, num_hid, L):
    self.num_vis = num_vis
    self.num_hid = num_hid
    self.L = L
    self.weights = numpy.random.randn(num_vis, num_hid)
    self.bias_hid = numpy.random.randn(num_hid)
    self.bias_vis = numpy.random.randn(num_vis)
    self.hj = numpy.random.randn(num_hid)
    self.obs = numpy.array([0 for x in range(LENGTH)])

  def train(self, data, epochs, lr):
    for ep in xrange(epochs):
      for obs in data:
        #Calculate activation values for hidden states using logistic function
        pos_activation = numpy.dot(obs,self.weights) + self.bias_hid
        pos_probs = self.sigmoid(pos_activation)
        #Turn hidden states on with probability = activation value
        updated_hid_states = numpy.array([1 if x > numpy.random.random() else 0 for x in pos_probs])
        positive = numpy.outer(obs, pos_probs)
        
        #Calculate activation values for visible states based on updated hidden states
        vis_activation = numpy.dot(updated_hid_states, self.weights.T) + self.bias_vis
        vis_probs = self.sigmoid(vis_activation)

        #Turn Visible neurons on with probability = activation value
        updated_vis_states = numpy.array([1 if x > numpy.random.random() else 0 for x in vis_probs])

        #Calculate hidden neuron's activation value based on new visible neurons
        hid2_activation = numpy.dot(updated_vis_states, self.weights) + self.bias_hid
        hid2_probs = self.sigmoid(hid2_activation)

        #Again turn hidden neurons on with probability = activation value
        updated_hid_states2 = numpy.array([1 if x > numpy.random.random() else 0 for x in hid2_probs])
        negative = numpy.outer(obs, hid2_probs)

        #Update weights and biases or break if change is too small
        delta_w = lr*(positive - negative)
        delta_bv = lr*(updated_vis_states - obs)
        delta_bh = lr*(pos_probs - hid2_probs)
        self.hj = updated_hid_states2
        for i in range(LENGTH):
          self.obs[i] = obs[i]
        if(delta_w.min() < .005 or delta_bv.min() < .005 or delta_bh.min() < .005):
          break
        else:
          self.weights = self.weights + delta_w 
          self.bias_vis = self.bias_vis + lr*(updated_vis_states - obs)
          self.bias_hid = self.bias_hid + lr*(pos_probs - hid2_probs)

  def sigmoid(self, z):
    return 1/(1 + numpy.exp(-z))

def find_top_50_per():
    for i in range(PopSize):
        command = 'value = population%d.value' % i
        exec(command)
        Popu_value[i] = value



if __name__ == '__main__':
    MaxGeneration = 200 #max genrataion
    for ab_service in range(5, 51, 5):
        WORKER_NUM = int(ab_service/(CPU_NUM-1))+1
        for service_num in range(100, 701, 100):
            LENGTH = 1		#the chromosome length of 1st variable  
            while True:
                t = 2**LENGTH-1
                if t >= service_num:
                    break
                LENGTH += 1
            CHROMLENGTH = LENGTH*ab_service
            PopSize = int(service_num/4)		#size of sample
            Train_sample = int(PopSize/5)
            class Individual:
                def __init__(self):
                    self.chrom = ['']*CHROMLENGTH		#a chrom
                    self.value = 0.0	#destination value
                    self.fitness = 0.0	#shiyingdu 

            for task in range(0, ab_service):
                tmp = 'C%d' % (task+1) + '= [[' + repr('') + ']*4 for col in range(int(service_num))]'
                exec(tmp)
                tmp = 'MM%d' % (task+1) + '= [0]*6'
                exec(tmp)
            for i in range(int(service_num/4)):
                command = 'population%d = Individual()' % i
                exec(command)
            bestIndividual = Individual()       #the best individul of so far
            worstIndividual = Individual()     #the worst individual of so far
            currentBest = Individual()             #the best individul to fo far
            for i in range(int(service_num/4)):
                command = 'population%d = Individual()' % i
                exec(command)

            service_partition(ab_service)
            compute_c_max_min()
            path = './log/ab_%d/ser_%d/pop_%d/hid_%d/' % (ab_service, service_num, PopSize, HIDEN_LENGTH)
            if os.path.exists(path) == False:
                os.makedirs(path)
            name = path + 'time.txt'
            try:
                os.remove(name)
            except:
                pass
            for j in range(5):
                name = path + '%d.txt' % j
                try:
                    os.remove(name)
                except:
                    pass
                name = path + 'time.txt'
                fp = open(name, 'a')
                generation = 0
                starttime = datetime.datetime.now()
                generateInitialPopulation()
                evaluatePopulation()
                generateNextPopulation()
                while generation < MaxGeneration:
                    generation += 1
                    generateNextPopulation()
                    evaluatePopulation()
                    performEvolution()
                    outputTextReport(j)
                endtime = datetime.datetime.now()
                print (endtime - starttime)
                fp.write(str(endtime - starttime)+'\r\n')
                fp.close()

"""

A Quantum K nearest neighbour based on the paper 

Quantum Algorithm for K-Nearest Neighbors Classification Based on the Metric of Hamming Distance
by
Yue Ruan  Xiling Xue   Heng Liu  Jianing Tan  Xi Li

This implementation condiders a dataset of numbers 0-8 in binary form and classify it into even or odd.
the results are not stable as the vector dimension of data point is 3 and the number of points are 6 (too low!!)


"""


from qiskit import *
import matplotlib.pyplot as plt
from qiskit import tools
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit.circuit.library import MCMT, MCXGate, Measure
from qiskit.extensions import UnitaryGate
import numpy as np




import pprint
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
from sklearn.model_selection import train_test_split







class QKNN:

    def __init__(self, pattern, n, m, class_bit, k_neighbours, threshold, shots):

        self.pattern = pattern

        self.m = m
        self.n = n
        self.class_n = class_bit
        self.k_neighbours = k_neighbours
        self.t = threshold

        self.shots = shots

        self.n_total = n+class_bit

        self.main_pR = QuantumRegister(self.n_total, "p")
        self.main_uR = QuantumRegister(2,"u")
        self.main_mR = QuantumRegister(self.n_total, "m")

        self.main_circuit = QuantumCircuit(self.main_pR, self.main_uR, self.main_mR)
        self.one_state = [0,1]
        self.zero_state = [1,0]

    def trainSuperPosition(self):

       
        """

        This function creates a superposition of dataset as described in the paper.

        """

        for i in range(self.m):
            pR = QuantumRegister(self.n_total, "p")
            uR = QuantumRegister(2,"u")
            mR = QuantumRegister(self.n_total, "m")
            circuit = QuantumCircuit(pR,uR,mR, name="pattern"+str(i+1))

            for j in range(self.n_total):
                
                if self.pattern[i][j] == 0:
                    circuit.initialize(self.zero_state,pR[j])
                else:
                    circuit.initialize(self.one_state,pR[j])
                
                circuit.ccx(pR[j],uR[1],mR[j])

            for j in range(self.n_total):

                circuit.cx(pR[j],mR[j])
                circuit.x(mR[j])
                
            circuit.mcx(mR,uR[0])
                
            
            k = i+1
            data = np.array([[np.sqrt((k-1)/k),np.sqrt(1/k)],[-np.sqrt(1/k),np.sqrt((k-1)/k)]])
            gate = UnitaryGate(data=data)
            gate = gate.control(1,ctrl_state="1")
            circuit.append(gate,[uR[0],uR[1]],[])

            circuit.mcx(mR,uR[0])


            for j in range(self.n_total):

                circuit.x(mR[j])
                circuit.cx(pR[j],mR[j])

            for j in range(self.n_total):
                circuit.ccx(pR[j],uR[1],mR[j])
            
            """circuit.draw(output = "mpl")
            plt.tight_layout()
            plt.show()"""
            self.main_circuit.append(circuit.to_instruction(), self.main_pR[:self.n_total]+ self.main_uR[:2] + self.main_mR[:self.n_total])
        
        
        return self.main_circuit

    def fit(self, x):

        """

        A function to fit the test vector x with the superpositioned dataset.
        The circuit from previous set is appended to this circuit as there is no concept of saving the data!

        """
        l = 2**(self.k_neighbours)-self.n
        a = t+l
        a_binary = "{0:b}".format(a)
        a_len = self.k_neighbours+1

        if len(a_binary) < a_len:
            a_binary = "0"*(a_len-len(a_binary))+a_binary

        xR = QuantumRegister(self.n, "x")
        auR = QuantumRegister(1, "au")
        aR = QuantumRegister(a_len,"a")
        cR = ClassicalRegister(1, "c")
        oR = ClassicalRegister(self.class_n, "o")
        
        predictCircuit = QuantumCircuit(xR, self.main_mR, aR, auR, cR, oR)
        circuit = self.main_circuit + predictCircuit
        
        circuit.barrier()

        for k in range(len(x)):

            circuit.cx(self.main_mR[k],xR[k])
            circuit.x(xR[k])

        for i in range(a_len):

            if a_binary[::-1][i] == "0":
                circuit.initialize(self.zero_state,aR[i])
            else:
                circuit.initialize(self.one_state,aR[i])
        
        circuit.initialize(self.one_state,auR)
        for k in range(len(x)):
            
            for i in range(a_len):
                
                circuit.ccx(xR[k],auR, aR[i])
                ctrlString = "1"+"0"*(i)+"1"
                tempmc = MCXGate(i+2,ctrl_state=ctrlString)
                circuit.append(tempmc,[xR[k]]+aR[:i+1]+[auR],[])

            circuit.x(auR)
            ctrlString ="0"*(a_len-1)+"1"
            tempmc = MCXGate(a_len,ctrl_state=ctrlString)
            circuit.append(tempmc,[xR[k]]+aR[0:a_len-1]+[auR],[])

        circuit.barrier()

        circuit.measure(auR, cR)
        for i in range(self.class_n):
            circuit.measure(self.main_mR[self.n+i],oR[i])
        
        


        simulator = Aer.get_backend("qasm_simulator")
        results = execute(circuit,simulator, shots=self.shots).result()
    
        result_dict = results.get_counts(circuit)

        return result_dict
        







if __name__ == "__main__":

    
    class_bit = 1


    k = 3
    t = 1 # random guess


    data_size = 8 # higer number causes the creation of more Qubits. (hard to simulate in my personal laptop!!)
    test_data_points = 1

    exponent = int(math.log(data_size, 2))

    data = np.array(np.arange(data_size), dtype= np.uint8)
    label = np.zeros(data_size)
    label[1::2] = 1
    data= np.flip((((data[:,None] & (1 << np.arange(exponent)))) > 0).astype(int), axis=1)


    trainData,testData,trainLabel,testLabel = train_test_split(data,label,test_size=test_data_points)
   
    print("training data points: {}".format(len(trainLabel)))
    print("testing data points: {}".format(len(testLabel)))

    model = KNeighborsClassifier(n_neighbors=k,algorithm="brute")
    model.fit(trainData,trainLabel)
        
    # evaluate the model and update the accuracies list
    kpredict = model.predict(testData)
    score = accuracy_score(testLabel,kpredict,normalize=True)



    class_bit = 1
    pattern_np = np.concatenate((trainData,trainLabel.reshape(trainLabel.size,1)), axis=1)

    
    # Lesser shots often lead to class undetermined state.

    QKNN_obj = QKNN(pattern_np, n=pattern_np.shape[1]-class_bit,m=pattern_np.shape[0], 
                    class_bit=class_bit, k_neighbours=k, threshold =t, shots=1024)
    QKNN_obj.trainSuperPosition()
    QPredict = []
    
    for x in testData:
        predict = QKNN_obj.fit(x)

        ## Can be simplified Did it in a hasty way!!!
        key_List = np.array(list(predict.keys()))
        required_key = key_List[np.where(key_List.astype('<U1')=="1")[0]]
        if not required_key:
            assert False,"class not determined"
        else:
            val = []
            for key in required_key:
                val.append(predict[key])
            max_i = np.argmax(val)
            QPredict.append(int(required_key[max_i][2]))
    
    Qscore = accuracy_score(testLabel,QPredict,normalize=True)

    print("for  KNN k=%d, accuracy=%.2f%%" % (k, score * 100))
    print("for QKNN k=%d, accuracy=%.2f%%" % (k, Qscore * 100))
    print(testLabel)
    print(kpredict)
    print(QPredict)


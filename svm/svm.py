import numpy as np, random , math
from scipy . optimize import minimize
import matplotlib . pyplot as plt
import matplotlib.patches as mpatches



def kernel(x1,x2,type='RBF',sigma=0.3):
    if type == 'linear':
        x1t = np.transpose(x1)
        return np.dot(x1t,x2)
    elif type == 'polynomial2':
        x1t = np.transpose(x1)
        return math.pow(np.dot(x1t,x2)+1,2)
    elif type == 'RBF':
        return math.exp(-np.linalg.norm(x1-x2, 2)**2/(2*sigma**2))
    else:
        raise ValueError("Unsupported kernel type")

# sum(alpha_i) + 0.5 * sum(sum(alpha_i * alpha_j * target_i * target_j * kernel(x_i, x_j)))
# add slack
def objective(alpha_):
    alpha_product = -sum(alpha_)
    p_matrix = compute_p_matrix(inputs, targets, N)
    for i in range(N):
        for j in range(N):
            procedure = 0.5 * alpha_[i] * alpha_[j] * p_matrix[i][j]
            alpha_product += procedure

   
    return alpha_product

# return sum(alpha_i * target_i) = 0
def zerofun(alpha_):
    return np.dot(alpha_, inputs)

def compute_p_matrix(inputs, targets, N):
    P = []
    for i in range(N):
        A = []
        for j in range(N):
            k = kernel(inputs[i], inputs[j])
            A.append(targets[i]*targets[j]*k)
        P.append(np.array(A))

    return np.array(P)

#Extract the non-zero alpha values
def separate_alpha(alpha, inputs, targets, threshold):
    zeroPoints = []
    zeroTargets = []
    zeroAlpha = []
    svPoints = []
    svTargets = []
    svAlpha = []
    for i in range(len(alpha)):
        if alpha[i]<threshold:
            zeroAlpha.append(alpha[i])
            zeroPoints.append(inputs[i])
            zeroTargets.append(targets[i])
        else:
            svAlpha.append(alpha[i])
            svPoints.append(inputs[i])
            svTargets.append(targets[i])

    return svAlpha, svPoints, svTargets, zeroAlpha, zeroPoints, zeroTargets

def calculate_b(alphas, inputs, targets, C):
    si = 0
    for i in range(len(alphas)):
        if alphas[i] < C:
            si = i
            break
    ans = 0
    for i in range(len(inputs)):
        ans += alphas[i]*targets[i]*kernel(inputs[si], inputs[i])
    return ans - targets[si]

def indicator(sv, alphas, inputs, targets, b):
    sm = 0
    for i in range(len(alphas)):
        sm += alphas[i]*targets[i]*kernel(sv,inputs[i])
    sm -= b
    return sm

def data_generator(case='easy'):
    np.random.seed(100)
    
    if case == 'easy':
        
        classA = np.concatenate(
        (np.random.randn ( 10 , 2) *0.2+ [ 1.5, 0.5 ] ,
        np.random.randn ( 10 , 2)*0.2 + [ -1.5 ,0.5 ] ) )
        classB = np.random.randn(20,2) *0.2+[0.0,-0.5]
    
    elif case == 'hard':
       
        classA = np.concatenate((
        np.random.randn(10, 2) * 0.5 + [1.0, 1.0],
        np.random.randn(10, 2) * 0.5 + [-1.0, 1.0]
        ))
        classB = np.random.randn(20, 2) * 0.8 + [0.0, 0.5]

    
    elif case == 'harder':
        
        classA = np.concatenate((
        np.random.randn(10, 2) * 0.6 + [0.8, 0.8],
        np.random.randn(10, 2) * 0.6 + [-0.8, 0.8]
        ))
        classB = np.random.randn(20, 2) * 1.2 + [0.0, 0.0]

    
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(len(classA)), -np.ones(len(classB))))
    N = inputs.shape[0]
    
    return classA, classB, inputs, targets, N


def minimize_objective(inputs, targets, N, C):
    
    threshold = math.pow(10, -5)

    B=[(0, C) for b in range(N)]
    start=np.zeros(N)
    XC = {'type':'eq', 'fun':zerofun}
    alpha = minimize(objective, start, bounds=B,
    constraints=XC).x
    
    svAlpha, svPoints, svTargets, zeroAlpha, zeroPoints, zeroTargets=\
    separate_alpha(alpha, inputs, targets, threshold)
    b = calculate_b(svAlpha, svPoints, svTargets, C)
    print ("SVM with C={}, alpha={}".format(C, svAlpha))

    return svAlpha, svPoints, svTargets, b



if __name__ == "__main__":
    #cases = ['hard']
    cases = ['easy', 'hard', 'harder']
    
    C_seq = [1, 5, 10]
    #C_seq = [5]
    for C in C_seq:
        for case in cases:
            print(f"\nTesting {case} case:")
            classA, classB, inputs, targets, N = data_generator(case)
            
            try:
                svAlpha, svPoints, svTargets, b = minimize_objective(inputs, targets, N, C)
                
                plt.figure(figsize=(8, 6))
                plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b+')
                plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
                plt.axis('equal')
                plt.savefig(f'./figure/data_{case}.png')
                
                xgrid = np.linspace(-5, 5)
                ygrid = np.linspace(-4, 4)
                grid = np.array([[indicator(np.array([x,y]),
                    svAlpha, svPoints, svTargets, b)
                    for x in xgrid]
                    for y in ygrid])
                
                plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                    colors=('red', 'black', 'blue'),
                    linewidths=(1,3,1))
                
                plt.title(f'SVM Classification - {case} - {C} case')
                plt.savefig(f'./figure/svm_{case}_{C}.png')
                # plt.show()
                
            except Exception as e:
                print(f"Optimizer failed for {case} case:")
                print(e)
    
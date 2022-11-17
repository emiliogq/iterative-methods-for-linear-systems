
import numpy as np;
import math as m;

# np.set_printoptions(precision=8)
import fractions
np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

def diagonal(matrix):
    d = np.diag(np.diag(matrix))
    # print("Diagonal of A")
    # print("")
    # print(d)
    # print("")
    return d

def upper(matrix):
    u = np.triu(matrix, 1)
    # print("Upper of A")
    # print("")
    # print(u)
    # print("")
    return u
    
def inverse(matrix):
    inverse = np.linalg.inv(matrix)
    # print("A-1")
    # print("")
    # print(inverse)
    # print("")
    return inverse

def lower(matrix):
    l = np.tril(matrix,-1)
    # print("Lower of A")
    # print("")
    # print(l)
    # print("")
    return l

def get_jacobi_iter_matrix (matrix):
    print("-------------------")
    print("Jacobi iterative matrix (B)")
    print("-------------------")
    d_inverse = inverse(diagonal(matrix)) 
    l_u = lower(matrix)+upper(matrix)
    b = np.dot(-1 * d_inverse,(l_u))
    print("Jacobi B = -D^-1·(L+U)")
    print("")
    print(b)
    print("")
    return b

def get_jacobi_iter_vector(matrix,b):
    print("-------------------")
    print("Jacobi iterative array (c)")
    print("-------------------")
    c = np.dot(inverse(diagonal(matrix)), b)
    print(c)
    return c

def get_gauss_seidel_iter_matrix(matrix):
    print("-------------------")
    print("Gauss-Seidel iterative matrix (B)")
    print("-------------------")
    d = diagonal(matrix)
    l = lower(matrix)
    u = upper(matrix)
    b = np.dot((-1 * inverse(l+d)),u)
    np.savetxt("b_gauss.csv",b,delimiter=",")
    print("Gauss-Seidel B = -((L+D)^-1)·U")
    print("")
    print(b)
    print("")
    return b

def get_gauss_seidel_iter_vector(A, b):
    print("-------------------")
    print("Gauss-Seidel iterative array (c)")
    print("-------------------")
    c = np.dot(inverse(lower(A)+diagonal(A)), b) 
    print(c)
    return c

def eigval(matrix):
    eigvals = np.linalg.eigvals(matrix)
    # print("eigen values")
    # print("")
    # print(eigvals)
    # print("")
    return np.linalg.eigvals(matrix)

def spectral_radius(eigen_values):
    sp = np.max(np.absolute(eigen_values))
    print("spectral radius")
    print("")
    print(sp)
    print("")
    return sp

def generate_square_matrix(p, size=10):
    d = np.fromfunction(lambda i, j: (i == j) * (p-5), (size, size), dtype=int)
    u = np.fromfunction(lambda i, j: (j>i) * (1+j-i), (size, size), dtype=int)
    l = np.fromfunction(lambda i, j: (j<i) * (11+j-i), (size, size), dtype=int)
    A = u+d+l
    return A

def jacobi(A,b,end_condition, x = np.zeros((10,1)), x_start=0):
    print("-------------------")
    print("Jacobi Method")
    print("-------------------") 
    B = get_jacobi_iter_matrix(A)
    c = get_jacobi_iter_vector(A,b)                                                                                                                                             
    iterative(A, B,x,b,c,end_condition, x_start)

def gauss_seidel(A,b,end_condition, x = np.zeros((10,1)), x_start = 0):
    B = get_gauss_seidel_iter_matrix(A)
    c = get_gauss_seidel_iter_vector(A, b)
    iterative(A, B,x,b, c, end_condition, x_start)

def iterative(A, B,x, b, c, end_condition, x_start=0):
    k=int(0)
    r = (b - np.dot(A,x))
    e = np.linalg.norm(r)
    min_e = e
    min_k = k
    min_x = x
    while not end_condition(k,e):       
        print("x^"+str(k))
        print("----------")
        print("")
        print(x[x_start:])
        print("")
        print("r^"+str(k))
        print("----------")
        print("")
        print(r[x_start:])
        print("")
        print("E^"+str(k))
        print("----------")
        print("")
        print(e)
        print("")

        x = (np.dot(B,x)+c)
        r = (b - np.dot(A,x))
        e = np.linalg.norm(r)

        k=k+1

def estimate_iterations(spectral_radius, order_error):
    print("Iterations (k)")
    print("-----------------")
    print("")
    k = (order_error - m.log(0.5,10)) / -m.log(spectral_radius, 10)
    print("k >= "+str(k))

def exercise_2():
    print("EX 2")
    print("-----------")
    print("")
    A = np.array(
        [(6, 2, 0),
        (2, 3, 0),
        (1, -10, 3)
        ]
    )

    iter_matrix = get_jacobi_iter_matrix(A)
    sp = spectral_radius( eigval(iter_matrix) )
    estimate_iterations(sp, 4.0)
   

    iter_matrix = get_gauss_seidel_iter_matrix(A)
    sp = spectral_radius( eigval(iter_matrix) )
    estimate_iterations(sp, 4.0)


def exercise_3():
    print("EX 3")
    print("-----------")
    print("")
    b = np.zeros((10,1))
    b[0] = 1
    
    print("3a and 3b (Begin)")
    print("-----------")
    matrix = generate_square_matrix(26)
    end_condition = lambda k,e : k <= 4
    jacobi(matrix, b, end_condition, x_start=6)
    gauss_seidel(matrix,b,end_condition,x_start=6)

    print("-----------")
    print("3a and 3b (End)")

    print("3c (Begin)")
    print("-----------")
    euclidian_limit = m.pow(10,-10)
    end_condition = lambda k,e : k >= 100 or e <= euclidian_limit
    jacobi(matrix,b,end_condition, x_start=0)
    gauss_seidel(matrix,b,end_condition, x_start=0)
    print("-----------")
    print("3c (End)")

    
    print("3d (Begin)")
    print("-----------")
    matrix = generate_square_matrix(75)
    jacobi(matrix,b,end_condition, x_start=0)
    gauss_seidel(matrix,b,end_condition, x_start=0)
    # print("-----------")
    # print("3c (End)")

# exercise_2()

exercise_3()
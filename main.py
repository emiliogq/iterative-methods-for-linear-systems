
import numpy as np;
import math as m;

# np.set_printoptions(precision=8)
# import fractions
# np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

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
    # print("-------------------")
    # print("Jacobi iterative matrix (B)")
    # print("-------------------")
    d_inverse = inverse(diagonal(matrix)) 
    l_u = lower(matrix)+upper(matrix)
    b = np.dot(-1 * d_inverse,(l_u))
    # print("Jacobi B = -D^-1·(L+U)")
    # print("")
    # print(b)
    # print("")
    return b

def get_jacobi_iter_vector(matrix,b):
    # print("-------------------")
    # print("Jacobi iterative array (c)")
    # print("-------------------")
    c = np.dot(inverse(diagonal(matrix)), b)
    # print(c)
    return c

def get_gauss_seidel_iter_matrix(matrix):
    # print("-------------------")
    # print("Gauss-Seidel iterative matrix (B)")
    # print("-------------------")
    d = diagonal(matrix)
    l = lower(matrix)
    u = upper(matrix)
    # print("")
    # print("(L+D)^-1: ")
    inverse_ld = -1 * inverse(l+d)
    # print(inverse_ld)
    b = np.dot(inverse_ld,u)
    np.savetxt("b_gauss.csv",b,delimiter=",")
    # print("")
    # print("Gauss-Seidel B = -((L+D)^-1)·U")
    # print("")
    # print(b)
    # print("")
    return b

def get_gauss_seidel_iter_vector(A, b):
    # print("-------------------")
    # print("Gauss-Seidel iterative array (C)")
    # print("-------------------")
    c = np.dot(inverse(lower(A)+diagonal(A)), b) 
    # print("")
    # print("C = (L+D)^-1·b")
    # print(c)
    # print("")
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
    # print("spectral radius")
    # print("")
    # print(sp)
    # print("")
    return sp

def generate_square_matrix(p, size=10):
    d = np.fromfunction(lambda i, j: (i == j) * (p-5), (size, size), dtype=int)
    u = np.fromfunction(lambda i, j: (j>i) * (1+j-i), (size, size), dtype=int)
    l = np.fromfunction(lambda i, j: (j<i) * (11+j-i), (size, size), dtype=int)
    A = u+d+l
    return A

def print_iteration_variable( ivar : tuple):
        print(ivar[0])
        print("----------")
        print("")
        print(ivar[1])
        print("")
        print("----------")
        print("")

def iterative(B, x, c, end_condition):
    k = int(0)
    x_s = np.array([(1, 2, -1, 1)]).transpose()
    x_prev = x
    norm_x_s = np.linalg.norm(x_s)
    approximated_relative_error = int(1)
    previous_relative_error = relative_error = np.linalg.norm(x - x_s) / norm_x_s

    ivars = [
        ("x^"+str(k), x), 
        ("x^s", x_s), 
        # ("x^"+str(k-1), x_prev), 
        ("E^"+str(k)+"  = || x^"+str(k)+" - x^("+str(k-1)+")|| / || x^"+str(k)+" ||", approximated_relative_error),
        # ("E^"+str(k-1), previous_relative_error) 
        ("e^"+str(k)+" = || x^"+str(k)+" - x_s || / || x_s ||", relative_error), 
        ("e^"+str(k-1), previous_relative_error)
    ]
    
    for ivar in ivars:
        print_iteration_variable(ivar)
    
    while not end_condition(k,relative_error):       

        x_prev = x
        previous_relative_error = relative_error
        # x^(k+1) = B · x^(k) + c
        x = ( np.dot(B,x) + c )
        # Ex 3e
        approximated_relative_error = np.linalg.norm( x - x_prev) / np.linalg.norm(x)
        
        relative_error = np.linalg.norm(x - x_s) / norm_x_s
        k = k + 1

        ivars = [
            ("x^"+str(k), x), 
            ("x^s", x_s), 
            # ("x^"+str(k-1), x_prev), 
            ("E^"+str(k)+"  = || x^"+str(k)+" - x^("+str(k-1)+")|| / || x^"+str(k)+" ||", approximated_relative_error),
            # ("E^"+str(k-1), previous_relative_error) 
            ("e^"+str(k)+" = || x^"+str(k)+" - x_s || / || x_s ||", relative_error), 
            ("e^"+str(k-1), previous_relative_error)
        ]
        
        for ivar in ivars:
            print_iteration_variable(ivar)
        

def estimate_iterations(spectral_radius, order_error):
    # print("Iterations (k)")
    # print("-----------------")
    # print("")
    k = (-(m.log(0.5,10)) + order_error) / -m.log(spectral_radius, 10)
    # print("k >= "+str(k))

def exercise_3():
    print("EX 3")
    print("-----------")
    print("")
    A = np.array(
        [(10, -1, 2, 0),
        (-1, 11, -1, 3),
        (2,-1,10,-1),
        (0, 3,-1, 8)
        ]
    )
    b = np.array([(6, 25, -11, 15)]).transpose()
    x = np.zeros((4,1))

    print("3a")
    print("-----------")
    print("Jacobi iterative matrix (B)")
    print("-------------------")
    B_J = get_jacobi_iter_matrix(A)

    print("-----------")
    print("Jacobi iterative array (C)")
    print("-------------------")
    C_J = get_jacobi_iter_vector(A, b)
    
    print("-----------")
    print("Gauss-Seidel iterative Matrix (B)")
    print("-------------------")
    
    B_GS = get_gauss_seidel_iter_matrix(A)
    print("-----------")
    print("Gauss-Seidel iterative array (C)")
    print("-------------------")
    C_GS = get_gauss_seidel_iter_vector(A,b)

    # print("-----------")

    # print("3b")
    # print("-----------")
    # spectral_radius_jacobi = spectral_radius( eigval(B_J) )
    # spectral_radius_gs = spectral_radius( eigval(B_GS) )
    # print("-----------")
    
    # print("3c")
    # print("-----------")
    # estimate_iterations(spectral_radius_jacobi, 5.0)
    # estimate_iterations(spectral_radius_gs, 5.0)
    
    print("-----------")
    print("3d limited by iterations")
    print("-----------")

    end_condition = lambda k,e : k == 15
    iterative(B_J, x, C_J, end_condition)
    end_condition = lambda k,e : k == 6
    iterative(B_GS, x, C_GS, end_condition)
   
    print("-----------")

    print("3d limited by error")
    print("-----------")
    
    euclidian_limit = 0.5 * m.pow(10,-5)
    end_condition = lambda k,e : e <= euclidian_limit
    iterative(B_J, x, C_J, end_condition)
    iterative(B_GS, x, C_GS, end_condition)
    print("-----------")

    # print("3e")
    # print("-----------")

    # euclidian_limit = 0.5 * m.pow(10,-11)
    # end_condition = lambda k,e : e <= euclidian_limit
    # iterative(B_J, x, C_J, end_condition)
    # iterative(B_GS, x, C_GS, end_condition)
    
    # print("-----------")

   

exercise_3()
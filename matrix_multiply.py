
import random
from timeit import default_timer as timer

from pyspark.mllib.linalg.distributed import *
from pyspark.sql import SparkSession
from scipy.sparse import lil_matrix

app_name = 'PySpark Matrix Multiplication Example'
master = 'local'
spark = SparkSession.builder.appName(app_name).master(master).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

N = 2

def create_matrix(size, max_value):
    return [[random.randint((max_value * -1), max_value) for i in range(size)] \
            for j in range(size)]

def create_empty_matrix(size):
    return [[0 for i in range(size)] for j in range(size)]

def matrix_multiply(A, B, C, size):
    for i in range(size):
        for j in range(size):
            total = 0
            for k in range(size):
                total += A[i][k] * B[k][j] 
            C[i][j] = total

    return C

def as_block_matrix(rdd, rows, columns):
    return IndexedRowMatrix(
        rdd.zipWithIndex().map(lambda i: IndexedRow(i[1], i[0]))
    ).toBlockMatrix(rows, columns)

def indexedrowmatrix_to_array(matrix):
    result = lil_matrix((matrix.numRows(), matrix.numCols()))

    for indexed_row in matrix.rows.collect():
        result[indexed_row.index] = indexed_row.vector

    return result

A = create_matrix(N, 500)
B = create_matrix(N, 500)
C = create_empty_matrix(N)

print('Performing standard matrix multiplication')

start = timer() 
C = matrix_multiply(A, B, C, N) 
end = timer() 

print('Best Sequential execution time (seconds):', end - start)

A_rdd = spark.sparkContext.parallelize(A)
B_rdd = spark.sparkContext.parallelize(B)

start = timer() #
C_matrix = as_block_matrix(A_rdd, N, N).multiply(as_block_matrix(B_rdd, N, N)) #
end = timer() #

print('Apache Spark execution time (seconds):', end - start)

result = indexedrowmatrix_to_array(C_matrix.toIndexedRowMatrix())

if N <= 4:
    print("Printing sequential result matrix.")
    for row in C:
        print(row)
    print("Printing Spark result matrix")
    print(result)

import scala.io.Source
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import java.io.File

val filename = "iris-data.txt"
val data = io.Source.fromFile(filename).getLines().map(_.split(",").map(_.trim.toDouble)).toArray

val jf = new File("iris-data.txt")
val data = csvread(jf)

val features = data(::, 1 to 4)
val y_train = data(::, 1)

val ones = DenseMatrix.ones[Double](x_train.rows, 1)
val x_train = DenseMatrix.horzcat(ones, features)

val nrow = x_train.rows
val ncol = x_train.cols

var weights = DenseVector.ones[Double](ncol)

for (i <- 0 to 1000){
val output = x_train * weights
println(output(0 to 5))
println(y_train(0 to 5))
println("")
val error = y_train - output
print(sum(abs(error)) + "   ")
val gradient = (x_train.t * error) :/ nrow.toDouble
weights = weights + gradient :* .01
}

val predictions = x_train * weights
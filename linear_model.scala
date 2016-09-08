import scala.io.Source
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import java.io.File

val jf = new File("iris-data.txt")
val data = csvread(jf)

val x_train = data(::, 1 to 4)
val y_train = data(::, 0)

def linear_model_batch(x: BDM[Double], y_train: BDV[Double], lr:Double = .01,
					 num_iters:Int = 1000):BDV[Double] = {

	val ones = DenseMatrix.ones[Double](x.rows, 1)
	val x_train = DenseMatrix.horzcat(ones, x)

	val nrow = x_train.rows
	val ncol = x_train.cols

	var weights = DenseVector.ones[Double](ncol) :* .01

	for (i <- 0 to num_iters){
		val output = x_train * weights
		val error = y_train - output
		println("Train Error = " + (sum(abs(error)) / nrow.toDouble))
		println("")
		val gradient = (error.t * x_train) :/ nrow.toDouble
		weights = weights + (gradient :* lr).t
	}
	weights
}

def evaluate(weights: BDV[Double], x: BDM[Double]): BDV[Double] = {
	val ones = DenseMatrix.ones[Double](x.rows, 1)
	val x_test = DenseMatrix.horzcat(ones, x)
	val predictions = x_test * weights
	predictions
}

val weights = linear_model_batch(x_train, y_train)
val predictions = evaluate(weights, x_train)




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

def linear_model(x: BDM[Double], y_train: BDV[Double], lr:Double = .01,
					 num_iters:Int = 10000, optimizer:String = "batch"):BDV[Double] = {

	val ones = DenseMatrix.ones[Double](x.rows, 1)
	val x_train = DenseMatrix.horzcat(ones, x)

	if (optimizer == "batch"){

		val nrow = x_train.rows
		val ncol = x_train.cols
		var weights = DenseVector.ones[Double](ncol) :* .01

		for (i <- 0 to num_iters){
			val output = x_train * weights
			val error = y_train - output
			val gradient = (error.t * x_train) :/ nrow.toDouble
			weights = weights + (gradient :* lr).t
		}
		weights
	}

	else { //optimizer == "normal"
		val weights = pinv(x_train) * y_train
		weights
	}
}

def evaluate(weights: BDV[Double], x: BDM[Double]): BDV[Double] = {
	val ones = DenseMatrix.ones[Double](x.rows, 1)
	val x_test = DenseMatrix.horzcat(ones, x)
	val predictions = x_test * weights
	predictions
}

val weights = linear_model(x_train, y_train, num_iters=100000, optimizer="normal")
val predictions = evaluate(weights, x_train)




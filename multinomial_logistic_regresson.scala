import scala.io.Source
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import java.io.File

val jf = new File("iris-data.txt")
val data = csvread(jf)

val num_iters:Int = 1000

val features = data(::, 0 to 3)
val labels = data(::, 4)-1.0

def logistic_regression(x: BDM[Double], y: BDV[Double], lr:Double = .01,
					 num_iters:Int = 1000):BDM[Double] = {
	val ones = DenseMatrix.ones[Double](x.rows, 1)
	val x_train = DenseMatrix.horzcat(ones, x)
	val y_train = convert(y, Int)

	val nrow = x_train.rows
	val ncol = x_train.cols
	val nclasses = y.toArray.distinct.length

	var weights = DenseMatrix.ones[Double](ncol, nclasses) :* 1.0/nclasses

	for (iterations <- 0 to num_iters){
		val output = x_train * weights
		val scores = exp(output)
		val divisor = sum(scores(*, ::))

		for (i <- 0 to scores.cols-1){
			scores(::, i) := scores(::, i) :/ divisor
		}

		val zeroes = DenseMatrix.zeros[Double](scores.rows,scores.cols)

		def conditional(value: Int, seek: Int):Int = {if (value == seek){(-1)} else {0}} 

		for (i <- 0 to scores.cols-1){
			scores(::, i) := scores(::, i) + convert(y_train.map(x => conditional(x, i)), Double)
		}

		val gradient = (scores.t * x_train).t

		weights -= gradient :* lr
	}
weights
}

def evaluate(weights: BDM[Double], x: BDM[Double]): BDV[Int] = {
	val ones = DenseMatrix.ones[Double](x.rows, 1)
	val x_test = DenseMatrix.horzcat(ones, x)
	val predictions = argmax(x_test * weights, Axis._1)
	predictions
}

val w = logistic_regression(features, labels)
val predictions = evaluate(w, features)


import scala.io.Source
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import java.io.File
import scala.util.Random

val jf = new File("iris.csv")
val data = csvread(jf)

val features = data(::, 0 to 3)
val labels = data(::, 4)

val x = features
val y = labels
val lr = .01
val num_iters = 1000
val nhidden = 100

def neural_network(x: BDM[Double], y: BDV[Double], lr:Double = .01,
	nhidden:Int=100, num_iters:Int = 1000):(BDM[Double], BDM[Double]) = {

	val ones = DenseMatrix.ones[Double](x.rows, 1)
	val x_train = DenseMatrix.horzcat(ones, x)
	val y_train = convert(y, Int)

	val nrow = x_train.rows
	val ncol = x_train.cols
	val nclasses = y.toArray.distinct.length

	val r = scala.util.Random
	var weights1 = DenseMatrix.ones[Double](ncol, nhidden).map(x => r.nextDouble-0.5) :* .01
	var weights2 = DenseMatrix.ones[Double](nhidden, nclasses).map(x => r.nextDouble-0.5) :* .01

	for (iterations <- 0 to num_iters){

		// Forwards
		val hidden_raw = x_train * weights1
		val hidden_relu = hidden_raw.map(x => max(0.0, x))
		val output_raw = hidden_relu * weights2
		val softmax = exp(output_raw)
		val divisor = breeze.linalg.sum(softmax(*, ::))

		for (i <- 0 to softmax.cols-1){
			softmax(::, i) := softmax(::, i) :/ divisor
		}

		// Backwards
		val zeroes = DenseMatrix.zeros[Double](softmax.rows, softmax.cols)
		def conditional(value: Int, seek: Int):Int = {if (value == seek){(-1)} else {0}} 

		for (i <- 0 to softmax.cols-1){
			softmax(::, i) := softmax(::, i) + convert(y_train.map(x => conditional(x, i)), Double)
		}

		def relu_grad(value: Double):Double = {if (value <= 0) {0} else {1}}

		val gradient2 = hidden_relu.t * softmax :/ nrow.toDouble
		val grad_through_weights = softmax * weights2.t
		val grad_through_relu = grad_through_weights :* hidden_relu.map(relu_grad)
		val gradient1 = x_train.t * grad_through_relu

		weights1 -= lr * gradient1
		weights2 -= lr * gradient2
	}
	(weights1, weights2)
}

def evaluate(weights1:BDM[Double], weights2:BDM[Double], 
	x:BDM[Double]):BDV[Int] = {
	val ones = DenseMatrix.ones[Double](x.rows, 1)
	val x_test = DenseMatrix.horzcat(ones, x)
	val hidden_raw = x_test * weights1
	val hidden_relu = hidden_raw.map(x => max(0.0, x))
	val output_raw = hidden_relu * weights2
	val predictions = argmax(output_raw, Axis._1)
	predictions
}

val (w1, w2) = neural_network(features, labels)
val predictions = evaluate(w1, w2, x)

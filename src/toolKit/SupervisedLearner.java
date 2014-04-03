// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

package toolKit;

import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;

import clustering.Kmeans;

public abstract class SupervisedLearner {

	// Before you call this method, you need to divide your data
	// into a feature matrix and a label matrix.
	public abstract void train(Matrix features, Matrix labels) throws Exception;

	// A feature vector goes in. A label vector comes out. (Some supervised
	// learning algorithms only support one-dimensional label vectors. Some
	// support multi-dimensional label vectors.)
	public abstract void predict(double[] features, double[] labels) throws Exception;

	// The model must be trained before you call this method. If the label is nominal,
	// it returns the predictive accuracy. If the label is continuous, it returns
	// the root mean squared error (RMSE). If confusion is non-NULL, and the
	// output label is nominal, then confusion will hold stats for a confusion matrix.
	public double measureAccuracy(Matrix features, Matrix labels, Matrix confusion) throws Exception
	{
		labels = new Matrix(features, 0, 0, features.rows(), features.cols());
		if(features.rows() != labels.rows())
			throw(new Exception("Expected the features and labels to have the same number of rows"));

		if(features.rows() == 0)
			throw(new Exception("Expected at least one row"));

		int labelValues = labels.valueCount(0);
		double sse = 0.0;
		for (int i = 0; i < features.rows(); i++) {
			double[] feat = features.row(i);
			double[] targ = labels.row(i);

			predict(feat, targ);
			sse += Math.pow(this.calculateDistance(feat, targ), 2);
		}
		return sse;
	}

	public double calculateDistance(double[] feat, double[] targ) {
		return 0;
	}
}

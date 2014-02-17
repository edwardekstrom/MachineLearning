package backprop;

import toolKit.Matrix;

public class BackpropInstance {
	public double[] _features;
	public double[] _labels;
	
	public BackpropInstance(Matrix features, Matrix labels, int row){
		_features = new double[features.cols()];

		for(int i = 0 ; i < features.cols(); i++){
			_features[i] = features.get(row, i);
		}

		_labels = new double[labels.cols()];
		
		for(int i = 0 ; i < labels.cols(); i++){
			_labels[i] = labels.get(row, i);
		}
	}
}

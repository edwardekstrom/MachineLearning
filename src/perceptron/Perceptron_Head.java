package perceptron;

import java.util.Random;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class Perceptron_Head{
	Random _rand;
	int _epoch;
	static double LEARNING_RATE = .1;
	
	Epoch _bestEpoch;

	public Perceptron_Head(Random rand){
		_rand = rand;
		_epoch = 0;
		_bestEpoch = new Epoch(null);
	}
	
	public void train(Matrix features, Matrix labels) throws Exception {
		double[] _weightVector = new double[features.cols() + 1];
		
		for (int k = 0 ; k < _weightVector.length ; k++){
			_weightVector[k] = .1;
		}
		double net;
		double inference;
		double misinferred;
		int epochsWOImpovement = 0;
		boolean improvement = true;
		while (improvement) {
			misinferred = 0;
			for (int i = 0; i < features.rows(); i++) {
				net = 0;
				for (int j = 0; j < features.cols(); j++) {
					net = net + features.get(i, j) * _weightVector[j];
				}
				net = net + 1*_weightVector[_weightVector.length-1];
						
				if(net > 0){
					inference = 1;
				}else{
					inference = 0;
				}
				if(inference != labels.get(i, 0)){
					misinferred = misinferred + 1.0;
					for(int j = 0; j<features.cols();j++){
						double curW = _weightVector[j];
						double deltaW = LEARNING_RATE * (labels.get(i, 0) - inference) * features.get(i, j);
						_weightVector[j] = curW + deltaW;
					}
					double biasW = _weightVector[_weightVector.length - 1];
					double deltaBiasW = LEARNING_RATE * (labels.get(i, 0) - inference) * 1;
					_weightVector[_weightVector.length - 1] = biasW + deltaBiasW;
				}
				
			}
			double[] copy = new double[_weightVector.length];
			for(int k = 0; k<_weightVector.length ; k++) copy[k] = _weightVector[k];
			Epoch curEpoc = new Epoch(copy);
			curEpoc._error = misinferred / (double)features.rows();
//			System.out.println(_epoch + ", " + curEpoc._error);
//			System.out.println(curEpoc._error);
			if(curEpoc._error < _bestEpoch._error){
				_bestEpoch = curEpoc;
				epochsWOImpovement = 0;
			}else{
				epochsWOImpovement++;
			}
			
			if(epochsWOImpovement > 100){
				break;
			}
//			features.shuffle(_rand, labels);
			_epoch++;
		}
		
		System.out.println("Number of Epochs Required: " + _epoch);
		for(int i = 0 ; i < _bestEpoch._weights.length; i++){
			System.out.println(_bestEpoch._weights[i]);
		}
//		System.out.println(_epoch);
	}

	public void predict(double[] features, double[] labels) throws Exception {
		double net = 0.0;
		for(int i = 0 ;i<features.length; i++){
			net += features[i]*_bestEpoch._weights[i];
		}
		net += 1*_bestEpoch._weights[_bestEpoch._weights.length - 1];
//		if(net > 0){
			labels[0] = net;
//		}else{
//			labels[0] = 0;
//		}
		
	}
	
	private class Epoch{
		double[] _weights;
		double _error;
		
		private Epoch(double[] weights){
			_weights = weights;
			_error = 1;
		}
	}
}

package backprop;

import java.util.Random;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class Backprop extends SupervisedLearner {
	private Integer[] _structure;
	private int _numInputNodes;
	private int _numOutputNodes;
	
	public Backprop(Random rand, Integer[] structure){
		_structure = structure;
		_numInputNodes = _structure[0];
		_numOutputNodes = _structure[_structure.length];
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
		
	}
}

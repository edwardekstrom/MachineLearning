package backprop;

import java.util.Random;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class NeuralNet extends SupervisedLearner {
	private Random _rand;
	private Integer[] _structure;
	private int _numInputNodes;
	private int _numOutputNodes;
	private NeuralLayerI[] _layers;
	
	public NeuralNet(Random rand, Integer[] structure){
		_rand = rand;
		_structure = structure;
		_numInputNodes = _structure[0];
		_numOutputNodes = _structure[_structure.length];
		
		_layers = new NeuralLayerI[_structure.length];
		
		for(int i = 0; i < structure.length - 1; i++){
			_layers[i] = new NeuralLayerI(_structure[i] + 1, rand, false);
		}
		_layers[_layers.length - 1] = new NeuralLayerI(_structure[_structure.length - 1], rand, true);
		
		for(int i = 0; i < _layers.length - 1; i++){
			_layers[i].connectToLayer(_layers[i+1]);
		}
		
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		for(int i = 0; i < features.rows(); i++){
			BackpropInstance instance = new BackpropInstance(features, labels, i);
		}
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
		
	}
	
	
}

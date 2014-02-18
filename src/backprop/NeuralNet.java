package backprop;

import java.util.Random;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class NeuralNet extends SupervisedLearner {
	static double LEARNING_RATE = .1;
	
	private Random _rand;
	private int[] _structure;
	private int _numInputNodes;
	private int _numOutputNodes;
	private NeuralLayerI[] _layers;
	
	private double _expectedOutput;
	
	public NeuralNet(Random rand, int[] structure){
		_rand = rand;
		_structure = structure;
		_numInputNodes = _structure[0];
		_numOutputNodes = _structure[_structure.length - 1];
		
		_layers = new NeuralLayerI[_structure.length];
		
		for(int i = 0; i < structure.length - 1; i++){
			_layers[i] = new NeuralLayerI(_structure[i], rand, false);
		}
		_layers[_layers.length - 1] = new NeuralLayerI(_structure[_structure.length - 1], rand, true);
		
		for(int i = 0; i < _layers.length - 1; i++){
			_layers[i].connectToLayer(_layers[i+1]);
		}
		
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
//		features.shuffle(_rand, labels);
		
		int epochsWithoutImprovement = 0;
		
		while (epochsWithoutImprovement < 100) {
			double missedHypotheses = 0;
			for (int i = 0; i < features.rows(); i++) {
				BackpropInstance instance = new BackpropInstance(features, labels, i);
				double hypothesis = trainOnInstance(instance);
				
				if(hypothesis != labels.get(i, 0)) missedHypotheses++;
				
//				_expectedOutput = labels.get(i, 0);
				updateWeights(instance._labels);
			}
//			features.shuffle(_rand, labels);
			epochsWithoutImprovement++;
		}
	}
	
//	public void updateWeights() {
//
//
//		CalculateSignalErrors();
//		BackPropagateError();
//
//	}

	private void updateWeights(double[] labels) {
		
		NeuralNode[] outputNodes = _layers[_layers.length - 1].getNodes();
		NeuralNode[] prevNodes = _layers[_layers.length - 2].getNodes();
		for(int i = 0 ; i < prevNodes.length; i++){
			for(int j = 0 ; j < outputNodes.length ; j++){
				double deltaWeight;
				if ((int) labels[0] == j) {
					outputNodes[j].setDelta((1 - outputNodes[j].getOutput())
							* outputNodes[j].getOutput()
							* (1 - outputNodes[j].getOutput()));

					deltaWeight = LEARNING_RATE * outputNodes[j].getDelta()
							* prevNodes[i].getOutput();
				} else {
					outputNodes[j].setDelta((0 - outputNodes[j].getOutput())
							* outputNodes[j].getOutput()
							* (1 - outputNodes[j].getOutput()));

					deltaWeight = LEARNING_RATE * outputNodes[j].getDelta()
							* prevNodes[i].getOutput();
				}
				prevNodes[i].setToUpdate(outputNodes[j], deltaWeight);
			}
		}
		
		for(int layer = _layers.length - 2; layer > 0; layer--){
			NeuralNode[] curNodes = _layers[layer].getNodes();
			prevNodes = _layers[layer - 1].getNodes();
			
			for(int i = 0; i < prevNodes.length; i++){
				double deltaWeight;
				double delta;
				for(int j = 0; j < curNodes.length -1; j++){
					delta = curNodes[j].getDeltaSum() * prevNodes[i].getOutput();
					curNodes[j].setDelta(delta);
					deltaWeight = LEARNING_RATE * delta * prevNodes[i].getOutput();
					prevNodes[i].setToUpdate(curNodes[j], deltaWeight);
				}
			}
		}
		
		for(int layer = 0 ; layer < _layers.length - 1; layer++){
			NeuralNode[] curNodes = _layers[layer].getNodes();
			for(int i = 0 ; i < curNodes.length; i++){
				curNodes[i].update();
			}
		}
	}
	



	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		BackpropInstance instance = new BackpropInstance(features, labels);
		labels[0] = trainOnInstance(instance);
		
	}
	
	private double trainOnInstance(BackpropInstance instance){
		NeuralLayerI inputLayer = _layers[0];
		for(int i = 0; i < instance._features.length; i++){
			inputLayer.getNodes()[i].setOutput(instance._features[i]);
		}
		
		for(int i = 1; i < _layers.length ; i++){
			NeuralLayerI curLayer = _layers[i];
			NeuralNode[] curNodes = curLayer.getNodes();
			
			NeuralLayerI prevLayer = _layers[i - 1];
			NeuralNode[] prevNodes = prevLayer.getNodes();
			
			if(curLayer._isOutputLayer){
				for (int j = 0; j < curNodes.length; j++) {
					double net = 0;
					for (int k = 0; k < prevNodes.length; k++) {
						net += prevNodes[k].getOutput()
								* prevNodes[k].getWeightTo(curNodes[j]);
					}
					curNodes[j].setNet(net);
					curNodes[j]
							.setOutput(((double) 1.0) / (1 + Math.exp(-net)));
				}
			} else {
				for (int j = 0; j < curNodes.length - 1; j++) {
					double net = 0;
					for (int k = 0; k < prevNodes.length; k++) {
						net += prevNodes[k].getOutput()
								* prevNodes[k].getWeightTo(curNodes[j]);
					}
					curNodes[j].setNet(net);
					curNodes[j]
							.setOutput(((double) 1.0) / (1 + Math.exp(-net)));
				}
			}
		}
		
		double highest = -1.1;
		double toReturn = -1.0;
		NeuralNode[] outputNodes = _layers[_layers.length - 1].getNodes();
		for(int i = 0; i < outputNodes.length ; i ++){
			if(outputNodes[i].getOutput() > highest){
				highest = outputNodes[i].getOutput();
				toReturn = (double)i;
			}
		}
//		System.out.println(toReturn);
		return toReturn;
	}
	
	
//	private void CalculateSignalErrors() {
//
//		int i,j,k,OutputLayer;
//		double Sum;
//
//		OutputLayer = _layers.length-1;
//
//	       	// Calculate all output signal error
//		for (i = 0; i < _layers[OutputLayer].getNodes().length; i++){
//			double temp;
//			if(i == _expectedOutput) temp = 1;
//			else temp = 0;
//			_layers[OutputLayer].getNodes()[i].setDelta( 
//				 (temp - 
//					_layers[OutputLayer].getNodes()[i].getOutput()) * 
//					_layers[OutputLayer].getNodes()[i].getOutput() * 
//					(1-_layers[OutputLayer].getNodes()[i].getOutput())   );
//		}
//
//	       	// Calculate signal error for all nodes in the hidden layer
//		// (back propagate the errors)
//		for (i = _layers.length-2; i > 0; i--) {
//			for (j = 0; j < _layers[i].getNodes().length; j++) {
//				Sum = 0;
//
//				for (k = 0; k < _layers[i+1].getNodes().length; k++)
//					Sum = Sum + _layers[i].getNodes()[j].getWeightTo(_layers[i+1].getNodes()[k]) * 
//						_layers[i+1].getNodes()[k].getDelta();
//
//				_layers[i].getNodes()[j].setDelta( 
//					 _layers[i].getNodes()[j].getOutput()*(1 - 
//						_layers[i].getNodes()[j].getOutput())*Sum );
//			}
//		}
//
//	}
//	
//	private void BackPropagateError() {
//
//		int i,j,k;
//
//		// Update Weights
//		for (i = _layers.length-1; i > 0; i--) {
//			for (j = 0; j < _layers[i].getNodes().length; j++) {
//				// Calculate Bias weight difference to node j
//				Layer[i].Node[j].ThresholdDiff 
//					= LearningRate * 
//					Layer[i].Node[j].SignalError + 
//					Momentum*Layer[i].Node[j].ThresholdDiff;
//
//				// Update Bias weight to node j
//				Layer[i].Node[j].Threshold = 
//					Layer[i].Node[j].Threshold + 
//					Layer[i].Node[j].ThresholdDiff;
//
//				// Update Weights
//				for (k = 0; k < Layer[i].Input.length; k++) {
//					// Calculate weight difference between node j and k
//					Layer[i].Node[j].WeightDiff[k] = 
//						LearningRate * 
//						Layer[i].Node[j].SignalError*Layer[i-1].Node[k].Output +
//						Momentum*Layer[i].Node[j].WeightDiff[k];
//
//					// Update weight between node j and k
//					Layer[i].Node[j].Weight[k] = 
//						Layer[i].Node[j].Weight[k] + 
//						Layer[i].Node[j].WeightDiff[k];
//				}
//			}
//		}
//	}
	
}

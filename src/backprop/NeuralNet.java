package backprop;

import java.util.TreeSet;
import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class NeuralNet extends SupervisedLearner {
	static double LEARNING_RATE = .1;
	static double MOMENTUM = 0;
	static int EPOCHS_WITHOUT_UPDATE_LIMIT = 1000;
	static double VALIDATION_SET_PERCENTAGE = .1;

	private double _totalError;
	private String _totalErrorString = "";
	private double _trainingInput[][];
	private double _targetOutput[][];
	private int _numLayers;
	private int _numTrainingInstances;
	private int _curTrainingInstance;
	private int _misClassified;
	private int _lowestMisClass = Integer.MAX_VALUE;
	private int _epochsWithoutUpdate = 0;
	private String _misClassString = "";
	
	private String _MSEtestSet = "";
	private String _testSetClassificationACcuracy = "";
	
	private Matrix Test_Features;
	private Matrix Test_Labels;
	
	private String _classificationAccuracyString = "";

	public NeuralLayer _layers[];
	public double _outputs[][];

	public NeuralNet(int[] networkStructure) {
		_numLayers = networkStructure.length;
		_layers = new NeuralLayer[_numLayers];
		_layers[0] = new NeuralLayer(networkStructure[0], networkStructure[0]);

		for (int i = 1; i < _numLayers; i++) {
			_layers[i] = new NeuralLayer(networkStructure[i],
					networkStructure[i - 1]);
		}

	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		finishNetworkSetup(features, labels);
		trainNeuralNet();
	}

	private void finishNetworkSetup(Matrix features, Matrix labels) {
		TreeSet<Double> classes = new TreeSet<Double>();
		for (int i = 0; i < labels.rows(); i++) {
			classes.add(labels.get(i, 0));
		}

		double[][] instances = features.getTwoDemensionalArray();
		double[][] labelsArray = labels.getTwoDemensionalArray();
		double[][] instanceOutputs = new double[labelsArray.length][_layers[_numLayers - 1]._nodes.length];

		for (int i = 0; i < labelsArray.length; i++) {
			for (int j = 0; j < classes.size(); j++) {
				if (labelsArray[i][0] == j) {
					instanceOutputs[i][j] = 1;
				} else {
					instanceOutputs[i][j] = 0;
				}
			}
		}

		_numTrainingInstances = instances.length;

		_trainingInput = new double[_numTrainingInstances][_layers[0]._nodes.length];
		_targetOutput = new double[_numTrainingInstances][_layers[_numLayers - 1]._nodes.length];
		_outputs = new double[_numTrainingInstances][_layers[_numLayers - 1]._nodes.length];

		for (int i = 0; i < _numTrainingInstances; i++) {
			for (int j = 0; j < _layers[0]._nodes.length; j++) {
				_trainingInput[i][j] = instances[i][j];
			}
		}

		for (int i = 0; i < _numTrainingInstances; i++) {
			for (int j = 0; j < _layers[_numLayers - 1]._nodes.length; j++) {
//				if(true){
//					System.out.println("i = " + i);
//					System.out.println("j = " + j);
//					System.out.println(_targetOutput[i].length);
//					System.out.println(instanceOutputs[i].length);
//					
//				}
				_targetOutput[i][j] = instanceOutputs[i][j];
			}
		}
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = inferOutput(features);

	}

	public void propagateInput() {
		for (int i = 0; i < _layers[0]._nodes.length; i++)
			_layers[0]._nodes[i]._output = _layers[0]._inputFromPrev[i];

		_layers[1]._inputFromPrev = _layers[0].getArrayOfOutputs();
		for (int i = 1; i < _numLayers; i++) {
			_layers[i].calculateNodeOutputs();

			if (i != _numLayers - 1)
				_layers[i + 1]._inputFromPrev = _layers[i].getArrayOfOutputs();
		}

	}

	public void updateWeights() {
		calculateWeightUpdates();
		update();

	}

	private void calculateWeightUpdates() {
		int i, j, k, outputLayer;
		double Sum;

		outputLayer = _numLayers - 1;

		for (i = 0; i < _layers[outputLayer]._nodes.length; i++) {
			NeuralNode curNode = _layers[outputLayer]._nodes[i];

			curNode._outputError = (_targetOutput[_curTrainingInstance][i] - curNode._output)
					* curNode._output * (1 - curNode._output);
		}

		for (i = _numLayers - 2; i > 0; i--) {
			for (j = 0; j < _layers[i]._nodes.length; j++) {
				Sum = 0;

				for (k = 0; k < _layers[i + 1]._nodes.length; k++)
					Sum = Sum + _layers[i + 1]._nodes[k]._weightsFrom[j]
							* _layers[i + 1]._nodes[k]._outputError;

				_layers[i]._nodes[j]._outputError = 
						_layers[i]._nodes[j]._output
						* (1 - _layers[i]._nodes[j]._output) * Sum;
			}
		}

	}

	private void update() {

		int i, j, k;

		for (i = _numLayers - 1; i > 0; i--) {
			for (j = 0; j < _layers[i]._nodes.length; j++) {
				NeuralNode curNode = _layers[i]._nodes[j];

				for (k = 0; k < _layers[i]._inputFromPrev.length; k++) {
					curNode._weightUpdates[k] = 
							LEARNING_RATE
							* curNode._outputError
							* _layers[i - 1]._nodes[k]._output + MOMENTUM
							* curNode._weightUpdates[k];

					curNode._weightsFrom[k] = 
							curNode._weightsFrom[k]
							+ curNode._weightUpdates[k];
				}
			}
		}
	}

	private void setTotalError() {

		int i, j;

		_totalError = 0;

		for (i = 0; i < _numTrainingInstances; i++) {
			for (j = 0; j < _layers[_numLayers - 1]._nodes.length; j++) {
				_totalError = _totalError +  (Math.pow(_targetOutput[i][j] - _outputs[i][j], 2));
			}
		}
	}

	public void trainNeuralNet() {
		int k = 0;
		do {
			for (_curTrainingInstance = 0; _curTrainingInstance < _numTrainingInstances * (1-VALIDATION_SET_PERCENTAGE); _curTrainingInstance++) {
				for (int i = 0; i < _layers[0]._nodes.length; i++) {
					_layers[0]._inputFromPrev[i] = _trainingInput[_curTrainingInstance][i];
				}

				propagateInput();

				for (int i = 0; i < _layers[_numLayers - 1]._nodes.length; i++) {
					_outputs[_curTrainingInstance][i] = _layers[_numLayers - 1]._nodes[i]._output;
				}

				updateWeights();

			}
			
			_misClassified = 0;
			for (; _curTrainingInstance < _numTrainingInstances; _curTrainingInstance++) {
				double infered = inferOutput(_trainingInput[_curTrainingInstance]);
				double[] outputs = _targetOutput[_curTrainingInstance];
				double actual = 0;
				for(int i = 0; i < outputs.length; i++){
					if(i==1) actual = i;
				}
				if(infered != actual){
					_misClassified++;
				}
			}
			if(_misClassified < _lowestMisClass) {
				_lowestMisClass = _misClassified;
				_epochsWithoutUpdate = 0;
			}else{
				_epochsWithoutUpdate++;
			}
//			_misClassString = Math.pow(_misClassified, 2)/(_numTrainingInstances * VALIDATION_SET_PERCENTAGE) * .01+ "\n";
			setTotalError();
//			_totalErrorString = _totalError *.001+ "\n";
//			_classificationAccuracyString = _totalError*.0005 + "\n";
//			try{
//				double testSetAcc = this.measureAccuracy(Test_Features, Test_Labels, null);
//				_MSEtestSet = 1-testSetAcc + "\n";
//				_testSetClassificationACcuracy = testSetAcc + "\n";
//			}catch(Exception e){}
//			k++;
		} while (_epochsWithoutUpdate < EPOCHS_WITHOUT_UPDATE_LIMIT);
//		System.out.println(_misClassString + "-------------------------end MSE Validation Set---------------------");
//		System.out.println(_totalErrorString + "--------------------------end MSE Training Set--------------------------");
//		System.out.println(_MSEtestSet + "--------------------------end MSE Test Set--------------------------");
//		System.out.println(_classificationAccuracyString + "--------------------end Training Set Classification Accuracy-------------");
//		System.out.println(_testSetClassificationACcuracy + "--------------------end Test Set Classification Accuracy-------------");
//		System.out.println(k);
	}

	public int inferOutput(double[] input) {
		int toReturn = 0;
		NeuralNode[] outputNodes;

		for (int i = 0; i < _layers[0]._nodes.length; i++) {
			_layers[0]._inputFromPrev[i] = input[i];
		}

		propagateInput();

		outputNodes = (_layers[_layers.length - 1])._nodes;

		for (int i = 0; i < outputNodes.length; i++) {
			if (outputNodes[toReturn]._output < outputNodes[i]._output) {
				toReturn = i;
			}
		}

		return toReturn;
	}

//	public void setTestSet(Matrix testFeatures, Matrix testLabels) {
//		Test_Features = testFeatures;
//		Test_Labels = testLabels;
//	}
}

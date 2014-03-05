package backprop;

public class NeuralLayer {
	public double _inputFromPrev[];
	public NeuralNode _nodes[];
	
	public NeuralLayer(int NumberOfNodes, int NumberOfInputs) {
		_nodes = new NeuralNode[NumberOfNodes];

		for (int i = 0; i < NumberOfNodes; i++)
			_nodes[i] = new NeuralNode(NumberOfInputs);

		_inputFromPrev = new double[NumberOfInputs];
	}
	
	public void calculateNodeOutputs() {
		for (int i = 0; i < _nodes.length; i++) {
			double net = 0;

			for (int j = 0; j < _nodes[i]._weightsFrom.length; j++) {
				net = net + _inputFromPrev[j] * _nodes[i]._weightsFrom[j];
			}

			_nodes[i]._output = sigmoidFunction(net);
		}
	}

	private double sigmoidFunction(double Net) {
		return 1 / (1 + Math.exp(-Net));
	}

	public double[] getArrayOfOutputs() {

		double array[];

		array = new double[_nodes.length];

		for (int i = 0; i < _nodes.length; i++) {
			array[i] = _nodes[i]._output;
		}

		return array;
	}
}

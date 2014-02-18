package backprop;

public class NeuralLayer {
	private double _net;
	public double _inputFromPrev[];
	public NeuralNode _nodes[];

	public void calculateInstance() {
		for (int i = 0; i < _nodes.length; i++) {
			_net = _nodes[i]._bias;

			for (int j = 0; j < _nodes[i]._weightsFrom.length; j++) {
				_net = _net + _inputFromPrev[j] * _nodes[i]._weightsFrom[j];
			}

			_nodes[i]._output = sigmoidFunction(_net);
		}
	}

	private double sigmoidFunction(double Net) {
		return 1 / (1 + Math.exp(-Net));
	}

	public double[] resultArray() {

		double array[];

		array = new double[_nodes.length];

		for (int i = 0; i < _nodes.length; i++) {
			array[i] = _nodes[i]._output;
		}

		return array;
	}

	public NeuralLayer(int NumberOfNodes, int NumberOfInputs) {
		_nodes = new NeuralNode[NumberOfNodes];

		for (int i = 0; i < NumberOfNodes; i++)
			_nodes[i] = new NeuralNode(NumberOfInputs);

		_inputFromPrev = new double[NumberOfInputs];
	}

	public NeuralNode[] get_nodes() {
		return _nodes;
	}
}

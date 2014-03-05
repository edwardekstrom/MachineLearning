package backprop;

public class NeuralNode {
	public double _output;
	public double _weightsFrom[];
	public double _weightUpdates[];
	public double _outputError;
	
	public NeuralNode(int inputNodes) {
		_weightsFrom = new double[inputNodes];

		_weightUpdates = new double[inputNodes];

		initialiseWeights();
	}

	private void initialiseWeights() {

		for (int i = 0; i < _weightsFrom.length; i++) {
			_weightsFrom[i] = -1 + 2 * Math.random();

			_weightUpdates[i] = 0;
		}
	}
}

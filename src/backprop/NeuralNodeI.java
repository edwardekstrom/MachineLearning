package backprop;

import java.util.HashMap;
import java.util.Map;

public class NeuralNodeI implements NeuralNode {
	
	private double _output;
	private double _net;
	private Map<NeuralNode, Double> _weightsMap;
	
	
	
	
	public NeuralNodeI() {
		_weightsMap = new HashMap<NeuralNode, Double>();
	}
	
	@Override
	public double getOutput() {
		return _output;
	}

	@Override
	public double getNet() {
		return _net;
	}

	@Override
	public double getWeightTo(NeuralNode node) {
		return _weightsMap.get(node);
	}

	@Override
	public void setWeigthTo(NeuralNode node, double newWeight) {
		_weightsMap.put(node, newWeight);
	}

	@Override
	public void setOutput(double out) {
		_output = out;
	}

	@Override
	public void setNet(double net) {
		_net = net;
	}

}

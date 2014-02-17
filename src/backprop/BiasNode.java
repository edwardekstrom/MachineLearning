package backprop;

import java.util.HashMap;
import java.util.Map;

public class BiasNode implements NeuralNode {
	
	private Map<NeuralNode, Double> _weightsMap;
	
	public BiasNode(){
		_weightsMap = new HashMap<NeuralNode, Double>();
	}
	
	@Override
	public double getOutput() {
		return 1;
	}

	@Override
	public void setOutput(double out) {
	}

	@Override
	public double getNet() {
		return 1;
	}

	@Override
	public void setNet(double net) {
	}

	@Override
	public double getWeightTo(NeuralNode node) {
		return _weightsMap.get(node);
	}

	@Override
	public void setWeigthTo(NeuralNode node, double newWeight) {
		_weightsMap.put(node, newWeight);
	}

}

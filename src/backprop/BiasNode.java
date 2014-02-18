package backprop;

import java.util.HashMap;
import java.util.Map;

public class BiasNode implements NeuralNode {
	
	private Map<NeuralNode, Double> _weightsMap;
	private Map<NeuralNode, Double> _toUpdate;
	private double _delta;
	
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
//-----------------------------------------

	@Override
	public void setToUpdate(NeuralNode node, double updateValue) {
		_toUpdate.put(node, updateValue);
	}

	@Override
	public void setDelta(double d) {
		_delta = d;
	}

	@Override
	public double getDelta() {
		return _delta;
	}

	@Override
	public double getDeltaSum() {
		double sum = 0;
		for(NeuralNode nn : _weightsMap.keySet()){
			sum += nn.getDelta() * _weightsMap.get(nn);
		}
		return sum;
	}

	@Override
	public void update() {
		for(NeuralNode nn: _weightsMap.keySet()){
			double curWeight = _weightsMap.get(nn);
			_weightsMap.put(nn, curWeight + _toUpdate.get(nn));
		}
	}
}

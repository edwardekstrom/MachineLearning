package backprop;

import java.util.Random;

public class NeuralLayerI implements NeuralLayer {
	
	private NeuralNode[] _nodes;
	
	private Random _rand;
	
	public boolean _isOutputLayer;
	
	public NeuralLayerI(int numNodes, Random rand, boolean isOutputLayer) {
		_rand = rand;
		
		_nodes = new NeuralNodeI[numNodes];
		
		_isOutputLayer = isOutputLayer;
		
		for(int i = 0; i < numNodes; i++){
			_nodes[i] = new NeuralNodeI();
		}
		
		if(!isOutputLayer){
			NeuralNodeI node = new NeuralNodeI();
			node.setOutput(1);
			node.setNet(1);
			_nodes[numNodes - 1] = node;
		}
	}

	@Override
	public NeuralNode[] getNodes() {
		return _nodes;
	}

	@Override
	public void connectToLayer(NeuralLayerI nl) {
		NeuralNode[] nextLayerNodes = nl.getNodes();
		
		if (nl._isOutputLayer) {
			
			for (int i = 0; i < _nodes.length; i++) {
				for (int j = 0; j < nextLayerNodes.length; j++) {
					double nextRand = _rand.nextDouble();// * (-1^_rand.nextInt(2));
					_nodes[i].setWeigthTo(nextLayerNodes[j], nextRand);
				}
			}
			
		}else{
			
			for (int i = 0; i < _nodes.length; i++) {
				for (int j = 0; j < nextLayerNodes.length - 1; j++) {
					double nextRand = _rand.nextDouble();// * (-1^_rand.nextInt(2));
					_nodes[i].setWeigthTo(nextLayerNodes[j], nextRand);
				}
			}
			
		}
	}

}
